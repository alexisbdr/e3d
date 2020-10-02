import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from losses import DiceCoeffLoss
from PIL import Image
from segmentation import UNet
from segmentation.dataset import EvMaskDataset
from segmentation.params import Params
from torch import optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.manager import RenderManager


def get_args():

    parser = argparse.ArgumentParser(
        description="EvUnet Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=Params.epochs,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=Params.batch_size,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=Params.learning_rate,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="model_cpt",
        type=str,
        default=Params.model_cpt,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "--img_size",
        dest="img_size",
        type=float,
        default=Params.img_size,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val_split",
        type=float,
        default=Params.val_split,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        type=str,
        default=Params.config_file,
        help="Load Params dict from config file, file should be in cfg format",
    )

    return parser.parse_args()


def eval_seg_net(net, loader):
    """
    Evaluation approach for the segmentation net
    Uses the DiceCoefficient loss defined in losses.py
    """

    loss = 0
    with tqdm(total=len(loader), desc="Validation", unit="batch", leave=False) as pbar:
        for X, y in loader:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            X = X.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            with torch.no_grad():
                out = net(X)
            out = torch.sigmoid(out)
            out = (out > 0.5).float()
            loss += DiceCoeffLoss().forward(out, y)

            pbar.update()

    loss = loss / len(loader)
    return loss


def train(unet, params):

    # Create Train and Val DataLoaders
    if not params.train_dir:
        raise FileNotFoundError(
            "Training directory has not been set using cli or params"
        )
    datasets = [
        EvMaskDataset(dir_num + 1, params)
        for dir_num in range(len(os.listdir(params.train_dir)) - 1)
    ]
    dataset = ConcatDataset(datasets)

    val_size = int(len(dataset) * params.val_split)
    train_size = len(dataset) - val_size
    train, val = random_split(dataset, (train_size, val_size))

    train_loader = DataLoader(
        train, batch_size=params.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    val_loader = DataLoader(
        val, batch_size=params.batch_size, shuffle=False, num_workers=8, drop_last=True
    )

    optimizer = params.optimizer(
        unet.parameters(), lr=params.learning_rate, weight_decay=1e-8, momentum=0.9
    )

    # Or Maybe just use a cross entropy loss - need to eval this
    if unet.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(
        comment=f"LR_{params.learning_rate}_EPOCHS_{params.epochs}_BS_{params.batch_size}_IMGSIZE_{params.img_size}"
    )

    iters = []
    train_losses = []
    val_losses = []

    step = 0
    min_loss = np.inf

    unet.train()
    for epoch in range(params.epochs):
        # logging.info(f"Starting epoch: {epoch+1}")
        with tqdm(
            total=train_size, desc=f"Epoch {epoch+1}/{params.epochs}", unit="img"
        ) as pbar:

            for i, (X, y) in enumerate(train_loader):

                X = Variable(X).cuda()
                y = Variable(y).cuda()

                # Casting variables to float
                X = X.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)

                output = unet(X)
                loss = criterion(output, y)

                writer.add_scalar("Loss/Train:", loss.item(), step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(unet.parameters(), 0.1)
                optimizer.step()

                pbar.update(X.shape[0])

                step += 1
                if step % (train_size // (10 * params.batch_size)) == 0:

                    for tag, value in unet.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), step
                        )

                    iters.append(i)
                    train_losses.append(loss)

                    writer.add_scalar(
                        "learning_rate", optimizer.param_groups[0]["lr"], step
                    )

                    unet.eval()
                    val_loss = eval_seg_net(unet, val_loader)
                    unet.train()
                    val_losses.append(val_loss)

                    writer.add_scalar("DiceCoeff: ", val_loss, step)

                    writer.add_images("images", X, step)
                    writer.add_images("masks-gt", y, step)
                    writer.add_images("masks-pred", torch.sigmoid(output) > 0.5, step)

    torch.save(
        {"model": unet.state_dict(), "optimizer": optimizer.state_dict()},
        "dolphin_model_checkpoint.cpt",
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.cuda.set_device()

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)
    params.device = device

    logging.info(f"Using {device} as computation device")

    try:
        model = UNet.load(params)
        logging.info(f"Loaded UNet Model from {params.model_cpt}- Starting training")
        train(model, params)
    except KeyboardInterrupt:
        interrupted_path = "Interrupted.pth"
        logging.info(f"Received Interrupt, saving model at {interrupted_path}")
        torch.save({"model": model.state_dict()}, interrupted_path)
