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
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from segpose import SegPoseNet, UNet
from segpose.criterion import PoseCriterion
from segpose.dataset import EvMaskPoseDataset
from segpose.params import Params
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
        for X, y, pose in loader:
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


def train(segpose, params):

    # Create Train and Val DataLoaders
    if not params.train_dir:
        raise FileNotFoundError(
            "Training directory has not been set using cli or params"
        )

    datasets = [
        EvMaskPoseDataset(dir_num + 1, params)
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

    # Criterions
    if unet.n_classes > 1:
        unet_criterion = nn.CrossEntropyLoss()
    else:
        unet_criterion = nn.BCEWithLogitsLoss()
    pose_criterion = PoseCriterion(sax=0.0, saq=params.beta, srx=0.0, srq=params.gamma)

    # Optimizer
    unet_params = segpose.unet.parameters()
    unet_optimizer = params.unet_optimizer(
        unet_params,
        lr=params.unet_learning_rate,
        weight_decay=params.unet_weight_decay,
        momentum=params.unet_momentum,
    )
    if not params.train_unet:
        for p in unet_params:
            p.requires_grad = False

    pose_parameters = list(segpose.parameters()) + list(pose_criterion.parameters())

    pose_optimizer = params.pose_optimizer(
        params=pose_parameters,
        lr=params.pose_learning_rate,
        weight_decay=params.pose_weight_decay,
    )
    if not params.train_pose:
        for p in pose_params:
            p.requires_grad = False

    writer = SummaryWriter(
        comment=f"LR_{params.pose_learning_rate}_EPOCHS_{params.epochs}_BS_{params.batch_size}_IMGSIZE_{params.img_size}"
    )

    iters = []
    train_losses = []
    val_losses = []

    step = 0
    min_loss = np.inf

    segpose.train()
    for epoch in range(params.epochs):
        # logging.info(f"Starting epoch: {epoch+1}")
        with tqdm(
            total=train_size, desc=f"Epoch {epoch+1}/{params.epochs}", unit="img"
        ) as pbar:

            for i, (ev_frame, mask_gt, pose) in enumerate(train_loader):

                event_frame = Variable(ev_frame).cuda()
                mask_gt = Variable(mask_gt).cuda()

                # Casting variables to float
                ev_frame = ev_frame.to(device=device, dtype=torch.float)
                mask_gt = mask_gt.to(device=device, dtype=torch.float)

                R_gt, T_gt, pose_gt = pose.values()

                mask_pred, pose_pred = segpose(ev_frame)

                if (
                    step % (train_size // (10 * params.batch_size)) == 0
                    and params.train_unet
                ):
                    mask_pred_m = (
                        torch.sigmoid(output.detach().requires_grad_())
                        > params.threshold_conf
                    ).type(torch.uint8) * 255
                    logging.info(f"unet output shape: {output_m.shape}")
                    mask_pred, mesh_losses = MeshDeformationModel(
                        device
                    ).run_optimization(mask_pred_m, R, T, writer, step)
                    mask_pred = mask_pred.cuda()
                    logging.info(f"mesh defo shape: {output.shape}")
                    writer.add_images("masks-pred-mesh-deform", output, step)

                unet_loss = unet_criterion(mask_pred.requires_grad_(), mask_gt)
                pose_loss = pose_criterion(pose_pred, pose_gt)
                loss = unet_loss + pose_loss

                writer.add_scalar("UNetLoss/Train", pose_loss.item(), step)
                writer.add_scalar("PoseLoss/Train", pose_loss.item(), step)
                writer.add_scalar("Loss/Train:", loss.item(), step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                unet_optimizer.zero_grad()
                pose_optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_value_(segpose.unet.parameters(), 0.1)
                # nn.utils.clip_grad_norm(pose_params, )

                unet_optimizer.step()
                pose_optimizer.step()

                pbar.update(ev_frame.shape[0])

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
                        "unet_learning_rate", unet_optimizer.param_groups[0]["lr"], step
                    )
                    writer.add_scalar(
                        "pose_learning_rate", pose_optimizer.param_groups[0]["lr"], step
                    )

                    segpose.eval()
                    val_loss = eval_seg_net(segpose.unet, val_loader)
                    segpose.train()
                    val_losses.append(val_loss)

                    writer.add_scalar("DiceCoeff: ", val_loss, step)

                    writer.add_images("images", ev_frame, step)
                    writer.add_images("masks-gt", mask_gt, step)
                    writer.add_images(
                        "masks-pred",
                        torch.sigmoid(mask_pred) > params.threshold_conf,
                        step,
                    )

    torch.save(
        {"model": unet.state_dict(), "optimizer": optimizer.state_dict()},
        "epochs15_batch4_end_dolphin.cpt",
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set the device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device == "cuda:1":
        torch.cuda.set_device()

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)
    params.device = device

    logging.info(f"Using {device} as computation device")

    try:
        unet = UNet.load(params)
        segpose_model = SegPoseNet(unet, params)
        logging.info(segpose_model)
        logging.info(f"Loaded UNet Model from {params.model_cpt}- Starting training")
        train(segpose_model, params)
    except KeyboardInterrupt:
        interrupted_path = "Interrupted.pth"
        logging.info(f"Received Interrupt, saving model at {interrupted_path}")
        torch.save({"model": segpose_model.state_dict()}, interrupted_path)
