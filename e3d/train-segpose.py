import configargparse
import os
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from segpose import UNet, UNetDynamic
from segpose.dataset import (ConcatDataSampler,
                             EvMaskPoseDataset, EvimoDataset)
from utils.params import Params
from torch import optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.manager import RenderManager
import json
from utils.visualization import plot_camera_scene

def get_args():
    parser = configargparse.ArgumentParser(
        description="EvUnet Training",
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-cfg",
        "--config",
        dest="config_file",
        type=str,
        default=Params.config_file,
        help="Load Params dict from config file, file should be in cfg format",
    )
    parser.add_argument(
        "--gpu", dest="gpu_num", default="0", type=str, help="GPU Device Number",
    )


    return parser.parse_args()


def plot_cams_from_poses(pose_gt, pose_pred, device: str):
    """
    """
    R_gt, T_gt = pose_gt
    R_pred, T_pred = pose_pred

    cameras_gt = PerspectiveCameras(device=device, R=R_gt, T=T_gt)
    cameras_pred = PerspectiveCameras(device=device, R=R_pred, T=T_pred)

    fig = plot_camera_scene(cameras_pred, cameras_gt, "final_preds", params.device)

    return fig


def eval_seg_net(net, loader):
    """
    Evaluation approach for the segmentation net
    Uses the DiceCoefficient loss defined in losses.py
    """

    seg_loss, pose_loss = 0, 0
    with tqdm(total=len(loader), desc="Validation", unit="batch", leave=False) as pbar:
        for X, y, R, T in loader:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            X = X.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            with torch.no_grad():
                out = net(X)
            y = y.view(-1, *y.size()[2:]).unsqueeze(1)
            out = torch.sigmoid(out)
            out = (out > 0.5).float()
            seg_loss += DiceCoeffLoss().forward(out, y)
            # pose_loss += PoseCriterion(sax=0.0, saq=params.beta, srx=0.0, srq=params.gamma)

            pbar.update()

    loss = seg_loss / len(loader)
    return loss


def train_evimo(model, params):
    if not params.train_dir:
        raise FileNotFoundError(
            "Training directory has not been set using cli or params"
        )
    print('Prepare training data......')
    datasets = []
    for directory in os.listdir(params.train_dir):
        dir_path = os.path.join(params.train_dir, directory)
        if not os.path.isdir(dir_path):
            continue
        datasets.append(EvimoDataset(dir_path, obj_id=params.evimo_obj_id, is_train=True, slice_name=params.slice_name))
        print(len(datasets[-1]))

    dataset_all = ConcatDataset(datasets)
    val_size = max(int(len(dataset_all) * params.val_split), 1)
    train_size = max(len(dataset_all) - val_size, 1)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_all, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=params.unet_batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.unet_batch_size, num_workers=8, shuffle=True)

    # Criterions
    if model.module.n_classes > 1:
        model_criterion = nn.CrossEntropyLoss()
    else:
        model_criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    model_params = model.module.parameters()
    model_optimizer = params.unet_optimizer(
        model_params,
        lr=params.unet_learning_rate,
        weight_decay=params.unet_weight_decay,
        momentum=params.unet_momentum,
    )
    if params.model_cpt:
        cpt = torch.load(params.model_cpt)
        model_optimizer.load_state_dict(cpt["unet_optimizer"])
    if not params.train_unet:
        for p in model_params:
            p.requires_grad = False

    os.makedirs(os.path.join(params.exper_dir, 'runs'), exist_ok=True)
    log_dir = os.path.join(params.exper_dir, f"runs/{params.name}_LR_{params.unet_learning_rate}_EPOCHS_{params.unet_epochs}_BS_{params.unet_batch_size}")

    writer = SummaryWriter(
        log_dir=log_dir
    )

    val_losses = []
    step = 0

    model.train()
    for epoch in range(params.unet_epochs):
        epoch_loss = 0.0
        with tqdm(
                total=train_size, desc=f"Epoch {epoch + 1}/{params.unet_epochs}", unit="img"
        ) as pbar:

            for i, (ev_frame, mask_gt, R_gt, T_gt) in enumerate(train_loader):
                event_frame = Variable(ev_frame).to(params.device)
                mask_gt = Variable(mask_gt).to(params.device)

                # Casting variables to float
                ev_frame = event_frame.to(device=device, dtype=torch.float)
                mask_gt = mask_gt.to(device=device, dtype=torch.float)

                mask_pred = model(ev_frame)

                mask_gt = mask_gt.view(-1, *mask_gt.size()[2:]).unsqueeze(1)

                # Compute losses
                loss = model_criterion(mask_pred, mask_gt)

                writer.add_scalar("CompleteLoss/Train:", loss.item(), step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                model_optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_value_(model.module.parameters(), 0.1)

                model_optimizer.step()

                pbar.update(ev_frame.shape[0])

                epoch_loss += loss

                step += 1

                # Evaluation

            writer.add_scalar("epoch loss", epoch_loss, epoch)
            model.eval()
            val_loss = eval_seg_net(model, val_loader)
            model.train()
            val_losses.append(val_loss)
            print(f'Epoch: {epoch} Train Loss: {epoch_loss.item()}')
            print(f'Epoch: {epoch}  DiceCoeff: {val_loss.item()}')

            writer.add_scalar("DiceCoeff: ", val_loss, step)

            writer.add_images(
                "event frame",
                ev_frame.view(-1, *ev_frame.size()[2:]).unsqueeze(1),
                step,
            )
            writer.add_images("masks-gt", mask_gt, step)
            writer.add_images(
                "masks-pred-probs", mask_pred, step,
            )
            writer.add_images(
                "masks-pred",
                torch.sigmoid(mask_pred) > params.threshold_conf,
                step,
            )
    writer.close()
    model_dir = os.path.join(params.exper_dir, f"{params.name}_epochs{params.unet_epochs}_batch{params.unet_batch_size}_minibatch{params.unet_mini_batch_size}.cpt")
    torch.save(
        {
            "model": model.module.state_dict(),
            "unet_optimizer": model_optimizer.state_dict(),
        },
        model_dir,
    )

def train_synth(model, params):

    # Create Train and Val DataLoaders
    if not params.train_dir:
        raise FileNotFoundError(
            "Training directory has not been set using cli or params"
        )
    print('Prepare training data......')
    datasets = []
    for dir_num in range(len(os.listdir(params.train_dir)) - 1):
        dataset = EvMaskPoseDataset(dir_num + 1, params)
        if dataset.render_manager is not None:
            datasets.append(dataset)

    # Train, Val split according to datasets
    val_size = int(len(datasets) * params.val_split)
    train_dataset = ConcatDataset(datasets[val_size:])
    val_dataset = ConcatDataset(datasets[:val_size])
    train_size = len(train_dataset)  # We use this to calculate other stuff later

    train_sampler = ConcatDataSampler(
        train_dataset, batch_size=params.batch_size, shuffle=True
    )
    val_sampler = ConcatDataSampler(
        val_dataset, batch_size=params.batch_size, shuffle=True
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8)

    # Criterions
    if model.module.n_classes > 1:
        model_criterion = nn.CrossEntropyLoss()
    else:
        model_criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    model_params = model.module.parameters()
    model_optimizer = params.unet_optimizer(
        model_params,
        lr=params.unet_learning_rate,
        weight_decay=params.unet_weight_decay,
        momentum=params.unet_momentum,
    )
    if params.model_cpt:
        cpt = torch.load(params.model_cpt)
        model_optimizer.load_state_dict(cpt["unet_optimizer"])
    if not params.train_unet:
        for p in model_params:
            p.requires_grad = False

    log_dir = os.path.join(params.exper_dir,
                           f"runs/{params.name}_LR_{params.unet_learning_rate}_EPOCHS_{params.unet_epochs}_BS_{params.unet_batch_size}")

    writer = SummaryWriter(
        log_dir=log_dir
    )

    val_losses = []

    step = 0

    model.train()
    for epoch in range(params.unet_epochs):
        # logging.info(f"Starting epoch: {epoch+1}")
        epoch_loss = 0.0
        with tqdm(
            total=train_size, desc=f"Epoch {epoch+1}/{params.unet_epochs}", unit="img"
        ) as pbar:

            for i, (ev_frame, mask_gt, R_gt, T_gt) in enumerate(train_loader):

                event_frame = Variable(ev_frame).to(params.device)
                mask_gt = Variable(mask_gt).to(params.device)

                # Casting variables to float
                ev_frame = event_frame.to(device=device, dtype=torch.float)
                mask_gt = mask_gt.to(device=device, dtype=torch.float)

                mask_pred = model(ev_frame)

                mask_gt = mask_gt.view(-1, *mask_gt.size()[2:]).unsqueeze(1)

                # Compute losses
                loss = model_criterion(mask_pred, mask_gt)

                writer.add_scalar("CompleteLoss/Train:", loss.item(), step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                model_optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_value_(model.module.parameters(), 0.1)

                model_optimizer.step()

                pbar.update(ev_frame.shape[0])

                epoch_loss += loss

                step += 1

                # Evaluation

            writer.add_scalar("epoch loss", epoch_loss, epoch)
            model.eval()
            val_loss = eval_seg_net(model, val_loader)
            model.train()
            print(f'Epoch: {epoch} Train Loss: {epoch_loss.item()}')
            print(f'Epoch: {epoch}  DiceCoeff Loss: {val_loss.item()}')
            val_losses.append(val_loss)

            writer.add_scalar("DiceCoeff: ", val_loss, step)

            writer.add_images(
                "event frame",
                ev_frame.view(-1, *ev_frame.size()[2:]).unsqueeze(1),
                step,
            )
            writer.add_images("masks-gt", mask_gt, step)
            writer.add_images(
                "masks-pred-probs", mask_pred, step,
            )
            writer.add_images(
                "masks-pred",
                torch.sigmoid(mask_pred) > params.threshold_conf,
                step,
            )

    model_dir = os.path.join(params.exper_dir,
                             f"{params.name}_epochs{params.unet_epochs}_batch{params.unet_batch_size}_minibatch{params.unet_mini_batch_size}.cpt")
    torch.save(
        {
            "model": model.module.state_dict(),
            "unet_optimizer": model_optimizer.state_dict(),
        },
        model_dir,
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()
    args_dict = vars(args)
    params = Params()
    params.config_file = args_dict['config_file']
    params.__post_init__()

    params.gpu_num = args_dict['gpu_num']
    # Set the device
    dev_num = params.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = dev_num
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} as computation device")

    if device == f"cuda":
        torch.cuda.set_device()
    logging.info(f"Using {device} as computation device")
    params.device = device
    logging.info(
        f"Training {'Unet' if params.train_unet else 'Nothing'}"
    )

    try:
        unet = UNetDynamic.load(params)
        unet = nn.DataParallel(unet).to(device)
        logging.info(unet)
        logging.info(
            f"Loaded UNet Model from {params.model_cpt}- Starting training"
        )
        os.makedirs(params.exper_dir, exist_ok=True)
        with open(os.path.join(params.exper_dir, 'config.json'), 'w') as file:
            file.write(json.dumps(params.as_dict()))
        if params.is_real_data:
            train_evimo(unet, params)
        else:
            train_synth(unet, params)

    except (KeyboardInterrupt, SystemExit):
        interrupted_path = os.path.join(params.exper_dir,
                                 f"Interrupt_{params.name}_epochs{params.unet_epochs}_batch{params.unet_batch_size}_minibatch{params.unet_mini_batch_size}.cpt")
        logging.info(f"Received Interrupt, saving model at {interrupted_path}")
        torch.save(
            {
                "model": unet.state_dict(),
            },
            interrupted_path)
    except Exception as e:
        logging.info("Received Error: ", e)
        # error_path = f"{params.name}_errored.pth"
        # torch.save({"model": unet.state_dict()}, error_path)
