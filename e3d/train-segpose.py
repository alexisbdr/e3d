import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import logging
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from pytorch3d.renderer import SfMPerspectiveCameras
from segpose import SegPoseNet, UNet
from segpose.criterion import PoseCriterion, PoseCriterionRel
from segpose.dataset import (ConcatDataSampler, EvMaskPoseBatchedDataset,
                             EvMaskPoseDataset)
from segpose.params import Params
from torch import optim
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.manager import RenderManager
from utils.visualization import plot_camera_scene


def get_args():

    parser = argparse.ArgumentParser(
        description="EvUnet Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name", metavar="NAME", type=str, help="Model Name", dest="name",
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
        "-m",
        "--model",
        dest="segpose_model_cpt",
        type=str,
        default=Params.segpose_model_cpt,
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
    parser.add_argument(
        "--gpu", dest="gpu_num", default="0", type=str, help="GPU Device Number",
    )
    parser.add_argument(
        "-t",
        "--train-dir",
        dest="train_dir",
        type=str,
        default=Params.train_dir,
        help="Path to prediction directory",
    )

    return parser.parse_args()


def plot_cams_from_poses(pose_gt, pose_pred, device: str):
    """
    """
    R_gt, T_gt = pose_gt
    R_pred, T_pred = pose_pred

    cameras_gt = SfMPerspectiveCameras(device=device, R=R_gt, T=T_gt)
    cameras_pred = SfMPerspectiveCameras(device=device, R=R_pred, T=T_pred)

    fig = plot_camera_scene(cameras_pred, cameras_gt, "final_preds", params.device)

    return fig


def eval_seg_net(net, loader):
    """
    Evaluation approach for the segmentation net
    Uses the DiceCoefficient loss defined in losses.py
    """

    seg_loss, pose_loss = 0, 0
    with tqdm(total=len(loader), desc="Validation", unit="batch", leave=False) as pbar:
        for X, y, R, T, pose in loader:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            X = X.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            with torch.no_grad():
                out, pose_pred = net(X)
            y = y.view(-1, *y.size()[2:]).unsqueeze(1)
            out = torch.sigmoid(out)
            out = (out > 0.5).float()
            seg_loss += DiceCoeffLoss().forward(out, y)
            # pose_loss += PoseCriterion(sax=0.0, saq=params.beta, srx=0.0, srq=params.gamma)

            pbar.update()

    loss = seg_loss / len(loader)
    return loss


def train(segpose, params):

    # Create Train and Val DataLoaders
    if not params.train_dir:
        raise FileNotFoundError(
            "Training directory has not been set using cli or params"
        )

    datasets = []
    for dir_num in range(len(os.listdir(params.train_dir)) - 1):
        if params.train_pose and params.train_relative and False:
            dataset = EvMaskPoseBatchedDataset(
                params.mini_batch_size, dir_num + 1, params
            )
        else:
            dataset = EvMaskPoseDataset(dir_num + 1, params)
        if dataset.render_manager is not None:
            datasets.append(dataset)

    # Train, Val split according to datasets
    val_size = int(len(datasets) * params.val_split)
    train_size = len(datasets) - val_size
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
    if unet.n_classes > 1:
        unet_criterion = nn.CrossEntropyLoss()
    else:
        unet_criterion = nn.BCEWithLogitsLoss()

    if params.train_relative:
        pose_criterion = PoseCriterionRel(
            sax=0.0, saq=params.beta, srx=0.0, srq=params.gamma
        ).to(device)
    else:
        pose_criterion = PoseCriterion(sax=0.0, saq=params.beta, learn_beta=True).to(
            device
        )

    # Optimizer
    unet_params = segpose.unet.parameters()
    unet_optimizer = params.unet_optimizer(
        unet_params,
        lr=params.unet_learning_rate,
        weight_decay=params.unet_weight_decay,
        momentum=params.unet_momentum,
    )
    if params.segpose_model_cpt:
        cpt = torch.load(params.segpose_model_cpt)
        unet_optimizer.load_state_dict(cpt["unet_optimizer"])
    if not params.train_unet:
        for p in unet_params:
            p.requires_grad = False

    pose_parameters = list(segpose.parameters()) + list(pose_criterion.parameters())

    pose_optimizer = params.pose_optimizer(
        params=pose_parameters,
        lr=params.pose_learning_rate,
        weight_decay=params.pose_weight_decay,
    )
    if params.segpose_model_cpt:
        pose_optimizer.load_state_dict(cpt["pose_optimizer"])
    if not params.train_pose:
        for p in pose_parameters:
            p.requires_grad = False

    writer = SummaryWriter(
        comment=f"{params.name}_LR_{params.pose_learning_rate}_EPOCHS_{params.epochs}_BS_{params.batch_size}_IMGSIZE_{params.img_size}"
    )

    iters = []
    train_losses = []
    val_losses = []
    fine_tuning = (
        params.fine_tuning
    )  # True if we're running fine-tuning @ this step --> loss calc
    prev = defaultdict(list)

    step = 0
    min_loss = np.inf

    segpose.train()
    for epoch in range(params.epochs):
        # logging.info(f"Starting epoch: {epoch+1}")
        with tqdm(
            total=train_size, desc=f"Epoch {epoch+1}/{params.epochs}", unit="img"
        ) as pbar:

            for i, (ev_frame, mask_gt, R_gt, T_gt, pose_gt) in enumerate(train_loader):

                event_frame = Variable(ev_frame).to(params.device)
                mask_gt = Variable(mask_gt).to(params.device)
                pose_gt = Variable(pose_gt).to(params.device)

                # Casting variables to float
                ev_frame = ev_frame.to(device=device, dtype=torch.float)
                mask_gt = mask_gt.to(device=device, dtype=torch.float)

                mask_pred, pose_pred = segpose(ev_frame)

                if params.fine_tuning and step % (
                    train_size // (10 * params.batch_size)
                ) in [2, 3]:
                    prev["R_gt"].append(R_gt)
                    prev["T_gt"].append(T_gt)
                    prev["mask_pred"].append(mask_pred)

                    # Fine-tuning through Differentiable Renderer
                    if step % (train_size // (10 * params.batch_size)) == 3:
                        fine_tuning = True
                        # Concatenate results from previous step
                        R_gt = torch.cat(prev["R_gt"])
                        T_gt = torch.cat(prev["T_gt"])
                        mask_pred_m = torch.cat(prev["mask_pred"])

                        mask_pred_m = (
                            torch.sigmoid(mask_pred).squeeze() > params.threshold_conf
                        ).type(torch.uint8) * 255
                        writer.add_images(
                            "mask-pred-input", mask_pred_m.unsqueeze(1), step
                        )
                        # R_gt_m = R_gt.view(-1, *R_gt.size()[2:]).unsqueeze(1)
                        # T_gt_m = T_gt.view(-1, *T_gt.size()[2:]).unsqueeze(1)
                        logging.info(
                            f"unet output shape: {mask_pred_m.shape}, R shape {R_gt.shape}"
                        )
                        mesh_model = MeshDeformationModel(device)
                        mesh_losses = mesh_model.run_optimization(
                            mask_pred_m, R_gt, T_gt, writer, step
                        )
                        renders = mesh_model.render_final_mesh(
                            (R_gt, T_gt), "predict", mask_pred_m.shape[-2:]
                        )
                        mask_pred_m = renders["silhouettes"].to(device)
                        image_pred = renders["images"].to(device)

                        logging.info(f"mesh defo shape: {image_pred.shape}")
                        writer.add_images("masks-pred-mesh-deform", mask_pred_m, step)
                        writer.add_images(
                            "images-pred-mesh-deform",
                            image_pred.permute(0, 3, 1, 2),
                            step,
                        )
                        # Cut out batch_size from mask_pred for calculating loss
                        mask_pred_m = mask_pred_m[: ev_frame.shape[0]].requires_grad_()

                        prev = defaultdict(list)

                mask_gt = mask_gt.view(-1, *mask_gt.size()[2:]).unsqueeze(1)

                # Compute losses
                unet_loss = unet_criterion(mask_pred, mask_gt)
                if params.unet_end2end and fine_tuning:
                    unet_loss += IOULoss().forward(mask_pred_m, mask_gt) * 1.2
                    fine_tuning = False
                if params.train_relative:
                    pose_loss = pose_criterion(pose_pred, pose_gt)
                else:
                    pose_loss = pose_criterion(pose_pred.squeeze(1), pose_gt.squeeze(1))
                loss = pose_loss + unet_loss

                writer.add_scalar("UNetLoss/Train", unet_loss.item(), step)
                writer.add_scalar("PoseLoss/Train", pose_loss.item(), step)
                writer.add_scalar("CompleteLoss/Train:", loss.item(), step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                unet_optimizer.zero_grad()
                pose_optimizer.zero_grad()

                # if params.train_unet: unet_loss.backward()
                # if params.train_pose: pose_loss.backward()
                loss.backward()

                nn.utils.clip_grad_value_(segpose.unet.parameters(), 0.1)
                # nn.utils.clip_grad_norm(pose_params, )

                unet_optimizer.step()
                pose_optimizer.step()

                pbar.update(ev_frame.shape[0])

                # Evaluation

                if (
                    step % (train_size // (10 * params.batch_size)) == 0
                    and params.train_unet
                ):

                    """
                    if params.train_unet:
                        for tag, value in unet.named_parameters():
                            tag = tag.replace(".", "/")
                            writer.add_histogram(
                                "weights/" + tag, value.data.cpu().numpy(), step
                            )
                            writer.add_histogram(
                                "grads/" + tag, value.grad.data.cpu().numpy(), step
                            )
                    """
                    iters.append(i)
                    train_losses.append(
                        [
                            pose_loss if params.train_pose else 0.0,
                            unet_loss if params.train_unet else 0.0,
                        ]
                    )

                    writer.add_scalar(
                        "unet_learning_rate", unet_optimizer.param_groups[0]["lr"], step
                    )
                    writer.add_scalar(
                        "pose_learning_rate", pose_optimizer.param_groups[0]["lr"], step
                    )

                    segpose.eval()
                    val_loss = eval_seg_net(segpose, val_loader)
                    segpose.train()
                    val_losses.append(val_loss)

                    # Plot cam pose

                    # q_pred = qexp(pose_pred.squeeze(0)[:, 3:])
                    # R_pred = rc.quaternion_to_matrix(q_pred).unsqueeze(0)

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

                # End of loop -> increase step
                step += 1

    torch.save(
        {
            "model": segpose.state_dict(),
            "unet_optimizer": unet_optimizer.state_dict(),
            "pose_optimizer": pose_optimizer.state_dict(),
        },
        f"{params.name}_epochs{params.epochs}_batch{params.batch_size}_minibatch{params.mini_batch_size}.cpt",
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)

    # Set the device
    dev_num = params.gpu_num
    device = torch.device(f"cuda:{dev_num}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device} as computation device")
    if device == f"cuda:{dev_num}":
        torch.cuda.set_device()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = dev_num
    logging.info(f"Using {device} as computation device")
    params.device = device
    logging.info(
        f"Training {'POSE' if params.train_pose else ''} and {'UNET' if params.train_unet else ''}"
    )

    try:
        unet = UNet.load(params)
        segpose_model = SegPoseNet.load(unet, params)
        logging.info(segpose_model)
        logging.info(
            f"Loaded UNet Model from {params.segpose_model_cpt}- Starting training"
        )
        train(segpose_model, params)
    except (KeyboardInterrupt, SystemExit):
        interrupted_path = f"{params.name}_interrupted.pth"
        logging.info(f"Received Interrupt, saving model at {interrupted_path}")
        torch.save({"model": segpose_model.state_dict()}, interrupted_path)
    except Exception as e:
        logging.info("Received Error: ", e)
        error_path = f"{params.name}_errored.pth"
        torch.save({"model": segpose_model.state_dict()}, error_path)
