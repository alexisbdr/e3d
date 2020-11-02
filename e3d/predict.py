import argparse
import logging
import os
import sys
from os.path import abspath, join

import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from segpose import SegPoseNet, UNet
from segpose.dataset import EvMaskPoseDataset
from segpose.params import Params
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils.manager import RenderManager
from utils.pose_utils import qexp, qlog, quaternion_angular_error
from utils.visualization import plot_img_and_mask


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6

    return 1.0 - (intersect / union).sum() / intersect.nelement()


def t_error(predict, target):
    return torch.norm(predict - target)


def process_rotation(R):
    q = rc.matrix_to_quaternion(R)
    q = torch.sign(q[0])
    return qlog(q)


def get_args():

    parser = argparse.ArgumentParser(
        description="EvUnet Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "-p",
        "--path",
        dest="pred_dir",
        type=str,
        default=Params.pred_dir,
        help="Path to prediction directory",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold_conf",
        default=Params.threshold_conf,
        help="Probability threshold for masks",
    )

    return parser.parse_args()


def predict_mesh(mesh_model: MeshDeformationModel):

    raise NotImplementedError


def predict_segpose(segpose: SegPoseNet, img: Image, threshold: float, img_size: tuple):
    """Runs prediction for a single PIL Event Frame
    """
    ev_frame = torch.from_numpy(EvMaskPoseDataset.preprocess(img, img_size))
    ev_frame = ev_frame.unsqueeze(0).to(device=device, dtype=torch.float)
    with torch.no_grad():
        mask_pred, pose_pred = segpose(ev_frame)
        probs = torch.sigmoid(mask_pred).squeeze(0).cpu()
        pose_pred = pose_pred.squeeze(1).detach().cpu().numpy()
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img.size[1]),
                transforms.ToTensor(),
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy() > threshold

    plot_img_and_mask(img, full_mask)

    return (full_mask).astype(np.uint8) * 255, pose_pred


def main(models: dict, params: Params):

    if not params.pred_dir and not os.path.exists(params.pred_dir):
        raise FileNotFoundError(
            "Prediction directory has not been set or the file does not exist, please set using cli args or params"
        )
    pred_folders = [join(params.pred_dir, f) for f in os.listdir(params.pred_dir)]
    for p in pred_folders:
        try:
            manager = RenderManager.from_path(p)
        except FileNotFoundError:
            continue
        # Run Silhouette Prediction Network
        logging.info(f"Starting mask predictions")
        mask_priors, R_pred, T_pred = [], [], []
        q_loss, t_loss = 0, 0
        # Collect Translation stats
        R_gt, T_gt = manager._trajectory
        std_T, mean_T = torch.std_mean(T_gt)
        for idx in range(len(manager)):

            ev_frame = manager.get_event_frame(idx)
            mask_pred, pose_pred = predict_segpose(
                models["segpose"], ev_frame, params.threshold_conf, params.img_size
            )

            manager.add_pred(idx, mask_pred, "silhouette")
            mask_priors.append(torch.from_numpy(mask_pred))

            # Make qexp a torch function
            q_pred = torch.from_numpy(qexp(pose_pred[0, 3:])).squeeze(0)
            q_targ = rc.matrix_to_quaternion(R_gt[idx]).squeeze(0)
            t_pred = torch.from_numpy(pose_pred[:, :3]) * std_T + mean_T
            T_pred.append(t_pred)

            q_loss += quaternion_angular_error(q_pred, q_targ)
            t_loss += t_error(t_pred, T_gt[[idx]])

            r_pred = rc.quaternion_to_matrix(q_pred).unsqueeze(0)
            R_pred.append(r_pred)

        R_pred = torch.cat(R_pred, dim=0)
        T_pred = torch.cat(T_pred, dim=0)
        q_loss_mean = q_loss / idx
        t_loss_mean = t_loss / idx
        logging.info(
            f"Mean Translation Error: {t_loss_mean}; Mean Rotation Error: {q_loss_mean}"
        )

        logging.info("Starting Mesh Reconstruction from predicted silhouettes")

        R, T = manager._trajectory
        # Expects torch tensors as input to mesh deformation model
        input_m = torch.stack((mask_priors))
        logging.info(f"Input pred shape & max: {input_m.shape}, {input_m.max()}")
        # The MeshDeformation model will return silhouettes across all view by default
        results = models["mesh"].run_optimization(input_m, R, T)
        renders = models["mesh"].render_final_mesh(
            (R, T), "predict", input_m.shape[-2:]
        )

        mesh_silhouettes = renders["silhouettes"].squeeze(1)
        mesh_images = renders["images"].squeeze(1)
        # Calculate mean iou in comparison to groundtruth
        groundtruth_silhouettes = (
            manager._images("silhouette", img_size=mesh_silhouettes.shape[1:]) / 255.0
        )

        logging.info(
            f"Groundtruth shape & max: {groundtruth_silhouettes.shape}, {groundtruth_silhouettes.max()}, {groundtruth_silhouettes.dtype}"
        )
        logging.info(
            f"Mesh pred shape & max: {mesh_silhouettes.shape}, {mesh_silhouettes.max()}, {mesh_silhouettes.dtype}"
        )

        mesh_iou = neg_iou_loss(groundtruth_silhouettes, mesh_silhouettes)
        seg_iou = neg_iou_loss(groundtruth_silhouettes, input_m / 255.0)
        gt_iou = neg_iou_loss(groundtruth_silhouettes, groundtruth_silhouettes)

        results["t_mean_error"] = t_loss_mean
        results["q_mean_error"] = q_loss_mean
        results["mesh_iou"] = mesh_iou.detach().cpu().numpy().tolist()
        results["seg_iou"] = seg_iou.detach().cpu().numpy().tolist()
        logging.info(f"Mesh IOU list & results: {mesh_iou}")
        logging.info(f"Seg IOU list & results: {seg_iou}")
        logging.info(f"GT IOU list & results: {gt_iou} ")

        # results["mean_iou"] = IOULoss().forward(groundtruth, mesh_silhouettes).detach().cpu().numpy().tolist()
        # results["mean_dice"] = DiceCoeffLoss().forward(groundtruth, mesh_silhouettes)

        manager.set_pred_results(results)
        manager.add_pred_mesh(models["mesh"]._final_mesh)
        for idx in range(len(mesh_silhouettes)):
            manager.add_pred(
                idx,
                mesh_silhouettes[idx].cpu().numpy(),
                "silhouette",
                destination="mesh",
            )
            manager.add_pred(
                idx, mesh_images[idx].cpu().numpy(), "phong", destination="mesh"
            )
        manager.close()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set the device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if device == "cuda:1":
        torch.cuda.set_device(device)

    logging.info(f"Using {device} as computation device")

    args = get_args()
    args_dict = vars(args)
    params = Params(**args_dict)
    params.device = device

    try:
        unet = UNet.load(params)
        segpose = SegPoseNet.load(unet, params)
        logging.info("Loaded SegPose from params")
        mesh_model = MeshDeformationModel(device)
        logging.info("Loaded Mesh Deformation Model")
        models = dict(segpose=segpose, mesh=mesh_model)
        main(models, params)
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating prediction run")
        sys.exit(0)
