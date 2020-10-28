import argparse
import logging
import os
import sys
from os.path import abspath, join

import numpy as np
import torch
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from segpose import UNet, SegPoseNet
from segpose.dataset import EvMaskPoseDataset
from segpose.params import Params
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils.manager import RenderManager
from utils.visualization import plot_img_and_mask


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6

    return 1.0 - (intersect / union).sum() / intersect.nelement()


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


def predict_segpose(unet: SegPoseNet, img: Image, threshold: float, img_size: tuple):
    """Runs prediction for a single PIL Image
    """
    ev_frame = torch.from_numpy(EvMaskPoseDataset.preprocess(img, img_size))
    print(ev_frame.shape)
    ev_frame = ev_frame.unsqueeze(0).to(device=device, dtype=torch.float)
    print(ev_frame.shape)
    with torch.no_grad():
        
        mask_pred, pose_pred = unet(ev_frame)
        print(mask_pred.shape)
        probs = torch.sigmoid(mask_pred).squeeze(0).cpu()
    
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

    return (full_mask).astype(np.uint8) * 255


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
        mask_priors = []
        for idx in range(len(manager)):

            ev_frame = manager.get_event_frame(idx)

            mask_pred, pose_pred = predict_segpose(
                models["segpose"], ev_frame, params.threshold_conf, params.img_size
            )

            manager.add_pred(idx, mask_pred, "silhouette")
            mask_priors.append(torch.from_numpy(mask_pred))

        logging.info("Starting Mesh Reconstruction from predicted silhouettes")

        R, T = manager._trajectory
        # Expects torch tensors as input to mesh deformation model
        input_m = torch.stack((mask_priors))
        logging.info(f"Input pred shape & max: {input_m.shape}, {input_m.max()}")
        # The MeshDeformation model will return silhouettes across all view by default
        mesh_silhouettes, results = models["mesh"].run_optimization(input_m, R, T)
        mesh_pred = mesh_silhouettes.squeeze(1)
        # Calculate mean iou in comparison to groundtruth
        groundtruth = manager._images("silhouette", img_size=params.img_size) / 255.0

        logging.info(
            f"Groundtruth shape & max: {groundtruth.shape}, {groundtruth.max()}, {groundtruth.dtype}"
        )
        logging.info(
            f"Mesh pred shape & max: {mesh_pred.shape}, {mesh_pred.max()}, {mesh_pred.dtype}"
        )

        mesh_iou = neg_iou_loss(groundtruth, mesh_pred)
        seg_iou = neg_iou_loss(groundtruth, input_m / 255.0)
        gt_iou = neg_iou_loss(groundtruth, groundtruth)

        results["mean_iou"] = mesh_iou.detach().cpu().numpy().tolist()
        results["seg_iou"] = seg_iou.detach().cpu().numpy().tolist()
        logging.info(f"Mesh IOU list & results: {results['mean_iou']}")
        logging.info(f"Seg IOU list & results: {seg_iou}")
        logging.info(f"GT IOU list & results: {gt_iou} ")

        # results["mean_iou"] = IOULoss().forward(groundtruth, mesh_silhouettes).detach().cpu().numpy().tolist()
        # results["mean_dice"] = DiceCoeffLoss().forward(groundtruth, mesh_silhouettes)

        manager.set_pred_results(results)
        for idx, sil in enumerate(mesh_pred):
            manager.add_pred(idx, sil.cpu().numpy(), "silhouette", destination="mesh")

        manager.close()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
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
        models = {"segpose": unet, "mesh": mesh_model}
        main(models, params)
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating prediction run")
        sys.exit(0)
