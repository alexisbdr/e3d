import configargparse
import logging
import os
import random
import sys
from os.path import abspath, join

import cv2
import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
import torch.nn as nn
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (PerspectiveCameras, TexturesAtlas,
                                get_world_to_view_transform, look_at_rotation)
from pytorch3d.structures import Meshes
from segpose import UNetDynamic, UNet
from segpose.dataset import EvMaskPoseDataset, EvimoDataset
from utils.params import Params
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils.manager import RenderManager
from utils.pose_utils import qexp, qlog, quaternion_angular_error
from utils.visualization import (plot_camera_scene, plot_img_and_mask,
                                 plot_trajectory_cameras)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json


def neg_iou_loss_all(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6

    return 1.0 - (intersect / union).sum() / intersect.nelement()


def discoeff_loss(predict, target):
    dims = tuple(range(predict.ndim)[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target).sum(dims) + 1e-6
    return (2 * intersect + 1e-6) / union


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndim)[1:])
    intersect = (predict * target).sum(dims) + 1e-6
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1.0 - intersect / union


def scale_meshes(pred_meshes, gt_meshes, scale="gt-10"):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
        print(pred_scale, gt_scale)
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes



def get_args():
    parser = configargparse.ArgumentParser(
        description="EvUnet Prediction",
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cfg",
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
        '--model_cpt',
        dest='model_cpt',
        type=str,
        default=Params.model_cpt,
        help='model checkpoint path!'
    )
    parser.add_argument(
        '--name',
        dest='name',
        type=str,
        default=Params.name,
        help='test experiment name'
    )
    parser.add_argument(
        '--mesh_sphere_level',
        dest='mesh_sphere_level',
        type=int,
        default=Params.mesh_sphere_level,
        help='ico level of mesh'
    )
    parser.add_argument(
        '--mesh_sphere_scale',
        dest='mesh_sphere_scale',
        type=float,
        default=Params.mesh_sphere_scale,
        help='Mesh scale in real world'
    )
    parser.add_argument(
        '--lambda_laplacian',
        dest='lambda_laplacian',
        type=float,
        default=Params.lambda_laplacian
    )
    parser.add_argument(
        '--lambda_flatten',
        dest='lambda_flatten',
        type=float,
        default=Params.lambda_flatten
    )
    parser.add_argument(
        '--ransac_iou_threshold',
        dest='ransac_iou_threshold',
        type=float,
        default=0.8,
        help='Threshold used for inliers selected in RANSAC method'
    )
    parser.add_argument(
        '--ransac_model_num',
        dest='ransac_model_num',
        type=int,
        default=5,
        help='number of models used for ransac'
    )
    parser.add_argument(
        '--ransac_least_samples',
        dest='ransac_least_samples',
        type=int,
        default=30,
        help='least number of views used for ransac'
    )

    return parser.parse_args()


def plot_cams_from_poses(pose_gt, pose_pred, device: str):
    """
    """
    R_gt, T_gt = pose_gt
    cameras_gt = PerspectiveCameras(device=device, R=R_gt, T=T_gt)
    if pose_pred is None:
        fig = plot_trajectory_cameras(cameras_gt, device)
        return fig

    R_pred, T_pred = pose_pred
    cameras_pred = PerspectiveCameras(device=device, R=R_pred, T=T_pred)
    fig = plot_camera_scene(cameras_pred, cameras_gt, "final_preds", device)

    return fig


def predict_segpose(unet: UNetDynamic, img: Image, threshold: float, img_size: tuple):
    """Runs prediction for a single PIL Event Frame
    """
    ev_frame = torch.from_numpy(EvMaskPoseDataset.preprocess_images(img, img_size))
    ev_frame = ev_frame.unsqueeze(0).to(device=device, dtype=torch.float)
    with torch.no_grad():
        mask_pred = unet(ev_frame)
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


def save_mesh(path, mesh):
    save_obj(path, mesh._final_mesh.verts_packed().cpu(), mesh._final_mesh.faces_packed().cpu())


def save_pose(path ,mesh):
    init_R, init_t = mesh._get_init_pose
    with open(path, 'w') as file:
        file.write(str(init_R))
        file.write(str(init_t))


def pred_evimo(unet: UNetDynamic, params: Params, device: str):
    if not os.path.isdir(params.pred_dir):
        raise FileNotFoundError(
            "Testing directory has not been set "
        )
    logging.info('Prepare testing data......')
    for directory in os.listdir(params.pred_dir):
        dir_path = os.path.join(params.pred_dir, directory)
        if not os.path.isdir(dir_path):
            continue
        dataset = EvimoDataset(dir_path, obj_id=params.evimo_obj_id, is_train=False, slice_name=params.slice_name)
        logging.info(f'Data Loaded: {len(dataset)}')
        basedir = os.path.join(params.exper_dir, params.name, directory)
        os.makedirs(basedir, exist_ok=True)
        logging.info(f'Directory : {basedir} Made!')

        logging.info(f'Predict the mask of the event frames!')
        mask_preds, mask_gts, ev_frames, mask_probs = [], [], [], []
        R, T = [], []
        test_loader = DataLoader(dataset, batch_size=params.unet_batch_size, num_workers=8, shuffle=False)
        for count, (ev_frame, mask_gt, R_gt, T_gt) in enumerate(test_loader):
            R.append(R_gt)
            T.append(T_gt)
            ev_frames.append(ev_frame)
            ev_frame = Variable(ev_frame).to(device=device, dtype=torch.float)
            with torch.no_grad():
                mask_pred = unet(ev_frame)
            prob = torch.sigmoid(mask_pred).squeeze()
            mask_pred = prob > params.threshold_conf
            mask_preds.append(mask_pred.float())
            mask_gts.append(mask_gt)
            mask_probs.append(prob)

        # prepare the info for mesh reconstruction
        mask_preds = torch.cat(mask_preds, dim=0).detach().cpu()    # mask predicted by the NN
        mask_gts = torch.cat(mask_gts, dim=0).squeeze().detach().cpu()   # mask ground truth
        mask_probs = torch.cat(mask_probs, dim=0).detach().cpu()    # probability map of the mask
        R = torch.cat(R, dim=0).detach().cpu()
        T = torch.cat(T, dim=0).detach().cpu()
        ev_frames = torch.cat(ev_frames, dim=0).detach().cpu()  # event frames
        camera_intrinsics = torch.tensor(dataset.get_new_camera()).to(device=device, dtype=torch.float32)   # camera intrinsics

        iou_all = neg_iou_loss(mask_gts, mask_preds)
        dice_iou_all = discoeff_loss(mask_gts, mask_preds)
        logging.info(f'[Test IOU] The IOU of the Predict Masks and GT Dice IOU: {dice_iou_all.sum().item() / dice_iou_all.shape[0]}  Neg IOU: {iou_all.sum().item() / iou_all.shape[0]}')
        with open(os.path.join(basedir, 'loss.txt'), 'w') as file:
            file.write(f'Neg IOU:  {iou_all.sum().item() / iou_all.shape[0]}')
            file.write(f'Dice IOU: {dice_iou_all.sum().item() / dice_iou_all.shape[0]}')

        # For all GT masks run mesh reconstruction

        mesh_model_all_gt = MeshDeformationModel(device=device, params=params)
        logging.info("Set GT Masks Mesh Deformation Model")

        res_all_gt = mesh_model_all_gt.run_optimization(mask_gts.to(device), R.to(device), T.to(device), camera_settings=camera_intrinsics)
        renders_all_gt = mesh_model_all_gt.render_final_mesh((R.to(device), T.to(device)), "predict", mask_gts.shape[-2:], camera_settings=camera_intrinsics)
        mesh_all_gt_silhouettes = renders_all_gt["silhouettes"].squeeze(1)
        # mesh_all_gt_images = renders_all_gt["images"].squeeze(1)

        gt_mask_dir = os.path.join(basedir, "all_gt_masks")
        os.makedirs(gt_mask_dir, exist_ok=True)
        logging.info(f'Directory : {gt_mask_dir} Made!')
        save_mesh(os.path.join(gt_mask_dir, 'gt_mask_pred.obj'), mesh_model_all_gt)
        save_pose(os.path.join(gt_mask_dir, 'init_pose.txt'), mesh_model_all_gt)

        # gt_iou_gt_pred = discoeff_loss(mask_gts, mask_gts)
        gt_iou_gt_render = discoeff_loss(mask_gts, mesh_all_gt_silhouettes)
        gt_iou_all = neg_iou_loss(mask_gts, mesh_all_gt_silhouettes)
        # gt_iou_pred_render = discoeff_loss(mask_gts, mesh_all_gt_silhouettes)

        with open(os.path.join(basedir, 'loss.txt'), 'a+') as file:
            file.write(f'[All GT masks]IOU of GT and rendered masks Dice IOU: {gt_iou_gt_render.sum().item() / gt_iou_gt_render.shape[0]}  Neg IOU : {gt_iou_all.sum().item() / gt_iou_all.shape[0]}')

        # For all prediction masks for mesh R=reconstruction

        mesh_model_all_pred = MeshDeformationModel(device=device, params=params)
        logging.info("Set Pred Masks Mesh Deformation Model")

        res_all_pred = mesh_model_all_pred.run_optimization(mask_preds.to(device), R.to(device), T.to(device),
                                                        camera_settings=camera_intrinsics)
        renders_all_pred = mesh_model_all_pred.render_final_mesh((R.to(device), T.to(device)), "predict",
                                                             mask_preds.shape[-2:], camera_settings=camera_intrinsics)
        mesh_all_pred_silhouettes = renders_all_pred["silhouettes"].squeeze(1)
        mesh_all_pred_images = renders_all_pred["images"].squeeze(1)

        pred_mask_dir = os.path.join(basedir, "all_pred_masks")
        os.makedirs(pred_mask_dir, exist_ok=True)
        logging.info(f'Directory : {pred_mask_dir} Made!')
        save_mesh(os.path.join(pred_mask_dir, 'pred_mask_pred.obj'), mesh_model_all_pred)
        save_pose(os.path.join(pred_mask_dir, 'init_pose.txt'), mesh_model_all_pred)

        preds_iou_gt_pred = discoeff_loss(mask_gts, mask_preds)
        preds_iou_gt_render = discoeff_loss(mask_gts, mesh_all_pred_silhouettes)
        preds_iou_all = neg_iou_loss(mask_gts, mesh_all_pred_silhouettes)
        preds_iou_pred_render = discoeff_loss(mask_preds, mesh_all_pred_silhouettes)

        with open(os.path.join(basedir, 'loss.txt'), 'a+') as file:
            file.write(f'[All Pred masks]IOU of GT and rendered masks Dice IOU: {preds_iou_gt_render.sum().item() / preds_iou_gt_render.shape[0]}  Neg IOU: {preds_iou_all.sum().item() / preds_iou_all.shape[0]}')

        # For RANSAC Method Mesh Optimization

        logging.info("Start RANSAC Method Mesh Optimization!")
        best_num = 0
        best_mesh = None
        ransac_iou_threshold = params.ransac_iou_threshold
        ransac_model_num = params.ransac_model_num
        ransac_least_samples = params.ransac_least_samples
        for id in range(ransac_model_num):
            logging.info("The {} model for RANSAC: ".format(id))
            idx = random.sample(list(range(len(mask_preds))), ransac_least_samples)
            mask_preds_ran = torch.cat([mask_preds[i:i+1, ...] for i in idx], dim=0)
            R_ran = torch.cat([R[i] for i in idx], dim=0)
            T_ran = torch.cat([T[i] for i in idx], dim=0)

            mesh_model = MeshDeformationModel(device=device, params=params)

            res = mesh_model.run_optimization(mask_preds_ran.to(device), R_ran.to(device),
                                                       T_ran.to(device),
                                                        camera_settings=camera_intrinsics)
            renders = mesh_model.render_final_mesh((R.to(device), T.to(device)), "predict",
                                                                     mask_preds.shape[-2:],
                                                                     camera_settings=camera_intrinsics)

            mesh_silhouettes = renders["silhouettes"].squeeze(1)

            dice = discoeff_loss(mask_preds, mesh_silhouettes)
            num_over_threshold = (dice > ransac_iou_threshold).sum().item()
            if num_over_threshold > best_num:
                # best_idx = idx
                best_num = num_over_threshold
                best_mesh = mesh_model
        renders = best_mesh.render_final_mesh((R.to(device), T.to(device)), "predict",
                                               mask_preds.shape[-2:],
                                               camera_settings=camera_intrinsics)
        mesh_silhouettes = renders["silhouettes"].squeeze(1)
        dice = discoeff_loss(mask_preds, mesh_silhouettes)
        idx_over_threshold = dice > ransac_iou_threshold
        num_inliers = idx_over_threshold.sum().item()
        print(f'Inliers final: {num_inliers} / {mask_preds.shape[0]}')
        mask_preds_refine = mask_preds[idx_over_threshold]
        R_refine = R[idx_over_threshold]
        T_refine = T[idx_over_threshold]
        mesh_model = MeshDeformationModel(device=device, params=params)
        res = mesh_model.run_optimization(mask_preds_refine.to(device), R_refine.to(device),
                                          T_refine.to(device),
                                          camera_settings=camera_intrinsics)
        renders_ransac = mesh_model.render_final_mesh((R.to(device), T.to(device)), "predict",
                                               mask_preds.shape[-2:],
                                               camera_settings=camera_intrinsics)
        mesh_ransac_silhouettes = renders_ransac["silhouettes"].squeeze(1)
        mesh_ransac_images = renders_ransac["images"].squeeze(1)

        ransac_mask_dir = os.path.join(basedir, "ransac_masks")
        os.makedirs(ransac_mask_dir, exist_ok=True)
        print(f'Directory : {ransac_mask_dir} Made!')
        save_mesh(os.path.join(ransac_mask_dir, 'ransac_mask_pred.obj'), mesh_model)
        save_pose(os.path.join(ransac_mask_dir, 'init_pose.txt'), mesh_model)

        ransac_iou_gt_render = discoeff_loss(mask_gts, mesh_ransac_silhouettes)
        ransac_iou_all = neg_iou_loss(mask_gts, mesh_ransac_silhouettes)
        ransac_iou_pred_render = discoeff_loss(mask_preds, mesh_ransac_silhouettes)

        with open(os.path.join(basedir, 'loss.txt'), 'a+') as file:
            file.write(f'[RANSAC masks]IOU of GT and rendered masks Dice IOU: {ransac_iou_gt_render.sum().item() / ransac_iou_gt_render.shape[0]}  Neg IOU: {ransac_iou_all.sum().item() / ransac_iou_all.shape[0]}')

        res_dir = os.path.join(basedir, 'all_results')

        plt.rc('font', size=8)
        plt.rc('axes', titlesize=8)
        plt.rc('axes', labelsize=8)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)

        os.makedirs(res_dir, exist_ok=True)
        for count in range(mask_gts.shape[0]):
            plt.figure(figsize=(8, 4),dpi=288)
            plt.subplot(241)
            plt.title("event frame")
            plt.imshow(ev_frames[count].permute(1, 2, 0))
            plt.subplot(242)
            plt.title("gt mask ")
            plt.imshow(mask_gts[count])
            plt.subplot(243)
            plt.title("pred mask %.3f" % preds_iou_gt_pred[count])
            plt.imshow(mask_preds[count])
            plt.subplot(244)
            plt.title("pred mask prob")
            plt.imshow(mask_probs[count])
            plt.subplot(245)
            plt.title("all gt render %.3f" % gt_iou_gt_render[count])
            plt.imshow(mesh_all_gt_silhouettes[count])
            plt.subplot(246)
            plt.title("all pred render %.3f" % preds_iou_gt_render[count])
            plt.imshow(mesh_all_pred_silhouettes[count])
            plt.subplot(247)
            plt.title("IOU pred&render %.3f" % preds_iou_pred_render[count])
            plt.imshow(mesh_ransac_images[count])
            plt.subplot(248)
            plt.title("ransac render %.3f" % ransac_iou_gt_render[count] + " Inlier" if idx_over_threshold[count].item() else "")
            plt.imshow(mesh_ransac_silhouettes[count])
            plt.savefig(os.path.join(res_dir, f'{count}.png'))
            plt.show()
            plt.close()

        logging.info(f'Path over! : {dir_path}')


def pred_synth(models: dict, params: Params, mesh_type: str = "dolphin"):

    if not params.pred_dir and not os.path.exists(params.pred_dir):
        raise FileNotFoundError(
            "Prediction directory has not been set or the file does not exist, please set using cli args or params"
        )
    pred_folders = [join(params.pred_dir, f) for f in os.listdir(params.pred_dir)]
    count = 1
    for p in sorted(pred_folders):
        try:
            print(p)
            manager = RenderManager.from_path(p)
            manager.rectify_paths(base_folder=params.pred_dir)
        except FileNotFoundError:
            continue
        # Run Silhouette Prediction Network
        logging.info(f"Starting mask predictions")
        mask_priors = []
        # Collect Translation stats
        R_gt, T_gt = manager._trajectory
        for idx in range(len(manager)):
            try:
                ev_frame = manager.get_event_frame(idx)
            except Exception as e:
                print(e)
                break
            mask_pred = predict_segpose(
                models["unet"], ev_frame, params.threshold_conf, params.img_size
            )
            # mask_pred = smooth_predicted_mask(mask_pred)
            manager.add_pred(idx, mask_pred, "silhouette")
            mask_priors.append(torch.from_numpy(mask_pred))


        # Plot estimated cameras
        logging.info(f"Plotting pose map")
        count += 1
        groundtruth_silhouettes = manager._images("silhouette") / 255.0
        print(groundtruth_silhouettes.shape)
        print(torch.stack((mask_priors)).shape)
        seg_iou = neg_iou_loss(
            groundtruth_silhouettes, torch.stack((mask_priors)) / 255.0
        )
        print("Seg IoU", seg_iou)

        # RUN MESH DEFORMATION

        results = {}
        input_m = torch.stack((mask_priors))

        logging.info(f"Input pred shape & max: {input_m.shape}, {input_m.max()}")
        # The MeshDeformation model will return silhouettes across all view by default

        experiment_results = models["mesh"].run_optimization(input_m, R_gt, T_gt)
        renders = models["mesh"].render_final_mesh(
            (R_gt, T_gt), "predict", input_m.shape[-2:]
        )

        mesh_silhouettes = renders["silhouettes"].squeeze(1)
        mesh_images = renders["images"].squeeze(1)
        experiment_name = params.name
        for idx in range(len(mesh_silhouettes)):
            manager.add_pred(
                idx,
                mesh_silhouettes[idx].cpu().numpy(),
                "silhouette",
                destination=f"mesh_{experiment_name}",
            )
            manager.add_pred(
                idx,
                mesh_images[idx].cpu().numpy(),
                "phong",
                destination=f"mesh_{experiment_name}",
            )

        # Calculate chamfer loss:
        mesh_pred = models["mesh"]._final_mesh
        if mesh_type == "dolphin":
            path = params.gt_mesh_path
            mesh_gt = load_objs_as_meshes(
                [path],
                create_texture_atlas=False,
                load_textures=True,
                device=device,
            )
        # Shapenet Cars
        elif mesh_type == "shapenet":
            mesh_info = manager.metadata["mesh_info"]
            path = params.gt_mesh_path
            try:
                verts, faces, aux = load_obj(
                    path, load_textures=True, create_texture_atlas=True
                )

                mesh_gt = Meshes(
                    verts=[verts],
                    faces=[faces.verts_idx],
                    textures=TexturesAtlas(atlas=[aux.texture_atlas]),
                ).to(device)
            except:
                mesh_gt = None
                print("CANNOT COMPUTE CHAMFER LOSS")
        if mesh_gt and params.is_real_data:
            mesh_pred_compute, mesh_gt_compute = scale_meshes(
                mesh_pred.clone(), mesh_gt.clone()
            )
            pcl_pred = sample_points_from_meshes(
                mesh_pred_compute, num_samples=5000
            )
            pcl_gt = sample_points_from_meshes(mesh_gt_compute, num_samples=5000)
            chamfer_loss = chamfer_distance(
                pcl_pred, pcl_gt, point_reduction="mean"
            )
            print("CHAMFER LOSS: ", chamfer_loss)
            experiment_results["chamfer_loss"] = (
                chamfer_loss[0].cpu().detach().numpy().tolist()
            )

        mesh_iou = neg_iou_loss_all(groundtruth_silhouettes, mesh_silhouettes)

        experiment_results["mesh_iou"] = mesh_iou.cpu().numpy().tolist()

        results[experiment_name] = experiment_results

        manager.add_pred_mesh(mesh_pred, experiment_name)

        seg_iou = neg_iou_loss_all(groundtruth_silhouettes, input_m / 255.0)
        gt_iou = neg_iou_loss_all(groundtruth_silhouettes, groundtruth_silhouettes)

        results["mesh_iou"] = mesh_iou.detach().cpu().numpy().tolist()
        results["seg_iou"] = seg_iou.detach().cpu().numpy().tolist()
        logging.info(f"Mesh IOU list & results: {mesh_iou}")
        logging.info(f"Seg IOU list & results: {seg_iou}")
        logging.info(f"GT IOU list & results: {gt_iou} ")

        # results["mean_iou"] = IOULoss().forward(groundtruth, mesh_silhouettes).detach().cpu().numpy().tolist()
        # results["mean_dice"] = DiceCoeffLoss().forward(groundtruth, mesh_silhouettes)

        manager.set_pred_results(results)
        manager.close()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()
    args_dict = vars(args)
    params = Params()
    params.config_file = args_dict['config_file']
    params.__post_init__()
    params._set_with_dict(args_dict)
    params.ransac_iou_threshold = args_dict['ransac_iou_threshold']

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
    params.logger = logging

    try:
        unet = UNetDynamic.load(params)
        unet = nn.DataParallel(unet).to(device)
        logging.info("Loaded UNet from params")
        # mesh_model = MeshDeformationModel(device=device, params=params)
        # logging.info("Loaded Mesh Deformation Model")
        # models = dict(unet=unet, mesh=mesh_model)
        exper_path = os.path.join(params.exper_dir, params.name)
        os.makedirs(exper_path, exist_ok=True)
        with open(os.path.join(exper_path, 'config.json'), 'w') as file:
            file.write(json.dumps(params.as_dict()))
        if params.is_real_data:
            pred_evimo(unet, params, device)
        else:
            # pred_synth(models, params, mesh_type='shapenet')
            pass
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating prediction run")
        sys.exit(0)
