import argparse
import logging
import os
import random
import sys
from os.path import abspath, join

import cv2
import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
from losses import DiceCoeffLoss, IOULoss
from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (SfMPerspectiveCameras, TexturesAtlas,
                                get_world_to_view_transform, look_at_rotation)
from pytorch3d.structures import Meshes
from segpose import SegPoseNet, UNet
from segpose.dataset import EvMaskPoseDataset
from segpose.params import Params
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils.manager import RenderManager
from utils.pose_utils import qexp, qlog, quaternion_angular_error
from utils.visualization import (plot_camera_scene, plot_img_and_mask,
                                 plot_trajectory_cameras)


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


def smooth_predicted_mask(pred_mask):
    """Applies dilation and blurring to smooth mask edges
    Helpful for mesh reconstruction
    """
    # kernel_size = 15
    # morph_op = cv2.MORPH_CROSS

    # element = cv2.getStructuringElement(morph_op,(2*kernel_size + 1, 2*kernel_size+1),(kernel_size, kernel_size))
    # dilated_mask = cv2.dilate(pred_mask, element)

    # blurred_mask = cv2.blur(dilated_mask, (4, 4))

    smoothed_mask = cv2.morphologyEx(
        pred_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    return smoothed_mask


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
        type=float,
        default=Params.threshold_conf,
        help="Probability threshold for masks",
    )
    parser.add_argument(
        "--gpu", dest="gpu_num", default="0", type=str, help="GPU Device Number",
    )
    parser.add_argument(
        "--real",
        dest="real_data",
        action="store_true",
        help="Run the prediction for real data",
    )

    return parser.parse_args()


def plot_cams_from_poses(pose_gt, pose_pred, device: str):
    """
    """
    R_gt, T_gt = pose_gt
    cameras_gt = SfMPerspectiveCameras(device=device, R=R_gt, T=T_gt)
    if pose_pred is None:
        fig = plot_trajectory_cameras(cameras_gt, device)
        return fig

    R_pred, T_pred = pose_pred
    cameras_pred = SfMPerspectiveCameras(device=device, R=R_pred, T=T_pred)
    fig = plot_camera_scene(cameras_pred, cameras_gt, "final_preds", device)

    return fig


def predict_mesh(mesh_model: MeshDeformationModel):

    raise NotImplementedError


def predict_segpose(segpose: SegPoseNet, img: Image, threshold: float, img_size: tuple):
    """Runs prediction for a single PIL Event Frame
    """
    ev_frame = torch.from_numpy(EvMaskPoseDataset.preprocess_images(img, img_size))
    ev_frame = ev_frame.unsqueeze(0).to(device=device, dtype=torch.float)
    with torch.no_grad():
        mask_pred, pose_pred = segpose(ev_frame)
        probs = torch.sigmoid(mask_pred).squeeze(0).cpu()
        pose_pred = pose_pred.squeeze(1).cpu()
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


def real_data(models: dict, params: Params):
    """A utility function to run on real data
    We cannot compute any stats here we just run through the entire pipeline and save the results
    """
    # First we open the directory and read the event frames:
    for f in os.listdir(params.pred_dir):
        folder = join(params.pred_dir, f)
        print(folder)
        if f.endswith("predict"):
            continue
        render_manager = RenderManager(
            types=["silhouette", "events"],
            new_folder=f"event-data/",
            render_folder="data/eventdata",
            folder_name=f"{params.pred_dir.split('/')[-1]}/{f}_predict",
        )
        render_manager.init()
        event_paths = [join(folder, f) for f in os.listdir(folder)]
        R_pred, T_pred = [], []
        for count, ev_path in enumerate(event_paths):
            print(ev_path)
            ev_image = Image.open(ev_path).convert("L").resize((280, 280))
            mask_pred, pose_pred = predict_segpose(
                models["segpose"], ev_image, params.threshold_conf, params.img_size
            )
            q_pred = pose_pred[:, 3:]
            r_pred = rc.quaternion_to_matrix(q_pred)
            t_pred = pose_pred[:, :3]
            wtv_trans = (
                get_world_to_view_transform(R=r_pred, T=t_pred).inverse().get_matrix()
            )
            t_pred_wtv = wtv_trans[:, 3, :3]
            r_pred_wtv = wtv_trans[:, :3, :3]
            pred = {"silhouette": mask_pred}
            render_manager.add_images(count, pred, r_pred_wtv, t_pred_wtv)
            render_manager.add_event_frame(count, np.array(ev_image))
            R_pred.append(r_pred_wtv)
            T_pred.append(t_pred_wtv)
        R_pred = torch.cat(R_pred)
        T_pred = torch.cat(T_pred)
        pose_plot = plot_cams_from_poses((R_pred, T_pred), None, device=params.device)
        R_pred_lookat = look_at_rotation(T_pred)
        T_pred_lookat = -torch.bmm(R_pred_lookat.transpose(1, 2), T_pred[:, :, None])[
            :, :, 0
        ]
        pose_plot_lookat = plot_cams_from_poses(
            (R_pred_lookat, T_pred_lookat), None, device=params.device
        )
        render_manager.add_pose_plot(pose_plot, "rot+trans")
        render_manager.add_pose_plot(pose_plot_lookat, "trans")
        render_manager.close()


def main(models: dict, params: Params, mesh_type: str = "dolphin"):

    print("here")
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
            # print(params.pred_dir.split('/')[-1])
            # manager.rectify_paths(params.pred_dir.split('/')[-1])
        except FileNotFoundError:
            continue
        # Run Silhouette Prediction Network
        logging.info(f"Starting mask predictions")
        mask_priors, R_pred, T_pred = [], [], []
        q_loss, t_loss = 0, 0
        # Collect Translation stats
        R_gt, T_gt = manager._trajectory
        poses_gt = EvMaskPoseDataset.preprocess_poses(manager._trajectory)
        std_T, mean_T = torch.std_mean(T_gt)
        for idx in range(len(manager)):
            try:
                ev_frame = manager.get_event_frame(idx)
            except Exception as e:
                print(e)
                break
            mask_pred, pose_pred = predict_segpose(
                models["segpose"], ev_frame, params.threshold_conf, params.img_size
            )
            # mask_pred = smooth_predicted_mask(mask_pred)
            manager.add_pred(idx, mask_pred, "silhouette")
            mask_priors.append(torch.from_numpy(mask_pred))

            # Make qexp a torch function
            # q_pred = qexp(pose_pred[:, 3:])
            # q_targ = qexp(poses_gt[idx, 3:].unsqueeze(0))
            ####  SHOULD THIS BE NORMALIZED ??
            q_pred = pose_pred[:, 3:]
            q_targ = poses_gt[idx, 3:]

            q_pred_unit = q_pred / torch.norm(q_pred)
            q_targ_unit = q_targ / torch.norm(q_targ)
            # print("learnt: ", q_pred_unit, q_targ_unit)

            t_pred = pose_pred[:, :3] * std_T + mean_T
            t_targ = poses_gt[idx, :3] * std_T + mean_T
            T_pred.append(t_pred)

            q_loss += quaternion_angular_error(q_pred_unit, q_targ_unit)
            t_loss += t_error(t_pred, t_targ)

            r_pred = rc.quaternion_to_matrix(q_pred).unsqueeze(0)
            R_pred.append(r_pred.squeeze(0))

        q_loss_mean = q_loss / (idx + 1)
        t_loss_mean = t_loss / (idx + 1)

        # Convert R,T to world-to-view transforms --> Pytorch3d convention for the :

        R_pred_abs = torch.cat(R_pred)
        T_pred_abs = torch.cat(T_pred)
        # Take inverse of view-to-world (output of net) to obtain w2v
        wtv_trans = (
            get_world_to_view_transform(R=R_pred_abs, T=T_pred_abs)
            .inverse()
            .get_matrix()
        )
        T_pred = wtv_trans[:, 3, :3]
        R_pred = wtv_trans[:, :3, :3]
        R_pred_test = look_at_rotation(T_pred_abs)
        T_pred_test = -torch.bmm(R_pred_test.transpose(1, 2), T_pred_abs[:, :, None])[
            :, :, 0
        ]
        # Convert back to view-to-world to get absolute
        vtw_trans = (
            get_world_to_view_transform(R=R_pred_test, T=T_pred_test)
            .inverse()
            .get_matrix()
        )
        T_pred_trans = vtw_trans[:, 3, :3]
        R_pred_trans = vtw_trans[:, :3, :3]

        # Calc pose error for this:
        q_loss_mean_test = 0
        t_loss_mean_test = 0
        for idx in range(len(R_pred_test)):
            q_pred_trans = rc.matrix_to_quaternion(R_pred_trans[idx]).squeeze()
            q_targ = poses_gt[idx, 3:]
            q_targ_unit = q_targ / torch.norm(q_targ)
            # print("look: ", q_test, q_targ)
            q_loss_mean_test += quaternion_angular_error(q_pred_trans, q_targ_unit)
            t_targ = poses_gt[idx, :3] * std_T + mean_T
            t_loss_mean_test += t_error(T_pred_trans[idx], t_targ)
        q_loss_mean_test /= idx + 1
        t_loss_mean_test /= idx + 1

        logging.info(
            f"Mean Translation Error: {t_loss_mean}; Mean Rotation Error: {q_loss_mean}"
        )
        logging.info(
            f"Mean Translation Error: {t_loss_mean_test}; Mean Rotation Error: {q_loss_mean_test}"
        )

        # Plot estimated cameras
        logging.info(f"Plotting pose map")
        cam_idx = random.sample(range(len(R_gt)), k=2)
        pose_plot = plot_cams_from_poses(
            (R_gt[cam_idx], T_gt[cam_idx]),
            (R_pred[cam_idx], T_pred[cam_idx]),
            params.device,
        )
        pose_plot_test = plot_cams_from_poses(
            (R_gt[cam_idx], T_gt[cam_idx]),
            (R_pred_test[cam_idx], T_pred_test[cam_idx]),
            params.device,
        )
        manager.add_pose_plot(pose_plot, "rot+trans")
        manager.add_pose_plot(pose_plot_test, "trans")
        count += 1

        groundtruth_silhouettes = manager._images("silhouette") / 255.0
        print(groundtruth_silhouettes.shape)
        print(torch.stack((mask_priors)).shape)
        seg_iou = neg_iou_loss(
            groundtruth_silhouettes, torch.stack((mask_priors)) / 255.0
        )
        print("Seg IoU", seg_iou)
        """
        keep_silhouettes = []
        for i in range(len(mask_priors)):
            print(groundtruth_silhouettes[i:i+1].squeeze(0).shape, groundtruth_silhouettes[i:i+1].squeeze(0).max())
            this_iou = neg_iou_loss(groundtruth_silhouettes[i:i+1].squeeze(0), mask_priors[i] / 255.0)
            print(this_iou)
            if this_iou < seg_iou: keep_silhouettes.append(mask_priors[i])
        """

        # RUN MESH DEFORMATION
        # Run it 3 times: w/ Rot+Trans - w/ Trans+LookAt - w/ GT Pose
        experiments = {
            "GT-Pose": [R_gt, T_gt],
            # "Rot+Trans": [R_pred, T_pred],
            # "Trans+LookAt": [R_pred_test, T_pred_test]
        }

        results = {}
        input_m = torch.stack((mask_priors))
        for i in range(len(experiments.keys())):

            logging.info(f"Input pred shape & max: {input_m.shape}, {input_m.max()}")
            # The MeshDeformation model will return silhouettes across all view by default

            R, T = list(experiments.values())[i]
            experiment_results = models["mesh"].run_optimization(input_m, R, T)
            renders = models["mesh"].render_final_mesh(
                (R, T), "predict", input_m.shape[-2:]
            )

            mesh_silhouettes = renders["silhouettes"].squeeze(1)
            mesh_images = renders["images"].squeeze(1)
            experiment_name = list(experiments.keys())[i]
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
                path = "data/meshes/dolphin/dolphin.obj"
                mesh_gt = load_objs_as_meshes(
                    [path],
                    create_texture_atlas=False,
                    load_textures=True,
                    device=device,
                )
            # Shapenet Cars
            elif mesh_type == "shapenet":
                mesh_info = manager.metadata["mesh_info"]
                path = f"data/ShapeNetCorev2/{mesh_info['synset_id']}/{mesh_info['mesh_id']}/models/model_normalized.obj"
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
            if mesh_gt:
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

            mesh_iou = neg_iou_loss(groundtruth_silhouettes, mesh_silhouettes)

            experiment_results["mesh_iou"] = mesh_iou.cpu().numpy().tolist()

            results[experiment_name] = experiment_results

            manager.add_pred_mesh(mesh_pred, experiment_name)

        # logging.info(
        #    f"Groundtruth shape & max: {groundtruth_silhouettes.shape}, {groundtruth_silhouettes.max()}, {groundtruth_silhouettes.dtype}"
        # )
        # logging.info(
        #    f"Mesh pred shape & max: {mesh_images.shape}, {mesh_silhouettes.max()}, {mesh_silhouettes.dtype}"
        # )

        seg_iou = neg_iou_loss(groundtruth_silhouettes, input_m / 255.0)
        gt_iou = neg_iou_loss(groundtruth_silhouettes, groundtruth_silhouettes)

        results["R_pred"] = R_pred.cpu().numpy().tolist()
        results["T_pred"] = T_pred.cpu().numpy().tolist()
        results["t_mean_error"] = np.array(t_loss_mean).tolist()
        results["q_mean_error"] = np.array(q_loss_mean).tolist()
        results["R_pred_trans"] = R_pred_test.cpu().numpy().tolist()
        results["T_pred_trans"] = T_pred_test.cpu().numpy().tolist()
        results["t_mean_error_trans"] = np.array(t_loss_mean_test).tolist()
        results["q_mean_error_trans"] = np.array(q_loss_mean_test).tolist()

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
    real = args_dict.pop("real_data")
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

    try:
        unet = UNet.load(params)
        segpose = SegPoseNet.load(unet, params)
        logging.info("Loaded SegPose from params")
        mesh_model = MeshDeformationModel(device)
        logging.info("Loaded Mesh Deformation Model")
        models = dict(segpose=segpose, mesh=mesh_model)
        if real:
            real_data(models, params)
        else:
            main(models, params)
    except KeyboardInterrupt:
        logging.error("Received interrupt terminating prediction run")
        sys.exit(0)
