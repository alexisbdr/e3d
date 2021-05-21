import os
import random
from collections import OrderedDict
import cv2
import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
from PIL import Image
from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.transforms import Transform3d, Rotate
from utils.params import Params
from torch.utils.data import (BatchSampler, ConcatDataset, Dataset, Sampler, SubsetRandomSampler)
from utils.manager import RenderManager
import json
import matplotlib.pyplot as plt


class ConcatDataSampler(Sampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """A Custom sampler that does the very simple yet incredibly unthought of following:
            -Takes in a set of datasets in the form of ConcatDataset
            -Creates a batched random sampler FOR EACH dataset
            -returns batches from each INDIVIDUAL dataset during iteration
        -Shuffle: True if you want to sample the shufflers randomly
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.samplers: list = []
        self.generate_new_samplers()

    def generate_new_samplers(self):
        prev_end = 0
        for num, dset in enumerate(self.dataset.datasets):
            end = prev_end + len(dset)
            sampler = iter(
                BatchSampler(
                    SubsetRandomSampler(list(range(prev_end, end))),
                    self.batch_size,
                    self.drop_last,
                )
            )
            prev_end = end
            self.samplers.append(sampler)

        self.cum_size = end
        self.curr_sampler = 0

    def fetch_batch(self) -> list:
        batch_idx = next(self.samplers[self.curr_sampler])
        # batch_idx *= (self.curr_sampler  + 1)
        return batch_idx

    def __iter__(self):
        retries = 0
        while retries < len(self.samplers):
            if self.shuffle:
                self.curr_sampler = random.choice(range(len(self.samplers)))
            try:
                # Fetch a batch of indices
                yield self.fetch_batch()
                retries = 0
            except StopIteration:
                self.curr_sampler += 1
                retries += 1
        # We've reached the end of the epoch - generate a new set of samplers
        self.generate_new_samplers()

    def __len__(self):
        return self.cum_size // self.batch_size


class EvMaskPoseDataset(Dataset):
    def __init__(self, dir_num: int, params, transforms: list = []):

        self.img_size = params.img_size
        self.transforms = transforms
        try:
            self.render_manager = RenderManager.from_directory(
                dir_num=dir_num, render_folder=params.train_dir
            )
            self.render_manager.rectify_paths(base_folder=params.train_dir)
        except:
            self.render_manager = None

    @classmethod
    def preprocess_poses(cls, poses: tuple):
        """Generates (N, 6) vector of absolute poses
        Args:
            Tuple of batched rotations (N, 3, 3) and translations (N, 3) in Pytorch3d view-to-world coordinates. usually returned from a call to RenderManager._trajectory
            More information about Pytorch3D's coordinate system: https://github.com/facebookresearch/pytorch3d/blob/master/docs/notes/cameras.md

        1. Computes rotation and translation matrices in view-to-world coordinates.
        2. Generates unit quaternion from R and computes log q repr
        3. Normalizes translation according to mean and stdev

        Returns:
            (N, 6) vector: [t1, t2, t3, logq1, logq2, logq3]
        """
        R, T = poses
        cam_wvt = get_world_to_view_transform(R=R, T=T)
        pose_transform = cam_wvt.inverse().get_matrix()
        T = pose_transform[:, 3, :3]
        R = pose_transform[:, :3, :3]

        # Compute pose stats
        std_R, mean_R = torch.std_mean(R)
        std_T, mean_T = torch.std_mean(T)

        q = rc.matrix_to_quaternion(R)
        # q /= torch.norm(q)
        # q *= torch.sign(q[0])  # hemisphere constraint
        # logq = qlog(q)

        T -= mean_T
        T /= std_T

        return torch.cat((T, q), dim=1)

    @classmethod
    def preprocess_images(self, img: Image, img_size) -> np.ndarray:
        """Resize and normalize the images to range 0, 1
        """
        img = img.resize(img_size)

        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=2)

        # HWC to CHW
        img_trans = img_np.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __len__(self):
        return len(self.render_manager)

    def add_noise_to_frame(self, frame, noise_std=0.1, noise_fraction=0.1):
        """Gaussian noise + hot pixels
        """
        size = frame.size
        noise = noise_std * np.random.randn(*size) * 255
        if noise_fraction < 1.0:
            noise[np.random.rand(*size) >= noise_fraction] = 0
        return Image.fromarray((frame + noise).astype('uint8')).convert('L')

    def __getitem__(self, index: int):

        mask = self.render_manager.get_image("silhouette", index)

        event_frame = self.render_manager.get_event_frame(index)
        event_frame = self.add_noise_to_frame(event_frame)

        R, T = self.render_manager.get_trajectory_point(index)

        assert mask.size == event_frame.size, "Mask and event frame must be same size"

        mask = torch.from_numpy(self.preprocess_images(mask, self.img_size)).type(
            torch.FloatTensor
        )
        event_frame = torch.from_numpy(
            self.preprocess_images(event_frame, self.img_size)
        ).type(torch.FloatTensor)

        return event_frame, mask, R, T


class EvimoDataset(Dataset):
    """Dataset to manage Evimo Data"""

    def __init__(self, path: str, obj_id="1", is_train=True,slice_name=''):

        self.new_camera = None
        self.map1 = None
        self.map2 = None
        self.K = None
        self.discoef = None
        self.obj_id = obj_id

        self.slices_path = os.path.join(path, slice_name)
        self.frames_path = os.path.join(self.slices_path, 'slices')
        if not os.path.exists(self.slices_path) or not os.path.exists(self.frames_path):
            raise ValueError(f'Unavailable data in {self.slices_path}')
        if is_train:
            dataset_txt = eval(open(os.path.join(self.slices_path, "meta_train.txt")).read())
        else:
            dataset_txt = eval(open(os.path.join(self.slices_path, "meta_test.txt")).read())

        self.calib = dataset_txt["meta"]
        self.frames_dict = dataset_txt["frames"]
        self.set_undistorted_camera()

    @classmethod
    def preprocess_images(cls, img: np.ndarray) -> torch.Tensor:
        """Normalize and convert to torch"""
        if img.dtype == np.uint16:
            img = img.astype(np.uint8)

        if img.max() > 1:
            img = img / 255

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        torch_img = torch.from_numpy(img)
        torch_img = torch_img.permute(2, 0, 1).float()

        return torch_img


    def set_undistorted_camera(self):
        # evimo data is fisheye camera
        K = np.zeros([3, 3])
        K[0, 0] = self.calib['fy']
        K[0, 2] = self.calib['cy']
        K[1, 1] = self.calib['fx']
        K[1, 2] = self.calib['cx']
        K[2, 2] = 1.0
        # for fisheye
        # self.discoef = np.array([self.calib['k1'],
        # self.calib['k2'], self.calib['k3'], self.calib['k4']])
        # self.new_camera = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        # K, self.discoef, (w, h), R=None, new_size=(w, h))
        # self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
        # K, self.discoef, R=np.eye(3), P=self.new_camera, size=(w, h), m1type=cv2.CV_32FC1)
        w, h = self.calib['res_y'], self.calib['res_x']
        self.K = K
        # for rodtan undistortion
        alpha = 0.0
        self.discoef = np.array([self.calib['k1'], self.calib['k2'], 0.0, 0.0, self.calib['k3']])
        self.new_camera, _ = cv2.getOptimalNewCameraMatrix(K, self.discoef, (w, h), alpha, (w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, self.discoef, np.eye(3), self.new_camera, (w, h), cv2.CV_32FC1)

    @classmethod
    def evimo_to_pytorch3d_xyz(self, p: dict):
        x_pt3d = float(p["t"]["y"])
        y_pt3d = float(p["t"]["x"])
        z_pt3d = -float(p["t"]["z"])
        t = torch.Tensor([x_pt3d, y_pt3d, z_pt3d]).unsqueeze(0)
        return t

    @classmethod
    def evimo_to_pytorch3d_Rotation(self, p: dict):
        pos_q = torch.Tensor([float(e) for e in p['q'].values()])
        pos_R = rc.quaternion_to_matrix(pos_q)
        pos_R = pos_R.transpose(1, 0)
        R = torch.Tensor(np.zeros((3, 3), dtype=float))
        R[0, 0], R[0, 1], R[0, 2] = pos_R[1, 1], pos_R[1, 0], -pos_R[1, 2]
        R[1, 0], R[1, 1], R[1, 2] = pos_R[0, 1], pos_R[0, 0], -pos_R[0, 2]
        R[2, 0], R[2, 1], R[2, 2] = -pos_R[2, 1], -pos_R[2, 0], pos_R[2, 2]
        return R

    def prepare_pose(self, p: dict) -> Transform3d:
        # transform evimo coordinate system to pytorch3d coordinate system
        pos_t = self.evimo_to_pytorch3d_xyz(p)
        pos_R = self.evimo_to_pytorch3d_Rotation(p)
        R_tmp = Rotate(pos_R)
        w2v_transform = R_tmp.translate(pos_t)

        return Transform3d(matrix=w2v_transform.get_matrix())

    def get_new_camera(self):
        return self.new_camera

    def __len__(self):
        return len(self.frames_dict)

    def __getitem__(self, idx: int):

        # Get Event Frame and mask
        event_path = os.path.join(self.frames_path, self.frames_dict[idx]["event_frame"])
        event_frame = cv2.imread(event_path, cv2.IMREAD_UNCHANGED)
        event_frame = cv2.remap(event_frame, self.map1, self.map2, cv2.INTER_LINEAR)

        mask_path = os.path.join(self.frames_path, self.frames_dict[idx]["mask_frame"])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.remap(mask, self.map1, self.map2, cv2.INTER_LINEAR)

        # mask = mask[:, :, 2]
        mask[mask > 1] = 1
        event_frame = self.preprocess_images(event_frame)
        mask = self.preprocess_images(mask)

        # Cam Pose and Object Pose
        curr_frame = self.frames_dict[idx]
        obj_pos = self.prepare_pose(curr_frame[self.obj_id]["pos"])

        o2c_mat = obj_pos.get_matrix()
        R = o2c_mat[:, :3, :3]
        t = o2c_mat[:, 3, :3]

        return event_frame, mask, R, t




def test_sampler():

    dt1 = EvMaskPoseDataset(1, Params())
    dt2 = EvMaskPoseDataset(2, Params())
    dt3 = EvMaskPoseDataset(3, Params())
    dt4 = EvMaskPoseDataset(4, Params())
    cdt = ConcatDataset([dt1, dt2, dt3, dt4])
    custom_sampler = ConcatDataSampler(cdt, 4, True)

    assert len(custom_sampler) == len(cdt) // 4
    for n in custom_sampler:
        assert n is not None
