from collections import OrderedDict

import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.manager import RenderManager
from utils.pose_utils import qlog


class EvMaskPoseDataset(Dataset):
    def __init__(self, dir_num: int, params, transforms: list = []):

        self.img_size = params.img_size
        self.transforms = transforms
        self.render_manager = RenderManager.from_directory(
            dir_num=dir_num, render_folder=params.train_dir
        )

        self._pose_stats = None

    def pose_stats(self):
        """Computes or returns pose stats
        """
        if self._pose_stats:
            return self._pose_stats

        R, T = self.render_manager._trajectory

        std_R, mean_R = torch.std_mean(R)
        std_T, mean_T = torch.std_mean(T)

        self._pose_stats = dict(std_R=std_R, mean_R=mean_R, std_T=std_T, mean_T=mean_T)

        return self._pose_stats

    def __len__(self):
        return len(self.render_manager)

    @classmethod
    def preprocess(self, img: Image, img_size):
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

    def preprocess_poses(cls, pose: tuple):
        """Convert rotation matrix to log quaternion representation
        Returns a 1x7 vector containing
        """

        R, T = pose

        q = rc.matrix_to_quaternion(R)
        q = torch.sign(q[0])  # hemisphere constraint
        logq = qlog(q)

        T = T.squeeze(0)
        T -= cls.pose_stats()["mean_T"]
        T /= cls.pose_stats()["std_T"]

        return torch.cat((T, q))

    def __getitem__(self, index: int):

        mask = self.render_manager.get_image("silhouette", index)

        event_frame = self.render_manager.get_event_frame(index)

        R, T = self.render_manager.get_trajectory_point(index)
        tq = self.preprocess_poses((R, T))
        pose_dict = OrderedDict(R=R, T=T, tq=tq)

        assert mask.size == event_frame.size, "Mask and event frame must be same size"

        mask = torch.from_numpy(self.preprocess(mask, self.img_size)).type(
            torch.FloatTensor
        )
        event_frame = torch.from_numpy(
            self.preprocess(event_frame, self.img_size)
        ).type(torch.FloatTensor)

        return event_frame, mask, pose_dict