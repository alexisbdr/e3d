import random
from collections import OrderedDict

import numpy as np
import pytorch3d.transforms.rotation_conversions as rc
import torch
from PIL import Image
from pytorch3d.renderer.cameras import get_world_to_view_transform
from segpose.params import Params
from torch.utils.data import (BatchSampler, ConcatDataset, Dataset, Sampler,
                              SubsetRandomSampler)
from utils.manager import RenderManager
from utils.pose_utils import qlog


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
                dir_num=dir_num, render_folder=params.train_dir, datamode='jjp'
            )
            self.render_manager.rectify_paths(base_folder=params.train_dir)
        except:
            self.render_manager = None

        if self.render_manager is not None:
            self.poses = self.preprocess_poses(self.render_manager._trajectory)

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


# TODO ?
    def add_noise_to_frame(self, frame, noise_std=0.1, noise_fraction=0.1):
        """Gaussian noise + hot pixels
        """
        # noise = noise_std * np.randn_like(*frame.shape)
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
        tq = self.poses[index : index + 1]

        assert mask.size == event_frame.size, "Mask and event frame must be same size"

        mask = torch.from_numpy(self.preprocess_images(mask, self.img_size)).type(
            torch.FloatTensor
        )
        event_frame = torch.from_numpy(
            self.preprocess_images(event_frame, self.img_size)
        ).type(torch.FloatTensor)

        return event_frame, mask, R, T, tq


class EvMaskPoseBatchedDataset(Dataset):
    def __init__(self, steps: int, dir_num: int, params, transforms: list = []):
        """Provides a wrapper around EvMaskPoseDataset to batch the getitem call
        """
        self.steps = steps
        self.dataset = EvMaskPoseDataset(dir_num, params, transforms)

    def __getitem__(self, index: int):
        """Returns a set of items randomly distributed along a set interval size
        """
        subset = range(
            index * self.steps - self.steps, index * self.steps + 2 * self.steps
        )
        sublist = [max(min(x, len(self.dataset) - 1), 0) for x in subset]
        data = [self.dataset[i] for i in sorted(random.sample(sublist, k=self.steps))]

        out_data = []
        for elem in range(len(data[0])):
            out_data.append(torch.cat([d[elem] for d in data], dim=0))

        return out_data

    def __len__(self):
        return int(len(self.dataset) / self.steps)


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
