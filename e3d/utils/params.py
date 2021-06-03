import json
from utils.pyutils import _from_dict
from dataclasses import dataclass
from torch import optim
from typing import Tuple

@dataclass
class ParamsBase:
    def __post_init__(self):
        if self.config_file:
            with open(self.config_file) as f:
                dict_in = json.load(f)
                for key, value in dict_in.items():
                    if key in self.__annotations__:
                        setattr(self, key, value)

    def _set_with_dict(self, dict_in: dict):
        for key, value in dict_in.items():
            if key in self.__annotations__:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, dict_in: dict):
        return _from_dict(cls, dict_in)

    # used for writing the setting into config files
    def as_dict(self):
        dict = self.__dict__.copy()
        pop_keys = ["device", "logger"]
        for key in pop_keys:
            if key in dict.keys():
                dict.pop(key)
        return dict

@dataclass
class Params(ParamsBase):

    # Always used parameters
    # when your data is evimo, it is true
    is_real_data: bool = False
    # config file path
    config_file: str = ""
    # gpu id used
    gpu_num: str = ""
    img_size: tuple = (260, 346)
    # train data dir
    train_dir: str = ""
    # model path dir
    model_cpt: str = ""
    # segpose model checkpoint for Synthetic data
    segpose_model_cpt = ""
    # experiment output path
    exper_dir: str = ""
    # name of the experiments
    name: str = ""
    # for evimo data slices, you can check it on pydvs
    slice_name: str = 'slices_discre_0.001_width_0.025'

    # UNet Training Params
    unet_batch_size: int = 6
    unet_mini_batch_size: int = 1
    unet_epochs: int = 30
    val_split: float = 0.1
    train_unet: bool = True
    unet_optimizer = optim.RMSprop
    unet_learning_rate: float = 1e-4
    unet_weight_decay: float = 1e-8
    unet_momentum: float = 0.9
    fine_tuning: bool = True
    # UNet Architecture Params
    n_channels: int = 1
    n_classes: int = 1
    layers: int = 4
    depth: int = 64
    bilinear: bool = True
    # threshold used for deciding the silhouette
    threshold_conf: float = 0.5

    # Pose Training Params
    pose_optimizer = optim.Adam
    pose_learning_rate: float = 1e-4
    pose_weight_decay: float = 5e-4
    train_pose: bool = True
    train_relative: bool = True

    # Pose Architecture Params
    feat_dim: int = 2048
    droprate: float = 0.5
    beta: float = -3.0
    gamma: float = -3.0

    # For testing
    # Params for mesh reconstruction
    # show mesh reconstruction result
    im_show: bool = True
    # prediction dir
    pred_dir: str = ""
    # object id in EVIMO, "1" is car and "2" is plane
    evimo_obj_id: str = "1"

    # For mesh optimization
    mesh_steps: int = 300
    mesh_batch_size: int = 16
    mesh_sphere_level: int = 3
    mesh_sphere_scale: float = 0.10
    mesh_pose_init_noise_var: float = 50.0
    mesh_optimizer = optim.Adam
    mesh_learning_rate: float = 0.002
    mesh_betas: Tuple[float, float] = (0.9, 0.99)
    # step to log the info of the mesh reconstruction
    mesh_log: bool = False
    mesh_show_step: int = 50
    # mesh optimization loss functions
    lambda_iou: float = 1.0
    lambda_laplacian: float = .5
    lambda_flatten: float = 0.05

    # parameters used in RANSAC method applied on mesh reconstruction
    ransac_iou_threshold: int = 0.8
    ransac_least_samples: int = 30
    ransac_model_num: int = 5

    # for mesh load
    gt_mesh_path: str = ''
    # for synthesize data prediction
    synth_type: str = ''