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

    def as_dict(self):
        dict = self.__dict__.copy()
        pop_keys = ["device", "logger"]
        for key in pop_keys :
            if key in dict.keys():
                dict.pop(key)
        return dict

@dataclass
class Params(ParamsBase):

    # Always used parameters
    is_real_data: bool = False
    config_file: str = ""
    gpu_num: str = ""
    img_size: tuple = (260, 346)
    # Folder Params
    train_dir: str = ""

    model_cpt: str = ""
    exper_dir: str = ""
    name: str = ""

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
    # UNet Architecture Params
    n_channels: int = 1
    n_classes: int = 1
    layers: int = 4
    depth: int = 64
    bilinear: bool = True

    threshold_conf: float = 0.5

    # For testing
    # Params for mesh reconstruction

    im_show: bool = True
    pred_dir: str = ""
    evimo_obj_id: str = "1"

    # For mesh optimization
    mesh_steps: int = 300
    mesh_batch_size: int = 16
    mesh_sphere_level: int = 2
    mesh_sphere_scale: float = 0.10
    mesh_pose_init_noise_var: float = 50.0
    mesh_optimizer = optim.Adam
    mesh_learning_rate: float = 0.002
    mesh_betas: Tuple[float, float] = (0.9, 0.99)
    mesh_show_step: int = 50
    # mesh optimization loss functions
    lambda_iou: float = 1.0
    lambda_laplacian: float = .5
    lambda_flatten: float = 0.05

    ransac_iou_threshold = 0.8

    # for mesh load
    gt_mesh_path: str = ''
    # for synthesize data prediction
    synth_type: str = ''

    # for evimo data slices
    slice_name: str = 'slices_discre_0.001_width_0.025'