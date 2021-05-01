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

    @classmethod
    def from_dict(cls, dict_in: dict):
        return _from_dict(cls, dict_in)

@dataclass
class Params(ParamsBase):

    # For training
    is_real_data: bool = False
    config_file: str = ""
    name: str = ""
    gpu_num: str = ""
    # Folder Params
    train_dir: str = ""
    pred_dir: str = ""
    model_cpt: str = ""

    # Image Params
    img_size: tuple = (280, 280)

    # Training Params
    unet_batch_size: int = 6
    unet_mini_batch_size: int = 1
    unet_epochs: int = 30
    val_split: float = 0.1

    # UNet Training Params
    train_unet: bool = True
    unet_optimizer = optim.RMSprop
    unet_learning_rate: float = 1e-4
    unet_weight_decay: float = 1e-8
    unet_momentum: float = 0.9
    threshold_conf: float = 0.5
      # Flag for whether to learn the unet layers

    # UNet Architecture Params
    n_channels: int = 1
    n_classes: int = 1
    layers: int = 4
    depth: int = 64
    bilinear: bool = Truess

    # For testing
    # Params for mesh reconstruction

    test_dir: str = ""
    im_show: bool = True
    output_dir: str = ""

    evimo_obj_id: str = "1"

    mesh_steps: int = 300
    mesh_batch_size: int = 24
    mesh_sphere_level: int = 2
    mesh_sphere_scale: float = 0.10
    mesh_pose_init_noise_var: float = 50.0
    mesh_optimizer = optim.Adam
    mesh_learning_rate: float = 0.002
    mesh_betas: Tuple[float, float] = (0.9, 0.99)
    mesh_show_step: int = 50

    lambda_iou: float = 1.0
    lambda_laplacian: float = .5
    lambda_flatten: float = 0.05