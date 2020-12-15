import configparser
from dataclasses import asdict, dataclass

from torch import optim
from utils.params_base import ParamsBase


@dataclass
class Params(ParamsBase):

    name: str = ""
    gpu_num: str = ""
    # Folder Params
    train_dir: str = "data/renders/train2_dolphin"
    pred_dir: str = "data/renders/test_chair_shapenet"
    segpose_model_cpt: str = ""
    unet_model_cpt: str = ""
    config_file: str = ""

    # Image Params
    img_size: tuple = (280, 280)

    # Training Params
    batch_size: int = 6
    mini_batch_size: int = 1
    epochs: int = 2
    val_split: float = 0.1

    # UNet Training Params
    unet_optimizer = optim.RMSprop
    unet_learning_rate: float = 1e-4
    unet_weight_decay: float = 1e-8
    unet_momentum: float = 0.9
    threshold_conf: float = 0.5
    train_unet: bool = True  # Flag for whether to learn the unet layers
    fine_tuning: bool = False

    # UNet Architecture Params
    n_channels: int = 1
    n_classes: int = 1
    bilinear: bool = False

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
