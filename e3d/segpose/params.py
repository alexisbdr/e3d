import configparser
from dataclasses import asdict, dataclass

from torch import optim
from utils.params_base import ParamsBase


@dataclass
class Params(ParamsBase):

    # Folder Params
    train_dir: str = "data/renders/test_dolphin"
    pred_dir: str = "data/renders/test_dolphin"
    model_cpt: str = "model_checkpoints/epochs15_batch4_end_dolphin.cpt"
    config_file: str = ""

    # Image Params
    img_size: tuple = (560, 560)

    # Training Params
    batch_size: int = 4
    epochs: int = 12
    val_split: float = 0.15

    # UNet Training Params
    unet_optimizer = optim.RMSprop
    unet_learning_rate: float = 1e-4
    unet_weight_decay: float = 1e-8
    unet_momentum: float = 0.9
    threshold_conf: float = 0.5
    train_unet: bool = False  # Flag for whether to learn the unet layers

    # UNet Architecture Params
    n_channels: int = 1
    n_classes: int = 1
    bilinear: bool = False

    # Pose Training Params
    pose_optimizer = optim.Adam
    pose_learning_rate: float = 1e-4
    pose_weight_decay: float = 5e-4
    train_pose: bool = True

    # Pose Architecture Params
    feat_dim: int = 2048
    droprate: float = 0.5
    beta: float = 3.0
    gamma: float = -3.0
