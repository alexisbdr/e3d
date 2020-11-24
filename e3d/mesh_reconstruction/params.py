from dataclasses import dataclass
from typing import Dict, Tuple

from torch import optim
from utils.params_base import ParamsBase


@dataclass
class Params(ParamsBase):

    img_size: Tuple[int, int] = (128, 128)

    # Training Params
    steps: int = 2000  # optimization loop
    batch_size: int = 2  # mesh mini batch

    # Optimizer Params
    optimizer = optim.Adam
    learning_rate: float = 0.005
    betas: Tuple[float, float] = (0.9, 0.99)  # NMR: 0.5, 0.99

    # Weights to be applied to each loss
    lambda_iou: float = 1.0
    lambda_laplacian: float = 1.0
    lambda_flatten: float = 0.01

    # Experiment inforation
    experiment_name: str = ""
    test_path: str = ""
    config_file: str = ""
    show: bool = False
