"""
Segmentation network evaluation step
Uses DiceCoeffLoss to evaluate performance of the network
See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
"""
#Move to top-level import to catch losses
import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join("..", dirname(__file__))))

import torch.nn as nn
from torch.utils.data import DataLoader

from losses import DiceCoeffLoss

def eval_segmentation_net(net: nn.Module, loader: DataLoader):
    print()