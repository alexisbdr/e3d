import numpy as np
import torch
import torch.nn as nn

"""
Set of losses implemented as Pytorch nn Modules
"""


class IOULoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(IOULoss, self).__init__()
        self.eps = eps

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "Predict and target must be same shape"
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims) + self.eps
        union = (predict + target - predict * target).sum(dims) + self.eps

        return 1.0 - (intersect / union).sum() / intersect.nelement()


class DiceCoeffLoss(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super(DiceCoeffLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, target):
        assert (
            predict.shape[0] == target.shape[0]
        ), "Predict and target must be same shape"
        inter = torch.dot(predict.view(-1), target.view(-1))
        union = torch.sum(predict) + torch.sum(target) + self.eps

        t = (2 * inter.float() + self.eps) / union.float()
        return t

