import logging
import itertools

import torch
import torch.nn.functional as F
from segpose.layers import DoubleConv, Down, OutConv, Up
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.encoder_out_channel = 1024 // factor

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, self.encoder_out_channel)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        self.x5 = self.down4(x4)
        x = self.up1(self.x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @classmethod
    def load(cls, params):
        """Method for loading the UNet either from a dict or from params
        """
        net = cls(params.n_channels, params.n_classes, params.bilinear)
        if params.unet_model_cpt:
            checkpoint = torch.load(params.model_cpt, map_location=params.device)
            net.load_state_dict(checkpoint["model"])

        net.to(device=params.device)

        return net


class UNetUpdated(nn.Module):
    """
    UNet architecture adapted from
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.encoder = [
            DoubleConv(n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor),
        ]

        self.decoder = [
            Up(1024, 512 // factor, bilinear),
            Up(512, 256 // factor, bilinear),
            Up(256, 128 // factor, bilinear),
            Up(128, 64, bilinear),
            OutConv(64, n_classes),
        ]

    def forward(self, x):

        x = x.float()
        skips = {}
        for count, layer in enumerate(self.encoder):
            skips[count] = x
            x = layer(x)

        for count, layer in enumerate(self.decoder):
            x = layer(x, skips[len(skips.values()) - count])

        return x

    @classmethod
    def load(cls, params):
        """Method for loading the UNet either from a dict or from params
        """
        net = cls(params.n_channels, params.n_classes, params.bilinear)
        if params.unet_model_cpt:
            checkpoint = torch.load(params.model_cpt, map_location=params.device)
            net.load_state_dict(checkpoint["model"])

        net.to(device=params.device)

        return net


class SegPoseNet(nn.Module):
    """
    Joint UNet and pose estimation wrapper
    """

    def __init__(self, unet: nn.Module, params):
        super(SegPoseNet, self).__init__()

        self.droprate = params.droprate

        self.unet = unet

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        print("encoder out channel: ", self.unet.encoder_out_channel)
        print("params feat dim: ", params.feat_dim)
        self.fc = nn.Linear(self.unet.encoder_out_channel, params.feat_dim)

        self.fc_xyz = nn.Linear(params.feat_dim, 3)
        self.fc_wpqr = nn.Linear(params.feat_dim, 3)

    def forward(self, x):

        s = x.size()
        x = x.view(-1, *s[2:]).unsqueeze(1)
        mask_pred = self.unet(x)

        x = self.unet.x5

        x = self.avgpool(x)
        x = torch.flatten(x, 1).cuda()
        x = self.fc(x)
        x = F.relu(x)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        poses = torch.cat((xyz, wpqr), 1)
        poses = poses.view(s[0], s[1], -1)

        return mask_pred, poses

    def parameters(self):
        """Generator that returns parameters only for pose model
        """
        for name, param in self.named_parameters(recurse=True):
            if name.split(".")[0] == "unet":
                continue
            yield param
            
            
    @classmethod
    def load(cls, unet: nn.Module, params):
        """Method for loading the UNet either from a dict or from params
        """
        net = cls(unet, params)
        if params.segpose_model_cpt:
            checkpoint = torch.load(params.segpose_model_cpt, map_location=params.device)
            net.load_state_dict(checkpoint["model"])

        net.to(device=params.device)

        return net
