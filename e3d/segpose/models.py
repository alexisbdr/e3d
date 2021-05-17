import logging
import os
import torch
from segpose.layers import DoubleConv, Down, OutConv, Up
from torch import nn
import numpy as np

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
        if params.model_cpt:
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
        # Method for loading the UNet either from a dict or from params
        net = cls(params.n_channels, params.n_classes, params.bilinear)
        if params.model_cpt:
            checkpoint = torch.load(params.model_cpt, map_location=params.device)
            net.load_state_dict(checkpoint["model"])

        net.to(device=params.device)

        return net


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @classmethod
    def load(cls, params, net=None):
        """Method for loading models either from a dict or from params"""
        if not net:
            net = cls()

        if params.model_cpt:
            params.logger.info(
                f"Loading Model using checkpoint file {params.model_cpt}"
            )
            checkpoint = torch.load(params.model_cpt, map_location=params.device)
            net.load_state_dict(checkpoint["model"])
        else:
            # Apply weight initialization if there is no checkpoint to load from
            net.apply(weights_init)

        net.to(device=params.device)

        return net

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class UNetDynamic(BaseModel):
    def __init__(
        self,
        in_channels,
        n_classes,
        depth: int = 64,
        layers: int = 4,
        bilinear=True,
    ):
        super(UNetDynamic, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if depth not in [8, 16, 32, 64]:
            raise ValueError("Incorrect input kernel size")
        self.depth = depth

        if layers not in [1, 2, 3, 4]:
            raise ValueError("Incorrect Layer size")
        self.layers = layers

        factor = 2 if bilinear else 1

        inc = DoubleConv(in_channels, depth)
        self.encoder = nn.ModuleList()
        self.encoder.append(inc)
        for i in range(layers):
            self.encoder.append(
                Down(
                    depth * pow(2, i),
                    depth * pow(2, i + 1) // (factor if i == layers - 1 else 1),
                )
            )

        self.decoder = nn.ModuleList()
        for i in range(layers, 0, -1):
            self.decoder.append(
                Up(
                    depth * pow(2, i),
                    depth * pow(2, i - 1) // (factor if i != 1 else 1),
                    bilinear,
                )
            )
        self.outc = OutConv(depth, n_classes)

    def forward(self, x):
        res = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i != len(self.encoder) - 1:
                res.append(x)
        self.encoder_out = x
        for up in self.decoder:
            x = up(x, res.pop())

        out = self.outc(x)
        return out

    @classmethod
    def load(cls, params):

        net = cls(
            int(params.n_channels),
            int(params.n_classes),
            int(params.depth),
            int(params.layers),
            bool(params.bilinear),
        )

        return super().load(params, net)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight"):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
