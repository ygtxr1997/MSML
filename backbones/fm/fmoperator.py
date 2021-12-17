import torch
import torch.nn as nn
from torch import einsum

import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class resblock_bottle(nn.Module):  # bottle neck
    def __init__(self, in_channels, out_channels, bottle_channels=128):
        super(resblock_bottle, self).__init__()
        if in_channels <= 128:
            bottle_channels = in_channels // 2
        self.conv1 = conv1x1(in_channels, bottle_channels)
        self.bn1 = nn.BatchNorm2d(bottle_channels, eps=1e-05, )
        self.prelu1 = nn.PReLU(bottle_channels)

        self.conv2 = conv3x3(bottle_channels, bottle_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(bottle_channels, eps=1e-05, )
        self.prelu2 = nn.PReLU(bottle_channels)

        self.conv3 = conv1x1(bottle_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-05, )

        self.prelu3 = nn.PReLU(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.prelu3(out)
        return out


class FMCnn(nn.Module):
    def __init__(self, height, width, channel):
        super(FMCnn, self).__init__()

        self.height = height
        self.width = width

        self.same_conv = nn.Conv2d(18 + channel, channel, kernel_size=3, stride=1, padding=1)
        self.res_block = self._make_layer_light(resblock_bottle, 2, channel, channel)

        self.mask_norm = torch.tanh

    def _make_layer_light(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, yf, yo):
        identity = yf
        x = torch.cat((yf, yo), dim=1)
        x = self.same_conv(x)
        x = self.res_block(x)
        x = self.mask_norm(x) + identity
        return x


if __name__ == '__main__':

    for layer in range(1, 5):

        print('*************************** layer [%d] ***************************' % layer)

        if layer == 1:
            shape_f = (1, 512, 7, 7)
            shape_o = (1, 18, 7, 7)
            dim = 512
            channel = 512
        elif layer == 2:
            shape_f = (1, 256, 14, 14)
            shape_o = (1, 18, 14, 14)
            dim = 256
            channel = 256
        elif layer == 3:
            shape_f = (1, 128, 28, 28)
            shape_o = (1, 18, 28, 28)
            dim = 512
            channel = 128
        elif layer == 4:
            shape_f = (1, 64, 56, 56)
            shape_o = (1, 18, 56, 56)
            dim = 1024
            channel = 64

        yf_i = torch.randn(shape_f).cpu()
        yo_j = torch.randn(shape_o).cpu()

        ''' type 1 '''
        mlm = FMCnn(
            height=yf_i.shape[2],
            width=yf_i.shape[3],
            channel=channel
        )

        z_i = mlm(yf_i, yo_j)
        print('output shape:', z_i.shape)

        from torchinfo import summary
        summary(mlm, [shape_f, shape_o], depth=0)
        from thop import profile
        flops, params = profile(mlm, inputs=(yf_i.cuda(), yo_j.cuda()))
        print('fm flops:', flops / 1e9, 'params:', params / 1e6)