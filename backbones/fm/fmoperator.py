import torch
import torch.nn as nn
from torch import einsum

import numpy as np

from einops.layers.torch import Rearrange
from einops import rearrange


__all__ = ['FMCnn', 'FMNone']


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


def arith_add(face: torch.tensor, mask: torch.tensor):
    return face + mask

def arith_sub(face: torch.tensor, mask: torch.tensor):
    return face - mask

def arith_div(face: torch.tensor, mask: torch.tensor):
    return face / mask

def arith_mul(face: torch.tensor, mask: torch.tensor):
    return face * mask


class FMCnn(nn.Module):
    def __init__(self, height,
                 width,
                 channel_f,
                 kernel_size=3,
                 resblocks=2,
                 activation='tanh',
                 arith_strategy='add',
                 ):
        super(FMCnn, self).__init__()

        # Y_f and Y_s share the same height and width
        self.height = height
        self.width = width

        """ Part1. 2D Filter """
        self.same_conv = conv3x3(18 + channel_f, channel_f)
        if kernel_size == 1:
            self.same_conv = conv1x1(18 + channel_f, channel_f)

        """ Part2. ResBlock """
        self.res_block = self._make_resblocks(resblock_bottle,
                                              num_blocks=resblocks,
                                              in_channels=channel_f,
                                              out_channels=channel_f)

        """ Part3. Activation Function """
        act_dict = {
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
        }
        self.mask_norm = act_dict[activation]

        """ Part4. Arithmetic Strategy """
        arithmetic = {
            'add': arith_add,
            'sub': arith_sub,
            'div': arith_div,
            'mul': arith_mul,
        }
        self.arith = arithmetic[arith_strategy]

    def _make_resblocks(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, yf, yo):
        """
        :param yf: facial features
        :param yo: occlusion segmentation representations
        :return: Z_f, purified facial features have the same shape with yf
        """
        identity = yf
        x = torch.cat((yf, yo), dim=1)
        x = self.same_conv(x)
        x = self.res_block(x)
        x = self.mask_norm(x)
        x = self.arith(identity, x)
        x += identity
        return x


class FMNone(nn.Module):
    def __init__(self,):
        super(FMNone, self).__init__()

    def forward(self, yf, yo):
        """
        :param yf: facial features
        :param yo: occlusion segmentation representations
        :return: yf, do nothing!
        """
        return yf


if __name__ == '__main__':

    gray = False
    if not gray:
        # iresnet, input=(1, 3, 112, 112)
        # torch.Size([1, 64, 56, 56])                                                                            │
        # torch.Size([1, 128, 28, 28])                                                                            │
        # torch.Size([1, 256, 14, 14])                                                                           │
        # torch.Size([1, 512, 7, 7])
        heights = [56, 28, 14, 7]
        f_channels = [64, 128, 256, 512]
    else:
        # lightcnn, input=(1, 1, 128, 128)
        # torch.Size([1, 48, 64, 64])                                                                            │
        # torch.Size([1, 96, 32, 32])                                                                            │
        # torch.Size([1, 192, 16, 16])                                                                           │
        # torch.Size([1, 128, 8, 8])
        heights = [64, 32, 16, 8]
        f_channels = [48, 96, 192, 128]

    s_channels = [18, 18, 18, 18]

    for layer in range(1, 5):

        print('*************************** layer [%d] ***************************' % layer)

        idx = layer - 1
        height = heights[idx]
        f_channel = f_channels[idx]
        s_channel = s_channels[idx]

        yf_i = torch.randn((1, f_channel, height, height)).cpu()
        yo_j = torch.randn((1, s_channel, height, height)).cpu()

        ''' type 1 '''
        fm = FMCnn(
            height=yf_i.shape[2],
            width=yf_i.shape[3],
            channel_f=f_channel
        )

        z_i = fm(yf_i, yo_j)
        print('output shape:', z_i.shape)
        assert z_i.shape == yf_i.shape

        # from torchinfo import summary
        # summary(mlm, [shape_f, shape_s], depth=0)
        from thop import profile
        flops, params = profile(fm.cuda(), inputs=(yf_i.cuda(), yo_j.cuda()))
        print('fm flops:', flops / 1e9, 'params:', params / 1e6)