import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np


__all__ = ['unet',]

""" 
Implementation of Large Kernel Matters Paper (face++)
Author: Xiangtai(lxtpku@pku.edu.cn)
"""
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


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


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class Unet(nn.Module):
    def __init__(self, block,
                 layers,
                 groups=1,
                 num_classes=2,
                 kernel_size=7,
                 dap_k=3,
                 gray=True,  # important
                 input_size=128,  # important
                 ):
        super(Unet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = 64

        if gray:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )

        self.gcm1 = _GlobalConvModule(512, num_classes * 4, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(256, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(128, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(64, num_classes * dap_k ** 2, (kernel_size, kernel_size))
        self.gcm5 = _GlobalConvModule(64, num_classes * dap_k ** 2, (kernel_size, kernel_size))

        if input_size == 128:
            self.deconv1 = nn.ConvTranspose2d(num_classes * 4, num_classes * dap_k ** 2, kernel_size=4, stride=2,
                                              padding=1, bias=False)
        elif input_size == 112:
            self.deconv1 = nn.ConvTranspose2d(num_classes * 4, num_classes * dap_k ** 2, kernel_size=3, stride=2,
                                              padding=1, bias=False)
        else:
            print('Error in input_size.')
        self.deconv2 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4, stride=2, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(2 * num_classes * dap_k ** 2, num_classes * dap_k ** 2, kernel_size=4, stride=2,
                                          padding=1, bias=False)

        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k, dap_k))
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        msk_feats = []

        # suppose input = (3, 112, 112) | (1, 128, 128)
        t = self.conv1(x)
        s = self.bn1(t)
        x0 = self.prelu(s)  # (64, 56, 56) | (64, 64, 64)

        x1 = self.layer1(x0)  # (64, 28, 28) | (128, 32, 32)

        x2 = self.layer2(x1)  # (128, 14, 14) | (256, 16, 16)

        x3 = self.layer3(x2)  # (256, 7, 7) | (512, 8, 8)

        x4 = self.layer4(x3)  # (512, 4, 4) | (1024, 4, 4)

        xx = self.bn2(x4)

        x_ = self.gcm1(xx)  # (2*4, 4, 4) | (2*4, 4, 4)

        seg0 = self.deconv1(x_)  # (2*9, 7, 7) | (2*9, 8, 8)

        x3_ = self.gcm2(x3)
        seg1 = self.deconv2(torch.cat((seg0, x3_), 1))  # (2*9, 14, 14) | (2*9, 16, 16)

        x2_ = self.gcm3(x2)
        seg2 = self.deconv3(torch.cat((seg1, x2_), 1))  # (2*9, 28, 28) | (2*9, 32, 32)

        x1_ = self.gcm4(x1)
        seg3 = self.deconv4(torch.cat((seg2, x1_), 1))  # (2*9, 56, 56) | (2*9, 64, 64)

        x0_ = self.gcm5(x0)
        seg5_ = self.deconv5(torch.cat((seg3, x0_), 1))  # (2*9, 112, 112) | (2*9, 128, 128)

        seg5 = self.DAP(seg5_)  # (2, 112, 112) | (2, 128, 128)

        """ Use detach link or not """
        # # 1. Use detach link (default)
        msk_feats.append(seg0.detach())
        msk_feats.append(seg1.detach())
        msk_feats.append(seg2.detach())
        msk_feats.append(seg3.detach())
        # # # 2. Do not use detach link
        # msk_feats.append(seg0)
        # msk_feats.append(seg1)
        # msk_feats.append(seg2)
        # msk_feats.append(seg3)
        """ End """

        msk_feats.append(seg5)

        return msk_feats


def unet(pre_trained=False,
         backbone='r18',
         gray=True,
         input_size=128,
         **kwargs):
    if pre_trained:
        print('No pretrained model for mskfuse29_light_y_seg')
    else:
        if 'r18' in backbone:
            model = Unet(IBasicBlock, [2, 2, 2, 2],
                         num_classes=2,
                         gray=gray,
                         input_size=input_size,
                         **kwargs)
        elif 'r34' in backbone:
            model = Unet(IBasicBlock, [3, 4, 6, 3], num_classes=2,
                         gray=gray,
                         input_size=input_size,
                         **kwargs)
        elif 'r50' in backbone:
            model = Unet(IBasicBlock, [3, 4, 14, 3], num_classes=2,
                         gray=gray,
                         input_size=input_size,
                         **kwargs)
        elif 'r100' in backbone:
            model = Unet(IBasicBlock, [3, 13, 30, 3], num_classes=2,
                         gray=gray,
                         input_size=input_size,
                         **kwargs)
        elif 'r200' in backbone:
            model = Unet(IBasicBlock, [6, 26, 60, 6], num_classes=2,
                         gray=gray,
                         input_size=input_size,
                         **kwargs)
        else:
            print('Error backbone type in OSB.')
    return model


if __name__ == '__main__':

    """ 1. IResNet accepts RGB images """
    print('-------- Test for osb-r18-rgb-112 --------')
    osb = unet(backbone='r18',
               gray=False,
               input_size=112)
    img = torch.zeros((1, 3, 112, 112))
    out = osb(img)
    print(out[-1].shape)

    import thop
    flops, params = thop.profile(osb, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

    """ 2. IResNet accepts Gray-Scale images """
    print('-------- Test for osb-r18-gray-128 --------')
    osb = unet(backbone='r18',
               gray=True,
               input_size=128)
    img = torch.zeros((1, 1, 128, 128))
    out = osb(img)
    for idx in range(4):
        print(idx, out[idx].shape)

    import thop
    flops, params = thop.profile(osb, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
