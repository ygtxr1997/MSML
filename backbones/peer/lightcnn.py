'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['lightcnn29_v2',]


model_dir = {
    'LightCNN-9': '/home/yuange/code/SelfServer/MSML/backbones/pretrained/LightCNN_9Layers_checkpoint.pth.tar',
    'LightCNN-29': '/home/yuange/code/SelfServer/MSML/backbones/pretrained/LightCNN_29Layers_checkpoint.pth.tar',
    'LightCNN-29v2': '/gavin/code/MSML/backbones/pretrained/LightCNN_29Layers_V2_checkpoint.pth.tar'
}


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1,
                         padding=1)
        # NOTE: the 'input' and 'output' of 'mfm-based resblock' share the same (height, width)

    def forward(self, x):  # if x=66, in_c=66, out_c=96
        res = x
        out = self.conv1(x)  # out=96
        out = self.conv2(out)  # out=96
        out = out + res  # res=66
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block,
                 layers,
                 dim_feature=256,
                 dropout=0.,
                 ):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, dim_feature)
        self.drop = nn.Dropout(p=dropout, inplace=True)

        """
        Peer lightcnn backbones only extract embedded features without classification layers
        """
        # self.fc2 = nn.Linear(dim_feature, num_classes, bias=False)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, ori):
        """ LightCNN in FRB
        input:
            ori     - (B, 1, 128, 128)
        output:
            feature - (B, dim_feature)
            inter   - [(B, 48, 64, 64),    # ft0
                       (B, 96, 32, 32),   # ft1
                       (B, 192, 16, 16),   # ft2
                       (B, 128, 8, 8),]    # ft3
        """
        inter = []
        x = self.conv1(ori)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        inter.append(x.detach())

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        inter.append(x.detach())

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        inter.append(x.detach())

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        inter.append(x.detach())

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)

        return x, inter


""" Vanilla LightCNN
"""
def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model

""" Peer version of LightCNN
"""
def lightcnn29_v2(pretrained=True,
                  dim_feature=256,
                  dropout=0.,
                  ):
    if pretrained:
        # customized model based on 'lightcnn'
        model = network_29layers_v2(resblock, [1, 2, 3, 4],
                                    dim_feature=dim_feature,
                                    dropout=dropout,
                                    )

        # load pretrained weight
        if os.path.isfile(model_dir['LightCNN-29v2']):
            pre_trained_dict = torch.load(model_dir['LightCNN-29v2'],
                                          map_location=torch.device('cpu'))['state_dict']
        else:
            error_info = 'Make sure the file {' + model_dir['LightCNN-29v2'] + '} exists!'
            raise FileNotFoundError(error_info)

        # get pretrained 'lightcnn' layers and insert to tmp
        tmp_dict = OrderedDict()
        for key in pre_trained_dict:
            # print(key)
            # > module.block4.3.conv2.filter.bias
            if 'fc2' not in key:  # skip classification FC layer
                tmp_dict[key[7:]] = pre_trained_dict[key]  # len('module.') == 7

        # get customized model layers which don't exist in 'lightcnn' and insert to tmp
        model_dict = model.state_dict()
        for key in model_dict:
            # print(key)
            # > block4.3.conv2.filter.bias
            if key not in tmp_dict:
                tmp_dict[key] = model_dict[key]

        model.load_state_dict(tmp_dict)
        print('=> [Peer] Pre-trained LightCNN-29v2 Loaded.')

    else:
        model = network_29layers_v2(resblock, [1, 2, 3, 4],
                                    dim_feature=dim_feature,
                                    dropout=dropout)

    return model


if __name__ == '__main__':

    """ Test for lightcnn29_v2 """
    light = lightcnn29_v2(
        pretrained=True,
    )
    img = torch.randn((1, 1, 128, 128))
    feature, inter = light(img)
    print(feature.shape)
    for ft in inter:
        print(ft.shape)

    import thop
    flops, params = thop.profile(light, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
