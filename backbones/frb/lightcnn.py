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


__all__ = ['lightcnn29',]


model_dir = {
    'LightCNN-9': '/home/yuange/code/SelfServer/MSML/backbones/pretrained/LightCNN_9Layers_checkpoint.pth.tar',
    'LightCNN-29': '/home/yuange/code/SelfServer/MSML/backbones/pretrained/LightCNN_29Layers_checkpoint.pth.tar',
    'LightCNN-29v2': '/home/yuange/code/SelfServer/MSML/backbones/pretrained/LightCNN_29Layers_V2_checkpoint.pth.tar'
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


class network_9layers(nn.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8 * 8 * 128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x


class network_29layers(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = mfm(8 * 8 * 128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_v2(nn.Module):
    def __init__(self, block,
                 layers,
                 fm_ops,
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

        """ Different from vanilla LightCNN,
        frb backbones only extract embedded features without classification layers
        """
        # self.fc2 = nn.Linear(dim_feature, num_classes, bias=False)

        """ List of Feature Masking Operators: [x, x, x, x] """
        assert len(fm_ops) == 4
        self.fm_ops = nn.ModuleList(fm_ops)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, segs):
        """ LightCNN in FRB
        input:
            img     - (B, 1, 128, 128)
            segs    - [(B, 18, 64, 64),   # seg3
                       (B, 18, 32, 32),   # seg2
                       (B, 18, 16, 16),   # seg1
                       (B, 18, 8, 8),]    # seg0
        output:
            feature - (B, dim_feature)
        """
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.fm_ops[0](x, segs[0])

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.fm_ops[1](x, segs[1])

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.fm_ops[2](x, segs[2])

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = self.fm_ops[3](x, segs[3])

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)

        return x


""" Vanilla LightCNN
"""
def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model


def LightCNN_29Layers(**kwargs):
    model = network_29layers(resblock, [1, 2, 3, 4], **kwargs)
    return model


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model

""" FRB version of LightCNN
"""
def lightcnn29(fm_ops,
               pretrained=True,
               dim_feature=256,
               dropout=0.,
               ):
    if pretrained:
        # customized model based on 'lightcnn'
        model = network_29layers_v2(resblock, [1, 2, 3, 4],
                                    fm_ops=fm_ops,
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
                tmp_dict[key[7:]] = pre_trained_dict[key]

        # get customized model layers which don't exist in 'lightcnn' and insert to tmp
        model_dict = model.state_dict()
        for key in model_dict:
            # print(key)
            # > block4.3.conv2.filter.bias
            if key not in tmp_dict:
                tmp_dict[key] = model_dict[key]

        print('=> Loading pre-trained LightCNN-29v2 ...')
        model.load_state_dict(tmp_dict)
        print('=> Loaded.')

    else:
        model = network_29layers_v2(resblock, [1, 2, 3, 4],
                                    fm_ops=fm_ops,
                                    dim_feature=dim_feature)

    return model


if __name__ == '__main__':

    """ Prepare for Feature Masking Operators """
    from backbones.fm import FMCnn, FMNone
    heights = [64, 32, 16, 8]
    f_channels = [48, 96, 192, 128]
    s_channels = [18, 18, 18, 18]
    fm_layers = [0, 1, 1, 0]

    fm_ops = []
    for i in range(4):
        fm_type = fm_layers[i]
        if fm_type == 0:
            fm_ops.append(FMNone())
        elif fm_type == 1:
            fm_ops.append(FMCnn(
                height=heights[i],
                width=heights[i],
                channel_f=f_channels[i]
            ))
        else:
            raise ValueError

    """ Prepare for Occlusion Segmentation Representations """
    segs = [torch.randn(1, 18, 64, 64),  # seg3
            torch.randn(1, 18, 32, 32),  # seg2
            torch.randn(1, 18, 16, 16),  # seg1
            torch.randn(1, 18, 8, 8),]  # seg0

    """ Test for lightcnn29 """
    light = lightcnn29(
        fm_ops=fm_ops,
        pretrained=True,
    )
    img = torch.randn((1, 1, 128, 128))
    feature = light(img, segs)
    print(feature.shape)

    import thop
    flops, params = thop.profile(light, inputs=(img, segs))
    print('flops', flops / 1e9, 'params', params / 1e6)