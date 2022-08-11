
import os
import torch
from torch import nn

__all__ = ['arcface18', 'arcface34', 'arcface50',]


model_dir = {
    'arcface18': './backbones/pretrained/r18-backbone.pth',
    'arcface34': './backbones/pretrained/r34-backbone.pth',
    'arcface50': './backbones/pretrained/r50-backbone.pth',
    'arcface100': './backbones/pretrained/r100-backbone.pth',
}


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


class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block,
                 layers,
                 dim_feature=512,
                 dropout=0,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, dim_feature)
        self.features = nn.BatchNorm1d(dim_feature, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
        """ IResNet for peer (or teacher)
        input:
            img     - (B, 3, 112, 112)

        output:
            feature - (B, dim_feature)
            inter   - [(B, 64, 56, 56),    # ft0
                       (B, 128, 28, 28),   # ft1
                       (B, 256, 14, 14),   # ft2
                       (B, 512, 7, 7),]    # ft3
        """
        inter = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)  # stem stage

            x = self.layer1(x)  # (64, 56, 56)
            inter.append(x.detach())

            x = self.layer2(x)  # (128, 28, 28)
            inter.append(x.detach())

            x = self.layer3(x)  # (256, 14, 14)
            inter.append(x.detach())

            x = self.layer4(x)  # (512, 7, 7)
            inter.append(x.detach())

            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x, inter


""" Vanilla IResNet
"""
def _iresnet_v(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)

    if pretrained:
        # load pretrained weight
        if os.path.isfile(model_dir[arch]):
            weight = torch.load(model_dir[arch], map_location=torch.device('cpu'))
            model.load_state_dict(weight)
        else:
            error_info = 'Make sure the file {' + model_dir[arch] + '} exists!'
            raise FileNotFoundError(error_info)

    model = model.eval()
    # model = model.cuda()
    return model

def arcface18(pretrained=True, progress=True, **kwargs):
    return _iresnet_v('arcface18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def arcface34(pretrained=True, progress=True, **kwargs):
    return _iresnet_v('arcface34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def arcface50(pretrained=True, progress=True, **kwargs):
    return _iresnet_v('arcface50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def arcface100(pretrained=True, progress=True, **kwargs):
    return _iresnet_v('arcface100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


# def iresnet152_v(pretrained=False, progress=True, **kwargs):
#     return _iresnet_v('iresnet152', IBasicBlock, [3, 21, 48, 3], pretrained,
#                     progress, **kwargs)
#
#
# def iresnet200_v(pretrained=False, progress=True, **kwargs):
#     return _iresnet_v('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
#                     progress, **kwargs)


if __name__ == '__main__':

    batch_size = 1

    """ Test for arcface18 """
    ires = arcface18()
    img = torch.randn((batch_size, 3, 112, 112))
    ires.eval()
    feature, inter = ires(img)
    print(feature.shape)
    for ft in inter:
        print(ft.shape)

    import thop
    flops, params = thop.profile(ires, inputs=(img, ))
    print('flops', flops / 1e9, 'params', params / 1e6)
