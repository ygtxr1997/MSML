import torch
from torch import nn

__all__ = ['iresnet18', 'iresnet34', 'iresnet50',]


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
                 fm_ops,
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

        """ List of Feature Masking Operators: [x, x, x, x] """
        assert len(fm_ops) == 4
        self.fm_ops = nn.ModuleList(fm_ops)

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

    def forward(self, x, segs):
        """ IResNet in FRB
        input:
            img     - (B, 3, 112, 112)
            segs    - [(B, 18, 56, 56),   # seg3
                       (B, 18, 28, 28),   # seg2
                       (B, 18, 14, 14),   # seg1
                       (B, 18, 7, 7),]    # seg0
        output:
            feature - (B, dim_feature)
        """
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)  # stem stage

            x = self.layer1(x)  # (64, 56, 56)
            x = self.fm_ops[0](x, segs[0])

            x = self.layer2(x)  # (128, 28, 28)
            x = self.fm_ops[1](x, segs[1])

            x = self.layer3(x)  # (256, 14, 14)
            x = self.fm_ops[2](x, segs[2])

            x = self.layer4(x)  # (512, 7, 7)
            x = self.fm_ops[3](x, segs[3])

            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        print(x.shape)
        x = self.features(x)
        return x


""" Vanilla IResNet
"""
def _iresnet_v(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model

def iresnet18_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet152_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet152', IBasicBlock, [3, 21, 48, 3], pretrained,
                    progress, **kwargs)


def iresnet200_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


""" FRB version of IResNet
"""
def _iresnet(block, layers, fm_ops, pretrained, **kwargs):
    model = IResNet(block, layers, fm_ops, **kwargs)
    if pretrained:
        raise ValueError('No pretrained model for iresnet')
    return model

def iresnet18(fm_ops,
           pretrained=False,
           dim_feature=512):
    return _iresnet(IBasicBlock, [2, 2, 2, 2],
                 fm_ops, pretrained,
                 dim_feature=dim_feature)

def iresnet34(fm_ops,
           pretrained=False,
           dim_feature=512):
    return _iresnet(IBasicBlock, [3, 4, 6, 3],
                 fm_ops, pretrained,
                 dim_feature=dim_feature)

def iresnet50(fm_ops,
           pretrained=False,
           dim_feature=512):
    return _iresnet(IBasicBlock, [3, 4, 14, 3],
                 fm_ops, pretrained,
                 dim_feature=dim_feature)


if __name__ == '__main__':

    batch_size = 1

    """ Prepare for Feature Masking Operators """
    from backbones.fm import FMCnn, FMNone
    heights = [56, 28, 14, 7]
    f_channels = [64, 128, 256, 512]
    s_channels = [18, 18, 18, 18]
    fm_layers = [1, 1, 1, 1]

    fm_ops = []
    for i in range(4):
        fm_type = fm_layers[i]
        print('fm_type', i, '=', fm_type)
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
    segs = [torch.randn(batch_size, 18, 56, 56),  # seg3
            torch.randn(batch_size, 18, 28, 28),  # seg2
            torch.randn(batch_size, 18, 14, 14),  # seg1
            torch.randn(batch_size, 18, 7, 7),]  # seg0

    """ Test for iresnet18 """
    ires = iresnet18(
        fm_ops=fm_ops,
    )
    img = torch.randn((batch_size, 3, 112, 112))
    ires.eval()
    feature = ires(img, segs)
    print(feature.shape)

    import thop
    flops, params = thop.profile(ires, inputs=(img, segs))
    print('flops', flops / 1e9, 'params', params / 1e6)
