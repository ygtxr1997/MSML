import os.path

import torch
from torch import nn

__all__ = ['iresnet18', 'iresnet34', 'iresnet50',
           'iresnet18_v', 'iresnet28_v', 'iresnet34_v', 'iresnet50_v',]


model_dir = {
    'arc18': '/gavin/code/MSML/backbones/pretrained/',
    'arc34': '/gavin/code/MSML/backbones/pretrained/',
    'arc50': '/gavin/code/MSML/out/arc50_no_occ_2/backbone.pth'
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
                 fm_ops,
                 dim_feature=512,
                 dropout=0,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 fp16=False,
                 peer_params: dict = None,
                 ):
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

        """ Peer """
        from backbones.peer import arcface18, arcface34, arcface50
        from backbones.peer import cosface50_casia
        self.peer = None  # peer type is consistent with msml.header_type
        self.header_type = peer_params.get('header_type').lower()
        if peer_params.get('use_ori'):
            if 'arc' in self.header_type:
                if layers == [2, 2, 2, 2]:  # iresnet18
                    self.peer = arcface18().requires_grad_(False)
                elif layers == [3, 4, 6, 3]:  # iresnet34
                    self.peer = arcface34().requires_grad_(False)
                elif layers == [3, 4, 14, 3]:  # iresnet50
                    self.peer = arcface50().requires_grad_(False)
            elif 'cos' in self.header_type:
                if layers == [3, 4, 14, 3]:  # iresnet50
                    print('[Peer] cos50_casia loaded.')
                    self.peer = cosface50_casia().requires_grad_(False)
            else:
                raise ValueError('Error type of iresnet, cannot decide peer network.')

        """ Recover """
        from backbones.decoder import dm_decoder
        self.decoder = lambda a, b: (None, 0)
        if peer_params.get('use_decoder'):
            self.decoder = dm_decoder(n_init=dim_feature)

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

    def forward(self, x, segs, ori):
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

        # Peer knowledge
        ft0, ft1, ft2, ft3 = None, None, None, None
        if ori is not None:
            _, inter = self.peer(ori)
            ft0, ft1, ft2, ft3 = inter[0], inter[1], inter[2], inter[3]

        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)  # stem stage

            x = self.layer1(x)  # (64, 56, 56)
            x, l = self.fm_ops[0](x, segs[0], ft0)

            x = self.layer2(x)  # (128, 28, 28)
            x, l1 = self.fm_ops[1](x, segs[1], ft1)

            x = self.layer3(x)  # (256, 14, 14)
            x, l2 = self.fm_ops[2](x, segs[2], ft2)

            x = self.layer4(x)  # (512, 7, 7)
            x, l3 = self.fm_ops[3](x, segs[3], ft3)

            x = self.bn2(x)

            # recover
            _rec, l4 = self.decoder(x, ori) if ori is not None else None, 0.

            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        l = l + l1 + l2 + l3 if ori is not None else 0.  # KD
        l = l + l4 if ori is not None else 0.  # Recover
        return x, l * 1.0


""" Vanilla IResNet
"""
class IResNetVanilla(nn.Module):
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
        super(IResNetVanilla, self).__init__()
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
        return x


def _iresnet_v(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNetVanilla(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet28_v(pretrained=False, progress=True, **kwargs):
    return _iresnet_v('iresnet28', IBasicBlock, [3, 4, 3, 3], pretrained,
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
def _iresnet(arch, block, layers, fm_ops, pretrained, **kwargs):
    model = IResNet(block, layers, fm_ops, **kwargs)
    if pretrained:
        model_type = arch.replace('iresnet', 'arc')  # 'iresnet50' to 'arc50'
        if os.path.isfile(model_dir[model_type]):
            pre_trained_weights = torch.load(model_dir[model_type],
                                             map_location=torch.device('cpu'))
        else:
            error_info = 'Make sure the file {' + model_dir['arc50'] + '} exists!'
            raise FileNotFoundError(error_info)

        # get pretrained 'arcxx' layers and insert to tmp
        from collections import OrderedDict
        tmp_dict = OrderedDict()
        for key in pre_trained_weights:
            # print(key)
            # > frb.conv1.weight
            # if 'fc2' not in key:  # do not skip classification FC layer
            if 'frb' in key:  # only load frb weights
                tmp_dict[key[len('frb.'):]] = pre_trained_weights[key]

        # get 'iresnet' model layers which don't exist in 'arcxx' and insert to tmp
        model_dict = model.state_dict()
        for key in model_dict:
            # print(key)
            # > conv1.weight
            if key not in tmp_dict:  # 'iresnet' may include 'fm_ops', but 'arcxx' does not
                tmp_dict[key] = model_dict[key]

        print('=> Loading pre-trained %s ...' % model_type)
        model.load_state_dict(tmp_dict)
        print('=> Loaded.')
    return model

def iresnet18(fm_ops,
              pretrained=False,
              dim_feature=512,
              dropout=0.,
              peer_params=None,
              ):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2],
                    fm_ops, pretrained,
                    dim_feature=dim_feature,
                    dropout=dropout,
                    peer_params=peer_params,
                    )

def iresnet34(fm_ops,
              pretrained=False,
              dim_feature=512,
              dropout=0.,
              peer_params=None,
              ):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3],
                    fm_ops, pretrained,
                    dim_feature=dim_feature,
                    dropout=dropout,
                    peer_params=peer_params,
                    )

def iresnet50(fm_ops,
              pretrained=False,
              dim_feature=512,
              dropout=0.,
              peer_params=None,
              ):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3],
                    fm_ops, pretrained,
                    dim_feature=dim_feature,
                    dropout=dropout,
                    peer_params=peer_params,
                    )


if __name__ == '__main__':
    import thop

    batch_size = 1
    default_peer_params = {
        'use_ori': False,
        'use_conv': False,
        'mask_trans': 'conv',
        'use_decoder': False,
    }

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
                channel_f=f_channels[i],
                peer_params=default_peer_params,
            ))
        else:
            raise ValueError

    """ Prepare for Occlusion Segmentation Representations """
    segs = [torch.randn(batch_size, 18, 56, 56),  # seg3
            torch.randn(batch_size, 18, 28, 28),  # seg2
            torch.randn(batch_size, 18, 14, 14),  # seg1
            torch.randn(batch_size, 18, 7, 7),]  # seg0

    """ Test for iresnet50 (msml) """
    # ires = iresnet50(
    #     fm_ops=fm_ops,
    #     pretrained=True,
    #     peer_params=default_peer_params,
    # )
    # img = torch.randn((batch_size, 3, 112, 112))
    # ires.eval()
    # feature, loss = ires(img, segs, None)
    # print(feature.shape)
    #
    # import thop
    # flops, params = thop.profile(ires, inputs=(img, segs, None))
    # print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

    """ Test for iresnet (vanilla) """
    ires_v = iresnet50_v()
    img = torch.randn((batch_size, 3, 112, 112))
    ires_v.eval()
    feature = ires_v(img)
    print(feature.shape)

    flops, params = thop.profile(ires_v, inputs=(img, ), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

    """ IResNet vanilla """
    # model = iresnet18_v().cuda()
    # weight = torch.load('demo/backbone.pth')
    # model.load_state_dict(weight)
    # model.eval()
    #
    # from PIL import Image
    # img_1 = Image.open('demo/retina_1.png').convert('RGB')
    # img_2 = Image.open('demo/retina_2.png').convert('RGB')
    #
    # import torchvision.transforms as transforms
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5])
    # ])
    # x = trans(img_1).cuda()
    # y = trans(img_2).cuda()
    # x = x[None, :, :, :]
    # y = y[None, :, :, :]
    #
    # x = model(x)
    # y = model(y)
    #
    # res = torch.cosine_similarity(x, y, dim=-1).cpu()
    # print(res)
    # # print('cosine sim = %.4f' % res)