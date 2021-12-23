import torch
import torch.nn as nn

from backbones.osb import unet
from backbones.frb import lightcnn29
from backbones.fm import FMCnn, FMNone


class MSML(nn.Module):
    def __init__(self,
                 frb_type,
                 osb_type,
                 fm_layers,
                 num_classes,
                 fp16=False,
                 ):
        super(MSML, self).__init__()
        assert len(fm_layers) == 4
        self._prepare_shapes(frb_type, osb_type)
        self._prepare_fm(fm_layers)
        self._prepare_frb(frb_type)
        self._prepare_osb(osb_type)

        self.classification = nn.Linear(self.dim_feature, num_classes, bias=False)

        self.fp16 = fp16

    def _prepare_shapes(self, frb_type, osb_type):
        """ We should know the output shapes of each stage in FRB and OSB """
        if 'lightcnn' in frb_type:
            self.input_size = 128
            self.gray = True
            self.heights = [64, 32, 16, 8]
            self.f_channels = [48, 96, 192, 128]
            self.dim_feature = 256
        else:
            raise ValueError('FRB type error')

        if 'unet' in osb_type:
            self.s_channels = [18, 18, 18, 18]
        else:
            raise ValueError('OSB type error')

    def _prepare_fm(self, fm_layers):
        fm_ops = []
        for i in range(4):
            fm_type = fm_layers[i]
            if fm_type == 0:
                fm_ops.append(FMNone())
            elif fm_type == 1:
                fm_ops.append(FMCnn(
                    height=self.heights[i],
                    width=self.heights[i],
                    channel_f=self.f_channels[i]
                ))
            else:
                raise ValueError('FM Operators type error')
        self.fm_ops = nn.ModuleList(fm_ops)

    def _prepare_frb(self, frb_type):
        if 'lightcnn' in frb_type:
            self.frb = lightcnn29(
                self.fm_ops,
            )

    def _prepare_osb(self, osb_type):
        if 'unet' in osb_type:
            self.osb = unet(
                backbone='r18',
                gray=self.gray,
                input_size=self.input_size,
            )

    def forward(self, x):
        seg_list = self.osb(x)  # [seg0, seg1, seg2, seg3, seg5] small to big
        seg_list.reverse()  # [seg5, seg3, seg2, seg1, seg0]
        segs = seg_list[1:]  # [seg3, seg2, seg1, seg0] big to small

        with torch.cuda.amp.autocast(self.fp16):
            feature = self.frb(x, segs)

        feature = feature.float() if self.fp16 else feature
        if self.training:
            x = self.classification(feature)
            return x
        else:
            return feature


if __name__ == '__main__':

    """ 1. IResNet accepts RGB images """
    # print('-------- Test for msml(iresnet-18, unet-r18, rgb-112) --------')
    # msml = MSML(
    #     frb_type='iresnet',
    #     osb_type='unet',
    #     fm_layers=[0, 1, 1, 0],
    #     num_classes=98310,
    #     fp16=True
    # ).cuda()
    # img = torch.zeros((1, 3, 112, 112)).cuda()
    # msml.eval()
    # out = msml(img)
    # print(out.shape)
    #
    # import thop
    # flops, params = thop.profile(msml, inputs=(img,))
    # print('flops', flops / 1e9, 'params', params / 1e6)

    """ 2. IResNet accepts Gray-Scale images """
    print('-------- Test for msml(lightcnn, unet-r18, gray-128) --------')
    msml = MSML(
        frb_type='lightcnn',
        osb_type='unet',
        fm_layers=[0, 0, 0, 0],
        num_classes=98310,
        fp16=False
    ).cuda()
    img = torch.zeros((1, 1, 128, 128)).cuda()
    msml.eval()
    out = msml(img)
    print(out.shape)

    import thop
    flops, params = thop.profile(msml, inputs=(img,))
    print('flops', flops / 1e9, 'params', params / 1e6)
