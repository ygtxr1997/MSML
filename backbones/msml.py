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

        self.classification = nn.Linear(self.dim_feature, 1000, bias=False)

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
        self.fm_ops = fm_ops

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
        segs = self.osb(x)
        feature = self.frb(x, segs)
        x = self.classification(feature)
        return x


def _prepare_fm(self, fm_op, fm_layers):
    self.heights = [64, 32, 16, 8]
    self.f_channels = [48, 96, 192, 128]
    self.s_channels = [18, 18, 18, 18]

    fm_ops = []
    for i in range(4):
        if i in fm_layers:
            fm_ops.append(fm_op(
                height=self.heights[i],
                width=self.heights[i],
                channel_f=self.f_channels[i]
            ))
        else:
            fm_ops.append(fm_op())
    self.fm_ops = fm_ops