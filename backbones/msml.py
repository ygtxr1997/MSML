import torch
import torch.nn as nn

from backbones.osb import unet
from backbones.frb import lightcnn29
from backbones.frb import iresnet18, iresnet34, iresnet50
from backbones.fm import FMCnn, FMNone

from headers import Softmax, AMCosFace, AMArcFace


__all__ = ['MSML', ]


class MSML(nn.Module):
    frb_type_list = ('lightcnn',
                     'iresnet18', 'iresnet34', 'iresnet50',)
    osb_type_list = ('unet',)
    head_type_list = ('Softmax', 'AMArcFace', 'AMCosFace')
    def __init__(self,
                 frb_type: str,
                 osb_type: str,
                 fm_layers: tuple,
                 num_classes: int,
                 fp16: bool = False,
                 frb_pretrained: bool = False,
                 fm_params: tuple = (3, 2, 'tanh', 'add'),  # (S, N, act, arith)
                 header_type: str = 'Softmax',
                 header_params: tuple = (64.0, 0.5, 0.0, 0.0),  # (s, m, a, k)
                 dropout: float = 0.,
                 use_osb: bool = True,
                 peer_params: dict = None,
                 ):
        super(MSML, self).__init__()
        assert len(fm_layers) == 4
        self._prepare_shapes(frb_type, osb_type)
        self._prepare_fm(fm_layers, fm_params, peer_params)
        self._prepare_frb(frb_type, dropout, peer_params, header_type, pretrained=frb_pretrained)
        self._prepare_osb(osb_type)

        self.num_classes = num_classes
        self._prepare_header(header_type, header_params)

        self.fp16 = fp16
        self.use_osb = use_osb

    def _prepare_shapes(self, frb_type, osb_type):
        """ We should know the output shapes of each stage in FRB and OSB """
        if 'lightcnn' in frb_type:
            self.input_size = 128
            self.gray = True
            self.heights = (64, 32, 16, 8)
            self.f_channels = (48, 96, 192, 128)
            self.dim_feature = 256
        elif 'iresnet' in frb_type:
            self.input_size = 112
            self.gray = False
            self.heights = (56, 28, 14, 7)
            self.f_channels = (64, 128, 256, 512)
            self.dim_feature = 512
        else:
            raise ValueError('FRB type error')

        if 'unet' in osb_type:
            self.s_channels = (18, 18, 18, 18)
        else:
            raise ValueError('OSB type error')

    def _prepare_fm(self, fm_layers, fm_params, peer_params):
        fm_ops = []
        for i in range(4):
            fm_type = fm_layers[i]
            if fm_type == 0:  # 0: don't use FM Operator
                fm_ops.append(FMNone())
            elif fm_type == 1:  # 1: use CNN FM Operator
                kernel_size, num_res, act, arith = fm_params
                fm_ops.append(FMCnn(
                    height=self.heights[i],
                    width=self.heights[i],
                    channel_f=self.f_channels[i],
                    kernel_size=kernel_size,
                    resblocks=num_res,
                    activation=act,
                    arith_strategy=arith,
                    peer_params=peer_params,
                ))
            else:
                raise ValueError('FM Operators type error')
        self.fm_ops = fm_ops  # should be a 'List'

    def _prepare_frb(self, frb_type, dropout=0., peer_params: dict = None,
                     header_type: str = "",
                     pretrained=False):
        peer_params["header_type"] = header_type
        if 'lightcnn' in frb_type:
            self.frb = lightcnn29(
                self.fm_ops,
                pretrained=pretrained,
                dropout=dropout,
                peer_params=peer_params
            )
        elif 'iresnet' in frb_type:
            if '18' in frb_type:
                self.frb = iresnet18(self.fm_ops, pretrained=pretrained,
                                     dropout=dropout, peer_params=peer_params)
            elif '34' in frb_type:
                self.frb = iresnet34(self.fm_ops, pretrained=pretrained,
                                     dropout=dropout, peer_params=peer_params)
            elif '50' in frb_type:
                self.frb = iresnet50(self.fm_ops, pretrained=pretrained,
                                     dropout=dropout, peer_params=peer_params)
            else:
                error_info = 'IResNet type {} not found'.format(frb_type)
                raise ValueError(error_info)

    def _prepare_osb(self, osb_type):
        if 'unet' in osb_type:
            self.osb = unet(
                backbone='r18',
                gray=self.gray,
                input_size=self.input_size,
            )

    def _prepare_header(self, head_type, header_params):
        assert head_type in self.head_type_list
        dim_in = self.dim_feature
        dim_out = self.num_classes

        """ Get hyper-params of header """
        s, m, a, k = header_params

        """ Choose the header """
        if 'Softmax' in head_type:
            self.classification = Softmax(dim_in, dim_out, device_id=None)
        elif 'AMCosFace' in head_type:
            self.classification = AMCosFace(dim_in, dim_out,
                                            device_id=None,
                                            s=s, m=m,
                                            a=a, k=k,
                                            )
        elif 'AMArcFace' in head_type:
            self.classification = AMArcFace(dim_in, dim_out,
                                            device_id=None,
                                            s=s, m=m,
                                            a=a, k=k,
                                            )
        else:
            raise ValueError('Header type error!')

    def forward(self, x, label=None, ori=None):
        """ Part 1. OSB
        The output order of OSB should be carefully processed.
        """
        if self.use_osb:
            seg_list = self.osb(x)  # [seg0, seg1, seg2, seg3, seg5] small to big
            seg_list.reverse()  # [seg5, seg3, seg2, seg1, seg0]
            final_seg = seg_list[0]  # seg5, the final segmentation for calculating seg_loss
            segs = seg_list[1:]  # [seg3, seg2, seg1, seg0] big to small
        else:
            segs = (None, None, None, None)
            final_seg = None

        """ Part 2. FRB 
        Note that FRB only returns latent feature rather than classification prediction.
        """
        with torch.cuda.amp.autocast(self.fp16):
            feature, kd = self.frb(x, segs, ori)

        feature = feature.float() if self.fp16 else feature
        if self.training:
            final_cls = self.classification(feature, label) + kd
            return final_cls, final_seg, kd
        else:
            return feature, final_seg


if __name__ == '__main__':

    default_peer_params = {
        "use_ori": True,
        "use_conv": True,
        "mask_trans": 'conv',
        "use_decoder": False
    }

    """ 1. IResNet accepts RGB images """
    print('-------- Test for msml (iresnet-18, unet-r18, rgb-112) --------')
    msml = MSML(
        frb_type='iresnet50',
        osb_type='unet',
        fm_layers=(1, 1, 1, 1),
        num_classes=98310,
        fp16=True,
        header_type='AMArcFace',
        header_params=(64.0, 0.4, 0.0, 0.0),
        peer_params=default_peer_params,
    ).cuda()
    img = torch.zeros((1, 3, 112, 112)).cuda()
    msml.eval()
    pred_cls, pred_seg = msml(img)
    print(pred_cls.shape, pred_seg.shape)

    import thop
    flops, params = thop.profile(msml, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))

    """ 2. LightCNN accepts Gray-Scale images """
    print('-------- Test for msml(lightcnn, unet-r18, gray-128) --------')
    msml = MSML(
        frb_type='lightcnn',
        osb_type='unet',
        fm_layers=(1, 1, 1, 1),
        frb_pretrained=True,
        num_classes=98310,
        fp16=False,
        header_type='Softmax',
        header_params=(64.0, 0.5, 0.0, 0.0),
        peer_params=default_peer_params,
    ).cuda()
    img = torch.zeros((1, 1, 128, 128)).cuda()
    msml.eval()
    pred_cls, pred_seg = msml(img)
    print(pred_cls.shape, pred_seg.shape)

    import thop
    flops, params = thop.profile(msml, inputs=(img,), verbose=False)
    print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))
