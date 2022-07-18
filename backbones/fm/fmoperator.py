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
        self.channel_f = channel_f

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
        self.arith_strategy = arith_strategy
        self.arith = arithmetic[arith_strategy]

        """ Part5. Peer Distillation """
        self.en_conv = True
        if self.en_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channel_f, channel_f, 3, 1, 1),
                nn.BatchNorm2d(channel_f, eps=1e-05,),
                nn.PReLU(channel_f),
                nn.Conv2d(channel_f, channel_f, 3, 1, 1),
                nn.BatchNorm2d(channel_f, eps=1e-05, ),
                nn.PReLU(channel_f),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(channel_f, channel_f, 3, 1, 1),
                nn.BatchNorm2d(channel_f, eps=1e-05, ),
                nn.PReLU(channel_f),
                nn.Conv2d(channel_f, channel_f, 3, 1, 1),
                nn.BatchNorm2d(channel_f, eps=1e-05, ),
                nn.PReLU(channel_f),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(channel_f, channel_f, 3, 1, 1),
                nn.BatchNorm2d(channel_f, eps=1e-05, ),
                # nn.ReLU(inplace=True),
                # nn.Sigmoid(),
            )

        """ If true, some tensors will be deep copied. """
        self.en_save = False

    def _make_resblocks(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def _save_intermediate_features(self, feat_type: str, feat_tensor: torch.tensor):
        """
        Save 'contaminated', 'mask', and 'purified' features (torch.tensor),
        whose shape is (Batch * width * height * channel_f).
        Warning: If the number of GPUs is larger than 1, this setting will be useless!
                 Accessing to model.module.frb.fm_ops[0].xxx is illegal!
        :return: None
        """
        if not self.en_save:
            return

        with torch.no_grad():
            clone_tensor = feat_tensor.clone().flatten()
            clone_tensor = clone_tensor.cpu().numpy()

        if feat_type == 'contaminated':
            self.contaminated_feat = clone_tensor
        elif feat_type == 'mask':
            self.mask_feat = clone_tensor
            print(self.mask_feat.shape, 'mask saved.')
        elif feat_type == 'purified':
            self.purified_feat = clone_tensor
        else:
            raise ValueError('Intermediate feature type error!')

    def plot_intermediate_features(self,
                                   gt_occ_msk: torch.tensor,
                                   save_folder: str = ".",
                                   ):
        """
        Plot distributions of Y_f, M, and Z_f.
        :param save_folder:
        :param gt_occ_msk: 0 - no occlusion,
                           1 - occluded.
        :return:
        """
        import os.path
        import matplotlib.pyplot as plt
        from PIL import Image

        # convert gt_occ_msk to np.array
        print(gt_occ_msk.shape)  # (B, H0, W0)
        batch = gt_occ_msk.shape[0]
        print('start resize gt_occ_msk...')
        gt_occ_msk = gt_occ_msk.numpy().astype(np.uint8) * 255
        resized_occ_msk = np.zeros(shape=(batch, self.height, self.width))
        for b in range(batch):
            tmp = gt_occ_msk[b]
            tmp = Image.fromarray(tmp, mode='L')
            tmp = tmp.resize(size=(self.width, self.height))
            resized_occ_msk[b] = np.array(tmp) // 255
        resized_occ_msk = torch.tensor(resized_occ_msk, dtype=torch.uint8)
        print(resized_occ_msk.shape,
              resized_occ_msk.min(), resized_occ_msk.max())  # (B, H, W), [0, 1]

        resized_occ_msk = resized_occ_msk[:, None, :, :]
        resized_occ_msk = resized_occ_msk.repeat(1, self.channel_f, 1, 1)
        resized_occ_msk = resized_occ_msk.flatten().numpy().astype(np.uint8)
        assert resized_occ_msk.size == self.mask_feat.size

        print('start plot...')
        color_map = np.array([0.3, 0.7, ])  # (0-occ-0.3-purple, 1-clean-0.7-yellow)
        colors = resized_occ_msk.astype(np.float)
        colors[resized_occ_msk == 0] = color_map[0]
        colors[resized_occ_msk == 1] = color_map[1]

        ''' Plot 1. Contaminated vs. Mask '''
        img_name = 'fm_cm_{}_{}.jpg'.format(self.height, self.arith_strategy)
        plt.figure(dpi=300)
        plt.title(img_name)
        plt.xlabel('Contaminated Face Feature')
        plt.ylabel('Mask Generated by FM Operators')
        plt.scatter(x=self.contaminated_feat,
                    y=self.mask_feat,
                    s=1,
                    c=colors,
                    alpha=0.4,
                    )

        plt.savefig(os.path.join(save_folder, img_name))
        plt.clf()

        ''' Plot 2. Contaminated vs. Purified '''
        img_name = 'fm_cp_{}_{}.jpg'.format(self.height, self.arith_strategy)
        plt.figure(dpi=300)
        plt.title(img_name)
        plt.xlabel('Contaminated Face Feature')
        plt.ylabel('Face Feature Purified by FM Operators')
        plt.scatter(x=self.contaminated_feat,
                    y=self.purified_feat,
                    s=1,
                    c=colors,
                    alpha=0.4,
                    )
        x_min, x_max = self.contaminated_feat.min(), self.contaminated_feat.max()
        plt.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=1)  # curve y=x

        plt.savefig(os.path.join(save_folder, img_name))
        plt.clf()

    def forward(self, yf, yo, yt=None):
        """
        :param yf: facial features
        :param yo: occlusion segmentation representations
        :param yt: peer knowledge
        :return: Z_f, purified facial features have the same shape with yf
        """
        identity = yf
        x = torch.cat((yf, yo), dim=1)
        x = self.same_conv(x)
        x = self.res_block(x)
        x = self.mask_norm(x)
        self._save_intermediate_features('contaminated', identity)
        self._save_intermediate_features('mask', x)

        # m_bar = 1 - x
        m_bar = self.conv_m(x)
        f_out = m_bar * identity
        f_out = self.conv1(f_out) if self.en_conv else f_out
        if yt is not None:
            f_occ = m_bar * yt
            f_occ = self.conv2(f_occ) if self.en_conv else f_occ

        x = self.arith(identity, x)
        self._save_intermediate_features('purified', x)

        x += f_out  # close or open ?
        l2 = None
        if yt is not None:
            l2 = torch.nn.MSELoss()(f_occ, f_out)

        x += identity  # Using skip connection is better
        return x, l2


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