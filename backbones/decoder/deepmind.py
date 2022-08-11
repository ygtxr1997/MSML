
import os
import torch
from torch import nn

__all__ = ['dm_decoder',]


"""
Patch Encoders / Decoders as used by DeepMind in their sonnet repo example:
https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F

# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(n_hid, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            nn.ConvTranspose2d(2 * n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(n_hid, 2 * n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            ResBlock(2 * n_hid, 2 * n_hid // 4),
            nn.ConvTranspose2d(2 * n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # nn.Conv2d(n_hid, 2 * n_hid, 3, padding=1),
            # nn.ReLU(),
            # ResBlock(2 * n_hid, 2 * n_hid // 4),
            # ResBlock(2 * n_hid, 2 * n_hid // 4),
            # nn.ConvTranspose2d(2 * n_hid, n_hid, 4, stride=2, padding=1),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def l2_loss(self, recover, ori):
        return torch.nn.MSELoss()(recover, ori)

    def forward(self, x, ori=None):
        recover = self.net(x)
        loss = self.l2_loss(recover, ori) if ori is not None else 0.
        return recover, loss


def _deepmind_decoder(arch, block, layers, pretrained, **kwargs):
    model = DeepMindDecoder(n_hid=64,
                            **kwargs)

    if pretrained:
        raise NotImplementedError('Pretrained model not support!')

    # model = model.eval()
    # model = model.cuda()
    return model


def dm_decoder(pretrained=False, **kwargs):
    return _deepmind_decoder('arcface18', ResBlock, None, pretrained, **kwargs)


if __name__ == '__main__':

    batch_size = 1
    dim_feature = 512

    """ Test for DeepMindDecoder """
    ires = dm_decoder(n_init=dim_feature)
    vector = torch.ones((batch_size, dim_feature, 7, 7))
    ori = torch.ones((batch_size, 3, 112, 112))
    ires.eval()
    recover, loss = ires(vector, ori)
    print(recover.shape, 'loss =', loss)

    import thop
    flops, params = thop.profile(ires, inputs=(vector, ))
    print('flops', flops / 1e9, 'params', params / 1e6)
