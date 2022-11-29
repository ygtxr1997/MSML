import os
import torch

import backbones.peer.arcface as iresnet
from config import config_init, load_yaml


class Saver(object):
    def __init__(self,
                 msml_folder: str,
                 iresnet_pth: str,
                 arch: str,
                 ):
        self.msml_folder = msml_folder
        self.iresnet_pth = iresnet_pth
        self.arch = arch

        self._load_msml_from_folder()
        self._check_saved_iresnet_pth()

    def _load_msml_from_folder(self):
        print('loading msml model from %s...' % self.msml_folder)
        cfg = load_yaml(os.path.join(self.msml_folder, 'config.yaml'))
        config_init(cfg)
        msml_state_dict = torch.load(os.path.join(self.msml_folder, 'backbone.pth'))

        ires: torch.nn.Module = eval('iresnet.%s' % self.arch)(
            pretrained=False
        )
        ires_state_dict = ires.state_dict()
        for key in ires_state_dict:
            ires_state_dict[key] = None  # erase initial weights

        ''' load weights from msml '''
        for key in msml_state_dict:
            if 'frb' in key:  # 'frb.features.weight'
                frb_key = key[len('frb.'):]  # 'features.weight'
                if frb_key in ires_state_dict:
                    ires_state_dict[frb_key] = msml_state_dict[key]
        ires.load_state_dict(ires_state_dict)
        torch.save(ires.state_dict(), self.iresnet_pth)
        print('Save state_dict to %s' % self.iresnet_pth)

    def _check_saved_iresnet_pth(self):
        print('Checking saved pth: %s' % self.iresnet_pth)
        ires: torch.nn.Module = eval('iresnet.%s' % self.arch)(
            pretrained=False
        )
        ires.load_state_dict(torch.load(self.iresnet_pth))
        x_in = torch.randn(2, 3, 112, 112)
        feat, inter = ires(x_in)
        print('output feature shape:', feat.shape)


if __name__ == '__main__':
    msml_folder = '/gavin/code/MSML/out/cos50_no_occ_2/'
    iresnet_pth = '/gavin/code/MSML/backbones/pretrained/cos50_no_occ_2.pth'
    arch = 'arcface50'

    demo = Saver(
        msml_folder,
        iresnet_pth,
        arch,
    )
