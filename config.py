import os.path

import yaml
from easydict import EasyDict as edict

""" Only used for training """
if os.path.exists('config.yaml'):
    with open('config.yaml') as f:
        loaded = yaml.safe_load(f)
    conf = edict(loaded)

""" Main Function of Initializing Config """
def config_init(cfg: edict):
    config_dataset(cfg)
    config_recipe(cfg)
    config_model(cfg)
    config_exp(cfg)

""" 1. Dataset """
def config_dataset(cfg: edict):

    cfg.is_gray = False
    cfg.out_size = (112, 112)
    cfg.use_norm = True

    if cfg.dataset == 'ms1m-retinaface-t2':
        cfg.rec = '/tmp/train_tmp/ms1m-retinaface'  # mount on RAM
        cfg.nw = 32
        cfg.num_classes = 93431
        cfg.num_epoch = 25
        cfg.warmup_epoch = -1
        cfg.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
                [m for m in [11, 17, 22] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001

        import numpy as np
        cfg.min_lr = 0
        def lr_fun_cos(cur_epoch):
            """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
            lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.num_epoch))
            return (1.0 - cfg.min_lr) * lr + cfg.min_lr

        cfg.warmup_factor = 0.3
        def lr_step_func_cos(epoch):
            cur_lr = lr_fun_cos(cur_epoch=epoch) * cfg.lr
            if epoch < cfg.warmup_epoch:
                alpha = epoch / cfg.warmup_epoch
                warmup_factor = cfg.warmup_factor * (1.0 - alpha) + alpha
                cur_lr *= warmup_factor
            return lr_fun_cos(cur_epoch=epoch)
            # return cur_lr / cfg.lr

        cfg.lr_func = lr_step_func

    elif cfg.dataset == 'webface':
        cfg.rec = '/tmp/train_tmp/casia'  # mount on RAM
        cfg.nw = 32
        cfg.num_classes = 10572
        cfg.num_epoch = 34
        cfg.warmup_epoch = -1
        cfg.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
                [m for m in [20, 28, 32] if m - 1 <= epoch])
        cfg.lr_func = lr_step_func

""" 2. Training Recipe """
def config_recipe(cfg: edict):
    # cfg.fp16 = True
    cfg.momentum = 0.9
    cfg.weight_decay = 5e-4
    # cfg.batch_size = 128  # 128
    cfg.lr = 0.1  # 0.1 for batch size is 512

    cfg.lambda1 = 1  # l_total = l_cls + lambda1 * l_seg

""" 3. Model Setting """
def config_model(cfg: edict):
    """ FRB, OSB, FM Operators """
    # cfg.frb_type = 'lightcnn' # 'iresnet18'
    cfg.pretrained = False
    # cfg.osb_type = 'unet'
    # cfg.use_osb = False
    # cfg.fm_layers = (0, 0, 0, 0)  # (fm1, fm2, fm3, fm4)
    cfg.fm_layers = tuple(cfg.fm_layers)
    """ Classification Header """
    # cfg.header_type = 'Softmax'
    # cfg.header_params = (64.0, 0.4, 0.0, 0.0)  # (s, m, a, k)
    cfg.header_params = tuple(cfg.header_params)
    cfg.dim_feature = 512
    """ PartialFC """
    cfg.sample_rate = 1

    if cfg.frb_type == 'lightcnn':
        cfg.is_gray = True
        cfg.out_size = (128, 128)
        cfg.use_norm = False
        cfg.pretrained = True
        cfg.lr = 0.001  # 0.001 for pretrained model of batch size 512
        cfg.dim_feature = 256

    """ Peer-Guided Default Params """
    default_peer_params = edict({
        'use_ori': False,
        'use_conv': False,
        'mask_trans': 'conv',
        'use_decoder': False,
    })
    if cfg.get('peer_params') is None:
        cfg.peer_params = default_peer_params

""" 4. Experiment Record """
def config_exp(cfg: edict):
    # cfg.exp_id = 1
    out_folder = 'out'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    cfg.output = os.path.join(out_folder, cfg.output_prefix + '_' + str(cfg.exp_id))
    print('output path: ', cfg.output)

""" yaml to edict """
def load_yaml(file_name: str):
    with open(file_name) as f:
        loaded = yaml.safe_load(f)
    loaded = edict(loaded)
    return loaded


if __name__ == '__main__':
    print(conf)
    config_init(conf)
    print(conf)
