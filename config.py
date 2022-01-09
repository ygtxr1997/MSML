from easydict import EasyDict as edict

cfg = edict()
cfg.dataset = "ms1m-retinaface-t2"

""" Main Function of Initializing Config """
def config_init():
    config_dataset()
    config_recipe()
    config_model()
    config_exp()

""" 1. Dataset """
def config_dataset():

    cfg.is_gray = False
    cfg.out_size = (112, 112)
    cfg.use_norm = True

    if cfg.dataset == 'ms1m-retinaface-t2':
        cfg.rec = '/tmp/train_tmp/ms1m-retinaface'  # mount on RAM
        cfg.nw = 0
        cfg.num_classes = 93431
        cfg.num_epoch = 25
        cfg.warmup_epoch = -1
        cfg.val_targets = ['lfw', 'cfg_fp', 'agedb_30']

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
        cfg.rec = '/tmp/train_tmp/casia_webface'  # mount on RAM
        cfg.nw = 0
        cfg.num_classes = 10572
        cfg.num_epoch = 34
        cfg.warmup_epoch = -1
        cfg.val_targets = ['lfw', 'cfg_fp', 'agedb_30']

        def lr_step_func(epoch):
            return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
                [m for m in [20, 28, 32] if m - 1 <= epoch])
        cfg.lr_func = lr_step_func

""" 2. Training Recipe """
def config_recipe():
    cfg.fp16 = True
    cfg.momentum = 0.9
    cfg.weight_decay = 5e-4
    cfg.batch_size = 128  # 128
    cfg.lr = 0.1  # 0.1 for batch size is 512

    cfg.lambda1 = 1  # l_total = l_cls + lambda1 * l_seg

""" 3. Model Setting """
def config_model():
    # FRB, OSB, FM Operators
    cfg.frb_type = 'iresnet18'
    cfg.osb_type = 'unet'
    cfg.fm_layers = (1, 1, 1, 1)
    # Classification Header
    cfg.header_type = 'AMCosFace'
    cfg.header_params = (64.0, 0.4, 0.0, 0.0)  # (s, m, a, k)
    # PartialFC
    cfg.sample_rate = 1

    if cfg.frb_type == 'lightcnn':
        cfg.is_gray = True
        cfg.out_size = (128, 128)
        cfg.use_norm = False

""" 4. Experiment Record """
def config_exp():
    cfg.exp_id = 0
    cfg.output = "tmp_" + str(cfg.exp_id)
    print('output path: ', cfg.output)

