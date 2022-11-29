import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist

import sys
import warnings

import torch

import backbones

from PIL import Image
import numpy as np
import random
import os
from torchvision import transforms
import time
from tqdm import tqdm
import re
from scipy.special import expit

from datasets.augment.rand_occ import RandomBlock, RandomConnectedPolygon, RandomRealObject
from datasets.load_dataset import EvalDataset

from config import config_init, load_yaml

a = ["lfw", "cfp_fp", "agedb_30"]
divisor = 1
TASKS = {
    'lfw': {
        'img_root': '/home/yuange/dataset/PKU-Masked-Face',
        'list_file': 'ver24000.list',
        'save_path': './features',
        'task_name': 'lfw',
        'model_name': 'arcface_r18',
        'resume_path': '',
        'num_classes': 10575,
        'transform':  transforms.Compose([]),
        'ground_truth_label': list(np.zeros([3000 * 4 // divisor], dtype=np.int))
                              + list(np.ones([3000 * 4 // divisor], dtype=np.int)),
    },
    'cfp_fp': {
        'img_root': '/home/yuange/dataset/PKU-Masked-Face',
        'list_file': 'ver24000.list',
        'save_path': './features',
        'task_name': 'cfp_fp',
        'model_name': 'arcface_r18',
        'resume_path': '',
        'num_classes': 10575,
        'transform':  transforms.Compose([]),
        'ground_truth_label': list(np.zeros([3000 * 4 // divisor], dtype=np.int))
                              + list(np.ones([3000 * 4 // divisor], dtype=np.int)),
    },
    'agedb_30': {
        'img_root': '/home/yuange/dataset/PKU-Masked-Face',
        'list_file': 'ver24000.list',
        'save_path': './features',
        'task_name': 'agedb_30',
        'model_name': 'arcface_r18',
        'resume_path': '',
        'num_classes': 10575,
        'transform':  transforms.Compose([]),
        'ground_truth_label': list(np.zeros([3000 * 4 // divisor], dtype=np.int))
                              + list(np.ones([3000 * 4 // divisor], dtype=np.int)),
    },
}


class ExtractFeature(object):
    '''特征提取类'''
    def __init__(self, task, cfg, args):
        self.img_root = task['img_root']  # not used
        if not os.path.exists(self.img_root):
            self.img_root = '/GPUFS/sysu_zhenghch_1/yuange/datasets/' + self.img_root[len('/home/yuange/dataset/'):]
        self.list_file = task['list_file']
        self.save_path = task['save_path']
        self.task_name = task['task_name']
        self.model_name = task['model_name']
        self.resume_path = task['resume_path']
        self.num_classes = task['num_classes']
        self.transform = task['transform']
        self.weight_folder = task['weight_folder']

        self.cfg = cfg
        self.gray = cfg.is_gray
        self.channel = 1 if self.gray else 3
        if cfg.is_gray:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                self.transform,
            ])

        self.pre_trained = args.pre_trained
        self.vis = args.vis

    def _load_model(self):
        cfg = self.cfg
        if self.model_name == 'arcface_r18':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r18-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_occ6/backbone.pth'
            self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/arcface_r18_angle/backbone.pth'
            # self.weight_path = './tmp_47018/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet18'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r34':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r34-backbone.pth'
            self.weight_path = './arcface_r34_7occ/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet34'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r50-backbone.pth'
            self.weight_path = 'ms1mv3_arcface_r50_occ6/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet50'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r100':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r100-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_onlysmooth/backbone.pth'
            self.weight_path = 'ms1mv3_arcface_r100_mg/backbone.pth'
            # weight = torch.load('/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_bot50_lr3/backbone.pth')
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet100'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'msml':
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb_r18_aaai/backbone.pth'
            # self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb18_mlm4_1115_drop01_swinmei/backbone.pth'
            self.weight_path = os.path.join(self.weight_folder, 'backbone.pth')
            weight = torch.load(self.weight_path)
            model = eval("backbones.{}".format('MSML'))(frb_type=cfg.frb_type,
                                                        osb_type=cfg.osb_type,
                                                        fm_layers=cfg.fm_layers,
                                                        header_type=cfg.header_type,
                                                        header_params=cfg.header_params,
                                                        num_classes=cfg.num_classes,
                                                        fp16=False,
                                                        use_osb=cfg.use_osb,
                                                        fm_params=cfg.fm_params,
                                                        peer_params=cfg.peer_params,
                                                        ).cuda()
            if not self.pre_trained:
                model.load_state_dict(weight)
        elif 'from2021' in self.model_name:
            self.weight_path = ''
            print('loading TPAMI2021 FROM model...')
            model = backbones.From2021()
        else:
            raise ValueError('Error model type\n')

        model.eval()
        model = torch.nn.DataParallel(model).cuda()

        if self.resume_path:
            print("=> loading checkpoint '{}'".format(self.resume_path))
            checkpoint = torch.load(self.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(self.resume_path))

        return model

    def _load_one_input(self, img, index, flip=False, protocol='NB'):

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        width, height = self.cfg.out_size
        resize_trans = transforms.CenterCrop((height, width))
        img = resize_trans(img)

        common_trans = transforms.Compose([transforms.ToTensor()])

        if protocol == 'NB':
            img = self.transform(img) if index % 2 == 0 else common_trans(img)
        elif protocol == 'BB':
            img = self.transform(img)

        return img  # torch.tensor, (C, H, W)

    def _visualize_features(self, features):  # Not used
        """ Visualize the 256-D features """
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne.fit_transform(features)
        print(tsne.embedding_.shape)  # (12000, 2)

        embedding = tsne.embedding_
        min_val = embedding.min()
        max_val = embedding.max()
        heat_map = np.zeros((100, 100), dtype=np.uint8)
        for pairs in embedding:
            px = int((pairs[0] - min_val) / (max_val - min_val) * 98)
            py = int((pairs[1] - min_val) / (max_val - min_val) * 98)
            heat_map[px][py] += 1
        heat_map = ((heat_map / heat_map.max()) * 15).astype(np.uint8)

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.tick_params(labelsize=35)

        import seaborn as sns
        sns.heatmap(heat_map, xticklabels=20, yticklabels=20, cbar=None)

        save_name = 'features_' + self.task_name[:-12] + '.jpg'
        plt.savefig(os.path.join(self.save_path, save_name))
        plt.clf()

        self.heat_map = heat_map

    def _visualize_feature_map(self, feature_map, save_name):  # Not used
        val_min, val_max = feature_map.min(), feature_map.max()  # [-1, 1]
        feature_map = feature_map.cpu().data.numpy()

        channel = feature_map.shape[1]
        h, w = feature_map.shape[-2], feature_map.shape[-1]

        heat_map = np.zeros((h, w), dtype=np.uint8)
        for b in range(1):
            for c in range(channel):
                for i in range(h):
                    for j in range(w):
                        heat_map[i][j] = int((feature_map[b][c][i][j] + 1.0) / 2 * 255)

                import matplotlib as mpl
                mpl.use('Agg')
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5, 5))
                plt.tick_params(labelsize=35)
                plt.axis('off')

                import seaborn as sns
                sns.heatmap(heat_map, cbar=None, cmap='jet')

                save_path = os.path.join(self.save_path, 'latent')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, str(b))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, str(h) + 'x' + str(w))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                plt.savefig(os.path.join(save_path, str(c) + save_name), pad_inches=0, bbox_inches='tight', dpi=100)
                plt.clf()

    def _visualize_attn(self, mask, identity, in_image): # Not used
        # print(mask.shape)  # (32, 512, 7, 7)
        # print(identity.shape)  # (32, 512, 7, 7)

        # save input image
        for b in range(1):
            arr = in_image[b].cpu().data.numpy()
            rgb = (arr * 127 + 128)
            rgb_img = np.zeros([112, 112, 3])
            rgb_img[:, :, 0] = rgb[0]
            rgb_img[:, :, 1] = rgb[1]
            rgb_img[:, :, 2] = rgb[2]
            img = Image.fromarray(rgb_img.astype(np.uint8), mode='RGB')
            save_path = os.path.join(self.save_path, 'latent')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, str(b))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img.save(os.path.join(save_path, 'input.jpg'))

        # visualize feature map
        self._visualize_feature_map(mask, 'attn.jpg')
        self._visualize_feature_map(identity, 'identity.jpg')
        self._visualize_feature_map(identity + identity * mask, 'cleaned.jpg')

    def start_extract(self, all_img, protocol='NB'):
        print("=> extract task started, task is '{}'".format(self.task_name))
        cfg = self.cfg
        model = self._load_model()

        if self.vis:
            for fm_idx in range(4):
                fm_op = model.module.frb.fm_ops[fm_idx]
                fm_op.en_save = True

        num = len(all_img)
        features = np.zeros((num, cfg.dim_feature))
        features_flip = np.zeros((num, cfg.dim_feature))

        # img to tensor
        channel = 1 if cfg.is_gray else 3
        width, height = cfg.out_size
        all_input = torch.zeros(num, channel, height, width)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i, protocol=protocol)
            all_input[i, :, :, :] = one_img_tensor

        all_flip = torch.zeros(num, channel, height, width)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i, flip=True, protocol=protocol)
            all_flip[i, :, :, :] = one_img_tensor

        # start
        print("=> img-to-tensor is finished, start inference ...")
        # all_input = all_input.cuda()
        with torch.no_grad():
            all_input_var = torch.autograd.Variable(all_input)
            if cfg.use_norm:
                all_input_var = all_input_var.sub_(0.5).div_(0.5)  # [0, 1] to [-1, 1]
            # print(all_input_var.min(), all_input_var.max())
            all_flip_var = torch.autograd.Variable(all_flip)
            if cfg.use_norm:
                all_flip_var = all_flip_var.sub_(0.5).div_(0.5)  # [0, 1] to [-1, 1]

        batch_size = 25 if not self.vis else 1
        total_step = num // batch_size
        assert batch_size * total_step == num
        for i in range(total_step):
            patch_input = all_input_var[i * batch_size : (i + 1) * batch_size]
            # feature, mask, identity = model(patch_input)
            output = model(patch_input.cuda())
            feature = output[0] if type(output) is tuple else output
            final_seg = output[1] if type(output) is tuple else None
            features[i * batch_size : (i + 1) * batch_size] = feature.data.cpu().numpy()

            # vis
            if i == -1:
                self._visualize_attn(mask, identity, patch_input)

            """ Visualization """
            if i <= 400 and i == 3 and self.vis:
                print('Start Visualizing...')
                """ Visualize Predicted Masks """
                # some_tensor.max(0)[0]: value of max_value
                # some_tensor.max(0)[1]: index of max_value
                mask = final_seg[0].cpu().max(0)[1].data.numpy() * 255  # (height, width)
                mask = mask.astype(np.uint8)
                mask = Image.fromarray(mask.astype(np.uint8))
                mask.save(os.path.join(self.save_path, 'lfw' + str(i) + '_learned.jpg'))

                if self.gray:
                    img = np.zeros((112, 112))
                    img = patch_input[0][0].cpu().data.numpy() * 255
                    img = Image.fromarray(img.astype(np.uint8), mode='L')
                else:
                    img = np.zeros((112, 112, 3))
                    img[:, :, 0] = (patch_input[0][0].cpu().data.numpy() + 1.0) * 127.5
                    img[:, :, 1] = (patch_input[0][1].cpu().data.numpy() + 1.0) * 127.5
                    img[:, :, 2] = (patch_input[0][2].cpu().data.numpy() + 1.0) * 127.5
                    img = Image.fromarray(img.astype(np.uint8), mode='RGB')

                img.save(os.path.join(self.save_path, 'lfw' + str(i) + '_truth.jpg'))

                """ Visualize Intermediate Features of FM Operators """
                B, C, H, W = final_seg.shape
                mask = torch.zeros((B, H, W))
                final_seg = final_seg.cpu()
                for b in range(B):
                    mask[b] = final_seg[b].max(0)[1]
                mask = mask.data  # 0-occ, 1-clean
                for fm_idx in range(4):
                    fm_op = model.module.frb.fm_ops[fm_idx]
                    fm_op.plot_intermediate_features(gt_occ_msk=mask,
                                                     save_folder=self.save_path)
                raise ValueError('Visualization Finished. Stop evaluating.')

        for i in range(total_step):
            patch_input = all_flip_var[i * batch_size: (i + 1) * batch_size]
            # feature, mask, identity = model(patch_input)
            output = model(patch_input.cuda())
            feature = output[0] if type(output) is tuple else output
            final_seg = output[1] if type(output) is tuple else None
            features_flip[i * batch_size: (i + 1) * batch_size] = feature.data.cpu().numpy()

            # vis
            if i == -1:
                self._visualize_attn(mask, identity, patch_input)

        features = features_flip + features

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        save_file = os.path.join(self.save_path, '{}_{}.npy'.format(self.task_name,
                                                                    self.weight_folder.replace('/', '_')))
        np.save(save_file, features)
        return features
        # print("=> extract task finished, file is saved at '{}'".format(save_file))

        # print("=> visualization started")
        # self._visualize_features(features)
        # print("=> visualization finished")

        # return ret_vector


class Verification(object):
    """人脸验证类"""
    def __init__(self, task):
        self.save_path = task['save_path']
        self.task_name = task['task_name']
        self.weight_folder = task['weight_folder']
        self.ground_truth_label = task['ground_truth_label']
        self._prepare()

    def _prepare(self):
        feature = np.load(os.path.join(self.save_path, '{}_{}.npy'.format(self.task_name,
                                                                          self.weight_folder.replace('/', '_'))))
        feature = sklearn.preprocessing.normalize(feature)
        self.feature = feature

    def start_verification(self):
        # print("=> verification started, caculating ...")
        predict_label = []
        num = self.feature.shape[0]
        for i in range(num // 2):
            dis_cos = cdist(self.feature[i * 2: i * 2 + 1, :],
                            self.feature[i * 2 + 1: i * 2 + 2, :],
                            metric='cosine')
            predict_label.append(dis_cos[0, 0])

        """ (1) Calculate Accuracy """
        fpr, tpr, threshold = roc_curve(self.ground_truth_label, predict_label)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # choose proper threshold
        print("=> verification finished, accuracy rate is {}".format(acc))
        ret_acc = acc

        # plot auc curve
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        plt.savefig(os.path.join(self.save_path, 'auc.jpg'))
        plt.clf()

        """ (2) Calculate TAR@FAR<=1e-k """
        neg_cnt = len(predict_label) // 2
        pos_cnt = neg_cnt
        self.ground_truth_label = np.array(self.ground_truth_label)
        predict_label = np.array(predict_label)
        pos_dist = predict_label[self.ground_truth_label == 0].tolist()
        neg_dist = predict_label[self.ground_truth_label == 1].tolist()

        far_val = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ret_tarfar = np.zeros((len(far_val)))
        for idx in range(len(far_val)):

            """ Choose the far values """
            if idx > 3:
                continue

            threshold = []
            for T in neg_dist:
                neg_pair_smaller = 0.
                for i in range(neg_cnt):
                    if neg_dist[i] < T:
                        neg_pair_smaller += 1
                far = neg_pair_smaller / neg_cnt
                if far <= far_val[idx]:
                    threshold.append(T)

            acc = 0.
            print(len(threshold))
            for T in threshold:
                pos_pair_smaller = 0.
                for i in range(pos_cnt):
                    if pos_dist[i] <= T:
                        pos_pair_smaller += 1
                tar = pos_pair_smaller / pos_cnt
                acc = max(acc, tar)

            print("=> verification finished, accuracy rate (TAR@FAR<=1e-%d) is %.6f" % (idx + 1, acc))
            ret_tarfar[idx] = acc

        return ret_acc, ret_tarfar


def main():

    parser = argparse.ArgumentParser(description='PyTorch MSML Testing')
    parser.add_argument('--network', type=str, default='msml', help='backbone network')
    parser.add_argument('--dataset', type=str, default='lfw', help='lfw, cfp_fp, agedb_30')
    parser.add_argument('--weight_folder', type=str, help='the folder containing pre-trained weights')
    parser.add_argument('--pre_trained', type=bool, default=False, help='use pre-trained lightcnn model')
    parser.add_argument('--fill_type', type=str, default='black', help='block occlusion fill type')
    parser.add_argument('--vis', type=bool, default=False, help='visualization of FM arith')
    parser.add_argument('--no-occ', action='store_true', help='do not add occ')
    args = parser.parse_args()

    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    task_type = args.dataset
    my_task = TASKS[task_type]

    my_task['model_name'] = args.network
    my_task['task_name'] = my_task['task_name']
    my_task['weight_folder'] = args.weight_folder
    print('[model_name]: ', my_task['model_name'])
    print('[transform]: ', my_task['transform'])

    """ Pre-load images into memory """
    print("=> Pre-loading images ...")
    from datasets.load_dataset import ReadMXNet
    cfg = load_yaml(os.path.join(args.weight_folder, 'config.yaml'))
    config_init(cfg)
    mx_reader = ReadMXNet(my_task['task_name'], cfg.rec)
    path = os.path.join(cfg.rec, my_task['task_name'] + ".bin")
    all_img, issame_list = mx_reader.load_bin(path, (112, 112))

    if args.network == 'from2021':
        cfg.out_size = (96, 112)  # (w,h)

    """ Multi-Test """
    protocol = 'BB'
    # lo_list = [40]
    # hi_list = [41]
    lo_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] if not args.vis else [35, ]
    hi_list = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] if not args.vis else [36, ]
    if args.no_occ:
        lo_list, hi_list = [0], [1]
    assert len(lo_list) == len(hi_list)
    # lo_list.reverse()
    # hi_list.reverse()
    avg_acc_list = []
    fars = np.zeros((len(lo_list), 5))
    weight_path = ''

    for ind in range(0, len(lo_list)):
        print('================== [ %d ] ===============' % ind)

        lo, hi = lo_list[ind], hi_list[ind]
        print('random block range: [%d ~ %d)' % (lo, hi))
        my_task['transform'] = transforms.Compose([transforms.CenterCrop((cfg.out_size[1], cfg.out_size[0])),
                                                   RandomBlock(lo, hi, fill=args.fill_type),
                                                   # RandomConnectedPolygon(is_training=False),
                                                   transforms.ToTensor()])

        intsame_list = []
        for i in range(len(issame_list)):
            flag = 0 if issame_list[i] else 1  # 0:is same
            intsame_list.append(flag)
        my_task['ground_truth_label'] = intsame_list

        avg_acc = 0.
        repeat_time = 1 if (lo == 0 and hi == 1) or (lo == 100 and hi == 101) else 10
        for repeat in range(repeat_time):
            ExtractTask = ExtractFeature(my_task, cfg=cfg, args=args)
            features = ExtractTask.start_extract(all_img, protocol=protocol)

            weight_path = ExtractTask.weight_path

            features = sklearn.preprocessing.normalize(features)

            import eval.verification as ver
            _, _, accuracy, val, val_std, far = ver.evaluate(features, issame_list)
            acc2, std2 = np.mean(accuracy), np.std(accuracy)
            print('acc2 = [%.6f]' % acc2)
            avg_acc += acc2

            VerificationTask = Verification(my_task)
            _, tarfar = VerificationTask.start_verification()
            fars[ind] += tarfar

        avg_acc = avg_acc / repeat_time
        fars[ind] /= repeat_time

        avg_acc_list.append(avg_acc)
        print('[avg_acc]: %.4f' % (avg_acc))

    # print results
    print(cfg)
    print('[protocol]:', protocol, '[fill_type]', args.fill_type)
    print('[model_name]:', my_task['model_name'])
    print('[weight_path]:', weight_path)
    for ind in range(0, len(avg_acc_list)):
        print('[%d ~ %d] | [avg_acc]: %.4f'
              % (lo_list[ind], hi_list[ind], avg_acc_list[ind]))
        far = fars[ind]
        print('          | [tar@far]: %.4f, %.4f, %.4f, %.4f, %.4f'
              % (far[0], far[1], far[2], far[3], far[4]))
