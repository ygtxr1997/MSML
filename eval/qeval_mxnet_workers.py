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
from torch.utils.data import DataLoader
from torchvision.transforms.functional import hflip

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


class MXNetEvaluator(object):
    def __init__(self,
                 all_img: list,
                 issame_list: list,
                 pre_trans,
                 cfg,
                 args,
                 ):
        """ MXNetEvaluator """
        ''' yaml config '''
        self.cfg = cfg
        self.is_gray = cfg.is_gray
        self.channel = 1 if self.is_gray else 3
        if self.is_gray:
            pre_trans = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(cfg.out_size),
                pre_trans
            ])
        self.out_size = cfg.out_size  # (w,h)
        self.dim_feature = cfg.dim_feature

        ''' dataset & dataloader '''
        self.num = len(all_img)
        self.batch_size = 40
        self.eval_dataset = EvalDataset(
            all_img=all_img, issame_list=issame_list, pre_trans=pre_trans, norm_0_1=self.is_gray
        )
        self.eval_loader = DataLoader(
            self.eval_dataset, self.batch_size,
            num_workers=12, shuffle=False, drop_last=False
        )
        self.issame_list = issame_list  # True:same
        self.intsame_list = [0 if x else 1 for x in issame_list]  # 0:same

        ''' args '''
        self.model_name = args.network
        self.weight_folder = args.weight_folder
        self.is_vis = args.is_vis
        self.dataset_name = args.dataset

        ''' model '''
        self.model = self._load_model()

        ''' visualization '''
        save_folder = os.path.join('./vis', self.dataset_name)
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

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
                                                        frb_pretrained=False,
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
            model.load_state_dict(weight)
        elif 'from2021' in self.model_name:
            self.weight_path = ''
            print('loading TPAMI2021 FROM model...')
            model = backbones.From2021()
        else:
            raise ValueError('Error model type\n')

        model.eval()
        model = torch.nn.DataParallel(model).cuda()

        return model

    def _infer(self, x):
        output = self.model(x)
        if type(output) is tuple:
            feature = output[0]
            final_seg = output[1]
        else:
            feature = output
            final_seg = None
        return feature, final_seg

    def _vis_segmentation_result(self,
                                 img1: torch.Tensor, img2: torch.Tensor,
                                 seg1: torch.Tensor, seg2: torch.Tensor,
                                 index_list: list):
        save_folder = self.save_folder
        n = self.num
        assert len(index_list) * 2 <= n
        for idx in range(n // 2):
            if not idx in index_list:
                continue
            ''' predicted segmentation masks '''
            seg1_pil, seg2_pil = self.__t2p_segmentation_result(seg1[idx], seg2[idx])
            seg1_pil.save(os.path.join(save_folder, str(idx * 2) + '_predict.jpg'))
            seg2_pil.save(os.path.join(save_folder, str(idx * 2 + 1) + '_predict.jpg'))

            ''' input faces '''
            img1_pil, img2_pil = self.__t2p_input(img1[idx], img2[idx], is_gray=self.is_gray)
            img1_pil.save(os.path.join(save_folder, str(idx * 2) + '_input.jpg'))
            img2_pil.save(os.path.join(save_folder, str(idx * 2 + 1) + '_input.jpg'))

    @staticmethod
    def __t2p_segmentation_result(seg1: torch.Tensor, seg2: torch.Tensor):
        assert seg1.ndim == 3  # (C,H,W)
        ''' predicted segmentation masks '''
        seg1_np = seg1.max(0)[1].cpu().numpy() * 255  # (h,w)
        seg2_np = seg2.max(0)[1].cpu().numpy() * 255  # (h,w)
        seg1_pil = Image.fromarray(seg1_np.astype(np.uint8))
        seg2_pil = Image.fromarray(seg2_np.astype(np.uint8))
        return seg1_pil, seg2_pil

    @staticmethod
    def __t2p_input(img1: torch.Tensor, img2: torch.Tensor, is_gray: bool = False):
        assert img1.ndim == 3  # (C,H,W)
        ''' input faces '''
        if is_gray:
            img1_pil = Image.fromarray((img1.cpu().numpy() * 255).astype(np.uint8), mode='L')
            img2_pil = Image.fromarray((img2.cpu().numpy() * 255).astype(np.uint8), mode='L')
        else:
            img1_pil = Image.fromarray(((img1.permute(1, 2, 0).cpu().numpy() + 1.) * 127.5).astype(np.uint8),
                                       mode='RGB')
            img2_pil = Image.fromarray(((img2.permute(1, 2, 0).cpu().numpy() + 1.) * 127.5).astype(np.uint8),
                                       mode='RGB')
        return img1_pil, img2_pil

    def _vis_intermediate_feature_map(self):
        pass

    def _vis_embeddings_tsne(self,
                             embeddings: torch.Tensor,
                             index: list = None,
                             ):
        pass
        # from sklearn.manifold import TSNE
        # import matplotlib as mpl
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        #
        # if index is None:
        #     index = np.arange(100).tolist()
        # embeddings = embeddings[index]

    def _vis_embeddings_heat(self,
                             embeddings: np.ndarray,
                             index_list: list = None):
        save_folder = self.save_folder
        from eval.vis_heat import vis_feature_1d
        embeddings = embeddings[index_list].astype(np.float)
        for idx, embedding in enumerate(embeddings):
            save_name = os.path.join(save_folder, '%d_heat.jpg' % idx)
            vis_feature_1d(embedding, save_name=save_name)

    @torch.no_grad()
    def start_eval(self):
        w, h = self.out_size
        n = self.num
        dim_feature = self.dim_feature
        features = np.zeros((n, dim_feature))  # [feat1,...,feat1,feat2,...,feat2]
        features_flip = np.zeros_like(features)
        img1s = torch.zeros((n, self.channel, h, w))
        # img2s =

        ''' 1. Extract Features '''
        print("=> start inference ...")
        batch_idx = 0
        for batch in tqdm(self.eval_loader):
            img1, img2, same = batch

            ''' a. original input '''
            img1 = img1.cuda()
            img2 = img2.cuda()

            feat1, seg1 = self._infer(img1)
            feat2, seg2 = self._infer(img2)
            features[batch_idx * self.batch_size:
                     (batch_idx + 1) * self.batch_size] = feat1.cpu().numpy()
            features[batch_idx * self.batch_size + (n // 2):
                     (batch_idx + 1) * self.batch_size + (n // 2)] = feat2.cpu().numpy()

            ''' b. flipped input '''
            img1_flip = hflip(img1)
            img2_flip = hflip(img2)

            feat1_flip, _ = self._infer(img1_flip)
            feat2_flip, _ = self._infer(img2_flip)
            features_flip[batch_idx * self.batch_size:
                          (batch_idx + 1) * self.batch_size] = feat1_flip.cpu().numpy()
            features_flip[batch_idx * self.batch_size + (n // 2):
                          (batch_idx + 1) * self.batch_size + (n // 2)] = feat2_flip.cpu().numpy()

            batch_idx += 1

        features = features_flip + features
        features = sklearn.preprocessing.normalize(features)

        ''' 2. Calculate Metrics '''
        predict_label = []
        features_reorder = np.zeros_like(features)  # [feat1,feat2,feat1,feat2,...,feat1,feat2]
        for i in range(n // 2):
            dis_cos = cdist(features[i: i + 1, :],
                            features[i + (n // 2): i + 1 + (n // 2), :],
                            metric='cosine')
            features_reorder[i * 2] = features[i]
            features_reorder[i * 2 + 1] = features[i + (n // 2)]
            predict_label.append(dis_cos[0, 0])

        """ (0) Visualization """
        if self.is_vis:
            vis_index = np.arange(400).tolist()
            self._vis_segmentation_result(img1, img2, seg1, seg2, index_list=vis_index)
            self._vis_embeddings_heat(features_reorder, index_list=vis_index)
            exit()

        """ (1) Calculate Accuracy """
        fpr, tpr, threshold = roc_curve(self.intsame_list, predict_label)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # choose proper threshold
        print("=> verification finished, accuracy rate is {}".format(acc))

        # plot auc curve
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        # plt.savefig(os.path.join(self.save_path, 'auc.jpg'))
        # plt.clf()

        import eval.verification as ver
        _, _, accuracy, val, val_std, far = ver.evaluate(features_reorder, self.issame_list)
        acc2, std2 = np.mean(accuracy), np.std(accuracy)
        print('acc2 = [%.6f]' % acc2)
        ret_acc = acc2

        """ (2) Calculate TAR@FAR<=1e-k """
        neg_cnt = len(predict_label) // 2
        pos_cnt = neg_cnt
        ground_truth_label = np.array(self.intsame_list)
        predict_label = np.array(predict_label)
        pos_dist = predict_label[ground_truth_label == 0].tolist()
        neg_dist = predict_label[ground_truth_label == 1].tolist()

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
    parser.add_argument('--protocol', type=str, default='BB', help='add occlusions to the one or two of a pair')
    parser.add_argument('--fill_type', type=str, default='black', help='block occlusion fill type')
    parser.add_argument('--is_vis', type=str, default='no', help='visualization of FM arith')
    parser.add_argument('--no-occ', action='store_true', help='do not add occ')
    args = parser.parse_args()

    args.is_vis = True if args.is_vis == 'yes' else False

    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    """ Pre-load images into memory """
    print("=> Pre-loading images ...")
    from datasets.load_dataset import ReadMXNet
    cfg = load_yaml(os.path.join(args.weight_folder, 'config.yaml'))
    config_init(cfg)
    mx_reader = ReadMXNet(args.dataset, cfg.rec)
    path = os.path.join(cfg.rec, args.dataset + ".bin")
    all_img, issame_list = mx_reader.load_bin(path, (112, 112))

    if args.network == 'from2021':
        cfg.out_size = (96, 112)  # (w,h)

    """ Multi-Test """
    lo_list = [40,]
    hi_list = [41,]
    # lo_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] if not args.is_vis else [35, ]
    # hi_list = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] if not args.is_vis else [36, ]
    if args.no_occ:
        lo_list, hi_list = [0], [1]
    assert len(lo_list) == len(hi_list)

    avg_acc_list = []
    fars = np.zeros((len(lo_list), 5))

    for ind in range(0, len(lo_list)):
        print('================== [ %d ] ===============' % ind)

        lo, hi = lo_list[ind], hi_list[ind]
        print('random block range: [%d ~ %d)' % (lo, hi))
        pre_trans = transforms.Compose([transforms.CenterCrop((cfg.out_size[1], cfg.out_size[0])),
                                        RandomBlock(lo, hi, fill=args.fill_type),
                                        # RandomConnectedPolygon(is_training=False),
                                        ])

        avg_acc = 0.
        repeat_time = 1 if (lo == 0 and hi == 1) or (lo == 100 and hi == 101) else 10
        for repeat in range(repeat_time):
            evaluator = MXNetEvaluator(all_img, issame_list, pre_trans, cfg, args)
            acc, far = evaluator.start_eval()

            avg_acc += acc
            fars[ind] += far

        avg_acc = avg_acc / repeat_time
        fars[ind] /= repeat_time

        avg_acc_list.append(avg_acc)
        print('[avg_acc]: %.4f' % (avg_acc))

    ''' print results '''
    print(cfg)
    print('[target]:', args.dataset, '[protocol]:', args.protocol, '[fill_type]', args.fill_type)
    print('[model_name]:', args.network)
    print('[weight_path]:', args.weight_folder)
    for ind in range(0, len(avg_acc_list)):
        print('[%d ~ %d] | [avg_acc]: %.4f'
              % (lo_list[ind], hi_list[ind], avg_acc_list[ind]))
        far = fars[ind]
        print('          | [tar@far]: %.4f, %.4f, %.4f, %.4f, %.4f'
              % (far[0], far[1], far[2], far[3], far[4]))


if __name__ == "__main__":
    main()