import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import transforms
import sklearn
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist

from config import config_init, load_yaml
import backbones


class EvaluatorFolder(object):
    def __init__(self,
                 weight_folder: str,
                 dataset_folder: str = '/gavin/datasets/msml/mfr2/mfr2_mtcnn',
                 pair_txt: str = '/gavin/datasets/msml/mfr2/pairs.txt',
                 ):
        self.img_dict = {}
        self.img_size = (112, 112)  # (H,W)
        self.save_folder = './eval/snapshot'
        os.makedirs(self.save_folder, exist_ok=True)

        img_pairs, ground_truth_label = self._prepare_img_pairs(dataset_folder, pair_txt)
        self.img_pairs = img_pairs
        self.ground_truth_label = ground_truth_label

        model, cfg = self._load_model(weight_folder)
        self.model = model
        self.cfg = cfg

        self.vis = False

        self.features = None

    def _prepare_img_pairs(self, dataset_folder, pair_txt):
        """"""
        ''' load image list'''
        id_list = os.listdir(dataset_folder)
        for identity in id_list:
            self.img_dict[identity] = []
            img_sublist = os.listdir(os.path.join(dataset_folder, identity))
            img_sublist.sort()
            for img_name in img_sublist:
                pil_img = Image.open(os.path.join(dataset_folder, identity, img_name), 'r').convert('RGB')
                self.img_dict[identity].append(np.array(pil_img))

        ''' load pairs '''
        with open(pair_txt, 'r') as txt:
            lines = txt.readlines()
        pair_cnt = len(lines)
        ground_truth_label = np.zeros(pair_cnt)  # 0:diff, 1:same
        # img_pairs = np.zeros((pair_cnt * 2, self.img_size, self.img_size, 3), dtype=np.uint8)
        img_pairs = []
        for idx in range(pair_cnt):
            words = lines[idx].strip().split(' ')
            if len(words) == 3:
                identity1 = words[0]
                identity2 = identity1
                idx1, idx2 = words[1], words[2]
            else:
                identity1, identity2 = words[0], words[2]
                idx1, idx2 = words[1], words[3]
            arr1 = self.img_dict[identity1][int(idx1) - 1]  # 1st images is named as 0001 instead of 0000
            arr2 = self.img_dict[identity2][int(idx2) - 1]
            # img_pairs[idx * 2] = arr1
            # img_pairs[idx * 2 + 1] = arr2
            img_pairs.append(Image.fromarray(arr1))
            img_pairs.append(Image.fromarray(arr2))
            ground_truth_label[idx] = len(words) - 3  # 0:pos(same), 1:neg(diff)

        return img_pairs, ground_truth_label

    def _load_model(self, weight_folder):
        weight_path = os.path.join(weight_folder, 'backbone.pth')
        cfg = None

        if 'msml' in weight_folder or 'out' in weight_folder:
            print('loading msml model...')
            cfg = load_yaml(os.path.join(weight_folder, 'config.yaml'))
            config_init(cfg)
            weight = torch.load(weight_path)
            model = eval("backbones.{}".format('MSML'))(
                frb_type=cfg.frb_type,
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
            self.img_size = cfg.out_size
            model.load_state_dict(weight)
        elif 'cosface2018' in weight_folder:
            print('loading cosface2018 sphere model...')
            self.img_size = (112, 96)
            model = backbones.cosface2018(self.img_size)
        elif 'from2021' in weight_folder:
            print('loading TPAMI2021 FROM model...')
            self.img_size = (112, 96)
            model = backbones.From2021()
        else:
            print('loading vanilla iresnet...')
            weight = torch.load(weight_path)
            model = eval("backbones.{}".format('iresnet18_v'))(dropout=0, fp16=False).cuda()
            model.eval()
            model.load_state_dict(weight)

        print(cfg)
        model.eval()
        model = torch.nn.DataParallel(model).cuda()
        return model, cfg

    def _load_one_input(self, img, index, flip=False):
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        common_trans = transforms.Compose([
            transforms.Resize(self.cfg.out_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor()
        ])
        if self.cfg.is_gray:
            common_trans = transforms.Compose([
                transforms.Grayscale(),
                common_trans,
            ])

        img = common_trans(img)

        return img  # torch.tensor, (C, H, W)

    def start_extract(self):
        model = self.model
        cfg = self.cfg
        all_img = self.img_pairs

        if cfg is None:
            channel = 3
            height, width = self.img_size
            use_norm = True
            dim_feature = 512
        else:
            channel = 1 if cfg.is_gray else 3
            height, width = cfg.out_size
            use_norm = cfg.use_norm
            dim_feature = cfg.dim_feature

        num = len(all_img)
        features = np.zeros((num, dim_feature))
        features_flip = np.zeros((num, dim_feature))

        # img to tensor
        all_input = torch.zeros(num, channel, height, width)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i)
            all_input[i, :, :, :] = one_img_tensor

        all_flip = torch.zeros(num, channel, height, width)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i, flip=True)
            all_flip[i, :, :, :] = one_img_tensor

            # start
            print("=> %d img-to-tensor is finished, start inference ..." % num)
            # all_input = all_input.cuda()
            with torch.no_grad():
                all_input_var = torch.autograd.Variable(all_input)
                if use_norm:
                    all_input_var = all_input_var.sub_(0.5).div_(0.5)  # [0, 1] to [-1, 1]
                # print(all_input_var.min(), all_input_var.max())
                all_flip_var = torch.autograd.Variable(all_flip)
                if use_norm:
                    all_flip_var = all_flip_var.sub_(0.5).div_(0.5)  # [0, 1] to [-1, 1]

            batch_size = 16 if not self.vis else 1
            total_step = num // batch_size
            assert batch_size * total_step == num
            for i in range(total_step):
                patch_input = all_input_var[i * batch_size: (i + 1) * batch_size]
                # feature, mask, identity = model(patch_input)
                output = model(patch_input.cuda())
                feature = output[0] if type(output) is tuple else output
                final_seg = output[1] if type(output) is tuple else None
                features[i * batch_size: (i + 1) * batch_size] = feature.data.cpu().numpy()

                """ Visualization """
                if i <= 200 and self.vis:
                    """ Visualize Predicted Masks """
                    # some_tensor.max(0)[0]: value of max_value
                    # some_tensor.max(0)[1]: index of max_value
                    if final_seg is not None:
                        mask = final_seg[0].cpu().max(0)[1].data.numpy() * 255  # (height, width)
                        mask = mask.astype(np.uint8)
                        mask = Image.fromarray(mask.astype(np.uint8))
                        mask.save(os.path.join(self.save_folder, 'lfw' + str(i) + '_learned.jpg'))

                    if channel == 1:
                        img = np.zeros((112, 112))
                        img = patch_input[0][0].cpu().data.numpy() * 255
                        img = Image.fromarray(img.astype(np.uint8), mode='L')
                    else:
                        img = np.zeros((112, 112, 3))
                        img[:, :, 0] = (patch_input[0][0].cpu().data.numpy() + 1.0) * 127.5
                        img[:, :, 1] = (patch_input[0][1].cpu().data.numpy() + 1.0) * 127.5
                        img[:, :, 2] = (patch_input[0][2].cpu().data.numpy() + 1.0) * 127.5
                        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

                    img.save(os.path.join(self.save_folder, 'lfw' + str(i) + '_truth.jpg'))

                    """ Visualize Intermediate Features of FM Operators """
                    # B, C, H, W = final_seg.shape
                    # mask = torch.zeros((B, H, W))
                    # final_seg = final_seg.cpu()
                    # for b in range(B):
                    #     mask[b] = final_seg[b].max(0)[1]
                    # mask = mask.data  # 0-occ, 1-clean
                    # for fm_idx in range(4):
                    #     fm_op = model.module.frb.fm_ops[fm_idx]
                    #     fm_op.plot_intermediate_features(gt_occ_msk=mask,
                    #                                      save_folder=self.save_folder)
                    # raise ValueError('Visualization Finished. Stop evaluating.')

            for i in range(total_step):
                patch_input = all_flip_var[i * batch_size: (i + 1) * batch_size]
                output = model(patch_input.cuda())
                feature = output[0] if type(output) is tuple else output
                features_flip[i * batch_size: (i + 1) * batch_size] = feature.data.cpu().numpy()

            features = features + features_flip
            self.features = features
            return features

    def start_verification(self):
        # print("=> verification started, caculating ...")
        predict_label = []
        # self.features = sklearn.preprocessing.normalize(self.features)
        num = self.features.shape[0]
        for i in range(num // 2):
            dis_cos = cdist(self.features[i * 2: i * 2 + 1, :],
                            self.features[i * 2 + 1: i * 2 + 2, :],
                            metric='cosine')
            predict_label.append(dis_cos[0, 0])

        """ (1) Calculate Accuracy """
        fpr, tpr, threshold = roc_curve(self.ground_truth_label, predict_label)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # choose proper threshold
        print("=> verification finished, accuracy rate is %.2f%%" % (acc * 100))
        ret_acc = acc
        roc_auc = auc(fpr, tpr)
        # print("=> verification finished, accuracy rate is %.2f%%" % (roc_auc * 100))

        # plot auc curve
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        # plt.savefig(os.path.join(self.save_folder, 'auc.jpg'))
        # plt.clf()

        """ (2) Calculate TAR@FAR<=1e-k """
        neg_cnt = len(predict_label) // 2
        pos_cnt = neg_cnt
        self.ground_truth_label = np.array(self.ground_truth_label)
        predict_label = np.array(predict_label)
        pos_dist = predict_label[self.ground_truth_label == 0].tolist()
        neg_dist = predict_label[self.ground_truth_label == 1].tolist()

        far_val = [1e-1, 1e-2, 1e-3]
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

    def stat_params_flops(self):
        model = self.model.module.eval()
        img = torch.zeros((1, 3, self.img_size[0], self.img_size[1])).cuda()
        import thop
        flops, params = thop.profile(model, inputs=(img,), verbose=False)
        print('#Params=%.2fM, GFLOPS=%.2f' % (params / 1e6, flops / 1e9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MSML Testing')
    parser.add_argument('--weight_folder', type=str, help='the folder containing pre-trained weights')
    args = parser.parse_args()

    eva = EvaluatorFolder(
        weight_folder=args.weight_folder
    )
    eva.start_extract()
    eva.start_verification()
    eva.stat_params_flops()

    import thop
    img = torch.zeros((1, 3, 112, 112))
    # from backbones.frb.resnet import resnet28
    # net = resnet28(pretrained=False, num_classes=512)
    # flops, params = thop.profile(net, inputs=(img,))
    # print('flops', flops / 1e9, 'params', params / 1e6)

    from backbones.frb.iresnet import iresnet28_v
    # net = iresnet28_v(pretrained=False)
    # flops, params = thop.profile(net, inputs=(img,))
    # print('flops', flops / 1e9, 'params', params / 1e6)
