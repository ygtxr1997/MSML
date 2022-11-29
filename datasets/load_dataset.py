import os
import numbers

import torch
import mxnet as mx
from torch.utils import data
from torchvision import transforms

import numpy as np
import PIL.Image as Image

from datasets.augment.rand_occ import NoneOcc
from datasets.augment.rand_occ import RandomRect, RandomEllipse, RandomConnectedPolygon
from datasets.augment.rand_occ import RandomGlassesList, RandomScarf, RandomRealObject
from datasets.augment.rand_occ import RealOcc


""" Read face mxnet dataset and its offline masked version.
    Another 6 types of occlusions (excluding mask) will also be added online.
"""
class FaceByRandOccMask(data.Dataset):
    def __init__(self,
                 root_dir: str,
                 local_rank: int,
                 is_train: bool = True,
                 out_size: tuple = (112, 112),  # (h, w)
                 is_gray: bool = False,
                 use_norm: bool = False,
                 ):
        super(FaceByRandOccMask, self).__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank

        """ Refer to 'README.MD' to see how to convert original face dataset to masked face dataset.
        - train.rec/idx: original face dataset
        - mask_out.rec/idx: faces occluded by masks based on 3D-method
        - mask.rec/idx: the binary mask indicating pixels are occluded or non-occluded (0: occluded; 255: clean)
        """
        path_img_rec = os.path.join(root_dir, 'train.rec')
        path_img_idx = os.path.join(root_dir, 'train.idx')
        path_mask_out_rec = os.path.join(root_dir, 'mask_out.rec')
        path_mask_out_idx = os.path.join(root_dir, 'mask_out.idx')
        path_mask_rec = os.path.join(root_dir, 'mask.rec')
        path_mask_idx = os.path.join(root_dir, 'mask.idx')

        self.img_rec = mx.recordio.MXIndexedRecordIO(path_img_idx, path_img_rec, 'r')
        self.mask_out_rec = mx.recordio.MXIndexedRecordIO(path_mask_out_idx, path_mask_out_rec, 'r')
        self.mask_rec = mx.recordio.MXIndexedRecordIO(path_mask_idx, path_mask_rec, 'r')

        """ Read header from original face dataset """
        s = self.img_rec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:  # flag will be set as len(label) while packing images
            self.header0 = (int(header.label[0]), int(header.label[1]))  # (num of images, classes)
            self.img_idx = np.array(range(1, int(header.label[0])))  # mask_out.rec starts from index 1
        else:
            self.img_idx = np.array(list(self.img_rec.keys))

        """ Resize images as you need """
        self.out_size = out_size
        self.resize = transforms.Resize(self.out_size)

        """ The other 6 types of occlusion (excluding mask)
        1. No occlusion
            - NoneOcc
        2. Geometric shapes
            - Rectangle, Ellipse, Connected Polygon
        3. Real-life objects
            - Glasses, Scarf, RealObject
        """
        self.no_occ = NoneOcc()
        self.trans_occ = (
            RandomRect(),
            RandomEllipse(),
            RandomConnectedPolygon(),
            RandomGlassesList(['./datasets/augment/occluder/glasses_crop',
                               './datasets/augment/occluder/eleglasses_crop']),
            RandomScarf('./datasets/augment/occluder/scarf_crop'),
            RandomRealObject('./datasets/augment/occluder/object_train'),
            # RealOcc(occ_type='rand'),
            # RealOcc(occ_type='hand'),
            # RealOcc(occ_type='coco'),
        )
        self.all_trans = tuple(list(self.trans_occ) + [self.no_occ])

        """ Convert face and mask to tensor """
        self.face_to_tensor = transforms.ToTensor()
        self.mask_to_tensor = Msk2Tenser()

        """ Gray or RGB & Normalize or not """
        self.is_gray = is_gray
        if self.is_gray:
            norm = transforms.Normalize(mean=0.5, std=0.5)
        else:
            norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.norm = norm if use_norm else transforms.Compose([])

        """ Other settings """
        self.is_train = is_train

    def __getitem__(self, index):

        """ 1. Read original face and its class label """
        idx = self.img_idx[index]  # map 'x' to 'x+1'
        s = self.img_rec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):  # label may be a tuple
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)

        """ 2. Add mask or other 6 types of occlusion """
        mask_flag = True if np.random.randint(1, 11) >= 9 else False  # P{mask} = 2/10
        ori, _ = self._get_occluded_face_and_mask(img, idx, False)
        img, msk = self._get_occluded_face_and_mask(img, idx, mask_flag)

        """ 3. Resize to out_size """
        img = self.resize(img)
        msk = self.resize(msk)
        ori = self.resize(ori)

        """ 4. Random Horizontal Flip """
        if np.random.randint(1, 11) >= 5:  # P{flip}=0.5
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
            ori = ori.transpose(Image.FLIP_LEFT_RIGHT)

        """ 5. For img, we convert it to tensor (in [0, 1]) and add Gaussian light to it. """
        img_tensor = self._add_gauss_to_face(img)
        ori_tensor = self.face_to_tensor(ori)  # [0, 1]

        """ 6. For mask, we convert it to tensor (in [0, 1]) and use multiple augmentation methods. """
        img_tensor, msk_tensor = self._add_gauss_to_mask(img_tensor, msk, mask_flag)

        """ 7. Normalize """
        img_tensor = self.norm(img_tensor)
        ori_tensor = self.norm(ori_tensor)

        return img_tensor, msk_tensor, ori_tensor, label

    def __len__(self):
        return len(self.img_idx)

    def _get_occluded_face_and_mask(self, src_img, img_idx: int, mask_flag: bool):
        """
        Get occluded face by 6 types occlusion transforms or 'mask_out.rec'.
        :param src_img: Source image read from mxnet.recordio
        :param img_idx: The index of source image, which is same as masked face image
        :param mask_flag: Use mask or other 6 types of occlusion
        :return: occluded face (PIL.Image, RGB), binary mask (PIL.Image, Gray)
        """
        if not mask_flag:  # Option-A. Add 6 types of occlusion
            src_img = mx.image.imdecode(src_img, to_rgb=1).asnumpy()  # original face, np:(112, 112, 3)
            src_img = Image.fromarray(src_img)
            if 'ms1m' in self.img_rec:
                rand_trans = self.all_trans[np.random.randint(0, len(self.all_trans))]
                out_img, out_mask = rand_trans(src_img)
            elif 'casia' in self.img_rec:
                if np.random.randint(0, 8) >= 4:  # P{rand_occ}=8/10*4/8=4/10
                    random_occ_trans = self.trans_occ[np.random.randint(0, len(self.trans_occ))]
                    out_img, out_mask = random_occ_trans(src_img)
                else:  # P{no_occ}=8/10*4/8=4/10
                    out_img, out_mask = self.no_occ(src_img)
            else:
                raise ValueError('self.img_rec %s not supported' % self.img_rec)

        else:  # Option-B. Add facial mask based on 3D method
            s1 = self.mask_out_rec.read_idx(img_idx)
            header, image = mx.recordio.unpack(s1)
            img_masked = mx.image.imdecode(image, to_rgb=1).asnumpy()  # masked face, rgb

            s2 = self.mask_rec.read_idx(img_idx)
            header, image = mx.recordio.unpack(s2)
            mask = mx.image.imdecode(image, to_rgb=1).asnumpy()  # mask, rgb

            out_img = Image.fromarray(img_masked)
            out_mask = Image.fromarray(mask).convert('L')

        out_img = out_img.convert('L') if self.is_gray else out_img

        return out_img, out_mask

    def _add_gauss_to_face(self, src_img):
        """ 1. Convert to tensor """
        out_img = self.face_to_tensor(src_img)  # [0, 1]

        """ 2. Add Gaussian light to occluded faces """
        height, width = self.out_size
        light = self._get_gauss(0, 0,
                                width, height,
                                center_x=(),
                                center_y=(),
                                radius=128)  # shape:(h, w), range:[0, 1]
        scale = np.random.uniform(0.7, 1.4)
        light = light.astype(np.float16) * scale  # shape:(h, w), range:[0, scale]
        out_img = out_img * light  # higher pixel values denote higher illumination

        """ 3. Revise pixel values into [0.0, 1.0] """
        out_img = out_img / out_img.max()

        return out_img

    def _add_gauss_to_mask(self, src_face_tensor, src_mask, mask_flag: bool):
        """ 6 types of occlusion don't need Gaussian light """
        if not mask_flag:
            out_mask = self.mask_to_tensor(src_mask)
            return src_face_tensor, out_mask

        """ 1. Binarization """
        msk = np.array(src_mask).astype(np.uint8)
        msk_tmp = np.ones(self.out_size) * 255  # white image
        msk_tmp[msk <= 128] = 0  # paint mask by black pixels, 0:occlusion, 255:clean

        """ 2. Randomly choose the transform type 
            - 0.7 ~ 1.0: Gaussian light, Color jitter
            - 0.5 ~ 0.7: Gaussian noise, Color jitter
            - 0.0 ~ 0.5: Rectangle block
        """
        trans_type = np.random.randint(0, 11)

        # the point of rectangle occlusion for mask
        left_top_y, left_top_x = 1, 40 + np.random.randint(-20, 21)
        right_down_y, right_down_x = 111, 100 + np.random.randint(-20, 11)

        height, width = self.out_size
        rescale_map = np.zeros((height, width), dtype=np.float16)

        """ 3. Add Gaussian light to masks """
        msk_light = np.zeros((3, height, width), dtype=np.float16)  # offset of mask
        msk_light[:] = (msk_tmp // 128 * (-1) + 1).astype(np.float16)  # 1:mask, 0:face

        # 3.1.a) Gaussian light
        if trans_type >= 7:
            gauss_map = self._get_gauss(left_top_x, left_top_y,
                                        right_down_x, right_down_y,
                                        center_x=(),
                                        center_y=())
            gauss_map = (gauss_map - 0.5) * 2 * 0.4 * (np.random.randint(0, 2) * 2 - 1)  # random [-1, 1] * 0.4

            rescale_map[left_top_y: right_down_y,
                        left_top_x: right_down_x] = gauss_map  # [-0.4, 0.4]

        # 3.1.b) Gaussian noise
        elif trans_type >= 5:
            rescale_map[left_top_y: right_down_y,
                        left_top_x: right_down_x] = np.random.randn(right_down_y - left_top_y,
                                                                    right_down_x - left_top_x)  # [0, 1]

        # 3.1.c) Rectangle block
        else:
            left_top_y = 40 + np.random.randint(-20, 20)
            right_down_y = 100 + np.random.randint(-20, 10)
            block_map = np.zeros((height, width), dtype=np.float16)
            block_map[left_top_y: right_down_y,
                      left_top_x: right_down_x] = np.random.randint(0, 2) * 2 - 1  # {-1, 1} black or white
            msk_light = msk_light * block_map

        # 3.2) Color jitter
        if trans_type >= 5:
            for c in range(3):
                msk_light[c] = msk_light[c] * rescale_map if np.random.randint(0, 2) >= 1 else 0

        # 3.3) The mask can be converted to gray scale
        if self.is_gray:
            msk_light_gray = (0.2989 * msk_light[0] +
                              0.5870 * msk_light[1] +
                              0.1140 * msk_light[2]) / 3
            msk_light = msk_light_gray

        """ Conduct the augmentation to mask pixels on masked face """
        out_face = src_face_tensor - msk_light
        # # maybe there is no need to revise the light of mask
        # max_light = out_face.max()
        # out_face[left_top_y: right_down_y,
        #          left_top_x: right_down_x] = out_face[left_top_y: right_down_y,
        #                                               left_top_x: right_down_x] / max_light  # revise

        out_mask = torch.from_numpy(msk_tmp // 255).int()

        return out_face, out_mask

    @staticmethod
    def _get_gauss(left_top_x: int, left_top_y: int,
                   right_down_x: int, right_down_y: int,
                   center_x: tuple = (1, 56, 111),
                   center_y: tuple = (1, 56, 111),
                   radius: int = -1,
                   metric: str = 'Euclidean'):
        """
        Generate a 2D Gaussian map (np.array) whose elements are in [0, 1].
        :param left_top_x: left-top point (x, y) of target rectangle
        :param left_top_y: left-top point (x, y) of target rectangle
        :param right_down_x: right-down point (x, y) of target rectangle
        :param right_down_y: right-down point (x, y) of target rectangle
        :param center_x: the center point (x, y) of 2D Gaussian distribution
        :param center_y: the center point (x, y) of 2D Gaussian distribution
        :param radius: the sigma of Gaussian distribution, denoting the highest radiant radius
        :param metric: the distance metric can be 'Euclidean' or 'Manhattan'
        :return:
        """

        image_height = right_down_y - left_top_y
        image_width = right_down_x - left_top_x

        """ The point of light source """
        if len(center_x) == 0 and len(center_y) == 0:
            center_x = left_top_x + (right_down_x - left_top_x) * np.random.random()
            center_y = left_top_y + (right_down_y - left_top_y) * np.random.random()
        else:
            center_x = center_x[np.random.randint(0, len(center_x))]
            center_y = center_y[np.random.randint(0, len(center_y))]

        """ The radiant radius of light source (higher denotes darker) """
        if radius < 0:
            edge = max(image_width, image_height)
            radius = np.random.uniform(int(edge / 1.5), int(edge * 1.5))

        """ Row matrix and column matrix """
        x1 = np.arange(image_width) - center_x
        x_map = x1[None, :].repeat(image_height, axis=0).astype(np.int16)

        y1 = np.arange(image_height) - center_y
        y_map = y1[:, None].repeat(image_width, axis=1).astype(np.int16)

        """ Distance metric
        Using Euclidean is 2.0x faster than using Manhattan.
        """
        if metric == 'Euclidean':
            gauss_map = np.sqrt(x_map ** 2 + y_map ** 2)  # Euclidean Distance, 156ms/1000samples
                                                          # Using np.sqrt can be faster than not using.
        elif metric == 'Manhattan':
            gauss_map = np.abs(x_map) + np.abs(y_map)  # Manhattan Distance, 362ms/1000samples
        else:
            raise ValueError('Distance Metric Error!')

        """ 2D Gaussian map ranges in [0, 1], where higher value denotes higher illumination. """
        gauss_map = np.exp(-0.5 * gauss_map ** 2 / radius ** 2)

        return gauss_map  # min:0, max:1


""" Original mxnet dataset
"""
class MXFaceDataset(data.Dataset):
    def __init__(self, root_dir, local_rank,
                 out_size=(112, 112),
                 use_norm=True,
                 is_gray=False,
                 ):
        super(MXFaceDataset, self).__init__()

        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(out_size)
             ])
        norm_mean, norm_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if is_gray:
            self.transform = transforms.Compose([self.transform,
                                                 transforms.Grayscale()])
            norm_mean, norm_std = 0.5, 0.5
        self.transform = transforms.Compose([self.transform,
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             ])
        if use_norm:
            self.transform = transforms.Compose([self.transform,
                                                 transforms.Normalize(mean=norm_mean, std=norm_std)])

        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


""" Convert occlusion mask (PIL.Image) to tensor (torch.tensor, ranging in [0, 1])
"""
class Msk2Tenser(object):
    def __call__(self, msk):
        msk = np.array(msk, dtype=np.uint8)
        msk[msk != 255] = 0
        msk[msk == 255] = 1
        msk = torch.from_numpy(msk).int()
        return msk


""" MXNet binary dataset reader. 
Refer to https://github.com/deepinsight/insightface.
"""
import pickle
from typing import List
from mxnet import ndarray as nd
class ReadMXNet(object):
    def __init__(self, val_targets, rec_prefix, image_size=(112, 112)):
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.rec_prefix = rec_prefix
        self.val_targets = val_targets

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = self.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def load_bin(self, path, image_size):
        try:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f)  # py2
        except UnicodeDecodeError as e:
            with open(path, 'rb') as f:
                bins, issame_list = pickle.load(f, encoding='bytes')  # py3
        data_list = []
        # for flip in [0, 1]:
        #     data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        #     data_list.append(data)
        for idx in range(len(issame_list) * 2):
            _bin = bins[idx]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))  # (C, H, W)

            img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            import PIL.Image as Image
            fig = Image.fromarray(img.asnumpy(), mode='RGB')
            data_list.append(fig)
            # data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            if idx % 1000 == 0:
                print('loading bin', idx)

            # # save img to '/home/yuange/dataset/LFW/rgb-arcface'
            # img = nd.transpose(img, axes=(1, 2, 0))  # (H, W, C)
            # # save_name = 'ind_' + str(idx) + '.bmp'
            # # import os
            # # save_name = os.path.join('/home/yuange/dataset/LFW/rgb-arcface', save_name)
            # import PIL.Image as Image
            # fig = Image.fromarray(img.asnumpy(), mode='RGB')
            # # fig.save(save_name)

        print('load finished', len(data_list))
        return data_list, issame_list


"""
Evaluation Benchmark
"""
class EvalDataset(data.Dataset):
    def __init__(self,
                 all_img: list = None,
                 issame_list: list = None,
                 pre_trans = None,
                 norm_0_1: bool = False,
                 ):
        """
        :param all_img: List[PIL.Image]
        :param issame_list: List[int], 0 is same, 1 is diff
        """
        self.all_img = all_img
        self.issame_list = issame_list  # 0:is same, 1:diff

        self.pre_trans = pre_trans
        self.post_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1,1]
        ])
        if norm_0_1:
            self.post_trans = transforms.Compose([
            transforms.ToTensor(),  # to [0,1]
        ])

    def __getitem__(self, index):
        img1 = self.all_img[index * 2]
        img2 = self.all_img[index * 2 + 1]
        same = self.issame_list[index]

        img1 = self.pre_trans(img1)
        img2 = self.pre_trans(img2)

        img1 = self.post_trans(img1)
        img2 = self.post_trans(img2)

        return img1, img2, same

    def __len__(self):
        return len(self.issame_list)


if __name__ == '__main__':

    import PIL.Image as Image
    import time

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    mx.random.seed(1)

    is_gray = False

    train_set = FaceByRandOccMask(
        root_dir='/tmp/train_tmp/casia',
        local_rank=0,
        use_norm=True,
        is_gray=is_gray,
    )
    start = time.time()
    for idx in range(100):
        face, mask, label = train_set.__getitem__(idx)
        if idx < 15:
            face = ((face + 1) * 128).numpy().astype(np.uint8)
            face = np.transpose(face, (1, 2, 0))
            if is_gray:
                face = Image.fromarray(face[:, :, 0], mode='L')
            else:
                face = Image.fromarray(face, mode='RGB')
            face.save('face_{}.jpg'.format(idx))
    print('time cost: %d ms' % (int((time.time() - start) * 1000)))
