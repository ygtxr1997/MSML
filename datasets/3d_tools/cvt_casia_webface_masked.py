import os
import sys
sys.path.append('..')
sys.path.append('./tools')
import random
import numbers
import time
from tqdm import tqdm

import mxnet as mx
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import PIL.Image

import insightface
from insightface.app import MaskRenderer

from torch.utils.data import DataLoader

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        print('[Dataset INFO]:', header)
        # The first Header saves 'cnt_samples' and 'cnt_classes'
        # > HEADER(flag=2, label=array([490624., 501196.], dtype=float32), id=0, id2=0)
        # flag will be set as len(label)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
            print('**** [casia-train] flag>0, cnt_sample=%d, cnt_class=%d ****' % (
                int(header.label[0]) - 1, int(header.label[1])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        """ For mask renderer """
        self.masks_list = ['masks/mask1.jpg', 'masks/mask2.jpg', 'masks/black-mask.png',
                      'masks/mask3.png', 'masks/mask4.jpg', 'masks/mask5.jpg',
                      'masks/mask6.png', 'masks/mask7.png', 'masks/mask8.png',
                      'masks/mask9.png', 'masks/mask10.png', 'masks/mask11.png']
        self.masks_cnt = len(self.masks_list)
        self.types = 1  # How many types of masks for each face
        self.group_cnt = self.masks_cnt // self.types
        assert self.masks_cnt == self.group_cnt * self.types
        masks_cv2_list = []
        for img in self.masks_list:
            masks_cv2_list.append(cv2.imread(img))
        self.masks_list = masks_cv2_list

        self.tool = MaskRenderer(name='antelope')
        self.tool.prepare(ctx_id=0, det_size=(128, 128))

        path_mask_out_rec = os.path.join(root_dir, 'mask_out.rec')
        path_mask_out_idx = os.path.join(root_dir, 'mask_out.idx')
        self.mask_out_rec = mx.recordio.MXIndexedRecordIO(path_mask_out_idx,
                                                          path_mask_out_rec, 'w')
        path_mask_rec = os.path.join(root_dir, 'mask.rec')
        path_mask_idx = os.path.join(root_dir, 'mask.idx')
        self.mask_rec = mx.recordio.MXIndexedRecordIO(path_mask_idx,
                                                      path_mask_rec, 'w')

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)

        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        id1, id2 = header.id, header.id2

        # Detect 68 landmarks from input face
        sample = mx.image.imdecode(img).asnumpy()  # (112, 112, 3), rgb
        face_image = cv2.merge([sample[:, :, 2], sample[:, :, 1], sample[:, :, 0]])
        params = self.tool.build_params(face_image)

        # Use different masks
        for i in range(self.types):

            idx = np.random.randint(i * self.group_cnt, (i + 1) * self.group_cnt)
            mask_image = self.masks_list[idx]

            # Render mask
            if params is not None:
                mask_out = self.tool.render_mask(face_image, mask_image, params)  # bgr
            else:
                mask_out = face_image

            # Save mask
            mask_out_np = np.asarray(mask_out, dtype=np.uint8)  # bgr
            header = mx.recordio.IRHeader(flag=2, label=float(label), id=id1, id2=id2)
            s1 = mx.recordio.pack_img(header, mask_out_np)
            self.mask_out_rec.write_idx(id1, s1)

            diff = (np.asarray(mask_out, dtype=np.int16) - np.asarray(face_image, dtype=np.int16))[:, :, 0]
            msk = np.ones([face_image.shape[1], face_image.shape[0]], dtype=np.uint8) * 255
            msk[diff != 0] = 0
            s2 = mx.recordio.pack_img(header, msk)
            self.mask_rec.write_idx(id1, s2)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


""" Write Record (Mask_Out & Mask) """
def write_record(root_dir):
    trainset = MXFaceDataset(root_dir=root_dir, local_rank=0)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    end = time.time()
    for step, (img, label) in enumerate(train_loader):
        if step % 5 == 0:
            print('[%d/%d]: time:%.3f' % (step, len(train_loader), time.time() - end))
        end = time.time()


""" Read Record (Mask_Out & Mask) """
def read_record(root_dir):
    path_mask_out_rec = os.path.join(root_dir, 'mask_out.rec')
    path_mask_out_idx = os.path.join(root_dir, 'mask_out.idx')
    mask_out_rec = mx.recordio.MXIndexedRecordIO(path_mask_out_idx,
                                                 path_mask_out_rec, 'r')
    path_mask_rec = os.path.join(root_dir, 'mask.rec')
    path_mask_idx = os.path.join(root_dir, 'mask.idx')
    mask_rec = mx.recordio.MXIndexedRecordIO(path_mask_idx,
                                             path_mask_rec, 'r')

    total_time = 0
    label = 0
    small = 9999999
    for index in range(1, 5179510):
        s1 = mask_out_rec.read_idx(index)
        header, image = mx.recordio.unpack(s1)
        small = min(small, header.label)
        label = max(label, header.label)
        start = time.time()
        # image = mx.image.imdecode(image, to_rgb=0).asnumpy()  # bgr
        total_time += time.time() - start
        # image = PIL.Image.fromarray(image)
        # image.save('mask_out_' + str(index) + '.jpg')
        # cv2.imwrite('mask_out_' + str(index) + '.jpg', image)

        # s2 = mask_rec.read_idx(index)
        # header, image = mx.recordio.unpack(s2)
        # image = mx.image.imdecode(image, to_rgb=0).asnumpy()
        # mask = cv2.merge([image[:, :, 2], image[:, :, 1], image[:, :, 0]])
        # # cv2.imwrite('mask_' + str(index) + '.jpg', mask)

    print('total cost time : %.6f' % total_time)
    print('max label:', label, 'small label:', small)


if __name__ == '__main__':

    root_dir = '/home/yuange/dataset/CASIA/insightface'

    write_record(root_dir)
    read_record(root_dir)


