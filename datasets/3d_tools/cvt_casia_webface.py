import torch
import torchvision
from torch.utils import data
from tqdm import tqdm

import mxnet as mx

import os
import numpy as np
import numbers
import time

from PIL import Image

dataset_path = '/home/yuange/dataset/CASIA/CASIA-Align-144'

train_data = torchvision.datasets.ImageFolder(
    root=dataset_path,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
)


def start_convert(target='train', dataset=train_data, num_classes=100, batch_size=128):

    idx_path = os.path.join(dataset_path, target + '.idx')
    rec_path = os.path.join(dataset_path, target + '.rec')
    write_record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    # The first Header saves 'cnt_samples' and 'cnt_classes'
    first_header = mx.recordio.IRHeader(flag=0, label=[len(dataset) + 1, num_classes],
                                        id=1, id2=0)  # flag will be set as len(label)
    print('total len: %d, total class: %d' % (len(dataset), num_classes))
    first_s = mx.recordio.pack_img(first_header, np.zeros((32, 32, 3)), quality=100, img_fmt='.png')
    header = first_header
    write_record.write_idx(0, first_s)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=14)

    last_time = time.time()
    total_len = len(dataloader)
    for idx, (images, labels) in enumerate(dataloader):

        images = np.asarray(images) * 255
        images = images.transpose((0, 2, 3, 1))  # (B, C, H, W) to (B, H, W, C)
        images = images[..., [2, 1, 0]]  # RGB to BGR

        batch_cnt = images.shape[0]
        for b in range(batch_cnt):
            img = images[b]
            label = labels[b].item()  # Convert Tensor to int
            assert type(label) == int

            header = mx.recordio.IRHeader(flag=0, label=label, id=0, id2=0)
            s = mx.recordio.pack_img(header, img, quality=97, img_fmt='.jpg')

            write_record.write_idx(1 + idx * batch_size + b, s)

        if idx % 20 == 0:
            speed = 20 * batch_size / (time.time() - last_time)
            print('[%d/%d]: Converting, target=%s, num_classes=%d, '
                  'speed=%d images/sec, need_time=%d hours' %
                  (idx, total_len, target, num_classes,
                   int(speed),
                   int((total_len - idx) / speed / 3600)))
            last_time = time.time()

    write_record.close()


if __name__ == '__main__':

    """ Start Convert """
    start_convert('train', train_data, 1000, 1)

    """ Check for train """
    print('=====> Checking training dataset')
    check_target = 'train'
    check_idx_path = os.path.join(dataset_path, check_target + '.idx')
    check_rec_path = os.path.join(dataset_path, check_target + '.rec')
    read_record = mx.recordio.MXIndexedRecordIO(check_idx_path, check_rec_path, 'r')

    for idx in range(10):
        item = read_record.read_idx(idx)
        header, s = mx.recordio.unpack(item)

        print('idx=', idx, 'flag=', header.flag, 'label=', header.label, )

        img = mx.image.imdecode(s).asnumpy()
        img = Image.fromarray(img)
        img.save('train' + str(idx) + '.jpg')

