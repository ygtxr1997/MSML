import os
import time
import copy

import numpy as np
import math

import cv2
from PIL import Image
from torchvision import transforms

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


""" Random Block (Square) Occlusion
This transform is only used for testing.
Init Params:
    - lo: lowest ratio (%) in range [lo, hi)
    - hi: highest ratio (%) in range [lo, hi)
    - fill: 'black' means black square;
            'gauss' means Gaussian noise square.
"""
class RandomBlockOcc(object):
    fill_list = ['black', 'gauss',]
    def __init__(self,
                 lo: int,
                 hi: int,
                 fill: str = 'black'):
        self.lo = lo
        self.hi = hi
        self.fill = fill
        assert fill in RandomBlockOcc.fill_list

    def __call__(self, img):
        ratio = np.random.randint(self.lo, self.hi) * 0.01
        img = self._block_occ(img, ratio)
        return img

    def _block_occ(self, img, ratio):
        width, height = img.size[0], img.size[1]
        assert width == height
        img_occ = copy.deepcopy(img)

        if ratio == 0:
            return img_occ

        block_width = int((ratio * width * width) ** 0.5)
        if self.fill == 'black':
            occ = Image.fromarray(np.zeros([block_width, block_width], dtype=np.uint8))
        elif self.fill == 'gauss':
            if img.mode == 'L':
                occ = Image.fromarray(np.random.randn(block_width, block_width) * 255)
            elif img.mode == 'RGB':
                occ_r = np.random.randn(block_width, block_width)
                occ_g = np.random.randn(block_width, block_width)
                occ_b = np.random.randn(block_width, block_width)
                occ = (np.stack((occ_r, occ_g, occ_b), axis=2) * 255).astype(np.uint8)
                occ = Image.fromarray(occ)
            else:
                raise ValueError('Error Image type.')

        randx = np.random.randint(0, width - block_width + 1)
        randy = np.random.randint(0, width - block_width + 1)
        img_occ.paste(occ, (randx, randy))

        return img_occ


""" Don't Add Any Occlusion
This transform is only used for training.
"""
class NoneOcc(object):
    def __init__(self, ret_msk:bool=True):
        self.ret_msk = ret_msk

    def __call__(self, img):
        width, height = img.size[0], img.size[1]
        assert width == height
        msk = np.ones((height, width), dtype=np.uint8) * 255  # white denotes no occlusion
        msk = Image.fromarray(msk)
        return img, msk


""" Random Rectangle Occlusion
This transform is only used for training.
Init Params:
    - lo: lowest ratio (%) in range [lo, hi)
    - hi: highest ratio (%) in range [lo, hi)
"""
class RandomRectOcc(object):
    def __init__(self,
                 lo: int = 0,
                 hi: int = 36,):
        self.lo = lo
        self.hi = hi

    def __call__(self, img):
        ratio = np.random.randint(self.lo, self.hi) * 0.01
        img, msk = self._rect_occ(img, ratio)
        return img, msk

    def _rect_occ(self, img, ratio):
        width, height = img.size[0], img.size[1]
        assert width == height
        img_occ = copy.deepcopy(img)

        occ_size = int(width * height * ratio)
        occ_width = np.random.randint(int(width * ratio) + 1, width + 1)
        occ_height = int(occ_size / occ_width)
        occ_randx = np.random.randint(0, width - occ_width + 1)
        occ_randy = np.random.randint(0, height - occ_height + 1)

        img_occ = np.array(img_occ, dtype=np.uint8)
        if img.mode == 'L':
            gray_val = np.random.randint(0, 256)
            img_occ[occ_randy:occ_randy + occ_height,
                    occ_randx:occ_randx + occ_width] = gray_val
        elif img.mode == 'RGB':
            for c in range(3):
                rgb_val = np.random.randint(0, 256)
                img_occ[occ_randy:occ_randy + occ_height,
                        occ_randx:occ_randx + occ_width,
                        c] = rgb_val
        else:
            raise ValueError('Error Image type.')
        img_occ = Image.fromarray(img_occ)

        msk = np.ones((height, width), dtype=np.uint8) * 255  # white denotes no occlusion
        msk[occ_randy:occ_randy + height,
            occ_randx:occ_randx + width] = 0  # black denotes occlusion
        msk = Image.fromarray(msk)

        return img_occ, msk

if __name__ == '__main__':

    face = Image.open('Dan_Kellner_0001.jpg', 'r')
    trans = RandomRectOcc(lo=0, hi=41,)
    for idx in range(100):
        ret, _ = trans(face)
        if idx < 15:
            ret.save('output_{}.jpg'.format(idx))