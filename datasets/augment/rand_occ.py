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
class RandomBlock(object):
    fill_list = ['black', 'gauss',]
    def __init__(self,
                 lo: int,
                 hi: int,
                 fill: str = 'black'):
        self.lo = lo
        self.hi = hi
        self.fill = fill
        assert fill in RandomBlock.fill_list

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
class RandomRect(object):
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


""" Random Ellipse Occlusion
This transform can be used for training and testing.
Init Params:
    - connected_num: the num of simply connected graphs (default: 1)
    - lo_ratio: the min area ratio of occlusion (default: 0.2)
    - hi_ratio: the max area ratio of occlusion (default: 0.4)
    - use_rand_color: use random color or not (default: True)
"""
class RandomEllipse(object):
    def __init__(self,
                 connected_num:int = 1,
                 lo_ratio: float = 0.2,
                 hi_ratio: float = 0.4,
                 use_rand_color:bool = True
                 ):
        self.connected_num = connected_num
        self.lo_ratio = lo_ratio
        self.hi_ratio = hi_ratio
        self.use_rand_color = use_rand_color

    def __call__(self, img):
        face_arr = np.array(img)
        height, width = img.size[1], img.size[0]
        channel = 1 if len(face_arr.shape) == 2 else 3

        # 1. Generate the ellipse (np.array), where 0 denotes no occlusion.
        # The shape of ellipse (np.array) is (height, width) without channel dimension.
        ellipse = self._get_ellipse(height, width)

        # 2. Add color to the ellipse
        color_list = np.array((0, 0, 0), dtype=np.uint8)
        for c in range(channel):
            color_list[c] = np.random.randint(1, 256) if self.use_rand_color else 255
        face_arr[ellipse != 0] = color_list if channel == 3 else color_list[0]

        # 3. Get occluded face and occlusion mask
        msk = np.ones([height, width], dtype=np.uint8) * 255
        msk[ellipse != 0] = 0

        img_surpass = Image.fromarray(face_arr)
        msk = Image.fromarray(msk)

        return img_surpass, msk

    def _get_ellipse(self, height, width):
        """
        Create random oval shape. \n
        0: no occlusion \n
        1~255: occluded by random gray value \n
        """
        ellipse = np.zeros([height, width], dtype=np.uint8)

        ch = np.random.randint(height // 5, 4 * height // 5)
        cw = np.random.randint(width // 5, 4 * width // 5)
        ah = np.random.randint(20, min(ch, height - ch))
        ratio = np.random.uniform(self.lo_ratio, self.hi_ratio)
        aw = int(height * width * ratio / (3.14 * ah))
        angle = 0
        gray_val = np.random.randint(1, 256) if self.use_rand_color else 255

        cv2.ellipse(ellipse, (cw, ch), (aw, ah), angle, 0, 360, gray_val, -1)

        return ellipse


""" Random Simply Connected Polygon Occlusion
This transform can be used for training and testing.
Init Params:
    - connected_num: the num of simply connected graphs (default: 1)
    - ratio: the area ratio of occlusion (default: 0.4)
    - use_rand_color: use random color or not (default: True)
    - lo_points_num: min number of points on a circle (default: 4) [lo, hi)
    - hi_points_num: max number of points on a circle (default: 11) [lo, hi)
    - use_circle: use circle or not (ellipse) (default: True)
"""
class RandomConnectedPolygon(object):
    # TODO: Use triangles, refer to: https://blog.csdn.net/islittlehappy/article/details/81533090
    def __init__(self,
                 connected_num: int = 1,
                 ratio: float = 0.4,
                 use_rand_color: bool = True,
                 lo_points_num: int = 4,
                 hi_points_num: int = 11,
                 use_circle: bool = True,
                 ):
        self.connected_num = connected_num
        self.ratio = ratio
        self.use_rand_color = use_rand_color
        self.lo_points_num = lo_points_num
        self.hi_points_num = hi_points_num
        self.use_circle = use_circle

    def __call__(self, img):
        face_arr = np.array(img)
        height, width = img.size[1], img.size[0]
        channel = 1 if len(face_arr.shape) == 2 else 3

        # 1. Generate the polygon shape (np.array), where 0 denotes no occlusion.
        # The size of polygon shape (np.array) is (height, width) without channel dimension.
        polygon = self._get_polygon(height, width)

        # 2. Add color to the polygon
        color_list = np.array((0, 0, 0), dtype=np.uint8)
        for c in range(channel):
            color_list[c] = np.random.randint(1, 256) if self.use_rand_color else 255
        face_arr[polygon != 0] = color_list if channel == 3 else color_list[0]

        # 3. Get occluded face and occlusion mask
        msk = np.ones((height, width), dtype=np.uint8) * 255
        msk[polygon != 0] = 0

        img_surpass = Image.fromarray(face_arr)
        msk = Image.fromarray(msk)

        return img_surpass, msk

    def _get_polygon(self, height, width):
        """
        Create random polygon (type: np.array).
            - 0: no occlusion
            - 1~255: occluded by random gray value
        Process:
            - Step1. Choose the center of a circle (or ellipse)
            - Step2. Generate a big circle (or ellipse) and a small circle (or ellipse)
            - Step3. Choose the first point on the big circle (or ellipse)
            - Step4. Get next point on the big circle (or ellipse) or the small circle (or ellipse) step by step
            - Step5. Paint the polygon according to these points with CV2
        """
        polygon = np.zeros((height, width), dtype=np.uint8)

        # Choose the center
        point_cnt = np.random.randint(self.lo_points_num, self.hi_points_num)
        points = np.zeros((2 * point_cnt + 2, 2), dtype=np.int32)  # Should be np.int32!

        center_x = np.random.randint(height // 5, 4 * height // 5)
        center_y = np.random.randint(width // 5, 4 * width // 5)

        # We need a big circle (or ellipse) and a small circle (or ellipse)
        big_radius = np.random.randint(height // 5, 1.3 * height // 5)
        small_radius = big_radius / np.random.uniform(1.3, 2.6)
        big_angle, small_angle = 0, 0  # range: [0, 2pi]
        get_next_point = self._calc_from_circle if self.use_circle else self._calc_from_ellipse

        # First point
        points[0] = get_next_point(big_radius, big_angle, center_x, center_y)

        # Subsequent points
        small_ind, big_ind = 0, 0
        for big_ind in range(point_cnt):
            big_angle += 2 * math.pi / point_cnt * np.random.uniform(0.7, 1.3)
            points[big_ind + small_ind + 1] = get_next_point(big_radius, big_angle, center_x, center_y)

            if np.random.random() > 0.5:
                small_ind += 1
                small_angle += 2 * math.pi / point_cnt * np.random.uniform(0.6, 1.4)
                points[big_ind + small_ind + 1] = get_next_point(small_radius, small_angle, center_x, center_y)

        # All points
        points = points[:1 + big_ind + small_ind + 1]  # (first, (big...), (small...),)
        points = np.array([points])  # Should be a 3d-array

        gray_val = np.random.randint(1, 256) if self.use_rand_color else 255
        cv2.fillPoly(polygon, points, gray_val)

        return polygon

    @staticmethod
    def _calc_from_circle(radius, angle, center_x, center_y):
        target_x = center_x + radius * math.cos(angle)
        target_y = center_y + radius * math.sin(angle)
        return np.array((int(target_x), int(target_y)))

    @staticmethod
    def _calc_from_ellipse(radius, angle, center_x, center_y):
        radius_a = radius * np.random.uniform(0.5, 1.5)
        radius_b = radius * np.random.uniform(0.5, 1.5)
        target_x = center_x + radius_a * math.cos(angle)
        target_y = center_y + radius_b * math.sin(angle)
        return np.array((int(target_x), int(target_y)))


""" Random Glasses Occlusion
This transform is only used for training.
Init Params:
    - glasses_path: the path to glasses image folder
    - occ_height: we should resize the glasses images into the same height (default: 40)
    - occ_width: we should resize the glasses images into the same width (default: 89)
    - height_scale: the resized images can be randomly rescaled by -h_s to +h_s (h_s >= 1.0, default: 1.1)
    - width_scale: the resized images can be randomly rescaled by -w_s to +w_s (w_s >= 1.0, default: 1.1)
"""
class Glasses(object):
    def __init__(self,
                 glasses_path: str = 'occluder/glasses_crop/',
                 occ_height: int = 40,
                 occ_width: int = 80,
                 height_scale: float = 1.1,
                 width_scale: float = 1.1,
                 ):
        self.glasses_root = glasses_path
        self.glasses_list = np.array(os.listdir(glasses_path))
        self.glasses_num = len(self.glasses_list)

        self.occ_height = occ_height
        self.occ_width = occ_width
        self.height_scale = height_scale
        self.width_scale = width_scale

        # Preload the image folder
        self.object_imgs = np.zeros((self.glasses_num,
                                     occ_height, occ_width, 4), dtype=np.uint8)  # (num, height, width, RGBA)
        for idx in range(self.glasses_num):
            object_path = os.path.join(self.glasses_root, self.glasses_list[idx])
            object = Image.open(object_path).convert('RGBA')  # [w, h]: (125, 40+)
            object = object.resize((occ_width, occ_height))
            self.object_imgs[idx] = np.array(object, dtype=np.uint8)  # [h, w, c=4]

    def __call__(self, img):
        mode = img.mode  # 'L' or 'RGB'
        height, width = img.size[1], img.size[0]

        """ 1. Get an occlusion image from the preloaded list, and resize it randomly """
        glasses = self.object_imgs[np.random.randint(0, self.glasses_num)]  # np-(h, w, RGBA)
        glasses = Image.fromarray(glasses, mode='RGBA')  # PIL-(h, w, RGBA)
        occ_width = int(self.occ_width * np.random.uniform(1 / self.width_scale, self.width_scale))  # w'
        occ_height = int(self.occ_height * np.random.uniform(1 / self.height_scale, self.height_scale))  # h'
        glasses = glasses.resize((occ_width, occ_height))  # PIL-(h', w', RGBA)

        """ 2. Split Alpha channel and RGB channels, and convert RGB channels into img.mode """
        alpha = np.array(glasses)[:, :, -1].astype(np.uint8)  # np-(h', w', A)
        glasses = glasses.convert(mode)  # PIL-(h', w', mode)

        """ 3. Generate top-left point (x, y) of occlusion """
        x_offset = int((0.12 + np.random.randint(-5, 6) * 0.02) * width)
        y_offset = int((0.3 + np.random.randint(-5, 6) * 0.01) * height)

        """ 4. Surpass the face by occlusion, based on np.array """
        face_arr = np.array(img)  # (H, W, mode)
        glasses_arr = np.array(glasses)  # (h', w', mode)

        face_crop = face_arr[y_offset: y_offset + occ_height,
                             x_offset: x_offset + occ_width]  # Crop the face according to the glasses position
        glasses_arr[alpha == 0] = face_crop[alpha == 0]  # 'Alpha == 0' denotes transparent pixel
        face_arr[y_offset: y_offset + occ_height,
                 x_offset: x_offset + occ_width] = glasses_arr  # Overlap, np-(H, W, mode)

        """ 5. Get occluded face and occlusion mask """
        img_glassesed = Image.fromarray(face_arr)  # PIL-(H, W, mode)

        msk_shape = (height, width) if mode == 'L' else (height, width, 3)
        msk = np.ones(msk_shape, dtype=np.uint8) * 255
        glasses_arr[alpha != 0] = 0  # occluded
        glasses_arr[alpha == 0] = 255  # clean
        msk[y_offset: y_offset + occ_height,
            x_offset: x_offset + occ_width] = glasses_arr
        msk = Image.fromarray(msk).convert('L')

        return img_glassesed, msk


if __name__ == '__main__':

    face = Image.open('375_face.jpg', 'r')
    trans = Glasses()
    for idx in range(1000):
        ret, _ = trans(face)
        if idx < 15:
            ret.save('output_{}.jpg'.format(idx))