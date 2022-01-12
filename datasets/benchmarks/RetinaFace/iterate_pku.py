from tqdm import tqdm
import os

from PIL import ImageFile
from PIL import Image

import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

""" [1] Test for one image """
g_image_path = 'images/test3.jpg'
# g_image_path = '/GPUFS/sysu_zhenghch_1/yuange/datasets/CASIA/CASIA-WebFace/0000045/001.jpg'

""" [2] Run for image dataset 
    Read dataset from 'g_image_folder', detect&align the faces, and write faces to 'g_target_folder'
"""
''' Example 1 '''
# g_image_folder = '/GPUFS/sysu_zhenghch_1/yuange/datasets/LFW/lfw/image'
# g_target_folder = '/GPUFS/sysu_zhenghch_1/yuange/datasets/LFW/lfw/align'

''' Example 2 '''
# g_image_folder = '/home/yuange/dataset/LFW/lfw-rgb/image'
# g_target_folder = '/home/yuange/dataset/LFW/rgb-mtcnn/image'

''' Example 3 '''
g_image_folder = '/home/yuange/dataset/PKU-Masked-Face/images'
g_target_folder = '/home/yuange/dataset/PKU-Masked-Face/images-retina-multi-pose'

''' Example 4 '''
# g_image_folder = 'fake_dataset'
# g_target_folder = 'output_dataset'


def main():
    # image = Image.open(g_image_path)
    # bounding_boxes, landmarks = detect_faces(image)
    # image_b = show_bboxes(image, bounding_boxes, landmarks)
    # image_b.show()
    # display(image)
    # image_a = align_img(image, bounding_boxes, landmarks)
    # display(image_a)

    """ Op1: LFW (not used) """
    # iterate_img_lfw()

    """ Op2: IJB (not used) """
    # iterate_img_ijb()

    """ Op3: IJB/img-crop (not used) """
    # iterate_img_ijb_crop()

    """ Op4: PKU-Masked-Face """
    iterate_img_pku()


def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def show_img(img, bounding_boxes, landmarks):
    img = show_bboxes(img, bounding_boxes, landmarks)
    img.show()
    # display(img)


def iterate_img_pku():
    if not os.path.exists(g_target_folder):
        os.mkdir(g_target_folder)

    """ load model """
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

    for identity in tqdm(os.listdir(g_image_folder)):  # '0_0/0_0.png'
        cur_path = os.path.join(g_image_folder, identity)
        target_path = os.path.join(g_target_folder, identity)

        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for img_name in os.listdir(cur_path):
            if '.png' not in img_name:
                print('error in %s\n' % img_name)
                continue

            """ 1. read image """
            img = cv2.imread(os.path.join(cur_path, img_name))
            im_shape = img.shape  # (height, width)
            height, width = im_shape[0], im_shape[1]

            thresh = 0.8
            scales = [128, 128]
            count = 1
            target_size = scales[0]
            max_size = scales[1]
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            # if im_size_min>target_size or im_size_max>max_size:
            im_scale = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)

            scales = [im_scale]
            flip = False

            """ 2. detect face """
            for c in range(count):
                """ faces: (num_faces, ?)
                    landmarks: (num_faces, [l_eye, r_eye, nose, l_mouse, r_mouse], [x_pos, y_pos])
                """
                faces, landmarks = detector.detect(img,
                                                   thresh,
                                                   scales=scales,
                                                   do_flip=flip)

            if faces.shape[0] == 0:
                print('no face here')
                continue

            """ 3. align face """
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                color = (0, 0, 255)
                # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    """ 3-1. rotate """
                    landmark5 = landmarks[i].astype(np.int)

                    eye_left = (landmark5[0][0], landmark5[0][1])
                    eye_right = (landmark5[1][0], landmark5[1][1])
                    eye_mid = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)

                    dy = eye_right[1] - eye_left[1]
                    dx = eye_right[0] - eye_left[0]
                    angle = math.atan2(dy, dx) * 180. / math.pi

                    rotate_matrix = cv2.getRotationMatrix2D(eye_mid, angle, scale=1)
                    img_r = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))  # 1:width, 0:height

                    for l in range(landmark5.shape[0]):
                        landmark5[l] = rotate(origin=eye_mid, point=landmark5[l], angle=angle,
                                              row=img.shape[0])
                        # cv2.circle(img_r, (landmark5[l][0], landmark5[l][1]), 1, (0, 0, 255), 2)

                    """ 3-2. crop """
                    eye_left = (landmark5[0][0], landmark5[0][1])
                    eye_right = (landmark5[1][0], landmark5[1][1])

                    vec_eye_left, vec_eye_right = np.array(eye_left), np.array(eye_right)
                    d_eye = 1 * (np.linalg.norm(vec_eye_left - vec_eye_right)) / 3

                    img_c = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
                    # img_c = img_c.crop((
                    #     eye_left[0] - 3.25 * d_eye,
                    #     (eye_left[1]) - 4.5 * d_eye,
                    #     eye_left[0] + (3 + 3.25) * d_eye,
                    #     (eye_left[1]) + 5 * d_eye
                    # ))

                    from torchvision.transforms import CenterCrop
                    center_crop = CenterCrop(max(im_shape[0], im_shape[1]))
                    img_c = center_crop(img_c)

                    # img_c = img_c.crop((
                    #     0, eye_mid[1] - width * 0.4,
                    #     width, eye_mid[1] + width * 0.6
                    # ))  # (x1, y1, x2, y2)

                    img_c = img_c.resize((112, 112), resample=Image.BICUBIC)
                    img_c.save(os.path.join(target_path, img_name))


def iterate_img_lfw():
    for identity in tqdm(os.listdir(g_image_folder)):
        cur_path = os.path.join(g_image_folder, identity)
        target_path = os.path.join(g_target_folder, identity)

        if not os.path.exists(target_path):
            os.mkdir(target_path)

        for img_name in os.listdir(cur_path):
            if '.jpg' not in img_name:
                print('error in %s\n' % img_name)
                continue

            img = Image.open(os.path.join(cur_path, img_name)).convert("RGB")
            bounding_boxes, landmarks = detect_faces(img)
            img_a = align_img(img, bounding_boxes, landmarks)
            img_a.save(os.path.join(target_path, img_name))


def iterate_img_ijb():
    # ijb_image_folder = "/data/yuange/IJB/IJB/IJB-C/images"
    # ijb_crop_folder = "/data/yuange/IJB/IJB/IJB-C/images/img-crop"
    # ijb_align_folder = "/data/yuange/IJB/IJB/IJB-C/images/img-align"
    # ijb_csv = "/data/yuange/IJB/IJB/IJB-C/protocols/ijbc_metadata_with_age.csv"

    ijb_image_folder = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/IJB-C/images"
    ijb_crop_folder = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/IJB-C/images-crop"
    ijb_align_folder = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/IJB-C/images-align"
    ijb_csv = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/IJB-C/protocols/ijbc_metadata_with_age.csv"

    """ 1. Crop """
    print("=> Crop started")
    import csv
    with open(ijb_csv, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        result = result[1:]

        for line in tqdm(result):
            csv_filename = line[1]  # 'img/xxxx.jpg'

            filemode = "RGB" if csv_filename[-3:] == "jpg" else "RGBA"

            csv_face_x, csv_face_y = int(line[3]), int(line[4])
            csv_face_w, csv_face_h = int(line[5]), int(line[6])

            img_source = os.path.join(ijb_image_folder, csv_filename)
            img_target = os.path.join(ijb_crop_folder, csv_filename)

            img = Image.open(img_source).convert(filemode)
            img_cropped = img.crop((csv_face_x, csv_face_y,
                                    csv_face_x + csv_face_w, csv_face_y + csv_face_h))
            img_cropped.save(img_target)
    print("=> Crop finished")

    """ 2. Align """
    print("=> Align started ...")
    for img_name in tqdm(os.listdir(ijb_crop_folder)):
        img = Image.open(os.path.join(ijb_crop_folder, img_name)).convert("RGB")
        bounding_boxes, landmarks = detect_faces(img)
        img_a = align_img(img, bounding_boxes, landmarks)
        img_a.save(os.path.join(ijb_align_folder, img_name))


def iterate_img_ijb_crop():
    ijb_crop_folder = "/home/yuange/dataset/IJB/rgb-lightcnn-crop"
    ijb_align_folder = "/home/yuange/dataset/IJB/rgb-lightcnn-align"

    # ijb_crop_folder = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/img-crop"
    # ijb_align_folder = "/GPUFS/sysu_zhenghch_1/yuange/datasets/IJB/IJB/img-align"

    test_file = os.path.join(ijb_crop_folder, 'test.list')

    with open(test_file, "r") as test_list:
        for img_name in tqdm(test_list.readlines()):
            img_name = img_name[:-1]
            img = Image.open(os.path.join(ijb_crop_folder, img_name)).convert("RGB")
            bounding_boxes, landmarks = detect_faces(img)
            img_a = align_img(img, bounding_boxes, landmarks)
            img_a.save(os.path.join(ijb_align_folder, img_name))


if __name__ == "__main__":
    main()
