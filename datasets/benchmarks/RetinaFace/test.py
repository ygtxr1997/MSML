import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

import math
import PIL.Image as Image

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

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

img = cv2.imread('2_2.png')
print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
#if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False

for c in range(count):
    faces, landmarks = detector.detect(img,
                                       thresh,
                                       scales=scales,
                                       do_flip=flip)
    print(c, faces.shape, landmarks.shape)

if faces is not None:
    print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
        #print('score', faces[i][4])
        box = faces[i].astype(np.int)
        #color = (255,0,0)
        color = (0, 0, 255)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        if landmarks is not None:
            ''' 3-1. rotate '''
            landmark5 = landmarks[i].astype(np.int)
            #print(landmark.shape)
            # for l in range(landmark5.shape[0]):
            #     if l == 0:
            #         color = (0, 0, 0)
            #         print(landmark5[l])
            #     elif l == 1:
            #         color = (0, 0, 255)
            #     elif l == 2:
            #         color = (0, 255, 0)
            #     else:
            #         color = (255, 0, 0)
            #     cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color,
            #                2)

            eye_left = (landmark5[0][0], landmark5[0][1])
            eye_right = (landmark5[1][0], landmark5[1][1])
            eye_mid = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)

            dy = eye_right[1] - eye_left[1]
            dx = eye_right[0] - eye_left[0]
            angle = math.atan2(dy, dx) * 180. / math.pi

            rotate_matrix = cv2.getRotationMatrix2D(eye_mid, angle, scale=1)
            img_r = cv2.warpAffine(img, rotate_matrix, (img.shape[1], img.shape[0]))

            for l in range(landmark5.shape[0]):
                landmark5[l] = rotate(origin=eye_mid, point=landmark5[l], angle=angle,
                                      row=img.shape[0])
                cv2.circle(img_r, (landmark5[l][0], landmark5[l][1]), 1, (0, 0, 255), 2)
            cv2.imwrite('rotate_test.jpg', img_r)

            eye_left = (landmark5[0][0], landmark5[0][1])
            eye_right = (landmark5[1][0], landmark5[1][1])
            eye_mid = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)

            ''' 3-2. crop '''
            vec_eye_left, vec_eye_right = np.array(eye_left), np.array(eye_right)
            d_eye = 1 * (np.linalg.norm(vec_eye_left - vec_eye_right)) / 3

            img_c = Image.fromarray(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
            img_c = img_c.crop((
                eye_left[0] - 3.25 * d_eye,
                (eye_left[1]) - 4.5 * d_eye,
                eye_left[0] + (3 + 3.25) * d_eye,
                (eye_left[1]) + 5 * d_eye
            ))
            img_c = img_c.resize((112, 112), resample=Image.BICUBIC)
            img_c.save('crop_test.jpg')

    filename = './detector_test.jpg'
    print('writing', filename)
    cv2.imwrite(filename, img)