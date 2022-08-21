import os

import PIL.Image
import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm

from eval.preprocess.PIPNet.lib.tools import get_lmk_model, demo_image
from eval.preprocess.utils import save, get_5_from_98, get_detector, get_lmk
from eval.preprocess.alignment import norm_crop, norm_crop_with_M, paste_back

from eval.preprocess.mtcnn import MTCNN

mtcnn = MTCNN()


def align_rmfrd(dataset_root: str = '/gavin/datasets/rmfrd',
                input_folder: str = 'masked_whn',
                output_folder: str = 'masked_whn_mtcnn',
                method: str = 'mtcnn',
                ):
    input_folder = os.path.join(dataset_root, input_folder)
    output_folder = os.path.join(dataset_root, output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    fail_cnt = 0
    identity_list = os.listdir(input_folder)
    for identity in tqdm(identity_list):
        in_id_folder = os.path.join(input_folder, identity)
        out_id_folder = os.path.join(output_folder, identity)
        if not os.path.exists(out_id_folder):
            os.mkdir(out_id_folder)

        img_list = os.listdir(in_id_folder)
        for img_name in img_list:
            img = Image.open(os.path.join(in_id_folder, img_name)).convert('RGB')

            if method == 'mtcnn':
                ''' mtcnn (fail: 228) '''
                faces = mtcnn.align_multi(img, min_face_size=16., thresholds=[0.3, 0.4, 0.5],
                                          factor=0.707, crop_size=(112, 112))
                if faces is not None:
                    face = faces[0]
                    face.save(os.path.join(out_id_folder, img_name))
                else:
                    face = None
                    print('[Warning] No face found in %s' % os.path.join(in_id_folder, img_name))
                    img = img.resize((112, 112))
                    img.save(os.path.join(out_id_folder, img_name))
                    fail_cnt += 1

            elif method == 'pipnet':
                ''' PIPNet (fail: 1016) '''
                source_img = np.array(img)
                net, detector = get_lmk_model()
                lmks = demo_image(source_img, net, detector)

                if len(lmks) == 0:  # no face
                    print('[Warning] No face found in %s' % os.path.join(in_id_folder, img_name))
                    source_img = PIL.Image.fromarray(source_img)
                    source_img = source_img.resize((112, 112))
                    fail_cnt += 1
                else:
                    lmk = get_5_from_98(lmks[0])
                    source_img = norm_crop(source_img, lmk, 112, mode='set1', borderValue=0.0)
                    source_img = PIL.Image.fromarray(source_img)
                source_img.save(os.path.join(out_id_folder, img_name))

    print('Total fail: %d' % fail_cnt)


if __name__ == '__main__':
    align_rmfrd(output_folder='masked_whn_mtcnn',
                method='mtcnn',
                )
    align_rmfrd(output_folder='masked_whn_pipnet',
                method='pipnet',
                )
