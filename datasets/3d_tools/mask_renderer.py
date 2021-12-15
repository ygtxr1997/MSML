import os, sys, datetime
import numpy as np
import os.path as osp
import cv2
import insightface
from insightface.app import MaskRenderer


def test_one_img():
    #make sure that you have download correct insightface model pack.
    #make sure that BFM.mat and BFM_UV.mat have been generated
    tool = MaskRenderer(name='antelope')
    tool.prepare(ctx_id=0, det_size=(128, 128))
    image = cv2.imread("./samples/Tom_Hanks_54745.png")
    mask_image  = cv2.imread("./masks/mask1.jpg")
    params = tool.build_params(image)
    mask_out = tool.render_mask(image, mask_image, params)

    cv2.imwrite('output_mask.jpg', mask_out)


def iterate_path(root_path='/home/yuange/dataset/CASIA/CASIA-Align-144',
                 out_path='/home/yuange/dataset/CASIA/CASIA-Align-144-Masked-1type',
                 types=1):

    tool = MaskRenderer(name='antelope')
    tool.prepare(ctx_id=0, det_size=(128, 128))

    masks_list = ['masks/mask1.jpg', 'masks/mask2.jpg', 'masks/black-mask.png',
                  'masks/mask3.png', 'masks/mask4.jpg', 'masks/mask5.jpg',
                  'masks/mask6.png', 'masks/mask7.png', 'masks/mask8.png',
                  'masks/mask9.png', 'masks/mask10.png', 'masks/mask11.png']
    masks_cnt = len(masks_list)
    group_cnt = masks_cnt // types
    assert masks_cnt == group_cnt * types

    from tqdm import tqdm
    import random
    import PIL.Image as Image

    for identity in tqdm(os.listdir(root_path)):

        cur_path = osp.join(root_path, identity)

        if not os.path.exists(os.path.join(out_path, identity)):
            os.makedirs(os.path.join(out_path, identity))

        for img in os.listdir(cur_path):

            # Detect 68 landmarks
            image = cv2.imread(os.path.join(cur_path, img))
            params = tool.build_params(image)

            # Use different masks
            for i in range(types):

                idx = np.random.randint(i * group_cnt, (i + 1) * group_cnt)
                mask_image = cv2.imread(masks_list[idx])

                if params is not None:
                    mask_out = tool.render_mask(image, mask_image, params)  # use single thread to test the time cost
                else:
                    mask_out = image

                cv2.imwrite(os.path.join(out_path, identity,  str(i) + '_' + img), mask_out)  # '2_001.jpg'

                # Save mask
                diff = (np.asarray(mask_out, dtype=np.int16) - np.asarray(image, dtype=np.int16))[:, :, 0]
                msk = np.ones([image.shape[1], image.shape[0]], dtype=np.uint8) * 255
                msk[diff != 0] = 0
                msk = Image.fromarray(msk)
                msk.save(os.path.join(out_path, identity, 'mask_' + str(i) + '_' + img))  # 'mask_2_001.jpg'


if __name__ == "__main__":

    """ Option 1: test for 1 image """
    test_one_img()

    """ Option 2: generate masked face training dataset """
    # iterate_path()