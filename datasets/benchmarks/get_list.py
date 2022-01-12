import os
import random
from tqdm import tqdm
import platform
import os.path as osp

"""
|---root_path
    |---img_folder
        |---identity_1
            |---image1.jpg
            |---image2.jpg
            |---...
        |---identity_2
            |---...
        |---identity_3
            |---...
        |---...
    |---relative.list
    |---full.list
    |---train.list
    |---val.list
    |---train_full.list
    |---val_full.list
    |---ver.list
"""

config = {
    'PKU': {
        'root_path': '/home/yuange/dataset/PKU-Masked-Face',
        # 'root_path': '/GPUFS/sysu_zhenghch_1/yuange/datasets/PKU-Masked-Face',
        'img_folder_name': 'images-retina-multi-pose',  # 图片集所在文件夹
        'list_file_name': 'relative.list',  # 相对路径列表
        'full_list_file_name': 'full.list',  # 绝对路径
        'train_list_file_name': 'train.list',  # 训练集相对路径q
        'val_list_file_name': 'val.list',  # 验证集相对路径
        'train_full_list_file_name': 'train_full.list',  # 训练集绝对路径
        'val_full_list_file_name': 'val_full.list',  # 验证集绝对路径
        'ver_list_file_name': 'ver.list',  # 人脸验证使用, 两两分配
        'out_path': '/home/yuange/dataset/PKU-Masked-Face'  # list文件保存位置
        # 'out_path': '/GPUFS/sysu_zhenghch_1/yuange/datasets/PKU-Masked-Face'  # list文件保存位置
    },
    'AR-pre': {
        'root_path': '/home/yuange/dataset/AR',  # 根路径
        'img_folder_name': 'AR-pre/input/',  # 图片集所在文件夹
        'list_file_name': 'relative.list',  # 相对路径列表
        'full_list_file_name': 'full.list',  # 绝对路径
        'train_list_file_name': 'train.list',  # 训练集相对路径q
        'val_list_file_name': 'val.list',  # 验证集相对路径
        'train_full_list_file_name': 'train_full.list',  # 训练集绝对路径
        'val_full_list_file_name': 'val_full.list',  # 验证集绝对路径
        'ver_list_file_name': 'test.list',  # 人脸验证使用, 两两分配
        'out_path': '/home/yuange/dataset/AR/AR-pre'  # list文件保存位置
    },
    'MegaFace': {
        'root_path': '/GPUFS/sysu_zhenghch_1/yuange/datasets/MegaFace',  # 根路径
        'img_folder_name': 'FlickrFinal2/',  # 图片集所在文件夹
        'list_file_name': 'relative.list',  # 相对路径列表
        'full_list_file_name': 'full.list',  # 绝对路径
        'train_list_file_name': 'distractor.list',  # 训练集相对路径q
        'val_list_file_name': 'probe.list',  # 验证集相对路径
        'train_full_list_file_name': 'distractor_full.list',  # 训练集绝对路径
        'val_full_list_file_name': 'probe_full.list',  # 验证集绝对路径
        'ver_list_file_name': 'test.list',  # 人脸验证使用, 两两分配
        'out_path': '/GPUFS/sysu_zhenghch_1/yuange/datasets/MegaFace'  # list文件保存位置
    }
}


def main():
    """ Choose the dataset from:
        - PKU
        - Mega
        - AR-pre
    """
    dataset_name = 'PKU'

    start_generate_list(dataset_name)


def get_filelist_ar(root, dir,
                    list_file, full_file,
                    train_file, val_file,
                    train_full_file, val_full_file,
                    ver_file,
                    ):
    """
    从图片文件夹根目录读取文件，把列表存储在 list_file 中\n
    :param root: 数据集根目录
    :param dir: 图片文件夹
    :param list_file: 存储列表文件
    :param full_file: 图片绝对路径列表
    :param train_file: 训练集列表
    :param val_file: 验证集列表
    :param train_full_file: 训练集绝对路径列表
    :param val_full_file: 验证集绝对路径列表
    :param ver_file: 人脸验证组合
    :return: None
    """
    cnt_label = 0
    cnt_train, cnt_val = 0, 0
    all_ids = []
    id2img = {}

    # 遍历文件夹
    for repeat in range(10):
        for img in tqdm(os.listdir(dir)):

            msk = img[:-4] + '_all_objects.png'
            label = int(img[2:5])
            offset = 0 if img[0] == 'm' else 50
            label += offset

            # 随机分配训练集与验证集
            num = random.randint(1, 100)

            line = dir + img + ' '
            line = line + str(label) + ' '
            line = line + root + '/' + 'AR-pre/mask/' + msk + '\n'

            list_file.write(line)

            if num <= 70:  # 70%设为train
                train_file.write(line)
                train_full_file.write(line)
                cnt_train += 1
            else:
                val_file.write(line)
                val_full_file.write(line)
                cnt_val += 1

            cnt_label += 1

    print("AR list generated successfully! Total classes = %d, "
          "train cnt = %d, val cnt = %d. \n" % (cnt_label, cnt_train, cnt_val))


def get_filelist_mega(root, dir,
                      list_file, full_file,
                      train_file, val_file,
                      train_full_file, val_full_file,
                      ver_file,
                      ):
    """
    从图片文件夹根目录读取文件，把列表存储在 list_file 中\n
    :param root: 数据集根目录
    :param dir: 图片文件夹
    :param list_file: 存储列表文件
    :param full_file: 图片绝对路径列表
    :param train_file: 训练集列表
    :param val_file: 验证集列表
    :param train_full_file: 训练集绝对路径列表
    :param val_full_file: 验证集绝对路径列表
    :param ver_file: 人脸验证组合
    :return: None
    """
    cnt_label = 0
    cnt_train, cnt_val = 0, 0
    all_ids = []
    id2img = {}

    distractor_path = os.path.join(root, 'megaface_images')  # 001a01.jpg [label]
    probe_path = os.path.join(root, 'facescrub_images')  # Adam01.jpg [label1] Adam02.jpg [label2]

    # [1/2]: For distractor, saved in train_file
    for root, dirs, files in tqdm(os.walk(distractor_path, topdown=False)):

        for img in files:

            if '.json' in img:  # skip .json
                continue

            if cnt_train >= 1000000:  # total distractors is 1000000
                continue

            line = os.path.join(root, img) + ' ' + str(9999) + '\n'

            list_file.write(line)

            train_file.write(line)
            train_full_file.write(line)
            cnt_train += 1

    # [2/2]: For probe, saved in val_file
    for identity in tqdm(os.listdir(probe_path)):

        cur_path = os.path.join(probe_path, identity)

        for img1 in os.listdir(cur_path):

            for img2 in os.listdir(cur_path):

                if img1 == img2:
                    continue

                line = os.path.join(cur_path, img1) + ' ' + str(cnt_label) + ' '
                line = line + os.path.join(cur_path, img2) + ' ' + str(cnt_label) + '\n'

                list_file.write(line)

                val_file.write(line)
                val_full_file.write(line)
                cnt_val += 1

        cnt_label += 1

    print("MegaFace list generated successfully! Total classes = %d, "
          "distractor cnt = %d, probe cnt = %d. \n" % (cnt_label, cnt_train, cnt_val))


def get_filelist_pku(root, dir,
                     list_file, full_file,
                     train_file, val_file,
                     train_full_file, val_full_file,
                     ver_file,
                     img_folder_name
                     ):
    """
    从图片文件夹根目录读取文件，把列表存储在 list_file 中\n
    :param root: 数据集根目录
    :param dir: 图片文件夹
    :param list_file: 存储列表文件
    :param full_file: 图片绝对路径列表
    :param train_file: 训练集列表
    :param val_file: 验证集列表
    :param train_full_file: 训练集绝对路径列表
    :param val_full_file: 验证集绝对路径列表
    :param ver_file: 人脸验证组合
    :return: None
    """
    divisor = 60
    cnt_ver_pair = 6000 * 4 // divisor
    cnt_pos, cnt_neg = 0, 0
    cur_ver = cnt_pos + cnt_neg

    pos_suffix = ['_0.png', '_0.png', '_0.png', '_0.png', '_0.png', '_0.png', '_0.png', '_0.png']  # position
    pos_len = len(pos_suffix)

    full_suffix = '_0'
    mask_suffix = '_1'

    # 1/2, same identity
    identity = 0
    while cur_ver < cnt_ver_pair // 2:
        cur_ver = cnt_neg + cnt_pos
        if cur_ver < cnt_ver_pair // 2:  # positive pair
            # 0: full
            # 1: mask
            op1 = random.randint(0, 1)
            op2 = random.randint(0, 1)

            face1_path = str(identity) + full_suffix if op1 == 0 else str(identity) + mask_suffix
            face2_path = str(identity) + full_suffix if op2 == 0 else str(identity) + mask_suffix

            face1_pos = pos_suffix[random.randint(0, pos_len - 1)]
            for j in range(10):
                face1 = os.path.join(face1_path, str(j) + face1_pos)
                if os.path.exists(os.path.join(dir, face1)):
                    break

            face2_pos = pos_suffix[random.randint(0, pos_len - 1)]
            for k in range(10):
                face2 = os.path.join(face2_path, str(k) + face2_pos)
                if os.path.exists(os.path.join(dir, face2)) and face2 != face1:
                    break

            if j != 9 and k != 9:
                ver_file.write(img_folder_name + '/' + face1 + '\n')
                ver_file.write(img_folder_name + '/' + face2 + '\n')
                cnt_pos += 1
        else:
            break

        identity += 1
        identity = (identity + 1018) % 1018

    # 2/2, diff identity
    identity = 0
    while cur_ver < cnt_ver_pair:
        cur_ver = cnt_neg + cnt_pos
        if cur_ver >= cnt_ver_pair // 2 and cur_ver < cnt_ver_pair:  # negative pair
            # 0: full
            # 1: mask
            op1 = random.randint(0, 1)
            op2 = random.randint(0, 1)

            face1_path = str(identity) + full_suffix if op1 == 0 else str(identity) + mask_suffix

            face1_pos = pos_suffix[random.randint(0, pos_len - 1)]
            for j in range(10):
                face1 = os.path.join(face1_path, str(j) + face1_pos)
                if os.path.exists(os.path.join(dir, face1)):
                    break

            identity += 1
            face2_path = str(identity) + full_suffix if op2 == 0 else str(identity) + mask_suffix

            face2_pos = pos_suffix[random.randint(0, pos_len - 1)]
            for k in range(10):
                face2 = os.path.join(face2_path, str(k) + face2_pos)
                if os.path.exists(os.path.join(dir, face2)):
                    break

            if j != 9 and k != 9:
                ver_file.write(img_folder_name + '/' + face1 + '\n')
                ver_file.write(img_folder_name + '/' + face2 + '\n')
                cnt_neg += 1
                identity += random.randint(0, 3)
        else:
            break

        identity = (identity + 1018) % 1018

    print("PKU ver list generated successfully! Ver cnt = %d. Pos cnt = %d, Neg cnt = %d\n"
          % (cnt_ver_pair, cnt_pos, cnt_neg))


def start_generate_list(task_name: str):
    functions = {
        'MegaFace': get_filelist_mega,
        'AR-pre': get_filelist_ar,
        'PKU': get_filelist_pku,
    }

    """ Get config and corresponding function """
    cur = config[task_name]
    func = functions[task_name]

    root_path = cur['root_path']  # 数据集根目录
    img_folder_name = cur['img_folder_name']  # 图片集所在文件夹

    list_file_name = cur['list_file_name']  # 相对路径列表
    full_list_file_name = cur['full_list_file_name']  # 绝对路径

    train_list_file_name = cur['train_list_file_name']  # 训练集相对路径
    val_list_file_name = cur['val_list_file_name']  # 验证集相对路径

    train_full_list_file_name = cur['train_full_list_file_name']  # 训练集绝对路径
    val_full_list_file_name = cur['val_full_list_file_name']  # 验证集绝对路径

    ver_list_file_name = cur['ver_list_file_name']  # 人脸验证使用, 两两分配

    out_path = cur['out_path']  # list文件保存位置

    with open(osp.join(out_path, list_file_name), 'w') as list_file, \
            open(osp.join(out_path, full_list_file_name), 'w') as full_file, \
            open(osp.join(out_path, train_list_file_name), 'w') as train_file, \
            open(osp.join(out_path, val_list_file_name), 'w') as val_file, \
            open(osp.join(out_path, train_full_list_file_name), 'w') as train_full_file, \
            open(osp.join(out_path, val_full_list_file_name), 'w') as val_full_file, \
            open(osp.join(out_path, ver_list_file_name), 'w') as ver_file:
        img_folder_path = osp.join(root_path, img_folder_name)
        func(root_path, img_folder_path,
             list_file, full_file,
             train_file, val_file,
             train_full_file, val_full_file,
             ver_file,
             img_folder_name)


if __name__ == '__main__':
    main()
