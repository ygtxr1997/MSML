# Dataset Preparation

## 1. Introduction

We use MXNet.recordio as the dataset format due to its convenience. 

Occlusion augmentation method is very important for occlusion-invariant CV tasks.
In our MSML, we use 3 types of occlusions (random connected geometric shapes, 
realistic objects collected from the web, 
and synthetic face masks rendered through 3D scheme (Zhu et al. 2016, 2015)).

In practice, we divide the augmentation methods into 2 categories: 
online augmentation and offline augmentation.
The default pytorch augmentation (torchvision.transform) is online.
So the 'random connected geometric shapes' and 
'realistic objects collected from the web' are both online augmentation,
which means the occlusions are added to faces during training.

However, the 'mask augmentation through 3D scheme' is very time-consuming.
We have to render the face masks before training and save them as file.
This is why we call it 'offline' augmentation.

## 2. Original Face Dataset

1. Download CASIA WebFace Dataset and MS1M-RetinaFace Dataset from 
[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

2. Put 'train.rec' and 'train.idx' to '/your/path/to/CASIA' and '/your/path/to/MS1M'.

3. Finished. In 'MSML/datasets/load_dataset.py', 
the *FaceByRandOccMask* class will read 'train.rec' and 'train.idx' 
and add random occlusions during training (online).

## 3. Masked Face Dataset

1. Read the instruction of 
'[mask_renderer.py](https://github.com/deepinsight/insightface/tree/master/recognition/_tools_)'.

2. Based on 'mask_renderer.py', we get codes 
'MSML/datasets/3d_tools/cvt_xxx_masked.py'.

3. In the '\_\_main\_\_' function of 'cvt_xxx_masked.py', 
you should change *root_dir* to '/your/path/to/some_dataset'.
This path should contain 'train.rec' and 'train.idx' as mentioned above.

4. *write_record* will start the 3D mask rendering process
and generate 'mask_out.rec', 'mask_out.idx', 'mask.rec' and 'mask.idx'.
(Make sure your old version of 'mask_out.rec/idx' and 'mask.rec/idx'
have been backed up.)

5. *read_record* will check if 'mask_out.rec/idx' and 'mask.rec/idx'
are generated correctly. 

6. Easily typing '$ python cvt_xxx.py' will start 3D rendering process.

7. The generated 'mask_out.rec/idx' and 'mask.rec/idx' will be directly 
read by 'MSML/datasets/load_dataset.py' 
to perform 'masked face augmentation' without much CPU time (offline).
