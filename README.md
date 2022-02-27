# MSML: Enhancing Occlusion-Robustness by Multi-Scale Segmentation-Based Mask Learning for Face Recognition (AAAI-2022)

## Introduction

This is an official implementation of AAAI 2022 paper 
"MSML: Enhancing Occlusion-Robustness by Multi-Scale Segmentation-Based Mask Learning for Face Recognition".
[https://www.aaai.org/AAAI22Papers/AAAI-50.YuanG.pdf](https://www.aaai.org/AAAI22Papers/AAAI-50.YuanG.pdf)

For occluded face recognition, 
the MSML network can effectively identify and remove the occlusions from 
feature representations at multiple levels and aggregate features from visible facial areas.

## Data preparation

The datasets include training datasets and testing datasets.

* Training: [CASIA WebFace, MS1M-RetinaFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_);

* Testing: [LFW, CFP_FP, AgeDB-30](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_);
[PKU-MASKED-FACE](https://pkuml.org/resources/pku-masked-face-dataset.html);
[AR Database](http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html);

We also use a 3D mask augmentation scheme to improve the robustness to masked faces. 
Please refer to [MSML/datasets](https://github.com/ygtxr1997/MSML/tree/main/datasets) for more details.

## Training

Before training, you should edit the config file ([MSML/config.yaml]()) as you need.
The config file ([MSML/config.yaml]()) will be first loaded.
Then some fixed settings in [MSML/config.py]() will be loaded according to the config file ([MSML/config.yaml]()).
You may need to change dataset folders (`cfg.dataset`) in [MSML/config.py]().

* Easily start training on 4 GPUs:

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 
--master_addr="127.0.0.1" --master_port=1234 train.py
```

After starting training, 
the model weights (`backbone.pth`), config file (`config.yaml`), and training log (`training.log`) 
will be generated in the output folder `{conf.output_prefix}_{conf.exp_id}`.

* Resume training from 13th epochs on 4 GPUs (13 indicates the epoch where you haven't finished):

```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 
--master_addr="127.0.0.1" --master_port=1234 train.py --resume 13
```

## Testing

* Easily test your saved model by indicating the folder:

```shell script
CUDA_VISIBLE_DEVICES=0 python test.py --network msml --weight_folder ires18_msml_2 
--dataset lfw --fill_type black --vis False
```

## Citation

Please cite our paper by:

```
@article{yuan2022msml,
    title={MSML: Enhancing Occlusion-Robustness by Multi-Scale Segmentation-Based Mask Learning for Face Recognition},
    author={Yuan, Ge and Zheng, Huicheng and Dong, Jiayu},
    journal={AAAI Conference on Artificial Intelligence},
    year={2022}
}
```

## Acknowledgements

This repo is mainly inspired by [InsightFace](https://github.com/deepinsight/insightface).
We thank the authors a lot for their valuable efforts.