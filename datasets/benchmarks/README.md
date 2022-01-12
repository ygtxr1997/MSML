# Benchmarks for evaluating

## Part 1. RetinaFace

### 1.1 Make

```shell script
cd benchmarks/RetinaFace
make clean
make all
```

### 1.2 Download pretrained models
Please refer to [RetinaFace](
https://github.com/deepinsight/insightface/tree/master/detection/retinaface).

### 1.3 Test for installation

```shell script
cd benchmarks/RetinaFace
python test.py
```

### 1.4 Detect and align faces

```shell script
cd benchmarks/RetinaFace
# after setting /your/output/dataset/folder in 'iterate_pku.py'
python iterate_pku.py
```

## Part 2. Generate Verification List File

```shell script
cd benchmarks
python get_list.py
```

For PKU-Masked-Face dataset, we provide 'ver400.list', 'ver6000.list', 
and 'ver24000.list' in 'PKU'. 
You can also copy these list files into /your/output/dataset/folder.

