# Few-shot Learning for 3D Shape Classification
In this project, we perform a detailed empirical study of 3d shape classification under few-shot setting using common 3D architectures, datasets and few-shot techniques.

## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/)
 - json

## Getting started
### ModelNet40 - Voxels
* Change directory to `./filelists/ModelNet40_voxels`
* download and unzip ModelNet40.zip from (https://modelnet.cs.princeton.edu/)
* use utils/binvox_convert.py to convert this data to voxel format

### ModelNet40 - Multi-view Images
* Change directory to `./filelists/ModelNet40_views`
* download and unzip rendered images from (https://github.com/jongchyisu/mvcnn_pytorch)

### ModelNet40 - Point Clouds
* Change directory to `./filelists/ModelNet40_points`
* download and unzip data from (https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

### Self-defined setting
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
* See utils/create_json.ipynb on how to generate these files for ModelNet40 dataset. Update data_dir['DATASETNAME'] in configs.py.  

## Train
In general, run
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

Specifically, below are some examples to run experiments on ModelNet40 dataset using different architectures and few-shot techniques:
* VoxNet: `python ./train.py --dataset modelnet40_voxels --method protonet --voxelized`
* MVCNN: `python ./train.py --dataset modelnet40_views --model Conv4 --method maml --num_views 12`
* PointNet: `python ./train.py --dataset modelnet40_points --method baseline --num_points 1024`
Similarly, you can use other 3D datasets. Commands below follow this example, and please refer to io_utils.py for additional options.

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
Run
```python ./save_features.py --dataset modelnet_views --model Conv4 --method protonet --train_aug```

## Test
Run
```python ./test.py --dataset modelnet_views --model Conv4 --method protonet --train_aug```

## Results
* The test results will be recorded in `./record/results.txt`

## References
We have modified and built upon the following publicly available code:
* Few-shot Framework and Methods: 
https://github.com/wyharveychen/CloserLookFewShot 
* VoxNet: 
https://github.com/dimatura/voxnet; https://github.com/MonteYang/VoxNet.pytorch; https://github.com/Ryanglambert/3d_model_retriever
* MVCNN: 
https://github.com/jongchyisu/mvcnn_pytorch; https://github.com/RBirkeland/MVCNN-PyTorch
* PointNet: 
https://github.com/charlesq34/pointnet; https://github.com/fxia22/pointnet.pytorch; https://github.com/yanx27/Pointnet_Pointnet2_pytorch
