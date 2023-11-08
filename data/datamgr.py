# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[
                                      0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter',
                              'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, vox, n_views, n_points, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.n_views = n_views
        self.n_points = n_points
        self.vox = vox

    # parameters that would change on train/val set
    def get_data_loader(self, data_file, aug):
        print("data_file",data_file)
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform,
                                self.vox, self.n_views, self.n_points)
        data_loader_params = dict(
            batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, vox, n_views, n_points, n_way, n_support, n_query, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query # 5+5=10
        self.n_eposide = n_eposide
        self.n_views = n_views
        self.n_points = n_points
        self.vox = vox

        self.trans_loader = TransformLoader(image_size)

    # parameters that would change on train/val set
    def get_data_loader(self, data_file, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        # 載入train dataset資料 base.json
        dataset = SetDataset(data_file, self.batch_size,
                             transform, self.vox, self.n_views, self.n_points)
        sampler = EpisodicBatchSampler(
            len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(
            batch_sampler=sampler, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, **data_loader_params)
        return data_loader
    
'''
这段代码是用于数据加载和数据增强的工具，通常用于深度学习的图像数据集。以下是对代码的解析：

1. `TransformLoader` 类：用于创建图像变换的实例。它接受以下参数：
   - `image_size`：图像的目标大小。
   - `normalize_param`：图像标准化参数，包括均值和标准差。
   - `jitter_param`：图像扭曲参数，包括亮度、对比度和颜色的扭曲程度。

   `parse_transform` 方法根据变换类型返回相应的图像变换方法。支持的变换类型包括 `ImageJitter`、`RandomSizedCrop`、`CenterCrop`、`Resize`、`Normalize` 等。

   `get_composed_transform` 方法返回一个组合了多个图像变换的复合变换，根据传入的 `aug` 参数来决定是否应用数据增强。

2. `DataManager` 类：一个抽象类，用于获取数据加载器。

3. `SimpleDataManager` 类：继承自 `DataManager`，用于加载普通数据集。接受以下参数：
   - `image_size`：图像的目标大小。
   - `vox`：是否使用 3D 图像。
   - `n_views`：视图数量。
   - `n_points`：点的数量。
   - `batch_size`：批处理大小。

   `get_data_loader` 方法根据数据文件和是否进行数据增强返回数据加载器。数据加载器用于加载图像数据集，可以在训练和验证集之间切换。

4. `SetDataManager` 类：继承自 `DataManager`，用于加载集合数据集，通常用于元学习任务。接受以下参数：
   - `image_size`：图像的目标大小。
   - `vox`：是否使用 3D 图像。
   - `n_views`：视图数量。
   - `n_points`：点的数量。
   - `n_way`：元学习中的类别数。
   - `n_support`：每个类别的支持集大小。
   - `n_query`：每个类别的查询集大小。
   - `n_episode`：元学习的回合数。

   `get_data_loader` 方法根据数据文件和是否进行数据增强返回数据加载器。数据加载器用于加载元学习任务中的集合数据集，其中包含支持集和查询集，并根据不同的采样策略生成元学习任务。

这些类和方法用于配置数据加载和数据增强，以准备深度学习任务所需的训练数据。根据具体的应用场景和数据集，可以使用不同的参数来配置数据加载器。
'''
