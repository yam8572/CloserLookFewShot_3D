# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

import sys
sys.path.insert(0, 'utils/')
# from ..utils import binvox_rw
import binvox_rw

def identity(x): return x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, vox, n_views, n_points):

        self.n_views = n_views
        self.n_points = n_points
        self.vox = vox

        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if self.n_views:
            stride = int(12 / self.n_views)
            for x, y in zip(self.meta['image_names'][::stride], self.meta['image_labels'][::stride]):
                self.sub_meta[y].append(x)
            shuffle = False
        else:
            for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
                self.sub_meta[y].append(x)
            shuffle = True

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        for cl in self.cl_list:
            if self.n_views:
                # shuffle
                rand_idx = np.random.permutation(
                    int(len(self.sub_meta[cl]) / self.n_views))
                sub_meta_new = []
                for i in range(len(rand_idx)):
                    sub_meta_new.extend(
                        self.sub_meta[cl][rand_idx[i] * self.n_views:(rand_idx[i] + 1) * self.n_views])
                self.sub_meta[cl] = sub_meta_new
            sub_dataset = SubDataset(
                self.sub_meta[cl], cl, transform=transform, vox=self.vox, n_views=self.n_views, n_points=self.n_points)
            self.sub_dataloader.append(torch.utils.data.DataLoader(
                sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, vox=False, n_views=None, n_points=None):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.n_views = n_views
        self.n_points = n_points
        self.vox = vox

    def __getitem__(self, i):
        #print( '%d -%d' %(self.cl,i))
        target = self.target_transform(self.cl)
        if self.n_views:
            imgs = []
            for j in range(self.n_views):
                image_path = os.path.join(self.sub_meta[i * self.n_views + j])
                img = Image.open(image_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
            return torch.stack(imgs), target
        elif self.n_points:
            image_path = os.path.join(self.sub_meta[i])
            point_set = np.loadtxt(
                image_path, delimiter=',').astype(np.float32)
            point_set = point_set[0:self.n_points, :]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            return point_set, target
        elif self.vox:
            image_path = os.path.join(self.sub_meta[i])
            with open(image_path, 'rb') as file:
                data = np.int32(binvox_rw.read_as_3d_array(file).data)
                data = data[np.newaxis, :]
            return data, target
        else:
            image_path = os.path.join(self.sub_meta[i])
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
