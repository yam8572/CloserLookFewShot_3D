#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # weight.fast (fast weight) is the temporaily adapted weight
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv3d_fw(nn.Conv3d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv3d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv3d(x, self.weight.fast, None,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv3d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv3d(x, self.weight.fast, self.bias.fast,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv3d_fw, self).forward(x)

        return out


class VoxNet(nn.Module):
    maml = False

    def __init__(self, n_classes=10, input_shape=(30, 30, 30)):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        if self.maml:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', Conv3d_fw(in_channels=1,
                                       out_channels=32, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', Conv3d_fw(
                    in_channels=32, out_channels=32, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3))
            ]))
        else:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                             out_channels=32, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', torch.nn.Conv3d(
                    in_channels=32, out_channels=32, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3))
            ]))
        x = self.feat(torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        if self.maml:
            self.mlp = torch.nn.Sequential(OrderedDict([
                ('fc1', Linear_fw(dim_feat, 128)),
                ('relu1', torch.nn.ReLU()),
                # ('drop3', torch.nn.Dropout(p=0.4)),
                # ('fc2', torch.nn.Linear(128, self.n_classes))
            ]))
        else:
            self.mlp = torch.nn.Sequential(OrderedDict([
                ('fc1', torch.nn.Linear(dim_feat, 128)),
                ('relu1', torch.nn.ReLU()),
                # ('drop3', torch.nn.Dropout(p=0.4)),
                # ('fc2', torch.nn.Linear(128, self.n_classes))
            ]))

        self.final_feat_dim = 128

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    voxnet = VoxNet()
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)
