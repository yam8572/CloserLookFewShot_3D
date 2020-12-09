import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


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


class Conv1d_fw(nn.Conv1d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv1d_fw, self).__init__(in_channels, out_channels,
                                        kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv1d(x, self.weight.fast, None,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv1d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv1d(x, self.weight.fast, self.bias.fast,
                               stride=self.stride, padding=self.padding)
            else:
                out = super(Conv1d_fw, self).forward(x)

        return out


# used in MAML to forward input with fast weight
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features):
        super(BatchNorm1d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast,
                               self.bias.fast, training=True, momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var,
                               self.weight, self.bias, training=True, momentum=1)
        return out


class STN3d(nn.Module):
    maml = False

    def __init__(self, channel):
        super(STN3d, self).__init__()
        if self.maml:
            self.conv1 = Conv1d_fw(channel, 64, 1)
            self.conv2 = Conv1d_fw(64, 128, 1)
            self.conv3 = Conv1d_fw(128, 1024, 1)
            self.fc1 = Linear_fw(1024, 512)
            self.fc2 = Linear_fw(512, 256)
            self.fc3 = Linear_fw(256, 9)
            self.relu = nn.ReLU()

            self.bn1 = BatchNorm1d_fw(64)
            self.bn2 = BatchNorm1d_fw(128)
            self.bn3 = BatchNorm1d_fw(1024)
            self.bn4 = BatchNorm1d_fw(512)
            self.bn5 = BatchNorm1d_fw(256)
        else:
            self.conv1 = torch.nn.Conv1d(channel, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 9)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    maml = False

    def __init__(self, k=64):
        super(STNkd, self).__init__()
        if self.maml:
            self.conv1 = Conv1d_fw(k, 64, 1)
            self.conv2 = Conv1d_fw(64, 128, 1)
            self.conv3 = Conv1d_fw(128, 1024, 1)
            self.fc1 = Linear_fw(1024, 512)
            self.fc2 = Linear_fw(512, 256)
            self.fc3 = Linear_fw(256, k * k)
            self.relu = nn.ReLU()

            self.bn1 = BatchNorm1d_fw(64)
            self.bn2 = BatchNorm1d_fw(128)
            self.bn3 = BatchNorm1d_fw(1024)
            self.bn4 = BatchNorm1d_fw(512)
            self.bn5 = BatchNorm1d_fw(256)
        else:
            self.conv1 = torch.nn.Conv1d(k, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k * k)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    maml = False

    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        if self.maml:
            self.conv1 = Conv1d_fw(channel, 64, 1)
            self.conv2 = Conv1d_fw(64, 128, 1)
            self.conv3 = Conv1d_fw(128, 1024, 1)
            self.bn1 = BatchNorm1d_fw(64)
            self.bn2 = BatchNorm1d_fw(128)
            self.bn3 = BatchNorm1d_fw(1024)
        else:
            self.conv1 = torch.nn.Conv1d(channel, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.final_feat_dim = 1024

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # if self.global_feat:
        #     return x, trans, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, N)
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss
