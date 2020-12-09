# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

from IPython import embed


class MAML(MetaTemplate):
    def __init__(self, model_func, vox, n_views, n_points, n_way, n_support, approx=False):
        super(MAML, self).__init__(model_func, vox, n_views,
                                   n_points, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

    def forward(self, x):
        out = self.feature.forward(x)
        # embed()
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda()
        x_var = Variable(x)
        # embed()
        if self.n_views:
            x_a_i = x_var[:, :self.n_support, :, :, :, :].contiguous().view(
                self.n_way * self.n_support * self.n_views, *x.size()[3:])  # support data
            x_b_i = x_var[:, self.n_support:, :, :, :, :].contiguous().view(
                self.n_way * self.n_query * self.n_views, *x.size()[3:])  # query data
        elif self.n_points:
            x_a_i = x_var[:, :self.n_support, :, :].contiguous().view(
                self.n_way * self.n_support, *x.size()[2:])  # support data
            x_b_i = x_var[:, self.n_support:, :, :].contiguous().view(
                self.n_way * self.n_query, *x.size()[2:])  # query data
            x_a_i = x_a_i.transpose(2, 1)
            x_b_i = x_b_i.transpose(2, 1)
        elif self.vox:
            # embed()
            x_a_i = x_var[:, :self.n_support, :, :, :, :].contiguous().view(
                self.n_way * self.n_support, *x.size()[2:])  # support data
            x_b_i = x_var[:, self.n_support:, :, :, :, :].contiguous().view(
                self.n_way * self.n_query, *x.size()[2:])  # query data
        else:
            # embed()
            x_a_i = x_var[:, :self.n_support, :, :, :].contiguous().view(
                self.n_way * self.n_support, *x.size()[2:])  # support data
            x_b_i = x_var[:, self.n_support:, :, :, :].contiguous().view(
                self.n_way * self.n_query, *x.size()[2:])  # query data
        y_a_i = Variable(torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data

        # the first gradient calcuated in line 45 is based on original weight
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            # embed()
            set_loss = self.loss_fn(scores, y_a_i)
            # build full graph support gradient of gradient
            grad = torch.autograd.grad(
                set_loss, fast_parameters, create_graph=True)
            # grad = torch.autograd.grad(
            #     set_loss, fast_parameters, create_graph=True, allow_unused=True)
            if self.approx:
                # do not calculate gradient of gradient if using first order approximation
                grad = [g.detach() for g in grad]
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # if grad[k] == None:
                #     embed()
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * \
                        grad[k]  # create weight.fast
                else:
                    # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    weight.fast = weight.fast - self.train_lr * grad[k]
                # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                fast_parameters.append(weight.fast)

        scores = self.forward(x_b_i)
        return scores

    # overwrite parrent function
    def set_forward_adaptation(self, x, is_feature=False):
        raise ValueError(
            'MAML performs further adapation simply by increasing task_upate_num')

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        y_b_i = Variable(torch.from_numpy(
            np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)

        return loss

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, _) in enumerate(train_loader):
            if self.vox:
                x = x.float()
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"

            loss = self.set_forward_loss(x)
            # avg_loss = avg_loss + loss.data[0]
            avg_loss = avg_loss + loss.data.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        i, len(train_loader), avg_loss / float(i + 1)))

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if self.vox:
                x = x.float()
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
