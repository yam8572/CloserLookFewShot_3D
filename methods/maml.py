# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

from IPython import embed

# 子類: 繼承MetaTemplate >> 看 MetaTemplate.py
class MAML(MetaTemplate):
    def __init__(self, model_func, vox, n_views, n_points, n_way, n_support, approx=False):
        super(MAML, self).__init__(model_func, vox, n_views,
                                   n_points, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)

        # self.n_task = 4
        self.n_task = 2

        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx  # first order approx.

    def forward(self, x):
        out = self.feature.forward(x)
        # embed()
        # backbone.py backbone.Linear_fw..forward(out)
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda()
        # x_var = Variable(x) 就寫法
        # 多為數組(張量)已經預設啟用了自動微分。 如果你需要計算梯度，只要設定 requires_grad=True
        x_var = torch.Tensor(x)
        x.requires_grad = True  # 設置需要梯度跟踪
        # print(f"x.shape={x.shape}") # torch.Size([5, 21, 1024, 6]
        # print(f"x_var.shape={x_var.shape}")# torch.Size([5, 21, 1024, 6]
        # embed()
        """
        .contiguous():當你需要重新整形（reshape）一個張量時，有時會出現錯誤，這是因為 PyTorch 張量在記憶體中不一定是連續儲存的。 
        為了解決這個問題，你可以使用 `.contiguous()` 方法，它會建立一個連續儲存的新張量，以便進行後續操作。
        .view():reshape tensor size
        """
        if self.n_views:
            x_a_i = x_var[:, :self.n_support, :, :, :, :].contiguous().view(
                self.n_way * self.n_support * self.n_views, *x.size()[3:])  # support data
            x_b_i = x_var[:, self.n_support:, :, :, :, :].contiguous().view(
                self.n_way * self.n_query * self.n_views, *x.size()[3:])  # query data
        elif self.n_points: # x.size()[2:]第三維度大小
            x_a_i = x_var[:, :self.n_support, :, :].contiguous().view(
                self.n_way * self.n_support, *x.size()[2:])  # support data
            # print("x_a_i.shape=",x_a_i.shape) # torch.Size([5*5=25, 1024, 6])
            x_b_i = x_var[:, self.n_support:, :, :].contiguous().view(
                self.n_way * self.n_query, *x.size()[2:])  # query data
            # print("x_b_i.shape",x_b_i.shape)# torch.Size([5*16=80, 1024, 6])
            
            x_a_i = x_a_i.transpose(2, 1)# torch.Size([25, 6, 1024])
            x_b_i = x_b_i.transpose(2, 1) # torch.Size([80, 6, 1024])

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
        # tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        y_a_i = Variable(torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data

        # the first gradient calcuated in line 58 is based on original weight
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            # embed()
            set_loss = self.loss_fn(scores, y_a_i)
            # build full graph support gradient of gradient
            # torch.autograd.grad時設定allow_unused=True
            # PyTorch允許未被使用的張量。 但要小心，確保這是你所期望的行為，因為允許未使用的張量可能導致不正確的梯度計算。
            grad = torch.autograd.grad(
                set_loss, fast_parameters, create_graph=True, allow_unused=True)
            # grad = torch.autograd.grad(
            #     set_loss, fast_parameters, create_graph=True)
            if self.approx:
                # do not calculate gradient of gradient if using first order approximation
                # .detach()` 方法用於建立一個新的張量，其值與原始張量相同，但不再與計算圖相關聯。 
                # 這意味著對新的張量進行操作不會影響計算圖中的梯度計算。 原始張量仍保留在計算圖中，以便進行後續的反向傳播。
                # 將梯度從計算圖中分離，以便進一步的計算不會影響梯度。
                # grad = [g.detach() for g in grad]
                grad = [g.detach() if g is not None else None for g in grad]

            fast_parameters = []
            # print(f"self.parameters()={self.parameters()}")
            for k, weight in enumerate(self.parameters()):
                print(f"k={k} weight={weight}")
                # if grad[k] == None:
                #     embed()
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * \
                        grad[k]  # create weight.fast
                else:
                    # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                    if grad[k] is not None:
                        weight.fast = weight.fast - self.train_lr * grad[k]
                    # weight.fast = weight.fast - self.train_lr * grad[k]
                # gradients calculated in line 58 are based on newest fast weight, but the graph will retain the link to old weight.fasts
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
            # if self.vox:
            #     x = x.float()
            # x.size(1) = 21=16+5
            # self.n_query=16,self.n_support=5
            # self.n_way=5,x.size(0)=5
            # print(f"i={i} (x,_)={x,_}") 
            print(f"i={i}") 
            print("x.size(1)",x.size(1))# 21
            self.n_query = x.size(1) - self.n_support
            print(f"self.n_query={self.n_query},self.n_support={self.n_support}")
            print(f"self.n_way={self.n_way},x.size(0)={x.size(0)}")
            assert self.n_way == x.size(0), "MAML do not support way change"

            # loss = self.set_forward_loss(x)
            # # avg_loss = avg_loss + loss.data[0]
            # avg_loss = avg_loss + loss.data.item()
            # loss_all.append(loss)

            # task_count += 1

            # if task_count == self.n_task:  # MAML update several tasks at one time
            #     loss_q = torch.stack(loss_all).sum(0)
            #     loss_q.backward()

            #     optimizer.step()
            #     task_count = 0
            #     loss_all = []
            # optimizer.zero_grad()
            # if i % print_freq == 0:
            #     print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
            #                                                             i, len(train_loader), avg_loss / float(i + 1)))

    def test_loop(self, test_loader, return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            print(f"i={i}") 
            print("x.size()[2:]",x.size()[2:]) # torch.Size([1024, 6])
            if self.vox:
                x = x.float()
            self.n_query = x.size(1) - self.n_support
            print("x.size(1)",x.size(1))# 21
            print("x.size(0)",x.size(0))# 5
        
            assert self.n_way == x.size(0), "MAML do not support way change"

            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        print(f"i={i} (x,_)={x,_}") 

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

"""
對訓練資料的敏感度: MAML 的表現高度依賴訓練集，需要足夠的任務來進行元訓練。 如果元訓練資料不足或不具代表性，MAML 可能無法產生良好的泛化效能。

計算和記憶體開銷: MAML 在訓練和測試時需要大量的運算資源和記憶體。 元訓練階段需要多次進行梯度更新，而元測試階段也需要保存多個快速權重。 這可能使 MAML 在資源受限的環境中不夠實際。

梯度更新次數的選擇: MAML 中的一個關鍵參數是元訓練中的梯度更新次數。 選擇合適的次數可能需要實驗和調整，過多或過少的次數都可能影響表現。

可能不穩定: MAML 可能會受到任務之間相似性和訓練任務數量的影響，導致不穩定的表現。 這種不穩定性可能使 MAML 難以在一些實際應用中獲得一致的好結果。

需要大量的訓練任務: MAML 需要大量的不同任務進行元訓練，這可能在某些領域中限制了它的應用。 獲取足夠的任務數據可能是具有挑戰性的。

總的來說，MAML 是一種強大的元學習方法，但它也有一些限制。 在應用 MAML 之前，需要仔細考慮這些缺點，並根據特定問題的需求和資源情況來決定是否選擇 MAML 或其他方法。
"""