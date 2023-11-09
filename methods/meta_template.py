import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

from pointnet import PointNetEncoder
from voxnet import VoxNet
# import provider
from IPython import embed

# parent 父類
class MetaTemplate(nn.Module):
    def __init__(self, model_func, vox, n_views, n_points, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.n_views = n_views
        self.n_points = n_points
        self.vox = vox
        if self.n_points:
            self.feature = PointNetEncoder(
                global_feat=True, feature_transform=True, channel=6)
        elif self.vox:
            self.feature = VoxNet()
        else:
            self.feature = model_func(n_views=self.n_views)
        self.feat_dim = self.feature.final_feat_dim
        # some methods allow different_way classification during training and test
        self.change_way = change_way

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            if self.n_views:
                x = x.contiguous().view(self.n_way * (self.n_support + self.n_query)
                                        * self.n_views, *x.size()[3:])
            else:
                x = x.contiguous().view(
                    self.n_way * (self.n_support + self.n_query), *x.size()[2:])
                if self.n_points:
                    x = x.transpose(2, 1)

            z_all = self.feature.forward(x)
            # embed()
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            # embed()
            if self.vox:
                x = x.float()
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            # embed()
            # avg_loss = avg_loss + loss.data[0]
            avg_loss = avg_loss + loss.data.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch,
                                                                        i, len(train_loader), avg_loss / float(i + 1)))

    def test_loop(self, test_loader, record=None):
        correct = 0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if self.vox:
                x = x.float()
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %
              (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

    # further adaptation, default is fixing feature and train a new softmax clasifier
    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Feature is fixed in further adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(
            np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(
        ), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(
                    rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        scores = linear_clf(z_query)
        return scores

'''
繼承用法範例
# 父类（基类）
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

# 子类（派生类），继承自 Animal
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# 创建子类的实例
dog = Dog("Buddy")
cat = Cat("Whiskers")

# 调用子类的方法
print(dog.speak())  # 输出：Buddy says Woof!
print(cat.speak())  # 输出：Whiskers says Meow!

'''

"""
@abstractmethod 是 Python 中的一個裝飾器，用於定義抽象方法。 抽象方法是在抽象類別中聲明的方法，但沒有提供具體的實作。 抽象類別本身不能被實例化，而是用作其他類別的基類，要求子類別提供抽象方法的具體實作。
使用 @abstractmethod 裝飾器可以告訴 Python 解釋器，這是一個抽象方法，必須在子類別中實作。

from abc import ABC, abstractmethod

class MyAbstractClass(ABC):  # 继承 ABC（Abstract Base Class）
    @abstractmethod
    def my_abstract_method(self):
        pass

class MyConcreteClass(MyAbstractClass):
    def my_abstract_method(self):
        print("Implemented abstract method")

# 试图实例化抽象类会引发错误
# my_obj = MyAbstractClass()  # 会引发 TypeError

my_obj = MyConcreteClass()
my_obj.my_abstract_method()  # 输出：Implemented abstract method


"""