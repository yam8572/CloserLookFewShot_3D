import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
import pointnet
import voxnet
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
# from methods.matchingnet import MatchingNet
# from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        # model are called by reference, no need to return
        model.train_loop(epoch, base_loader, optimizer)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        if acc > max_acc:  # for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir,
                                   '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    print("params",params)

    # if params.dataset == 'cross':
    #     base_file = configs.data_dir['miniImagenet'] + 'all.json'
    #     val_file = configs.data_dir['CUB'] + 'val.json'
    # elif params.dataset == 'cross_char':
    #     base_file = configs.data_dir['omniglot'] + 'noLatin.json'
    #     val_file = configs.data_dir['emnist'] + 'val.json'
    # else:
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'

    print("base_file",base_file)
    print("val_file",val_file)


    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    print("image_size",image_size)

    # # if params.dataset in ['omniglot', 'cross_char']:
    # #     assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
    # #     params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
                params.stop_epoch = 200
            # elif params.dataset in ['miniImagenet', 'cross']:
            #     params.stop_epoch = 400
            else:
                # params.stop_epoch = 400  # default
                params.stop_epoch = 10
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
                print("stop_epoch",params.stop_epoch)
            else:
                params.stop_epoch = 600  # default

    print("stop_epoch",params.stop_epoch)

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(
            image_size, vox=params.voxelized, n_views=params.num_views, n_points=params.num_points, batch_size=16)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)
        val_datamgr = SimpleDataManager(
            image_size, vox=params.voxelized, n_views=params.num_views, n_points=params.num_points, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        # if params.dataset == 'omniglot':
        #     assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        # if params.dataset == 'cross_char':
        #     assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(model_dict[params.model], params.voxelized,
                                  params.num_views, params.num_points, params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(
                model_dict[params.model], params.num_classes, loss_type='dist')

    elif params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way)) # max(1,16 *5/5) = 16
        print("n_query",n_query)
        # n_query = 8

        train_few_shot_params = dict(
            n_way=params.train_n_way, n_support=params.n_shot)

        # 載入train dataset資料 base.json
        base_datamgr = SetDataManager(
            image_size, vox=params.voxelized, n_views=params.num_views, n_points=params.num_points, n_query=n_query, **train_few_shot_params)
        # print("base_datamgr=",base_datamgr)
        base_loader = base_datamgr.get_data_loader(
            base_file, aug=params.train_aug)
        # print("base_loader",base_loader)

        test_few_shot_params = dict(
            n_way=params.test_n_way, n_support=params.n_shot)
        
        # 載入train dataset資料 val.json
        val_datamgr = SetDataManager(
            image_size, vox=params.voxelized, n_views=params.num_views, n_points=params.num_points, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model],
                             params.voxelized, params.num_views, params.num_points, **train_few_shot_params)
        # elif params.method == 'matchingnet':
        #     model = MatchingNet(
        #         model_dict[params.model], **train_few_shot_params)
        # elif params.method in ['relationnet', 'relationnet_softmax']:
        #     if params.model == 'Conv4':
        #         feature_model = backbone.Conv4NP
        #     elif params.model == 'Conv6':
        #         feature_model = backbone.Conv6NP
        #     elif params.model == 'Conv4S':
        #         feature_model = backbone.Conv4SNP
        #     else:
        #         def feature_model(): return model_dict[params.model](
        #             flatten=False)
        #     loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

        #     model = RelationNet(
        #         feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            params.resume=True
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            pointnet.STN3d.maml = True
            pointnet.STNkd.maml = True
            pointnet.PointNetEncoder.maml = True
            voxnet.VoxNet.maml = True
            model = MAML(model_dict[params.model], params.voxelized, params.num_views, params.num_points, approx=(
                params.method == 'maml_approx'), **train_few_shot_params)
            """
            model_dict = dict(
            Conv4=backbone.Conv4,
            Conv4S=backbone.Conv4S,
            Conv6=backbone.Conv6,
            ResNet10=backbone.ResNet10,
            ResNet18=backbone.ResNet18,
            ResNet34=backbone.ResNet34,
            ResNet50=backbone.ResNet50,
            ResNet101=backbone.ResNet101)

            train_few_shot_params = dict(
            n_way=params.train_n_way, n_support=params.n_shot)

            model MAML(
            (feature): PointNetEncoder(
            (stn): STN3d(
            (conv1): Conv1d_fw(6, 64, kernel_size=(1,), stride=(1,))
            (conv2): Conv1d_fw(64, 128, kernel_size=(1,), stride=(1,))
            (conv3): Conv1d_fw(128, 1024, kernel_size=(1,), stride=(1,))
            (fc1): Linear_fw(in_features=1024, out_features=512, bias=True)
            (fc2): Linear_fw(in_features=512, out_features=256, bias=True)
            (fc3): Linear_fw(in_features=256, out_features=9, bias=True)
            (relu): ReLU()
            (bn1): BatchNorm1d_fw(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn2): BatchNorm1d_fw(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn3): BatchNorm1d_fw(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn4): BatchNorm1d_fw(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn5): BatchNorm1d_fw(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (conv1): Conv1d_fw(6, 64, kernel_size=(1,), stride=(1,))
            (conv2): Conv1d_fw(64, 128, kernel_size=(1,), stride=(1,))
            (conv3): Conv1d_fw(128, 1024, kernel_size=(1,), stride=(1,))
            (bn1): BatchNorm1d_fw(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn2): BatchNorm1d_fw(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn3): BatchNorm1d_fw(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (fstn): STNkd(
            (conv1): Conv1d_fw(64, 64, kernel_size=(1,), stride=(1,))
            (conv2): Conv1d_fw(64, 128, kernel_size=(1,), stride=(1,))
            (conv3): Conv1d_fw(128, 1024, kernel_size=(1,), stride=(1,))
            (fc1): Linear_fw(in_features=1024, out_features=512, bias=True)
            (fc2): Linear_fw(in_features=512, out_features=256, bias=True)
            (fc3): Linear_fw(in_features=256, out_features=4096, bias=True)
            (relu): ReLU()
            (bn1): BatchNorm1d_fw(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn2): BatchNorm1d_fw(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn3): BatchNorm1d_fw(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn4): BatchNorm1d_fw(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (bn5): BatchNorm1d_fw(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
        )
        (loss_fn): CrossEntropyLoss()
        (classifier): Linear_fw(in_features=1024, out_features=5, bias=True)
        )
            """
            # maml use different parameter in omniglot
            # if params.dataset in ['omniglot', 'cross_char']:
            #     model.n_task = 32
            #     model.task_update_num = 1
            #     model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (
            params.train_n_way, params.n_shot)
        print("checkpoint_dir",params.checkpoint_dir)


    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        # maml use multiple tasks in one update self.n_task = 4
        stop_epoch = params.stop_epoch * model.n_task # 400*4=1600

    # params.resume: continue from previous trained model with largest epoch
    if params.resume:
        # /home/g111056119/Documents/7111056426/CloserLookFewShot_3D//checkpoints/modelnet40_points/Conv4_maml_5way_5shot
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
            configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    print("base_loader=",base_loader)
    print("val_loader",val_loader)
    print("optimization",optimization)
    print("start_epoch",start_epoch)
    print("stop_epoch",stop_epoch)
    print("params",params)

    model = train(base_loader, val_loader, model, optimization,
                  start_epoch, stop_epoch, params)
