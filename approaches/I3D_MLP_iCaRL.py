
from asyncio import create_task, current_task
import imp
import sys,time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.I3D_Backbone import I3D_backbone
from networks.MLP_Regressor import MLP_block
from scipy import stats
from utils import *
from config.I3D_MLP_iCaRL_config import *


class I3D_MLP_iCaRL(object):
    def __init__(self, dataloaders, args):
        self.dataloaders = dataloaders
        self.epochs = args.epochs
        
        self.i3d = I3D_backbone(I3D_class=400)
        self.i3d.load_pretrain(args.pretrained_i3d_weight)
        self.mlp = MLP_block(in_dim=2049, out_dim=1)

        self.lr = args.lr
        self.lr_factor = args.lr_factor
        self.weight_decay = args.weight_decay
        self.fix_bn = args.fix_bn

        self.mse = torch.nn.MSELoss()
        
        self.task_list = self.create_task_list()
        self.action_list = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        
        self.exemplar_set = []
        self.memory_size=memory_size

        self.exp_name = args.exp_name
        self.log_path = os.path.join(args.log_root, args.exp_name + '_log.txt')
        self.ckpt_root = args.ckpt_root

    def create_task_list(self):
        task_list = np.arange(6)
        np.random.shuffle(task_list)
        task_list = task_list.tolist()
        return task_list
    
    def _get_optimizer(self):
        optimizer = optim.Adam([
            {'params': self.i3d.parameters(), 'lr': self.lr * self.lr_factor},
            {'params': self.mlp.parameters()}
        ], lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        return optimizer, scheduler

    def to_cuda(self):
        self.i3d = self.i3d.cuda()
        self.mlp = self.mlp.cuda()
        self.mse = self.mse.cuda()
        torch.backends.cudnn.benchmark = True
        return

    def dp(self):
        self.i3d = nn.DataParallel(self.i3d)
        self.mlp = nn.DataParallel(self.mlp)

    def save_checkpoit(self):
        return
    
    def load_checkpoint(self):
        return

    def train(self):
        for i in range(len(self.task_list)):
            self.beforeTrain()
            self.train_current_task(i)
            self.aterTrain(i)


    def beforeTrain(self):
        self.mlp.eval()
        self.i3d.eval()
        # 获取当前任务

        # 获取训练数据

        # 增加头数
        self.mlp.train()
        self.i3d.train()
        return
    
    def train_current_task(self, task_id):
        current_task = self.task_list[task_id]
        optimizer, scheduler=self._get_optimizer()
        self.dp()
        steps = ['train', 'test']
        for epoch in range(self.epochs):
            for step in steps:
                print(step + ' step:')
                true_scores = []
                pred_scores = []
                if self.fix_bn:
                    self.i3d.apply(fix_bn)  # fix bn
                if step == 'train':
                    self.i3d.train()
                    self.mlp.train()
                    torch.set_grad_enabled(True)
                    data_loader = train_dataloader
                else:
                    self.i3d.eval()
                    self.mlp.eval()
                    torch.set_grad_enabled(False)
                    data_loader = test_dataloader
                for batch_idx, data in enumerate(data_loader):
                    true_scores.extend(data['final_score'].numpy())
                    # data preparing
                    video = data['video'].float().cuda() # N, C, T, H, W
                    label = data['final_score'].float().reshape(-1,1).cuda()
                    action_id = data['action_id'].cuda()
                    batch_size = video.shape[0]
                    # forward
                    # if num_iter == step_per_update:
                    #     num_iter = 0
                    feat = self.i3d(video)
                    pred_score = self.mlp(feat)
                    if (self.i3d_pre is not None) or (self.mlp_pre is not None):
                        feat_pre = self.i3d_pre(video)
                        pred_score_pre = self.mlp_pre(feat_pre)
                        loss_distillarion = 0.0
                        loss_rgs = 0.0
                        K = self.mlp.regressor[4].out_features
                        for k in range(K):
                            if k < K-1:
                                loss_distillarion += self.mse(pred_score_pre[:, k].unsqueeze(-1), pred_score[:, k].unsqueeze(-1))
                            for b in range(batch_size):
                                if k == action_id[b]:
                                    loss_rgs += self.mse(pred_score[b, k], label[b])
                        loss = loss_distillarion + loss_rgs
                    else:
                        loss = self.mse(pred_score[:, -1].unsqueeze(-1), label)

                    if step == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    for b in range(batch_size):
                        if action_id[b] == current_task:
                            pred_scores.append(pred_score[b,-1].item())
                            true_scores.append(label[b].item())
                pred_scores = np.array(pred_scores)
                true_scores = np.array(true_scores)
                rho, p = stats.spearmanr(pred_scores, true_scores)
                L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
                RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
                        true_scores.shape[0]
                    
                print('correlation: %.6f' % (rho))
                if step == 'test':
                    if rho > rho_best:
                        rho_best = rho
                        epoch_best = epoch
                        L2_min = L2
                        RL2_min = RL2
                        print('-----New best found!-----')
                        self.save_checkpoint(self.i3d, self.mlp, epoch, rho, RL2,
                                        self.exp_name + '_task{}_{}'.format(task_id, self.action_list[task_id]) +'_best')
                    if epoch == self.epochs:
                        self.save_checkpoint(self.i3d, self.mlp, epoch, rho, RL2,
                                        self.exp_name + '_task{}_{}'.format(task_id, self.action_list[task_id]) +'_final')
                    print('Current best————Corr: %.6f , L2: %.6f , RL2: %.6f @ epoch %d \n' % (rho_best, L2_min, RL2_min, epoch_best))

    def valuate_previous_tasks(self, task_id):
        print('Valuate step: ')
        print('Current task:', self.action_list[task_id])
        # load best model for current task
        best_model_path = './ckpt/' + self.exp_name + '_task{}_{}'.format(task_id, self.action_list[task_id]) + '_best.pth'
        _,_,_,_ = self.load_ckeckpoint((self.i3d, self.i3d) ,ckpt_path=best_model_path)
        self.to_cuda()
        self.dp()
        # for each seen task
        for pre_task_id in range(task_id):
            pre_task_name = self.task_list[pre_task_id]
            _, dataloader = buid_dataloaders(pre_task_name, loading_info=False)
            self.i3d.eval()  # set model to val mode
            self.mlp.eval()
            true_scores = []
            pred_scores = []
            torch.set_grad_enabled(False)
            for batch_idx, data in enumerate(dataloader):
                true_scores.extend(data['final_score'].numpy())
                # data preparing
                video = data['video'].float().cuda() # N, C, T, H, W
                # forward
                feat = self.i3d(video)
                pred_score = self.mlp(feat)
                pred_scores.extend([i.item() for i in pred_score[:, pre_task_id]])

                progress_bar(batch_idx, len(dataloader), '')
            pred_scores = np.array(pred_scores)
            true_scores = np.array(true_scores)
            rho, p = stats.spearmanr(pred_scores, true_scores)
            L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
            RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
                    true_scores.shape[0]
            print('    %20s:\t\t Corr: %.6f , L2: %.6f , RL2: %.6f' % (self.action_list[pre_task_id], rho, L2, RL2))
        return

    def afterTrain(self, task_id):
        best_model_path = './ckpt/' + self.exp_name + '_task{}_{}'.format(task_id, self.action_list[task_id]) + '_best.pth'
        _,_,_,_ = self.load_ckeckpoint((self.i3d, self.i3d) ,ckpt_path=best_model_path)
        
        # TODO:减少原本在memory中的样本
        
        m=int(self.memory_size / (task_id+1))
        return 
