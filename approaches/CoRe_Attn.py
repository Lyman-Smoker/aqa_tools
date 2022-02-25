
import imp
import sys,time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.I3D_Backbone import I3D_backbone
from networks.MLP_Regressor import MLP_block
from networks.Bidir_Attention import Bidir_Attention
from scipy import stats
from utils import *
from config.CoRe_Attn_config import *

class CoRe_Attn(object):
    def __init__(self, dataloaders, args):
        self.dataloaders = dataloaders
        self.epochs = args.epochs
        
        self.i3d = I3D_backbone(I3D_class=400)
        self.i3d.load_pretrain(args.pretrained_i3d_weight)
        self.bidir_attention = Bidir_Attention(dim=feature_dim, mask=mask, return_attn=div_loss)
        self.mlp = MLP_block(in_dim=2049, out_dim=1)

        self.lr = args.lr
        self.lr_factor = args.lr_factor
        self.weight_decay = args.weight_decay
        self.fix_bn = args.fix_bn

        self.mse = torch.nn.MSELoss()
        

        self.exp_name = args.exp_name
        self.log_path = os.path.join(args.log_root, args.exp_name + '_log.txt')
        self.ckpt_root = args.ckpt_root


    def _get_optimizer(self):
        optimizer = optim.Adam([
            {'params': self.i3d.parameters(), 'lr': self.lr * self.lr_factor},
            {'params': self.bidir_attention.parameters()},
            {'params': self.mlp.parameters()}
        ], lr=self.lr, weight_decay=self.weight_decay)
        scheduler = None
        return optimizer, scheduler

    def to_cuda(self):
        self.i3d = self.i3d.cuda()
        self.bidir_attention = nn.DataParallel(self.bidir_attention)
        self.mlp = self.mlp.cuda()
        self.mse = self.mse.cuda()
        torch.backends.cudnn.benchmark = True
        return

    def dp(self):
        self.i3d = nn.DataParallel(self.i3d)
        self.mlp = nn.DataParallel(self.mlp)
        self.bidir_attention = nn.DataParallel(self.bidir_attention)

    def get_dataloader(self):
        return self.dataloaders['train'], self.dataloaders['test']

    def train_net(self):
        with open(self.log_path, 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write(time.asctime(time.localtime(time.time())) + '\n')
            f.write('train net: \n')
            f.write('\n')

        
        optimizer, scheduler=self._get_optimizer()
        self.dp()
        # this function contains training and testing steps
        train_dataloader, test_dataloader = self.get_dataloader()
        steps = ['train', 'test']

        # parameter setting
        start_epoch = 0
        global epoch_best, rho_best, L2_min, RL2_min
        epoch_best = 0
        rho_best = 0
        L2_min = 1000
        RL2_min = 1000

        for epoch in range(self.epochs):
            print('EPOCH:', epoch)

            for step in steps:
                print(step + ' step:')
                true_scores = []
                pred_scores = []
                if self.fix_bn:
                    self.i3d.apply(fix_bn)  # fix bn
                if step == 'train':
                    self.i3d.train()
                    self.mlp.train()
                    self.bidir_attention.train()
                    torch.set_grad_enabled(True)
                    data_loader = train_dataloader
                else:
                    self.i3d.eval()
                    self.mlp.eval()
                    self.bidir_attention.eval()
                    torch.set_grad_enabled(False)
                    data_loader = test_dataloader
                for batch_idx, (data_1, data_2_list) in enumerate(data_loader):
                    num_voters = len(data_2_list)
                    batch_size = data_1['final_score'].shape[0]
                    pred_score_1_sum = torch.zeros((batch_size, 1)).cuda()
                    # Data preparing for data_1
                    video_1 = data_1['video'].float().cuda()  # N, C, T, H, W
                    label_1 = data_1['final_score'].float().reshape(-1, 1).cuda()
                    for data_2 in data_2_list:
                        
                        # Data preparing for data_2
                        video_2 = data_2['video'].float().cuda()  # N, C, T, H, W
                        label_2 = data_2['final_score'].float().reshape(-1, 1).cuda()

                        # Forward
                        # 1: pass backbone
                        # print(video_1.shape)
                        feat_1,feat_2 = self.i3d(video_1, video_2)  # [B, 10, 1024]
                        if attn:
                            feature_2to1, feature_1to2, attn_1, attn_2 = self.bidir_attention(feat_1, feat_2)  # [B, 10, 1024]
                            # 2: concat features
                            cat_feat_1 = torch.cat((feat_1, feature_2to1), 2)
                            cat_feat_2 = torch.cat((feat_2, feature_1to2), 2)
                            # 3: features fusion
                            # cat_feat_1 = ln_mlp(cat_feat_1)
                            # cat_feat_2 = ln_mlp(cat_feat_2)
                            aggregated_feature_1 = cat_feat_1.mean(1)
                            aggregated_feature_2 = cat_feat_2.mean(1)
                        else:
                            # 2: concat
                            # print(feat_1.shape)
                            # print(feat_2.shape)
                            # print(label_1.shape)
                            cat_feat_1 = torch.cat((feat_1, feat_2), 2)
                            cat_feat_2 = torch.cat((feat_2, feat_1), 2)
                            aggregated_feature_1 = cat_feat_1.mean(1)
                            aggregated_feature_2 = cat_feat_2.mean(1)
                        aggregated_feature_1 = torch.cat((aggregated_feature_1, label_2), 1)
                        aggregated_feature_2 = torch.cat((aggregated_feature_2, label_1), 1)
                        # print(combined_feature1.shape)
                        # 3: regress
                        pred_score1 = self.mlp(aggregated_feature_1)
                        pred_score2 = self.mlp(aggregated_feature_2)

                        # loss
                        loss1 = self.mse(pred_score1, label_1)
                        loss2 = self.mse(pred_score2, label_2)
                        loss = loss1 + loss2
                        
                        # Summing up each score
                        pred_score_1_sum += pred_score1
                    
                    # BP
                    if step == 'train':
                        # print('loss: ',loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    progress_bar(batch_idx, len(data_loader), 'loss: %.4f' % loss.item())
                    # else:
                    #     progress_bar(batch_idx, len(data_loader), ' ')
                        
                    # Updating score lists    
                    pred_score_1_avg = pred_score_1_sum / num_voters
                    true_scores.extend(data_1['final_score'].numpy())
                    pred_scores.extend([i.item() for i in pred_score_1_avg])
                
                # for name, parms in self.i3d.named_parameters():  
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:',parms.requires_grad)
                #     print('-->grad_value:',parms.grad)
                #     print("==============================")
                   
                pred_scores = np.array(pred_scores)
                true_scores = np.array(true_scores)
                # print(pred_scores)
                # print(true_scores)
                rho, p = stats.spearmanr(pred_scores, true_scores)
                L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
                RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
                print('[%s] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (step, epoch, rho, L2, RL2))

                # save checkpoint
                if step == 'test':
                    if rho > rho_best:
                        print('___________________find new best___________________')
                        rho_best = rho
                        epoch_best = epoch
                        RL2_min = RL2
                        self.save_checkpoint(self.i3d, self.mlp, self.bidir_attention, epoch, rho, RL2,
                                        self.exp_name + '_best')
                    print('current best----correlation: %.4f    RL2: %.4f   @ epoch %d' % (rho_best, RL2_min, epoch_best))
                    
                    with open(self.log_path, 'a') as f:
                        f.write(
                        '[%s] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (step, epoch, rho, L2, RL2) + '\n')
                
        self.save_checkpoint(self.i3d, self.mlp, self.bidir_attention, epoch, rho, RL2, self.exp_name + '_final')

        return
    
    def save_checkpoint(self, i3d, mlp, bidir_attn, epoch, rho, RL2, exp_name):
        if not os.path.isdir(os.path.join(self.ckpt_root ,  exp_name)):
            os.makedirs(os.path.join(self.ckpt_root ,  exp_name))
        torch.save({
            'i3d': i3d.state_dict(),
            'mlp': mlp.state_dict(),
            'bidir_attn': bidir_attn.state_dict(),
            'epoch': epoch,
            'rho': rho,
            'RL2': RL2,
        }, os.path.join(self.ckpt_root ,  exp_name , exp_name + '.pth'))
        return 

    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print('no checkpoint file from path %s...' % ckpt_path)
            return 0, 0, 0, 1000, 1000
        print('Loading weights from %s...' % ckpt_path)

        # load state dict
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # parameter resume of base model
        i3d_ckpt = {k.replace("module.", ""): v for k, v in state_dict['i3d'].items()}
        self.i3d.load_state_dict(i3d_ckpt)
        regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
        self.mlp.load_state_dict(regressor_ckpt)
        bidir_attention_ckpt = {k.replace("module.", ""): v for k, v in state_dict['bidir_attn'].items()}
        self.bidir_attention.load_state_dict(bidir_attention_ckpt)

        # parameter
        epoch_best = state_dict['epoch']
        rho_best = state_dict['rho']
        RL2_min = state_dict['RL2']

        return epoch_best, rho_best, RL2_min

