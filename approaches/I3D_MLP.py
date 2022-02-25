import sys,time
import numpy as np
import torch
from networks.I3D_Backbone import I3D_backbone
from networks.MLP_Regressor import MLP_block

class I3D_MLP(object):
    def __init__(self, Dataloader, args):
        self.Dataloader = Dataloader
        self.epochs = args.nepochs
        
        self.i3d = I3D_backbone(I3D_class=400)
        self.mlp = MLP_block(in_dim=1024, out_dim=1)

        self.mse = torch.nn.MSELoss()
        self.optimizer=self._get_optimizer()
        



    def _get_optimizer(self, lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train():
        return