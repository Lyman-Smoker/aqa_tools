
import imp
import sys,os,argparse,time
import numpy as np
import torch
from utils import *
import random

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='liym AQA collection')

# experiment settings
parser.add_argument('--experiment',default='',type=str,required=True,choices=['single', 'multi', 'IL'],help='experiment name')
parser.add_argument('--approach',default='',type=str,required=True,choices=['i3d-mlp','usdl','core-mlp','core-attn'],help='approach name')
parser.add_argument('--exp_name',type=str,default='debug',help='experiment name')                                                                            
# ckpt and log
parser.add_argument("--pretrained_i3d_weight", type=str, default='../pretrained_models/i3d_model_rgb.pth', help='path to the checkpoint model')
parser.add_argument('--log_root',type=str,default='./log',help='root to save log')
parser.add_argument('--ckpt_root',type=str,default='./ckpt',help='root to ckpt')
# dataset
parser.add_argument('--dataset',default='',type=str,required=True,choices=['aqa7', 'aqa7-pair', 'aqa7_multi', 'MTL', 'MTL_pair'],help='dataset name')      
parser.add_argument('--dataset_root',default='/mnt/gdata/AQA/AQA-7/',type=str,required=False,help='dataset root')  
parser.add_argument('--frame_length',default=102,type=int,required=False,help='frame_length')
parser.add_argument('--score_range',default=100,type=int,required=False,help='score range') 
parser.add_argument("--batch_size_test", type=int, default=16, help='batch size for testing')
parser.add_argument("--batch_size_train", type=int, default=16, help='batch size for training')
parser.add_argument("--workers", type=int, default=32, help='workers')
parser.add_argument("--action_id", type=int, default=1, help='action id from 1 to 6')

# run settings     
parser.add_argument('--seed',type=int,default=0,help='randn seed')                                                           
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--epochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.001,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--lr_factor',default=0.1,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--weight_decay',default=0.00001,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--fix_bn',default=True,type=bool,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--gpu',type=str,default='0',help='')

args=parser.parse_args()

# Args -- dataset
if args.dataset == 'aqa7':
    from dataloaders.Seven import Seven as Dataset
elif args.dataset == 'aqa7-pair':
    from dataloaders.Seven_Pairs import Seven_Pairs as Dataset
elif args.dataset == 'aqa7_multi':
    from dataloaders.Seven_Multi import Seven_Multi as Dataset
elif args.dataset == 'mtl_pair':
    from dataloaders.MTL_Pairs import MTL_Pairs as Dataset
elif args.dataset == 'mtl':
    from dataloaders.MTL_AQA import MTL_AQA as Dataset

# Args -- Approach
if args.approach == 'i3d-mlp':
    from approaches.I3D_MLP import I3D_MLP as Approach
elif args.approach == 'usdl':
    from approaches.UDSL import USDL as Approach
elif args.approach == 'core-mlp':
    from approaches.CoRe_MLP import Core_MLP as Approach
elif args.approach == 'core-attn':
    from approaches.CoRe_Attn import CoRe_Attn as Approach


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

dataloaders = build_dataloaders(Dataset, args)

approach = Approach(dataloaders=dataloaders, args=args)
approach.to_cuda()
approach.train_net()