import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
from torchvideotransforms import video_transforms, volume_transforms


###################################################################################
# fix BatchNorm
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
###################################################################################




###################################################################################
# 进度条
TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    #for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #    sys.stdout.write(' ')

    # Go back to the center of the bar.
    #for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #    sys.stdout.write('\b')
    sys.stdout.write(' %d/%d   ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

###################################################################################






###################################################################################
# build dataloaders
def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_dataloaders(dataset_creator, args):
    train_dataset, test_dataset = dataset_creator(args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_train,
                                                   shuffle=True, num_workers=int(args.workers),
                                                   pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)
    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['test'] = test_dataloader
    return dataloaders
###################################################################################





###################################################################################
# misc
def denormalize(label, class_idx, upper = 100.0): 
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label

def normalize(label, class_idx, upper = 100.0):
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper)
    return norm_label
###################################################################################





###################################################################################
# kaiming initialization
def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
###################################################################################