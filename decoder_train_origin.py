import os
import argparse
import logging
import h5py
import sys
sys.path.append('./')
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import common, yolo
from models.yolo import Model
from utils.torch_utils import initialize_weights, select_device, scale_img
from utils.datasets import HDF5FolderDataset
from tqdm import tqdm
from test_decoder import test
import numpy as np

def model_freeze(model, freeze, idx):
    # print(f'dec{idx}', freeze[idx])
    for dec in model.decoder:
        for name, param in dec.named_parameters():
            param.requires_grad = False
            # print(f'Freezing {name}: requires_grad={param.requires_grad}')
    for i, (name, param) in enumerate(model.decoder[idx].named_parameters()):
        if any(name.startswith(f) for f in freeze[idx]):
            param.requires_grad = True
            # print(f'Unfreezing {name}: requires_grad={param.requires_grad}')

def print_requires_grad_params(model):
    for i,dec in enumerate(model.decoder):
        for name, param in dec.named_parameters():
            if param.requires_grad:
                print(f"DEC{i}={name}: requires_grad={param.requires_grad}")

def init_schedulers(optimizers, max_lr, total_steps, final_div_factor):
    schedulers = [
        lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=final_div_factor
        ) for optimizer in optimizers
    ]
    return schedulers
        
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',default='cfg/deploy/e6e_SD_T.yaml')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--epochs', default=300)
parser.add_argument('--name')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--batch-size', default=16)
opt = parser.parse_args()

#Save
save_dir = f'runs/train/e6edec_{opt.name}'
os.makedirs(os.path.join(save_dir,'weights'), exist_ok=True)
logging.basicConfig(filename=save_dir+'/training_errors.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

device = select_device(opt.device)
#load model
model = Model(cfg=opt.cfg, ch=320, decoder= True)
model = model.to(device=device)
model.model.eval()
for name, param in model.model.named_parameters():
        param.requires_grad = False  # Freeze all layers
model.save = sorted(model.save + [11, 33, 38, 43, 48])

#resume
if opt.resume is not None:
    model = torch.load(opt.resume, map_location=device)
    with open(save_dir+'/result.txt','r') as f:
        start_epoch = int(f.readlines()[-1].split(' ')[0].split('/')[0]) + 1
else: start_epoch = 0

#freeze layer
freeze = [[['0.','1.'], ['0.','1.','2.'], ['0.','1.']],
          [['2.','3.'], ['3.','4.','5.'], ['2.','3.']],
          [['4.','5.'], ['6.','7.','8.'], ['4.','5.']]]
train_seq = [[(7,1),(26,8),(43,16)],
             [(3,3),(19,11),(38,18)],
             [(-1,5),(12,14),(33,20)]]
# 01-7, 03-5, 05-0
# 26-2, 19-5, 12-8
# 43-1, 38-3, 33-5
print('freeze set')

# Loss, Optimizer
# criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
criterion = nn.SmoothL1Loss()
optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.decoder.parameters()), lr=0.001) for _ in range(3)]
lr_schedulers = init_schedulers(optimizers, max_lr=0.01, total_steps=100, final_div_factor=0.1)
print('loss set')

#Dataset
dir = '/home/syslab-4090'
dir = os.path.join(dir, 'jongbeom','Dataset','coco','images','yolov7')
train_dataset = HDF5FolderDataset(dir+'/train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
nb = len(train_loader)  # number of batches
print('Dataset set')

print(int(opt.epochs//3),int(opt.epochs*2//3))
try:
    seq = (train_seq[0], 0)
    for epoch in range(start_epoch, opt.epochs):
        # pbar = enumerate(train_loader)
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        model.decoder.train()

        if int(opt.epochs//3) == epoch: 
            seq = (train_seq[1], 1)
        elif int(opt.epochs*2//3) == epoch: 
            seq = (train_seq[2], 2)

        losses = [[],[],[]]
        epoch_info = f"{epoch}/{opt.epochs - 1}"
        for i, img in pbar:
            img = img.to(device)
            # Forward pass
            y, y_dec = model(img, training=True)
            y.append(img)
            for idx, (m_i, d_i) in enumerate(seq[0]):
                print('this')
                # print(f'Iteration: {i}, m_i: {m_i}, d_i: {d_i}')
                model_freeze(model, freeze[seq[1]], idx)
                # print_requires_grad_params(model)
                optimizer = optimizers[idx]
                loss= criterion(y_dec[d_i],y[m_i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses[idx].append(loss.to('cpu').item())
            loss_info = f"D1Loss: {losses[0][-1]:.5f}, D2Loss: {losses[1][-1]:.5f}, D3Loss: {losses[2][-1]:.5f}"
            s = f"{epoch_info} {loss_info}"
            pbar.set_postfix_str(s)
        
        #lr schedulers
        for scheduler in lr_schedulers:
            scheduler.step()
        if (epoch + 1) % 100 == 0: 
            schedulers = init_schedulers(optimizers, max_lr=0.01, total_steps=100, final_div_factor=0.1)
        # Losses sum
        for i, l in enumerate(losses):
            losses[i] = sum(l) / nb
        loss_info = f"D1Loss: {losses[0]}, D2Loss: {losses[1]}, D3Loss: {losses[2]}"
        s = f"{epoch_info} {loss_info}\n"
        # Save per epoch
        with open(save_dir + '/result.txt', 'a') as f:
            f.write(s)  # append metrics, val_loss
        
        torch.save(model.decoder[0], save_dir+'/weights/dec1_last.pt')
        torch.save(model.decoder[1], save_dir+'/weights/dec2_last.pt')
        torch.save(model.decoder[2], save_dir+'/weights/dec3_last.pt')

        # test(data='data/coco_full.yaml',
        #      weights='runs/train/e6e640/weights/best.pt',
        #      decoder =save_dir+'/weights/dec1_last.pt',
        #      split_point=21,
        #      batch_size=16,
        #      device=opt.device)
        # test(data='data/coco_full.yaml',
        #      weights='runs/train/e6e640/weights/best.pt',
        #      decoder =save_dir+'/weights/dec2_last.pt',
        #      split_point=43,
        #      batch_size=16,
        #      device=opt.device)
        # test(data='data/coco_full.yaml',
        #      weights='runs/train/e6e640/weights/best.pt',
        #      decoder =save_dir+'/weights/dec3_last.pt',
        #      split_point=58,
        #      batch_size=16,
        #      device=opt.device)
        
except Exception as e:
    logging.error("Exception occurred", exc_info=True)

