import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import pandas as pd
import openpyxl
import torch.onnx

def export_model_to_onnx(model, inputs, file_path, dynamic_axes):
    torch.onnx.export(
        model, 
        inputs, 
        file_path, 
        export_params=True, 
        opset_version=11,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes=dynamic_axes
    )

parser = argparse.ArgumentParser()
parser.add_argument('--cfg')
opt = parser.parse_args()


model = Model(opt.cfg).eval()
# print(model_dict.keys())
yaml_ = Model('./cfg/deploy/yolov7-e6e.yaml')
pretrained = attempt_load('yolov7-e6e.pt')
yaml_dict = yaml_.state_dict()
model_dict = model.state_dict()
pre_dict = pretrained.state_dict()
pre_key = list(pre_dict.keys())
for _ in range(len(yaml_dict)-len(pre_key)):
    pre_key.append('')
print(len(pre_key), len(yaml_dict))
d = {'pretrained': pre_key,
     'yaml': yaml_dict.keys()
     }
d = pd.DataFrame(d)
d.to_excel('weight.xlsx')



# Define the dynamic axes
# dynamic_axes = {
#     'input': {0: 'batch_size', 2: 'height', 3: 'width'},
#     'output': {0: 'batch_size', 2: 'height', 3: 'width'}
# }
# inputs = torch.randn(1,3,640,640)
# # Export the model
# export_model_to_onnx(model, inputs, './e6enew.onnx', dynamic_axes)
