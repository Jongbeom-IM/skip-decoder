import argparse
import json
import os
from pathlib import Path
from threading import Thread
from models.yolo import Model

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized



parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--cfg', type=str)
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-med.yaml', help='hyperparameters path')
parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--task', default='val', help='train, val, test, speed or study')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--verbose', action='store_true', help='report mAP by class')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--project', default='runs/test', help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args()

with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
        
if opt.cfg:
    nc = len(data_dict['names'])
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))
    
x = torch.rand((1,3,640,640))
print(model(x))
for idx, module in enumerate(model.model):
    try:
        print(f'{idx}-Layer input Tensor: ',x.shape)
    except:
        print(f'{idx}-Layer input Tensor: ',[j.shape for j in x])
    try: x = module(x)
    except: x = module(x,idx)
