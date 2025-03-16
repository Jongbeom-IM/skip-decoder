import os
import torch
import torchvision.models as models
import torch.nn as nn
from models.yolo import Model, Detect, deepcopy
from models.experimental import Ensemble
import models.common as common 
import itertools
import argparse
from models.experimental import attempt_load
from utils.torch_utils import initialize_weights, select_device, scale_img
import h5py
from models.yolo import Model
from utils.datasets import HDF5FolderDataset
from torch.utils.data import DataLoader, Dataset
import test
import json
import os

from pathlib import Path
from threading import Thread

import numpy as np
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

def load_dict(cfg, weight='./yolov5s.pt',start=None,end=None):
    pretrained_model = torch.load(weight)
    pretrained_dict = pretrained_model['model'].state_dict()
    model = Model(cfg, ch=3, nc=80)
    model_dict = model.state_dict()
    
    prefix = [f'model.{i}.' for i in range(start)] + [f'model.{i}.' for i in range(end+1,80)]
        
    update_k = [k for k, v in model_dict.items() if any(k.startswith(p) for p in prefix)]
    update_v = [v for v in pretrained_dict.values()]
    update = {k: v for k, v in zip(update_k, update_v)}
    
    return update
    # print(len(pretrained_dict.keys()))
    # print(len(model_dict.keys()))
    
    
    
    # backbone_prefix = [f'model.{i}.' for i in range(10)]
    # model_prefix1 = [f'model.{i}.' for i in range(17, 22)]
    # model_prefix2 = [f'model.{i}.' for i in range(29, 36)]
    # model_prefix3 = [f'model.{i}.' for i in range(45,48)]
    # pretrained_prefix1 = [f'model.{i}.' for i in range(10, 15)]
    # pretrained_prefix2 = [f'model.{i}.' for i in range(15, 22)]
    # pretrained_prefix3 = [f'model.{i}.' for i in range(22,25)]
    # prefix_pair = list(itertools.chain(zip(backbone_prefix, backbone_prefix), 
    #                                 zip(pretrained_prefix1, model_prefix1), 
    #                                 zip(pretrained_prefix2, model_prefix2),
    #                                 zip(pretrained_prefix3, model_prefix3)))

    # dict_update = {}
    # for pretrained_prefix, model_prefix in prefix_pair:
    #     pretrained_key = [key for key in pretrained_dict.keys() if key.startswith(pretrained_prefix)]
    #     model_key = [key for key in model_dict.keys() if key.startswith(model_prefix)]
    #     selected_weights = {k_model: pretrained_dict[k_pt] for k_model, k_pt in zip(model_key, pretrained_key)}
    #     dict_update.update(selected_weights)

    # 새 모델의 state_dict 업데이트
    # return dict_update

def load_test_model(cfg, update, origin='./runs/train/e6e640/weights/best.pt'):
    origin = torch.load(origin, map_location='cpu')
    model = Model(cfg, ch=3, nc=80)
    
    origin_dict = origin['model'].state_dict()
    try: test_dict = update['model'].state_dict()
    except:
        update = torch.load(update, map_location='cpu')
        # update = Model(update, ch=3, nc=80)
        test_dict = update['model'].state_dict()
    model_dict = model.state_dict()
    
    prefix_update = [f'model.{i}.' for i in range(34)]
    prefix_origin = [f'model.{i}.' for i in range(22,64)]
        
    update_v = [v for k, v in test_dict.items() if any(k.startswith(p) for p in prefix_update)] + [v for k, v in origin_dict.items() if any(k.startswith(p) for p in prefix_origin)]
    update = {k: v for k, v in zip(model_dict.keys(), update_v)}
    model.load_state_dict(update, strict=False)  # load
    return model

def attempt_load_dec(weights, cfg, pretrained='./yolov5s.pt', device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    
    pretrained_model = torch.load(pretrained, map_location="cpu")
    pretrained_dict = pretrained_model['model'].model[-1].state_dict()
    
    ckpt = torch.load(weights, map_location="cpu")  # load
    print(ckpt.model[:-1].state_dict().keys())
    ckpt_dict = ckpt['model'].model[:-1].state_dict()
    ckpt_dict.update(pretrained_dict)
    
    ckpt = Model(cfg, ch=3, nc=80)
    ckpt.load_state_dict(ckpt_dict, strict=False)
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = ckpt.to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model

def freezing(weight = 'yolov5s.pt'):
    model = Model(cfg='models/yolov5s_cloud_run.yaml')
    model_dict = model.state_dict()
    for k, v in model.named_parameters():
        v.requires_grad = False
        
        

# parser = argparse.ArgumentParser()
# parser.add_argument('--weights')
# parser.add_argument('--cfg')
# parser.add_argument('--pretrained')
# opt = parser.parse_args()

# model = load_dict(cfg=opt.cfg, weight=opt.weights)
# print(dir(model.model))

# model = load_test_model('./cfg/deploy/e6e_SD1_D.yaml', './runs/train/exp6/weights/best.pt')
# input = torch.randn(1,3,640,640)
# pred = model(input)
# print(pred)
# 예시: 커스텀 디코더 클래스 정의
class Decoder(nn.Module):
    def __init__(self, decoder_selector, cfg='cfg/decoder/e6e_SD.yaml'):
        super(Decoder, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        
        # Define model
        if decoder_selector == 1:
            ch = self.yaml['ch'] = self.yaml.get('ch', 1280)  # input channels
            decoder_selector = 'Decoder1'
        elif decoder_selector == 2:
            ch = self.yaml['ch'] = self.yaml.get('ch', 160)  # input channels
            decoder_selector = 'Decoder2'
        elif decoder_selector == 3:
            ch = self.yaml['ch'] = self.yaml.get('ch', 640)  # input channels
            decoder_selector = 'Decoder3'

        self.model_blocks, self.save = parse_model(deepcopy(self.yaml), decoder=decoder_selector, ch=[ch])  # model, savelist
        self.model_blocks = nn.ModuleList(self.model_blocks)
        
        # Init weights, biases
        initialize_weights(self)
        
    def __getitem__(self, idx):
        return self.model_blocks[idx]
    
    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            if m.i in self.save: y.append(x)  # save output
        if profile:
            print('%.1fms total' % sum(dt))
        return y

# def parse_model(d,decoder,ch):
#     gd, gw = 1, 1
#     save, c2, blocks = [], ch[-1], []  # layers, savelist, ch out
#     for modules in d[decoder]:
#         layers = []
#         for i, (f, n, m, args) in enumerate(modules):
#             m = 'common.' + m
#             m = eval(m) if isinstance(m, str) else m  # eval strings
#             n = max(round(n * gd), 1) if n > 1 else n  # depth gain
#             if m in [common.DeConv, common.DeconvDownC]:
#                 c1, c2 = ch[f], args[0]
#                 args = [c1, c2, *args[1:]]
#                 save.append(i)
#                 if m is common.DeconvDownC:
#                     args.insert(2, n)  # number of repeats
#                     n = 1
#             elif m is common.SkipDeconvBlock:
#                 c1, c2 = ch[f], args[1]
#                 args = [c1, *args]
#             elif m is common.Maxpool_2:
#                 c2 = ch[f] // 2
#             else:
#                 c2 = ch[f]
#             m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
#             t = str(m)[8:-2].replace('__main__.', '')  # module type
#             np = sum([x.numel() for x in m_.parameters()])  # number params
#             m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#             layers.append(m_)
#             if i == 0:
#                 ch = []
#             ch.append(c2)
#         blocks.append(nn.Sequential(*layers))
#     return nn.ModuleList(blocks), sorted(save)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def attempt_load(model_path, map_location=None):
    # 모델을 파일에서 로드
    model = torch.load(model_path, map_location=map_location)
    return model

# def model_spliter(model_path, decoder_path, split_point, device):
#     # 모델 로드
#     pretrained = attempt_load(model_path, map_location=device)
#     # print(pretrained)
#     decoder = torch.load(decoder_path, map_location=device).half()
#     model = Model('/home/syslab-4090/jongbeom/reconfigurable/skip_yolov7/cfg/deploy/e6e_SD1.yaml',3)
#     decoder_dict = decoder.state_dict()
#     model_dict = pretrained['model'].state_dict()
    
#     prefix = list(model_dict.keys())[-1].split('.')[1]
#     prefix = ([f'model.{i}.' for i in range(split_point+1)], 
#               [f'model.{i}.' for i in range(split_point+1, int(prefix)+1)])
    
#     update_k = [k for k in model.state_dict().keys()]
#     update_v = [v for k, v in model_dict.items() if any(k.startswith(p) for p in prefix[0])] + \
#                 [v for v in decoder_dict.values()] + \
#                 [v for k, v in model_dict.items() if any(k.startswith(p) for p in prefix[1])]
#     update = {k: v for k, v in zip(update_k, update_v)}
#     print(len(update_k), len(update_v))
#     model.load_state_dict(update)
    
    
#     return model

def model_spliter(pretrained, decoder, index, device, cfg='cfg/deploy/e6e_SD1.yaml'):
    model = Model(cfg, 3).to(device)
    decoder = torch.load(decoder)
    pretrained = torch.load(pretrained)
    
    if index == 1:
        start_layer = 21
        end_layer = 27
    elif index == 2:
        start_layer = 43
        end_layer = 52
    else:
        start_layer = 58
        end_layer = 64
    
    decoder_layers = end_layer - start_layer
    
    decoder_dict = decoder.state_dict()
    pretrained_dict = pretrained['model'].state_dict()
    
    # Update layers outside the split region with the pretrained model's weights
    prefix = [f'{i}.' for i in range(start_layer+1)] + [f'{i}.' for i in range(end_layer+1, 64+decoder_layers)]
    prefix = [k for k in model.model.state_dict().keys() if any(k.startswith(p) for p in prefix)]
    update = {k: v for k, v in zip(prefix, pretrained_dict.values())}
    
    # Update layers within the split region with the decoder model's weights
    prefix = [f'{i}.' for i in range(start_layer+1, end_layer+1)]
    prefix = [k for k in model.model.state_dict().keys() if any(k.startswith(p) for p in prefix)]
    update.update({k: v for k, v in zip(prefix, decoder_dict.values())})
    
    model.model.load_state_dict(update)
    return model

# # 경로 지정
# model_path = 'e6e.pt'
# decoder_path = '/home/syslab-4090/jongbeom/reconfigurable/skip_yolov7/models_decoder/best_270.pt'

# # 모델 스플리터 호출
# model = model_spliter(model_path, decoder_path, 21, 'cpu')
# x = torch.randn(1,3,640,640)
# y = model(x)

#Save
# model = model_spliter('runs/train/e6e640/weights/best.pt','runs/train/decoder1_20240630/weights/last.pt', 1, 'cpu')
# model = Decoder('cfg/decoder/e6e_SD.yaml','Decoder1',3)
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.datasets import create_dataloader
import yaml
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
import models.yolo as yolo

l = [w for w in os.listdir('/home/syslab-4090/jongbeom/reconfigurable/skip_yolov7/runs/train/e6edec_20240704/weights') if w.endswith('last.pt')]
print(l)
l = [os.path.join('runs/train/e6edec_20240704/weights', l_) for l_ in l]
print([os.path.isfile(l_) for l_ in l])