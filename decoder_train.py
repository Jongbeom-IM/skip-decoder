import os
import argparse
import logging
import h5py
import sys
sys.path.append('./')
from copy import deepcopy
import yaml

import torch
from torch.utils.data import DataLoader, Dataset
from utils.loss import ComputeLoss
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import common, yolo
from models.yolo import Model
from utils.torch_utils import initialize_weights, select_device, scale_img
from utils.datasets import HDF5FolderDataset, create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from tqdm import tqdm
import test
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset_info = [path]
    def __len__(self):
        return len(self.dataset_info)
    def __getitem__(self, index):
        h5_file = self.dataset_info
        with h5py.File(h5_file[index], 'r') as f:
            data = np.array(f[str(9)], dtype=np.float32)
        data = torch.tensor(data.squeeze())
        return data

def model_freeze(model, freeze, idx):
    # print(f'dec{idx}', freeze[idx])
    for dec in model.decoder:
        for name, param in dec.named_parameters():
            param.requires_grad = False
            # print(f'Freezing {name}: requires_grad={param.requires_grad}')
    for i, (name, param) in enumerate(model.decoder[idx].named_parameters()):
        if any(name.startswith(f) for f in freeze):
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
parser.add_argument('--epochs', type=int, default=300)

parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', nargs='?', const=True, default=None, help='resume most recent training')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
parser.add_argument('--project', default='runs/train', help='save to project/name')
parser.add_argument('--entity', default=None, help='W&B entity')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--quad', action='store_true', help='quad dataloader')
parser.add_argument('--linear-lr', action='store_true', help='linear LR')
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
opt = parser.parse_args()

with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

#Save
save_dir = f'runs/train/e6edec_{opt.name}'
os.makedirs(os.path.join(save_dir,'weights'), exist_ok=True)
logging.basicConfig(filename=save_dir+'/training_errors.log', level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

device = select_device(opt.device)

#load model
model = Model(cfg=opt.cfg, ch=320, decoder= True)
pretrained = torch.load('runs/train/e6e640/weights/best.pt', map_location='cpu')
prefix = [f'model.{i}.' for i in range(10,59)]
model_dict = [v for k,v in pretrained['model'].state_dict().items() if any(k.startswith(p) for p in prefix)]
model_dict = {k:v for k,v in zip(model.model.state_dict().keys(), model_dict)}
model.model.load_state_dict(model_dict)
model = model.to(device=device)
model.model.eval()
for name, param in model.model.named_parameters():
        param.requires_grad = False  # Freeze all layers
model.save = sorted(model.save + [11, 33, 38, 43, 48])

#resume
if opt.resume is not None:
    for i in range(3):
        model.decoder[i] = torch.load(os.path.join(opt.resume,f'dec{i+1}_last.pt'), map_location=device)
    with open(opt.resume+'../result.txt','r') as f:
        start_epoch = int(f.readlines()[-1].split(' ')[0].split('/')[0]) + 1
    print('resume epoch:', start_epoch)
else: start_epoch = 0
    
#freeze layer
freeze = [['0.','1.'], ['0.','1.','2.'], ['0.','1.'],
          ['2.','3.'], ['3.','4.','5.'], ['2.','3.'],
          ['4.','5.'], ['6.','7.','8.'], ['4.','5.']]
train_seq = [((7,1),0), ((26,8),1) ,((43,16),2),
             ((3,3),0), ((19,11),1),((38,18),2),
             ((-1,5),0),((12,14),1),((33,20),2)]
# 01-7, 03-5, 05-0
# 26-2, 19-5, 12-8
# 43-1, 38-3, 33-5
print('freeze set')

# Loss, Optimizer
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()
optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.decoder.parameters()), lr=0.001) for _ in range(9)]
lr_schedulers = init_schedulers(optimizers, max_lr=0.01, total_steps=300, final_div_factor=0.1)
print('loss set')

#Dataset
with open('data/coco_full.yaml') as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)
nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names

dir = '/home/syslab-4090/jongbeom/Dataset/coco/images/yolov7'
# train_dataset = CustomDataset('/home/syslab-4090/jongbeom/Dataset/coco/images/yolov7/train/000000000009.h5')#
train_dataset = HDF5FolderDataset(dir+'/train')
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
nb = len(train_loader)
testloader, testset = create_dataloader(data_dict['val'], 640, opt.batch_size * 2, 64, opt,  # testloader
                                hyp=hyp, cache=False, rect=True, rank=-1,
                                world_size=opt.world_size, workers=8,
                                pad=0.5, prefix=colorstr('val: '))
print('Dataset set')

best_map, current_map50, current_map = [0,0,0], [0,0,0], [0,0,0]
try:
    for epoch in range(start_epoch, opt.epochs):
        # pbar = enumerate(train_loader)
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        model.decoder.train()

        losses = [[],[],[],[],[],[],[],[],[]]
        epoch_info = f"{epoch}/{opt.epochs - 1}"
        for bi, img in pbar:
            img = img.to(device)
            # Forward pass
            for dec in model.decoder:
                for name, param in dec.named_parameters():
                    param.requires_grad = True
            y, y_dec = model(img, training=True)
            y.append(img)
            for idx, ((m_i, d_i),i_) in enumerate(train_seq):
                model_freeze(model, freeze[idx], i_)
                # print(f'Iteration: {idx}, m_i: {m_i}, d_i: {d_i}')
                # print_requires_grad_params(model)
                # print(y_dec[d_i])
                optimizer = optimizers[idx]
                loss= criterion(y_dec[d_i],y[m_i])
                optimizer.zero_grad()
                loss.backward()
                # with h5py.File(f'feature{epoch}_{bi}.h5', 'a') as f:
                #     f.create_dataset('dec'+str(i_)+'_d'+str(d_i),data=y_dec[d_i].to('cpu').detach().numpy())
                #     f.create_dataset('dec'+str(i_)+'_m'+str(m_i),data=y[m_i].to('cpu').detach().numpy())
                optimizer.step()
                losses[idx].append(loss.to('cpu').item())
            loss_info = f"D1Loss: {losses[0][-1]:.3f} {losses[1][-1]:.3f} {losses[2][-1]:.3f}, D2Loss: {losses[3][-1]:.3f} {losses[4][-1]:.3f} {losses[5][-1]:.3f}, D3Loss: {losses[6][-1]:.3f} {losses[7][-1]:.3f} {losses[8][-1]:.3f}"
            s = f"{epoch_info} {loss_info}"
            pbar.set_postfix_str(s)
        
        #lr schedulers
        for scheduler in lr_schedulers:
            scheduler.step()
        # Losses sum
        for i, l in enumerate(losses):
            losses[i] = sum(l) / nb
        loss_info = f"D1Loss: {losses[0]:.5f} {losses[1]:.5f} {losses[2]:.5f}, D2Loss: {losses[3]:.5f} {losses[4]:.5f} {losses[5]:.5f}, D3Loss: {losses[6]:.5f} {losses[7]:.5f} {losses[8]:.5f}"
        
        torch.save(model.decoder[0], save_dir+'/weights/dec1_last.pt')
        torch.save(model.decoder[1], save_dir+'/weights/dec2_last.pt')
        torch.save(model.decoder[2], save_dir+'/weights/dec3_last.pt')

        #test
        for i in range(1,4):
            # Model parameters
            test_model = yolo.model_spliter(model_path='runs/train/e6e640/weights/best.pt',
                                            decoder_path=f'{save_dir}/weights/dec{str(i)}_last.pt',
                                            dec_v=i,
                                            device=device)
            nl = test_model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
            hyp['box'] *= 3. / nl  # scale to layers
            hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
            hyp['obj'] *= (640 / 640) ** 2 * 3. / nl  # scale to image size and layers
            hyp['label_smoothing'] = opt.label_smoothing
            test_model.nc = nc  # attach number of classes to model
            test_model.hyp = hyp  # attach hyperparameters to model
            test_model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
            test_model.class_weights = labels_to_class_weights(testset.labels, nc).to(device) * nc  # attach class weights

            compute_loss = ComputeLoss(test_model)  # init loss class
            result, maps, times = test.test(data=data_dict,
                                       batch_size=opt.batch_size*2,
                                       imgsz=640,
                                       model=test_model,
                                       single_cls=False,
                                       dataloader=testloader,
                                       save_dir=save_dir,
                                       verbose=False,
                                       plots=False,
                                       wandb_logger=None,
                                       is_coco=False,
                                       v5_metric=False
                                       )
            current_map50[i-1] = result[2]
            current_map[i-1] = result[3]
            if best_map[i-1] < current_map[i-1]: # save best.pt
                torch.save(model.decoder[i-1], save_dir+f'/weights/dec{str(i)}_epoch{epoch}_best.pt')
                best_map[i-1] = current_map[i-1]
        s = f"{epoch_info} {loss_info} {current_map50} {current_map}\n"
        # Save per epoch
        with open(save_dir + '/result.txt', 'a') as f:
            f.write(s)  # append metrics, val_loss
        
except Exception as e:
    logging.error("Exception occurred", exc_info=True)


