import torch
import torchvision
from torchvision import transforms
import os, sys, math, h5py
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir,'../')
sys.path.append(yolo_path)
from models.yolo import Model, Decoder, model_spliter
import argparse
from PIL import Image
import numpy as np

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_w = math.ceil(img_size[0] / int(s)) * int(s)
    new_h = math.ceil(img_size[1] / int(s)) * int(s)
    return (new_w, new_h)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str)
parser.add_argument('--decoder', type=int)
opt = parser.parse_args()

img = Image.open('h5_data/000000000139.jpg')
w, h = check_img_size(img.size)
transform = transforms.Compose([transforms.Resize((h,w)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
img = transform(img).unsqueeze(dim=0)
print(img.shape)
with h5py.File(f'h5_data/139dec{str(opt.decoder)}_blockup.h5', 'a') as f:
    # del f['origin']
    f.create_dataset('origin',data=img.to('cpu').detach().numpy())
layers = [[(9,23),(13,25),(17,27)], [(36,46),(29,49),(22,52)], [(43,60),(48,62),(53,64)]]
save = [[9,13,17], [22,29,36],[43,48,53]]
model = torch.load(opt.weights, 'cpu')

try:
    model['model'] = model['model'].float()
    model['model'].save = sorted(model['model'].save + save[opt.decoder-1])
except:
    model = model_spliter('runs/train/e6e640/weights/best.pt', opt.weights,opt.decoder, device='cpu')
    model.save = sorted(model.save + save[opt.decoder-1])
model(img)
print(len(model.y))
with h5py.File(f'h5_data/139dec{str(opt.decoder)}_blockup.h5', 'a') as f:
    for m, d in layers[opt.decoder-1]:
        f.create_dataset('m_'+str(m),data=model.y[m].to('cpu').detach().numpy())
        f.create_dataset('d_'+str(d),data=model.y[d].to('cpu').detach().numpy())
