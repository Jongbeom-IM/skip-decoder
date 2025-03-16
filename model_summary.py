import argparse
from models.yolo import Model

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
opt = parser.parse_args()

model = Model(opt.cfg, ch=3, nc=80)  # create
print(model)