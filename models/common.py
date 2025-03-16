# YOLOv5 common modules

import math
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import warnings

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized

glob_width_mult = [0.25, 0.5, 0.75, 1.0]

class TensorSplit(nn.Module):
    def __init__(self, i):
        super(TensorSplit, self).__init__()
        self.selecter = i
    def forward(self, x):
        return torch.split(x, int(x.shape[1]/2), dim=1)[self.selecter]

class SPConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, ind, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SPConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(c2) for _ in range(4)])
        #self.bn = nn.ModuleList([nn.BatchNorm2d(c2) for _ in range(2)])
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.ind = ind

    def forward(self, x, idx):
#         if idx == self.ind:
#             return self.act(self.bn[0](self.conv(x)))
#         else:
#             return self.act(self.bn[1](self.conv(x)))
        return self.act(self.bn[idx](self.conv(x)))
    
class RecoveryUnit(nn.Module):
    def __init__(self, c1, c2, ind, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RecoveryUnit, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        #self.bn = nn.BatchNorm2d(c2)
        self.bn = nn.ModuleList([nn.BatchNorm2d(c2) for _ in range(2)])
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.ind = ind

    def forward(self, x, idx):
        if idx == self.ind:
            #return self.act(self.bn[0](self.conv(x)))
            return self.bn[0](self.conv(x))
        else:
            return self.bn[1](x)
#        return self.bn[idx](x)
        
# class RecoveryUnit(nn.Module):
#     def __init__(self, c1, c2, ind, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(RecoveryUnit, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.ind = ind

#     def forward(self, x, ind):
#         if ind == self.ind:
#             return self.act(self.bn(self.conv(x)))
#         else:
#             return x
        
        
class SlimmableNeckConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=1, bias=True):
        super(SlimmableNeckConv2d, self).__init__(
            in_channel, out_channel,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups_list, bias=bias)
        self.width_mult_list = glob_width_mult
        self.in_channels_list = [int(in_channel*i) for i in self.width_mult_list]
        self.out_channels_list = [int(out_channel*i) for i in self.width_mult_list]
        self.groups_list = groups_list
        if self.groups_list == 1:
            self.groups_list = [1 for _ in range(len(self.in_channels_list))]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input, ind):
        weight = self.weight[
            0:self.out_channels_list[-1], 0:self.in_channels_list[ind], :, :]
        if self.bias:
            bias = self.bias[0:self.current_out_channels[ind]]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups_list[ind])
        return y
    
class SlimmableNeckConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SlimmableNeckConv, self).__init__()
        self.conv = SlimmableNeckConv2d(c1, c2, k, s, autopad(k, p), groups_list=g, bias=False)
        self.bn = SwitchableNeckBatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x, ind):
        return self.act(self.bn(self.conv(x, ind), ind))

    def fuseforward(self, x, ind):
        return self.act(self.conv(x, ind))

class SwitchableNeckBatchNorm2d(nn.Module):
    def __init__(self, channel):
        super(SwitchableNeckBatchNorm2d, self).__init__()
        self.width_mult = glob_width_mult
        self.num_features_list = [int(channel*i) for i in self.width_mult]
        self.num_features = max(self.num_features_list)
        self.bn = nn.ModuleList([nn.BatchNorm2d(self.num_features) for i in self.num_features_list])

    def forward(self, input, ind):
        y = self.bn[ind](input)
        return y
    
class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, channel):
        super(SwitchableBatchNorm2d, self).__init__()
        self.width_mult = glob_width_mult
        self.num_features_list = [int(channel*i) for i in self.width_mult]
        self.num_features = max(self.num_features_list)
        self.bn = nn.ModuleList([nn.BatchNorm2d(i) for i in self.num_features_list])

    def forward(self, input, ind):
        y = self.bn[ind](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=1, bias=True):
        super(SlimmableConv2d, self).__init__(
            in_channel, out_channel,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups_list, bias=bias)
        self.width_mult_list = glob_width_mult
        if in_channel == 12:
            self.in_channels_list = [12 for _ in self.width_mult_list]
        else:
            self.in_channels_list = [int(in_channel*i) for i in self.width_mult_list]
        self.out_channels_list = [int(out_channel*i) for i in self.width_mult_list]
        self.groups_list = groups_list
        if self.groups_list == 1:
            self.groups_list = [1 for _ in range(len(self.in_channels_list))]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input, ind):
        weight = self.weight[
            0:self.out_channels_list[ind], 0:self.in_channels_list[ind], :, :]
        if self.bias:
            bias = self.bias[0:self.current_out_channels[ind]]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups_list[ind])
        return y
    
    
class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

    
class SlimmableConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride=1, padding=0, output_padding=0,
                 groups_list=1, bias=True, dilation=1):
        super(SlimmableConvTranspose2d, self).__init__(
            in_channel, out_channel,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups_list, bias=bias, dilation=dilation)
        self.width_mult_list = glob_width_mult
        self.in_channels_list = [int(in_channel*i) for i in self.width_mult_list]
        self.out_channels_list = [int(out_channel*i) for i in self.width_mult_list]
        self.groups_list = groups_list
        if self.groups_list == 1:
            self.groups_list = [1 for _ in range(len(self.in_channels_list))]
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

    def forward(self, input, ind):
        weight = self.weight[
            0:self.in_channels_list[ind], 0:self.out_channels_list[ind], :, :]
        if self.bias:
            bias = self.bias[0:self.current_out_channels[ind]]
        else:
            bias = self.bias
        y = nn.functional.conv_transpose2d(
            input, weight, bias, self.stride, self.padding, self.output_padding,
            self.groups_list[ind], self.dilation)
        return y
    
class SlimmableConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SlimmableConv, self).__init__()
        self.conv = SlimmableConv2d(c1, c2, k, s, autopad(k, p), groups_list=g, bias=False)
        self.bn = SwitchableBatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x, ind):
        return self.act(self.bn(self.conv(x, ind), ind))

    def fuseforward(self, x, ind):
        return self.act(self.conv(x, ind))
    
class SlimmableDeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SlimmableDeConv, self).__init__()
        if s==2:
            self.deconv = SlimmableConvTranspose2d(c1, c2, k, s, autopad(k, p), output_padding=1, groups_list=g, bias=False)
        else:
            self.deconv = SlimmableConvTranspose2d(c1, c2, k, s, autopad(k, p), groups_list=g, bias=False)
        self.bn = SwitchableBatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x, ind):
        return self.act(self.bn(self.deconv(x, ind), ind))

    def fuseforward(self, x, ind):
        return self.act(self.deconv(x, ind))

class SlimmableBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(SlimmableBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SlimmableConv(c1, c_, 1, 1)
        self.cv2 = SlimmableConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x, ind):
        return x + self.cv2(self.cv1(x, ind), ind) if self.add else self.cv2(self.cv1(x, ind), ind)
    
class SlimmableC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SlimmableC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SlimmableConv(c1, c_, 1, 1)
        self.cv2 = SlimmableConv(c1, c_, 1, 1)
        self.cv3 = SlimmableConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[SlimmableBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x, ind):
        x_ = self.cv1(x, ind)
        for m in self.m:
            x_ = m(x_, ind)
        return self.cv3(torch.cat((x_, self.cv2(x, ind)), dim=1), ind)

class SlimmableDeconvBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(SlimmableDeconvBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SlimmableDeConv(c1, c_, 1, 1)
        self.cv2 = SlimmableDeConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x, ind):
        return x + self.cv2(self.cv1(x, ind), ind) if self.add else self.cv2(self.cv1(x, ind), ind)

class SlimmableDeconvC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(SlimmableDeconvC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SlimmableDeConv(c1, c_, 1, 1)
        self.cv2 = SlimmableDeConv(c1, c_, 1, 1)
        self.cv3 = SlimmableDeConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[SlimmableDeconvBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x, ind):
        x_ = self.cv1(x, ind)
        for m in self.m:
            x_ = m(x_, ind)        
        return self.cv3(torch.cat((x_, self.cv2(x, ind)), dim=1), ind)

class SlimmableSPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SlimmableSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SlimmableConv(c1, c_, 1, 1)
        self.cv2 = SlimmableConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x, ind):
        x = self.cv1(x, ind)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1), ind)

class SlimmableDeconvSPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SlimmableDeconvSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = SlimmableDeConv(c1, c_, 1, 1)
        self.cv2 = SlimmableDeConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([SlimmableConvTranspose2d(c_, c_, x, 1, x // 2, bias=False) for x in k])
        #self.m = nn.ModuleList([nn.Upsample(scale_factor=1.0) for x in k])
        #self.m = nn.ModuleList([nn.ConvTranspose2d(c_, c_*3, 3, 1, 1)])
        
    def forward(self, x, ind):
        x = self.cv1(x, ind)
        return self.cv2(torch.cat([x] + [m(x, ind) for m in self.m], 1), ind)

class SlimmableFocus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SlimmableFocus, self).__init__()
        self.conv = SlimmableConv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x, ind):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1), ind)
        # return self.conv(self.contract(x))
        
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

###
class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DeConv, self).__init__()
        if s==2:
            self.deconv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), output_padding=1, groups=g, bias=False)
        else:
            self.deconv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

    def fuseforward(self, x):
        return self.act(self.deconv(x))
###

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


#######
class DeconvBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(DeconvBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeConv(c1, c_, 1, 1)
        self.cv2 = DeConv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DeconvC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(DeconvC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeConv(c1, c_, 1, 1)
        self.cv2 = DeConv(c1, c_, 1, 1)
        self.cv3 = DeConv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[DeconvBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
########

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))



class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


########
class DeconvSPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(DeconvSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = DeConv(c1, c_, 1, 1)
        self.cv2 = DeConv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.ConvTranspose2d(c_, c_, x, 1, x // 2) for x in k])
        #self.m = nn.ModuleList([nn.Upsample(scale_factor=1.0) for x in k])
        #self.m = nn.ModuleList([nn.ConvTranspose2d(c_, c_*3, 3, 1, 1)])
        
    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
########

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
########

class DeconvFocus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DeconvFocus, self).__init__()
        self.conv = DeConv(c1, c2 * 4, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        y = self.conv(x)
        x = self.conv(x)
        #print(x.shape)
        y = torch.reshape(y, (x.shape[0], x.shape[1]//4, x.shape[2]*2, x.shape[3]*2))
        y[..., ::2, ::2], y[..., 1::2, ::2], y[..., ::2, 1::2], y[..., 1::2, 1::2] = torch.chunk(x, 4, 1)
        #print(torch.spit(x))
        return y
        # return self.conv(self.contract(x))

########

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im  # open
                im.filename = f  # for uri
            files.append(Path(im.filename).with_suffix('.jpg').name if isinstance(im, Image.Image) else f'image{i}.jpg')
            if not isinstance(im, np.ndarray):
                im = np.asarray(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with torch.no_grad(), amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

        t.append(time_synchronized())
        return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


