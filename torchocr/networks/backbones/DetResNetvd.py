from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging 
from collections import OrderedDict
import os 
import torch
from torch import nn

from torchocr.networks.CommonModules import HSwish

class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride=1,padding=0,groups=1,act=None):
        super().__init__()
        self.conv = nn.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def load_state_dict(self,_3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv.weight'] = torch.Tensor(_state[f'{_name_prefix}_weights'])
            if _name_prefix == 'conv1':
                _bn_name = f'bn_{_name_prefix}'
            else:
                bn_name = f'bn{_name_prefix[3:]}'
            to_load_state_dict['bn.weight'] = torch.Tensor(_state[f'{bn_name}_scale'])
            to_load_state_dict['bn.bias'] = torch.Tensor(_state[f'{bn_name}_offset'])
            to_load_state_dict['bn.running_mean'] = torch.Tensor(_state[f'{bn_name}_mean'])
            to_load_state_dict['bn.running_var'] = torch.Tensor(_state[f'{bn_name}_variance'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ConvBNACTWithPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, act=None):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size -1) //2,
                            groups=groups,
                            bias=False)
        self.bn = BatchNorm2d(out_channels)
        if act is None:
            self.act = None
        else:
            self.act = nn.ReLU()
    
    def load_3rd_state_dict(self, _3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv.weight'] = torch.Tensor(_state[f'{_name_prefix}_weights'])
            if _name_prefix == 'conv1':
                bn_name = f'bn_{_name_prefix}'
            else:
                bn_name = f'bn{_name_prefix[3:]}'
            to_load_state_dict['bn.weight'] = torch.Tensor(_state[f'{bn_name}_scale'])
            to_load_state_dict['bn.bias'] = torch.Tensor(_state[f'{bn_name}_offset'])
            to_load_state_dict['bn.running_mean'] = torch.Tensor(_state[f'{bn_name}_mean'])
            to_load_state_dict['bn.running_var'] = torch.Tensor(_state[f'{bn_name}_variance'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super.__init__()
        assert name is not None, 'shortcut must have name'

        self.name = name 
        if in_channels != out_channels or stride !=1:
            if if_first:
                self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=0, groups=1, act=None)
            else:
                self.conv = ConvBNACTWithPool(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              groups=1, act=None)
        elif if_first:
            self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=1, act=None)
        else:
            self.conv = None

    def load_3rd_state_dict(self,_3rd_name,_state):
        if _3rd_name == 'paddle':
            if self.conv:
                self.conv.load_3rd_state_dict(_3rd_name, _state, self.name)
        else:
            pass

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'bottleneck must have name'
        self.name = name
        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1,
                               padding=0, groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
                                 if_first=if_first, name=f'{name}_branch1')
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.conv0.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2a')
        self.conv1.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2b')
        self.conv2.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2c')
        self.shortcut.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'block must have name'
        self.name = name

        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                 name=f'{name}_branch1', if_first=if_first, )
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            self.conv0.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2a')
            self.conv1.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2b')
            self.shortcut.load_3rd_state_dict(_3rd_name, _state)
        else:
            pass

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)
    

