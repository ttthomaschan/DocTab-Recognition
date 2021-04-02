from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import torch
from torch import nn
from torchocr.networks.CommonModules import ConvBNACT, SEBlock


class ResidualUnit(nn.modules):
    def __init__(self, num_in_filter,num_mid_filter,num_out_filter,stride,kernel_size,act=None,use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter,out_channels=num_mid_filter,kernel_size=1,stride=1,padding=0,act=act)
        self.conv1 = ConvBNACT(in_channels=num_mid_filter,out_channels=num_mid_filter,kernel_size=kernel_size,stride=stride,padding=int((kernel_size-1)//2),act=act,groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter,out_channels=num_mid_filter)
        else:
            self.se = None
        self.conv2 = ConvBNACT(in_channels=num_mid_filter,out_channels=num_out_filter,kernel_size=1,stride=1,padding=0)
        self.not_add = num_in_filter != num_out_filter or stride !=1
        
    def load_3rd_state_dict(self, _3rd_name, _state, _convolution_index):
        if _3rd_name == 'paddle':
            self.conv0.load_3rd_state_dict(_3rd_name,_state,f'conv{_convolution_index}_expand')
            self.conv1.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_depthwise')
            if self.se is not None:
                self.se.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_se')
            self.conv2.load_3rd_state_dict(_3rd_name, _state, f'conv{_convolution_index}_linear')
        else:
            pass
        pass

    def forward(self,x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)
        if not self.not_add:
            y = x + y
        return y
    
class MobileNetV3(nn.Module):
    def __init__(self,in_channels,pretrained=True,**kwargs):
        
