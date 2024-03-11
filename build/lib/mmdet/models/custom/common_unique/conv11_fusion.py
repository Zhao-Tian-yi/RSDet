import warnings
import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from mmdet.utils import OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule
import os, sys

import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

        if self.pad_model == None:
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding,
                                        bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,
                                        bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class _Gate(nn.Module):
    def __init__(self, feature_nums, channel_nums, scale,num_gate,imgshape,num_ins):
        super(_Gate, self).__init__()
        W,H = imgshape
        self.feature_nums = feature_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.num_gate = num_gate

        self.gates_conv1 = nn.ModuleList()
        self.gates_conv2 = nn.ModuleList()
        self.gates_conv3 = nn.ModuleList()
        self.gates_fc = nn.ModuleList()
        self.flatten = nn.Flatten()

        for i in range(num_ins):
            gate_conv1 = ConvBlock(self.channel_nums[i],
                               int(self.channel_nums[i]/256*scale[i]), 3, 1, 1, activation=None, norm=None,
                               bias=False)  # common feature fusion
            self.gates_conv1.append(gate_conv1)
        for i in range(num_ins):
            gate_conv2 = ConvBlock(self.channel_nums[i],
                               int(self.channel_nums[i]/256*scale[i]), 3, 1, 1, activation=None, norm=None,
                               bias=False)  # common feature fusion
            self.gates_conv2.append(gate_conv2)
        for i in range(num_ins):
            gate_conv3 = ConvBlock(self.channel_nums[i],
                               int(self.channel_nums[i]/256*scale[i]), 3, 1, 1, activation=None, norm=None,
                               bias=False)  # common feature fusion
            self.gates_conv3.append(gate_conv3)
        for i in range(feature_nums):
            gate_fc = nn.Linear(int(num_ins*self.channel_nums[0]/256*scale[0]*W/scale[0]*H/scale[0]), self.num_gate) # common feature fusion
            self.gates_fc.append(gate_fc)
        # self.gates_fc = nn.Linear(int(num_ins*self.channel_nums[i]/256*scale[i]*W/scale[i]*H/scale[i]), self.num_gate),


    def forward(self, x_vis, x_lwir, x_common):
        x_v_list = []
        x_l_list = []
        x_c_list = []
        gates = []
        for i,(gate_conv1,gate_conv2,gate_conv3) in enumerate(zip(self.gates_conv1,self.gates_conv2,self.gates_conv3)):
            x_v_list.append(self.flatten(gate_conv1(x_vis[i])))
            x_l_list.append(self.flatten(gate_conv2(x_lwir[i])))
            x_c_list.append(self.flatten(gate_conv3(x_common[i])))

        gates.append(self.gates_fc[0](torch.cat(x_v_list,1)))
        gates.append(self.gates_fc[1](torch.cat(x_l_list,1)))
        gates.append(self.gates_fc[2](torch.cat(x_c_list,1)))
        gates = torch.stack(gates,dim=0)

        return gates


@MODELS.register_module()
class Conv11_Fusion(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    注：其实看作一个门控（或者是feature-wise的dropout)

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(
            self,
            feature_nums,
            num_ins,
            channel_nums,
            scale,
            num_gate,
            imgshape,
            neck: OptConfigType,
            start_level: int = 0,
            end_level: int = -1,

    ) -> None:
        super(Conv11_Fusion, self).__init__()

        self.feature_nums = feature_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.num_gate =num_gate

        self.Gate = _Gate(self.feature_nums, self.channel_nums, self.scale,self.num_gate, imgshape,num_ins)
        self.expert_vis = MODELS.build(neck)
        self.expert_lwir = MODELS.build(neck)
        self.expert_fusion = MODELS.build(neck)

    def forward(self, x_vis, x_lwir, x_common):
        """Forward function."""
        gate = self.Gate(x_vis, x_lwir, x_common)

        gate = F.softmax(gate, dim=0)
        x_vis = self.expert_vis(x_vis)
        x_lwir = self.expert_lwir(x_lwir)
        x_common = self.expert_fusion(x_common)

        outs = []
        for i in range(len(x_vis)):
            outs.append(gate[0,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_common[i] +
                        gate[1,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_vis[i] +
                        gate[2,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_lwir[i])
        outs = tuple(outs)
        for i in range(len(x_vis)):
            outs.append(gate[0,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_common[i] +
                        gate[1,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_vis[i] +
                        gate[2,:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * x_lwir[i])
        outs = tuple(outs)
        return outs