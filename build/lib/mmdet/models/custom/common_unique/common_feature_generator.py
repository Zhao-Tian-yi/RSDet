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


import os, sys

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

class ResnetBlock(nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu',
                 norm='batch', pad_model=None):
        super(ResnetBlock,self).__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale

        if self.norm == 'batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

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
        else:
            self.act = None

        if self.pad_model == None:
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None,
                        [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer,
                         self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out


class Extract_Scene_Semantic(BaseModule):
    """Extract Scene Semanti
    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """
    def __init__(self,
                 MILoss,
                 num_channels=3,
                 ):
        super(Extract_Scene_Semantic, self).__init__()
        # self.MILoss1= MODELS.build(MILoss)
        # self.MILoss2= MODELS.build(MILoss)

        self.CFF = ConvBlock(2*num_channels, num_channels, 3, 1, 1, activation='prelu', norm=None, bias=False) #common feature fusion
        n_resblocks = 1

        res_block_s1_lwir = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s1_lwir.append(ResnetBlock(32, 3, 1, 1, activation='prelu', norm=None))
        res_block_s1_lwir.append(ConvBlock(32, num_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s1_lwir = nn.Sequential(*res_block_s1_lwir)

        res_block_s2_lwir = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s2_lwir.append(ResnetBlock(32, 3, 1, 1, activation='prelu', norm=None))
        res_block_s2_lwir.append(ConvBlock(32, num_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s2_lwir = nn.Sequential(*res_block_s2_lwir)

        res_block_s1_vis = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s1_vis .append(ResnetBlock(32, 3, 1, 1, activation='prelu', norm=None))
        res_block_s1_vis .append(ConvBlock(32, num_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s1_vis  = nn.Sequential(*res_block_s1_vis )

        res_block_s2_vis = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(n_resblocks):
            res_block_s2_vis.append(ResnetBlock(32, 3, 1, 1, activation='prelu', norm=None))
        res_block_s2_vis .append(ConvBlock(32, num_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s2_vis  = nn.Sequential(*res_block_s2_vis )

    def forward(self,img_vis,img_lwir):

        s1_vis = self.res_block_s1_vis(img_vis)
        s1_lwir = self.res_block_s1_vis(img_lwir)
        # miloss1 = self.MILoss1(s1_vis ,s1_lwir)

        s2_vis = self.res_block_s2_vis(s1_vis)
        s2_lwir = self.res_block_s2_vis(s1_lwir)

        fused_common_feature = self.CFF(torch.cat([s2_vis,s2_lwir],dim=1))
        # miloss2 = self.MILoss1(s2_vis ,s2_lwir)
        # MILoss_value = miloss1+miloss2
        MILoss_value  = 0
        return fused_common_feature,MILoss_value


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class Extract_Edge(BaseModule):
    def __init__(self):
        super(Extract_Edge, self).__init__()
        self.conv_op =  nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 9
        self.sobel_kernel = self.sobel_kernel.reshape((1, 1, 3, 3))
        # 卷积输出通道，这里我设置为3
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=1)
        # 输入图的通道，这里我设置为3
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=0)
        self.conv_op.weight.data = torch.from_numpy(self.sobel_kernel)
        freeze(self.conv_op)

    def forward(self,img):
        edge_detect = self.conv_op(img)
        return edge_detect

@MODELS.register_module()
class CommonFeatureGenerator(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(
            self,
            loss_MI,
            method,
    ) -> None:
        super(CommonFeatureGenerator,self).__init__()
        num_channels =3
        self.MILoss = loss_MI
        self.ESS= Extract_Scene_Semantic(self.MILoss)
        self.EE = Extract_Edge()

        self.method = method

        self.CFF = ConvBlock(2 * num_channels, num_channels, 3, 1, 1, activation='prelu', norm=None,
                             bias=False)  # common feature fusion

    def edge_fusion(self,img_vis_edge,img_lwir_edge):
        #simple fusion
        fused_edge = img_vis_edge + img_lwir_edge
        fused_edge = fused_edge/fused_edge.std()
        return fused_edge


    def forward(self, img_vis,img_lwir):
        """Forward function."""

        common_features = []

        img_vis_edge = self.EE(img_vis)
        img_lwir_edge = self.EE(img_lwir)
        img_fused_edge = self.edge_fusion(img_vis_edge,img_lwir_edge)

        scene_semantic,MILoss_value = self.ESS.forward(img_vis,img_lwir)


        common_features.append(img_fused_edge)
        common_features.append(scene_semantic)
        if self.method == 'fusion':
            common_features = self.CFF(torch.cat([common_features[0],common_features[1]],dim=1))

            return common_features, MILoss_value
        else:
            return common_features,MILoss_value

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed."""
    #     super(CommonFeatureGenerator, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()


if __name__=='__main__':
    from PIL import Image

# edge extract test
    data_root = '/home/zhangguiwei/KK/Datasets/FLIR_align/test/'
    save_root = '/home/zhangguiwei/KK/data_preprocess/'
    img_filename = 'FLIR_08865_PreviewData.jpeg'
    im = cv2.imread(data_root+img_filename, flags=1)
    if len(im.shape)==3:
        im_tensor = np.transpose(im, (2, 0, 1))
    elif len(im.shape)==2:

        im = np.expand_dims(im, axis=2)
        im_tensor = np.transpose(im, (2, 0, 1))
    # 添加一个维度，对应于pytorch模型张量(B, N, H, W)中的batch_size
    im_tensor = im_tensor[np.newaxis, :]
    im_tensor = torch.Tensor(im_tensor)
    cfg = CommonFeatureGenerator()
    img_edge = cfg.extract_edge(im_tensor)
    cv2.imwrite(save_root+img_filename, np.array(im))
    cv2.imwrite(save_root+'edge_'+img_filename, img_edge )


# simple fusion test
    root = '/home/zhangguiwei/KK/data_preprocess/'
    file1 = 'edge_FLIR_08865_RGB.jpg'
    file2 =  'edge_FLIR_08865_PreviewData.jpeg'
    im1 = cv2.imread(root+file1, flags=1)
    im2 = cv2.imread(root+file2, flags=1)
    f= im1+im2
    cv2.imwrite(root+'f_'+file1, f)