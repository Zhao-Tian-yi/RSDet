import warnings
import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.transforms import Resize
from mmdet.registry import MODELS
from torch.autograd import Variable
import torch.nn.functional as F
import cv2


import os, sys


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False

class Extract_Edge(BaseModule):
    def __init__(self):
        super(Extract_Edge, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.conv_op =  nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 9
        self.sobel_kernel = self.sobel_kernel.reshape((1, 1, 3, 3))
        # 卷积输出通道，这里我设置为3
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=1)
        # 输入图的通道，这里我设置为3
        self.sobel_kernel = np.repeat(self.sobel_kernel, 3, axis=0)
        self.conv_op.weight.data = torch.from_numpy(self.sobel_kernel)
        freeze(self.conv_op)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,img):

        img = self.bn(img)
        edge_detect = self.conv_op(img)
        edge_detect = self.relu(edge_detect)
        return edge_detect

@MODELS.register_module()
class CommonFeatureGenerator2(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(
            self,
            loss_MI1,
            loss_MI2,
            strides,
            backbone,
            neck
    ) -> None:
        super(CommonFeatureGenerator2,self).__init__()
        num_channels =3
        self.MILoss1 = MODELS.build(loss_MI1)
        self.MILoss2 = MODELS.build(loss_MI2)
        self.EE = Extract_Edge()
        self.strides = strides

        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)
        self.neck_vis  = MODELS.build(neck)
        self.neck_lwir = MODELS.build(neck)
    def edge_fusion(self,img_vis_edge,img_lwir_edge):
        #simple fusion
        fused_edge = img_vis_edge + img_lwir_edge
        fused_edge = fused_edge/fused_edge.std()
        return fused_edge


    def forward(self, img_vis,img_lwir):
        """Forward function."""

        common_features = []
        #获取边缘特征
        img_vis_edge = self.EE(img_vis)
        img_lwir_edge = self.EE(img_lwir)
        img_fused_edge = self.edge_fusion(img_vis_edge,img_lwir_edge)


        x_vis_ = self.backbone_vis(img_vis)
        x_lwir_ = self.backbone_lwir(img_lwir)
        # x_lwir_ = self.backbone_lwir(img_lwir)
        img_fused_edge = 0.05*img_fused_edge


        miloss1 = self.MILoss1(x_vis_[1] ,x_lwir_[1])/(x_lwir_[1].shape[2]*x_lwir_[1].shape[3])
        miloss2 = self.MILoss2(x_vis_[2] ,x_lwir_[2])/(x_lwir_[2].shape[2]*x_lwir_[2].shape[3])
        MIloss =miloss1+miloss2

        x_vis = self.neck_vis(x_vis_)
        x_lwir = self.neck_lwir(x_lwir_)
        #TODO 加入边缘信息
        img_fused_edge_scale_list = []

        for stride in self.strides:
            img_fused_edge_down = F.interpolate(img_fused_edge, scale_factor=1/stride, mode='bicubic')
            img_fused_edge_scale_list.append(img_fused_edge_down)
        x_common_list = []

        for i in range(len(x_vis)):
            x_common = 0.5*(x_vis[i]+x_lwir[i])
            x_common = torch.cat([x_common,img_fused_edge_scale_list[i]],dim=1)
            x_common_list.append(x_common)

        common_features= tuple(x_common_list)

        return common_features,-MIloss



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