import pdb
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS

from PIL import Image
import torch
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class TinyGMask(nn.Module):
    def __init__(self, imgshape, patch_num=20):
        super(TinyGMask, self).__init__()
        W,H = imgshape
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(3, 16, 7, 4, 3)     #2,3,512,640
        self.conv2 = nn.Conv2d(16, 32, 7, 4, 3)    #2,3,100,124
        self.conv3 = nn.Conv2d(32, 64, 7, 4, 3)    #2,3,20,24
        self.flatten = nn.Flatten()

        self.trans2list = nn.Sequential(            #b,400
            nn.Linear(int(64 * H / 64 * W / 64), 1000),
            # nn.Linear(5000, 1000),
            nn.Linear(in_features=1000, out_features=np.power(self.patch_num, 2), bias=True),
            nn.Sigmoid())

    def forward(self, x):
        # Unet++
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        # import pdb
        # pdb.set_trace()
        x_final = self.trans2list(self.flatten(x))
        return x_final


class GMaskBinaryList(nn.Module):
    def __init__(self,
                 imgshape
                 ):
        super( GMaskBinaryList, self).__init__()

        self.g_mask_binary_list = TinyGMask(imgshape)
    def forward(self, x):
        mask_list = self.g_mask_binary_list(x)
        return mask_list

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class MaskFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_list_):
        mask_list_topk ,topk_index = mask_list_.topk(320)
        mask_list_min = torch.min(mask_list_topk, dim=1).values
        mask_list_min_ = mask_list_min.unsqueeze(-1)
        ge = torch.ge(mask_list_, mask_list_min_)
        zero = torch.zeros_like(mask_list_)
        one = torch.ones_like(mask_list_)
        mask_list = torch.where(ge, one, zero)
        return mask_list

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


@MODELS.register_module()
class UniqueMaskGenerator3(BaseModule):
    """
    Args:
        patch_num (int): raw or column patch number
        keep_low (bool):
    """
    def __init__(self,
                 imgshape,
                 keep_low,
                 patch_num=20,
                 ):
        super(UniqueMaskGenerator3, self).__init__()
        self.patch_num = patch_num
        self.keep_low  = keep_low
        self.imgshape=imgshape
        self.Gmaskbinarylist_vis = GMaskBinaryList(imgshape)
        self.Gmaskbinarylist_lwir = GMaskBinaryList(imgshape)
        self.MaskFun_vis = MaskFunction()
        self.MaskFun_lwir = MaskFunction()
    def forward(self, img_vis, img_lwir):
        """Forward function."""

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)  # 幅度谱，求模得到
        fre_m_vis = torch.fft.fftshift(fre_m_vis)
        # fre_p_vis = torch.angle(vis_fre)  # 相位谱，求相角得到

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)  # 幅度谱，求模得到
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        # fre_p_lwir = torch.angle(lwir_fre)  # 相位谱，求相角得到
        mask_vis_list_ = self.Gmaskbinarylist_vis(fre_m_vis)
        mask_lwir_list_ = self.Gmaskbinarylist_lwir(fre_m_lwir)
        mask_vis_list = self.MaskFun_vis.apply(mask_vis_list_).reshape((-1, 1,self.patch_num , self.patch_num))
        mask_vis_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1
        mask_lwir_list = self.MaskFun_lwir.apply(mask_lwir_list_).reshape((-1, 1,self.patch_num , self.patch_num))
        mask_lwir_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1

        mask_vis = F.interpolate(mask_vis_list, scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num], mode='nearest')
        mask_lwir = F.interpolate(mask_lwir_list, scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num], mode='nearest')

        return mask_vis, mask_lwir

