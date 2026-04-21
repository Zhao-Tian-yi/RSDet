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
        self.imgshape = imgshape
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2)     #input: 2,3
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)    #input: 2,16
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)    #input:  2,64
        # self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        # self.conv4_11 = nn.Conv2d(64, 1, 1, 1, 0)
        self.trans2list = nn.Sequential(
            # nn.Linear(in_features=int(H / 16 * W / 16), out_features=np.power(self.patch_num, 2), bias=False),
            nn.Linear(in_features=int(64 * H / 16 * W / 16), out_features=np.power(self.patch_num, 2), bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # # 第二个卷积层后同样使用ReLU激活函数和最大池化层
        x = self.conv1(x)                                       #input: 2,3,512,640
        # x = F.relu(x)
        # x = self.sigmoid(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) #input: 2,16,256,320
                                                                #input: 2,16,128,160
        x = self.conv2(x)
        # x = F.relu(x)
        # x = self.sigmoid(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) #input: 2,32,128,160
        x = self.conv3(x)                                      #input: 2,32,64,80
        # x = F.relu(x)
        # x = self.sigmoid(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) #input: 2,64,64,80
        # x = self.conv4_11(x)
        #input: 2,64,32,40

        # x = self.conv4(x)
        x_final = self.trans2list(self.flatten(x))                           #input: 2,1,32,40
        return x_final


import torch
import torch.nn.functional as F


class MaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_list_):
        mask_list_topk, topk_index = mask_list_.topk(300)
        mask_list_min = torch.min(mask_list_topk, dim=1).values
        mask_list_min_ = mask_list_min.unsqueeze(-1)
        ge = torch.ge(mask_list_, mask_list_min_)
        zero = torch.zeros_like(mask_list_)
        one = torch.ones_like(mask_list_)
        mask_list = torch.where(ge, one, zero)
        # mask_list = torch.where(ge, one, mask_list_)  #PartSoftMaskFunction
        # 保存用于反向传播的变量
        ctx.save_for_backward(mask_list_, mask_list_min_)
        return mask_list
        # return mask_list_.clone()                        #all soft
    @staticmethod
    def backward(ctx, grad_output):
        mask_list_, mask_list_min_ = ctx.saved_tensors
        grad_mask_list_ = grad_output.clone()
        # return grad_mask_list_ #PartSoftMaskFunction #all soft
        # 计算梯度
        grad_mask_list_[mask_list_ < mask_list_min_] = 0
        return grad_mask_list_


@MODELS.register_module()
class UniqueMaskGenerator4(BaseModule):
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
        super(UniqueMaskGenerator4, self).__init__()
        self.patch_num = patch_num
        self.keep_low  = keep_low
        self.imgshape=imgshape
        self.Gmaskbinarylist_vis = TinyGMask(imgshape)
        self.Gmaskbinarylist_lwir = TinyGMask(imgshape)
        self.MaskFun_vis = MaskFunction()
        self.MaskFun_lwir = MaskFunction()
    def forward(self, img_vis, img_lwir):
        """Forward function."""

        #### 频域
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
        mask_lwir_list = self.MaskFun_lwir.apply(mask_lwir_list_).reshape((-1, 1,self.patch_num , self.patch_num))
        # keep low
        mask_vis_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1
        mask_lwir_list[:, :, int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1,int(self.patch_num / 2) - 1:int(self.patch_num / 2) + 1] = 1

        mask_vis = F.interpolate(mask_vis_list, scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num], mode='nearest')
        mask_lwir = F.interpolate(mask_lwir_list, scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num], mode='nearest')

        return mask_vis, mask_lwir


if __name__ == '__main__':
    from PIL import Image
    import torch
    import cv2
    import numpy as np
    import torch

    # edge extract test
    data_root = 'data/FLIR_align/test/'
    save_root = 'debug_outputs/common_unique/'
    img_lwir_filename = 'FLIR_08865_PreviewData.jpeg'
    img_filename = 'FLIR_08865_RGB.jpg'
    im_lwir = cv2.imread(data_root + img_lwir_filename, flags=1)
    im = cv2.imread(data_root + img_filename, flags=1)

    fre = torch.fft.fft2(torch.tensor(im), dim=(0, 1))
    freq_view = torch.log(1 + torch.abs(fre))
    freq_view = (freq_view - freq_view.min()) / (freq_view.max() - freq_view.min()) * 255
    freq_view = torch.fft.fftshift(freq_view)
    cv2.imwrite(save_root + 'rgbfre_' + img_filename, np.array(freq_view))
    cv2.imwrite(save_root + img_filename, np.array(im))
    cv2.imwrite(save_root + img_lwir_filename, np.array(im_lwir))
