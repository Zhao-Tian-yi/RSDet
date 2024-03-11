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
        # import pdb
        # pdb.set_trace()
        mask_vis_list = self.Gmaskbinarylist_vis(fre_m_vis)
        mask_lwir_list = self.Gmaskbinarylist_lwir(fre_m_lwir)

        mask_vis_list_topk ,topk_index_vis = mask_vis_list.topk(320)
        mask_lwir_list_topk,topk_index_lwir = mask_lwir_list.topk(320)

        mask_vis_list_min = torch.min(mask_vis_list_topk, dim=1).values
        mask_lwir_list_min = torch.min(mask_lwir_list_topk, dim=1).values

        # repeat里的4和x的最后一维相同
        mask_vis_list_min_= mask_vis_list_min.unsqueeze(-1)
        mask_lwir_list_min_= mask_lwir_list_min.unsqueeze(-1)
        ge_vis = torch.ge(mask_vis_list, mask_vis_list_min_)
        ge_lwir = torch.ge(mask_lwir_list, mask_lwir_list_min_)
        # 设置zero变量，方便后面的where操作
        zero = torch.zeros_like(mask_vis_list)
        one = torch.ones_like(mask_vis_list)
        mask_vis_list = torch.where(ge_vis, one, zero).reshape((-1,1,self.patch_num,self.patch_num))
        mask_lwir_list = torch.where(ge_lwir, one, zero).reshape((-1,1,self.patch_num,self.patch_num))
        if self.keep_low:
            mask_vis_list[:, :, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1] = 1
            mask_lwir_list[:, :, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1] = 1

        else:
            mask_vis_list[:, :, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1] = 0
            mask_lwir_list[:, :, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1, int(self.patch_num/2) - 1:int(self.patch_num/2) + 1] = 0
        mask_vis = F.interpolate(mask_vis_list,
                                 scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num],
                                 mode='nearest')
        mask_lwir = F.interpolate(mask_lwir_list,
                                 scale_factor=[self.imgshape[0] / self.patch_num, self.imgshape[1] / self.patch_num],
                                 mode='nearest')
        # import cv2
        # import numpy as np
        # cv2.imwrite('/home/zhangguiwei/KK/codes/mmdet3-spectral/mmdet/test_vis.png',np.array(255*mask_vis[0].permute(1,2,0).cpu()))
        # cv2.imwrite('/home/zhangguiwei/KK/codes/mmdet3-spectral/mmdet/test_lwir.png',
        #             np.array(255 * mask_lwir[0].permute(1, 2, 0).cpu()))
        return mask_vis, mask_lwir


if __name__ == '__main__':
    from PIL import Image
    import torch
    import cv2
    import numpy as np
    import torch

    # edge extract test
    data_root = '/home/zhangguiwei/KK/Datasets/FLIR_align/test/'
    save_root = '/home/zhangguiwei/KK/codes/mmdet3-spectral/mmdet/models/custom/common_unique/'
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
