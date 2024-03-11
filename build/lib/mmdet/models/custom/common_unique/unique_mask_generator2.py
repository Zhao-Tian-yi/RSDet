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

class UNetBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(mid_c),
            nn.PReLU(),
            nn.Conv2d(mid_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, imgshape, patch_num=20):
        super(UNet, self).__init__()
        n_atoms = 6  # 2^N
        W,H = imgshape
        nfs = [n_atoms, n_atoms * 2, n_atoms * 4, n_atoms * 8]
        self.patch_num = patch_num
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.initial = nn.Conv2d(3, nfs[0] // 2, 3, 1, 1)

        self.conv0_0 = UNetBlock(3, nfs[0], nfs[0])
        self.conv1_0 = UNetBlock(nfs[0], nfs[1], nfs[1])
        self.conv2_0 = UNetBlock(nfs[1], nfs[2], nfs[2])
        self.conv3_0 = UNetBlock(nfs[2], nfs[3], nfs[3])

        self.conv0_1 = UNetBlock(nfs[0] + nfs[1], nfs[0], nfs[0])
        self.conv1_1 = UNetBlock(nfs[1] + nfs[2], nfs[1], nfs[1])
        self.conv2_1 = UNetBlock(nfs[2] + nfs[3], nfs[2], nfs[2])

        self.conv0_2 = UNetBlock(nfs[0] * 2 + nfs[1], nfs[0], nfs[0])
        self.conv1_2 = UNetBlock(nfs[1] * 2 + nfs[2], nfs[1], nfs[1])

        self.conv0_3 = UNetBlock(nfs[0] * 3 + nfs[1], nfs[0], nfs[0])

        self.final = nn.Conv2d(nfs[0] + nfs[0] // 2, 3, 5, 2, 2)
        self.flatten = nn.Flatten()

        self.trans2list = nn.Sequential(
            nn.Linear(int(3 * H / 2 * W / 2), 10000),
            nn.Linear(10000, 1000),
            nn.Linear(in_features=1000, out_features=np.power(self.patch_num, 2), bias=True),
            nn.Sigmoid())

    def forward(self, x):
        # Unet++
        xini = self.initial(x)  # [B,1,H,W]
        x0_0 = self.conv0_0(x)  # [B,n[0],H,W]
        x1_0 = self.conv1_0(self.pool(x0_0))  # [B,n[1],H/2,W/2]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  #

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x_f = self.final(torch.cat([x0_3, xini], 1))
        x_final = self.trans2list(self.flatten(x_f))
        return x_final


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
# class TinyGMask(nn.Module):
#     def __init__(self,):
#         super(TinyGMask, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)     #2,3,512,640
#         self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)    #2,3,100,124
#         self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)    #2,3,20,24
#
#         self.trans2mask = nn.Conv2d(64, 1, 1, 1, 0)
#
#     def forward(self, x):
#
#         x=self.conv1(x)
#         x=self.conv2(x)
#         x=self.conv3(x)
#
#         x_final = self.trans2mask(x)
#         return x_final

class GMaskBinaryList(nn.Module):
    def __init__(self,
                 imgshape
                 ):
        super( GMaskBinaryList, self).__init__()
        # self.g_mask_binary_list = UNet(imgshape)
        self.g_mask_binary_list = TinyGMask(imgshape)
    def forward(self, x):
        mask_list = self.g_mask_binary_list(x)
        return mask_list


@MODELS.register_module()
class UniqueMaskGenerator2(BaseModule):
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
        super(UniqueMaskGenerator2, self).__init__()
        self.patch_num = patch_num
        self.keep_low  = keep_low
        self.imgshape=imgshape
        self.Gmaskbinarylist_vis = GMaskBinaryList(imgshape)
        self.Gmaskbinarylist_lwir = GMaskBinaryList(imgshape)

    def forward(self, img_vis, img_lwir):
        """Forward function."""

        mask_vis_list = self.Gmaskbinarylist_vis(img_vis)
        mask_lwir_list = self.Gmaskbinarylist_lwir(img_lwir)

        mask_vis_list_topk ,topk_index_vis = mask_vis_list.topk(300)
        mask_lwir_list_topk,topk_index_lwir = mask_lwir_list.topk(300)

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
