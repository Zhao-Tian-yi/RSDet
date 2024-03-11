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

#########可视化
import numpy as np
import matplotlib
import matplotlib.pyplot as plt_sne
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import datetime

# ,features, labels, epoch, fileNameDir=None
def tSNEvis(x_vis_exclusive,x_lwir_exclusive,x_common):
    x_vis_exclusive_ = torch.flatten(x_vis_exclusive[0].cpu()[0][:-3], start_dim=1, end_dim=2)
    x_lwir_exclusive_ = torch.flatten(x_lwir_exclusive[0].cpu()[0][:-3], start_dim=1, end_dim=2)
    x_common_ = torch.flatten(x_common[0].cpu()[0][:-3], start_dim=1, end_dim=2)
    features = torch.cat([x_vis_exclusive_, x_lwir_exclusive_,x_common_],dim=0)
    labels_vis = torch.zeros((256,))
    labels_lwir = torch.zeros((256,))+1
    labels_com = torch.zeros((256,))+2
    labels = torch.cat([labels_vis , labels_lwir,labels_com ],dim=0)
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''

    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=521)
    import seaborn as sns

    # 查看标签的种类有几个
    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4

    try:
        tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    except:
        tsne_features = tsne.fit_transform(features)

    x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    tsne_features = (tsne_features - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    # 一个类似于表格的数据结构
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = tsne_features[:, 0]
    df["comp2"] = tsne_features[:, 1]

    # 颜色是根据标签的大小顺序进行赋色.
    hex = ["#c957db", "#dd5f57", "#b9db57", "#57db30", "#5784db"]  # 绿、红
    data_label = []
    for v in df.y.tolist():
        if v == 0:
            data_label.append("vis_specific")
        elif v == 1:
            data_label.append("ir_specific")
        elif v == 2:
            data_label.append("shared")
        elif v == 3:
            data_label.append("c3")
        elif v == 4:
            data_label.append("c4")

    df["value"] = data_label

    # hue=df.y.tolist()
    # hue:根据y列上的数据种类，来生成不同的颜色；
    # style:根据y列上的数据种类，来生成不同的形状点；
    # s:指定显示形状的大小
    sns.scatterplot(x=df.comp1.tolist(), y=df.comp2.tolist(), hue=df.value.tolist(), style=df.value.tolist(),
                    palette=sns.color_palette(hex, class_num),
                    markers={"vis_specific": ".", "ir_specific": ".", "shared": ".", "c3": ".", "c4": "."},
                    # s = 10,
                    data=df).set(title="")  # T-SNE projection

    # 指定图注的位置 "lower right"
    plt_sne.legend(loc="lower right")
    plt_sne.xticks([])
    plt_sne.yticks([])
    # 不要坐标轴
    # plt_sne.axis("off")
    # 保存图像
    fileroot = '/home/zhangguiwei/KK/codes/mmdet3-spectral/tSNE_vis/noMI'
    # 创建目标文件夹
    if not os.path.exists(fileroot):
        os.makedirs(fileroot)
    time = datetime.datetime.now()
    plt_sne.savefig(os.path.join(fileroot , "%s.jpg") % str(time), format="jpg", dpi=900)
    plt_sne.cla()
    # plt_sne.show()

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
    def __init__(self,):
        super(_Gate, self).__init__()


        self.flatten = nn.Flatten()
        # FLIR KAIST
        self.IA_fc1_1 = nn.Linear(in_features=32*40, out_features = 100)
        self.IA_fc1_2 = nn.Linear(in_features=100, out_features = 2)
        self.IA_fc2_1 = nn.Linear(in_features=16*20, out_features = 100)
        self.IA_fc2_2 = nn.Linear(in_features=100, out_features = 2)
        self.IA_fc3_1 = nn.Linear(in_features=80, out_features = 2)
        self.IA_fc4_1 = nn.Linear(in_features=20, out_features = 2)
        self.pool = nn.AvgPool2d(kernel_size=4) #FLIR KAIST
        # llvip
        # self.IA_fc1_1 = nn.Linear(in_features=32*40*4, out_features = 100)
        # self.IA_fc1_2 = nn.Linear(in_features=100, out_features = 2)
        # self.IA_fc2_1 = nn.Linear(in_features=16*20*4, out_features = 100)
        # self.IA_fc2_2 = nn.Linear(in_features=100, out_features = 2)
        # self.IA_fc3_1 = nn.Linear(in_features=80*4, out_features = 2)
        # self.IA_fc4_1 = nn.Linear(in_features=20*4, out_features = 2)
        # self.pool = nn.AvgPool2d(kernel_size=8) #llvip

        self.conv1 = torch.nn.Conv2d(512, 1, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(1024, 1, 1, 1, 0)
        self.conv3 = torch.nn.Conv2d(2048, 1, 1, 1, 0)
        self.conv4 = torch.nn.Conv2d(4096, 1, 1, 1, 0)

    def forward(self, x_vis, x_lwir):

        x1_1= self.conv1(self.pool(torch.cat([x_vis[0], x_lwir[0]], dim=1)))
        x1_2= self.conv2(self.pool(torch.cat([x_vis[1], x_lwir[1]], dim=1)))
        x1_3= self.conv3(self.pool(torch.cat([x_vis[2], x_lwir[2]], dim=1)))
        x1_4= self.conv4(self.pool(torch.cat([x_vis[3], x_lwir[3]], dim=1)))

        x2_1 = self.flatten(x1_1)
        x2_2 = self.flatten(x1_2)
        x2_3 = self.flatten(x1_3)
        x2_4 = self.flatten(x1_4)

        x3_1 = self.IA_fc1_2(self.IA_fc1_1(x2_1))
        x3_2 = self.IA_fc2_2(self.IA_fc2_1(x2_2))
        x3_3 = self.IA_fc3_1(x2_3)
        x3_4 = self.IA_fc4_1(x2_4)

        weights = []
        weights.append(x3_1)
        weights.append(x3_2)
        weights.append(x3_3)
        weights.append(x3_4)
        weights = tuple(weights)

        return weights


@MODELS.register_module()
class Conv11_Fusion3(BaseModule):
    """Common Feature Mask Generator
        This is the implementation of paper '  '

    注：其实看作一个门控（或者是feature-wise的dropout)

    Args:
        img_vis (Tensor): The RGB input
        img_lwir (Tensor): The infrared input
    """

    def __init__(
            self,
            loss_MI,
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
        super(Conv11_Fusion3, self).__init__()

        self.feature_nums = feature_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.num_gate =num_gate

        self.Gate = _Gate()
        self.expert_vis = MODELS.build(neck)
        self.expert_lwir = MODELS.build(neck)
        self.MILoss1 = MODELS.build(loss_MI)
        self.MILoss2 = MODELS.build(loss_MI)

    def forward(self, x_vis, x_lwir, x_common,img_vis,img_lwir):
        """Forward function."""
        weights = self.Gate(x_vis, x_lwir)
        gate=[]

        gate.append(F.softmax(weights[0], dim=1))
        gate.append(F.softmax(weights[1], dim=1))
        gate.append(F.softmax(weights[2], dim=1))
        gate.append(F.softmax(weights[3], dim=1))
        gate = tuple(gate)
        x_vis_exclusive = self.expert_vis(x_vis)
        x_lwir_exclusive = self.expert_lwir(x_lwir)

        # 10th
        miloss1_vis = self.MILoss1(x_vis_exclusive[1] ,x_lwir_exclusive[1])/(x_lwir_exclusive[1].shape[2]*x_lwir_exclusive[1].shape[3])
        miloss2_vis = self.MILoss2(x_vis_exclusive[2] ,x_lwir_exclusive[2])/(x_lwir_exclusive[2].shape[2]*x_lwir_exclusive[2].shape[3])
        MIloss_vis =miloss1_vis+miloss2_vis
        miloss1_lwir = self.MILoss1(x_vis_exclusive[1] ,x_lwir_exclusive[1])/(x_lwir_exclusive[1].shape[2]*x_lwir_exclusive[1].shape[3])
        miloss2_lwir = self.MILoss2(x_vis_exclusive[2] ,x_lwir_exclusive[2])/(x_lwir_exclusive[2].shape[2]*x_lwir_exclusive[2].shape[3])
        MIloss_lwir =miloss1_lwir+miloss2_lwir
        #
        # # tSNEvis(x_vis_exclusive,x_lwir_exclusive,x_common)
        unique_feature_fusion=[]
        for i in range(len(x_vis_exclusive)):
            unique_feature_fusion.append(gate[i][:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_vis_exclusive[i]+gate[i][:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*x_lwir_exclusive[i])
        unique_feature_fusion = tuple(unique_feature_fusion)

        outs = []
        for i in range(len(x_common)):
            outs.append(0.7*x_common[i]+0.3*unique_feature_fusion[i])
        outs = tuple(outs)

        ###############################################################
        # 11th 测试MOE
        # outs = []
        # for i in range(len(x_vis_exclusive)):
        #     outs.append(0.33*x_vis_exclusive[i]+0.33*x_lwir_exclusive[i]+0.33*x_common[i])
        # outs = tuple(outs)

        return outs,MIloss_vis,MIloss_lwir,x_vis_exclusive,x_lwir_exclusive,unique_feature_fusion,gate
        # return outs, MIloss_vis, MIloss_lwir
