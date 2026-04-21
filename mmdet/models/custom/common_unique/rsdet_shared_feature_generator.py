import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class EdgeExtractor(BaseModule):

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        sobel_kernel = np.array(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 9
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
        self.conv_op.weight.data = torch.from_numpy(sobel_kernel)
        freeze(self.conv_op)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):
        img = self.bn(img)
        edge_map = self.conv_op(img)
        return self.relu(edge_map)


@MODELS.register_module()
class SharedFeatureGenerator(BaseModule):
    """Paper-aligned shared/common feature generator used by RSDet."""

    def __init__(self, loss_MI1, loss_MI2, strides, backbone, neck) -> None:
        super().__init__()
        del loss_MI1
        del loss_MI2
        self.edge_extractor = EdgeExtractor()
        self.strides = strides
        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)
        self.neck_vis = MODELS.build(neck)
        self.neck_lwir = MODELS.build(neck)

    def edge_fusion(self, img_vis_edge, img_lwir_edge):
        fused_edge = img_vis_edge + img_lwir_edge
        return fused_edge / fused_edge.std()

    def forward(self, img_vis, img_lwir):
        img_vis_edge = self.edge_extractor(img_vis)
        img_lwir_edge = self.edge_extractor(img_lwir)
        img_fused_edge = 0.05 * self.edge_fusion(img_vis_edge, img_lwir_edge)

        x_vis = self.neck_vis(self.backbone_vis(img_vis))
        x_lwir = self.neck_lwir(self.backbone_lwir(img_lwir))

        fused_edge_pyramid = []
        for stride in self.strides:
            fused_edge_pyramid.append(
                F.interpolate(img_fused_edge, scale_factor=1 / stride, mode='bicubic'))

        shared_features = []
        for level in range(len(x_vis)):
            shared_feature = 0.5 * (x_vis[level] + x_lwir[level])
            shared_feature = torch.cat(
                [shared_feature, fused_edge_pyramid[level]], dim=1)
            shared_features.append(shared_feature)
        return tuple(shared_features)
