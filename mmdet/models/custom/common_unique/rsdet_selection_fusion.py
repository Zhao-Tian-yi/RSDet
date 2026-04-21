import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


class SelectionGate(nn.Module):

    def __init__(self, imgshape):
        super().__init__()
        img_h, img_w = imgshape
        self.flatten = nn.Flatten()
        self.pool = nn.AvgPool2d(kernel_size=8)
        self.conv1 = nn.Conv2d(512, 1, 1, 1, 0)
        self.conv2 = nn.Conv2d(1024, 1, 1, 1, 0)
        self.conv3 = nn.Conv2d(2048, 1, 1, 1, 0)
        self.conv4 = nn.Conv2d(4096, 1, 1, 1, 0)
        self.IA_fc1_1 = nn.Linear(
            in_features=max(1, (img_h // 32) * (img_w // 32)), out_features=2)
        self.IA_fc2_1 = nn.Linear(
            in_features=max(1, (img_h // 64) * (img_w // 64)), out_features=2)
        self.IA_fc3_1 = nn.Linear(
            in_features=max(1, (img_h // 128) * (img_w // 128)),
            out_features=2)
        self.IA_fc4_1 = nn.Linear(
            in_features=max(1, (img_h // 256) * (img_w // 256)),
            out_features=2)
        self.IA_fc5_1 = nn.Linear(
            in_features=max(1, (img_h // 256) * (img_w // 256)),
            out_features=2)

    def forward(self, x_vis, x_lwir):
        x1_1 = self.conv1(self.pool(torch.cat([x_vis[0], x_lwir[0]], dim=1)))
        x1_2 = self.conv2(self.pool(torch.cat([x_vis[1], x_lwir[1]], dim=1)))
        x1_3 = self.conv3(self.pool(torch.cat([x_vis[2], x_lwir[2]], dim=1)))
        x1_4 = self.conv4(self.pool(torch.cat([x_vis[3], x_lwir[3]], dim=1)))

        x2_1 = self.flatten(x1_1)
        x2_2 = self.flatten(x1_2)
        x2_3 = self.flatten(x1_3)
        x2_4 = self.flatten(x1_4)

        weights = [
            self.IA_fc1_1(x2_1),
            self.IA_fc2_1(x2_2),
            self.IA_fc3_1(x2_3),
            self.IA_fc4_1(x2_4),
            self.IA_fc5_1(x2_4),
        ]
        return torch.stack(weights, dim=0)


@MODELS.register_module()
class DynamicFeatureSelection(BaseModule):
    """Paper-aligned dynamic feature selection module used by RSDet."""

    def __init__(self,
                 loss_MI,
                 feature_nums,
                 num_ins,
                 channel_nums,
                 scale,
                 num_gate,
                 imgshape,
                 neck,
                 start_level: int = 0,
                 end_level: int = -1) -> None:
        super().__init__()
        del feature_nums
        del num_ins
        del channel_nums
        del scale
        del num_gate
        del start_level
        del end_level
        self.selection_gate = SelectionGate(imgshape)
        self.rgb_expert = MODELS.build(neck)
        self.ir_expert = MODELS.build(neck)
        self.mi_loss_level1 = MODELS.build(loss_MI)
        self.mi_loss_level2 = MODELS.build(loss_MI)

    def forward(self, x_vis, x_lwir, x_common, img_vis, img_lwir):
        del img_vis
        del img_lwir

        weights = self.selection_gate(x_vis, x_lwir)
        weights = F.softmax(weights, dim=2)

        thresholded_weights = torch.where(
            weights < 0.05,
            torch.zeros_like(weights),
            weights)
        gate = thresholded_weights.clone()
        gate[:, :, 1] = torch.where(
            thresholded_weights[:, :, 0] == 0,
            torch.ones_like(gate[:, :, 1]),
            gate[:, :, 1])
        gate[:, :, 0] = torch.where(
            thresholded_weights[:, :, 1] == 0,
            torch.ones_like(gate[:, :, 0]),
            gate[:, :, 0])

        x_vis_exclusive = self.rgb_expert(x_vis)
        x_lwir_exclusive = self.ir_expert(x_lwir)

        mi_loss_vis = self.mi_loss_level1(
            x_vis_exclusive[1], x_common[1]) / (
                x_vis_exclusive[1].shape[2] * x_vis_exclusive[1].shape[3])
        mi_loss_vis += self.mi_loss_level2(
            x_vis_exclusive[2], x_common[2]) / (
                x_vis_exclusive[2].shape[2] * x_vis_exclusive[2].shape[3])

        mi_loss_lwir = self.mi_loss_level1(
            x_lwir_exclusive[1], x_common[1]) / (
                x_vis_exclusive[1].shape[2] * x_vis_exclusive[1].shape[3])
        mi_loss_lwir += self.mi_loss_level2(
            x_lwir_exclusive[2], x_common[2]) / (
                x_vis_exclusive[2].shape[2] * x_vis_exclusive[2].shape[3])

        unique_feature_fusion = []
        for level in range(len(x_vis_exclusive)):
            unique_feature_fusion.append(
                gate[level][:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) *
                x_vis_exclusive[level] +
                gate[level][:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) *
                x_lwir_exclusive[level])
        unique_feature_fusion = tuple(unique_feature_fusion)

        fused_features = []
        for level in range(len(x_common)):
            fused_features.append(
                0.7 * x_common[level] + 0.3 * unique_feature_fusion[level])
        return tuple(fused_features), mi_loss_vis, mi_loss_lwir
