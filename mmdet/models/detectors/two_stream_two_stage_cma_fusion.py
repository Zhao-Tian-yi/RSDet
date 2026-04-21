import pdb

import cv2
import torch
import torch.nn as nn
import torchvision
from .faster_rcnn import FasterRCNN
from mmengine.config import ConfigDict
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
# from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
# from mmengine.runner import auto_fp16
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConvBlock(nn.Module):
    def __init__(self, channels):
        super(DynamicConvBlock, self).__init__()
        self.convk1d1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, groups=channels, bias=False)
        self.convk3d1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.convk5d1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels, bias=False)
        self.convk7d1 = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels, bias=False)
        self.convk3d3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, groups=channels,
                                  bias=False)
        self.convk3d5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, groups=channels,
                                  bias=False)
        self.convk3d7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=7, dilation=7, groups=channels,
                                  bias=False)
        self.convk1 = nn.Conv2d(channels, channels // 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a):
        c1 = self.convk1d1(x)
        c2 = self.convk3d1(x)
        c3 = self.convk5d1(x)
        c4 = self.convk7d1(x)
        c5 = self.convk3d1(x)
        c6 = self.convk3d5(x)
        c7 = self.convk3d7(x)

        out = self.relu(x * a[0] + c1 * a[1] + c2 * a[2] + c3 * a[3] + c4 * a[4] + c5 * a[5] + c6 * a[6] + c7 * a[7])
        return self.convk1(out)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv2d(out)
        # out = self.sigmoid(out)
        out = self.tanh(out) + 1
        return out


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(CrossModalAttentionBlock, self).__init__()
        self.ca = ChannelAttentionModule(channels * 3, 8)
        self.sa = SpatialAttentionModule()
        self.dc = DynamicConvBlock(channels * 3)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2, x1 + x2), dim=1)
        a = self.ca(out).view(-1)
        out = self.dc(out, a)
        out = self.sa(out) * out
        return out

class CrossModalGlobalContextBlock(nn.Module):
    def __init__(self, inplanes=256, ratio=16):
        super(CrossModalGlobalContextBlock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes // ratio)
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_mul_conv1 = nn.Sequential(
                                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        self.channel_mul_conv2 = nn.Sequential(
                                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=self.inplanes*2, out_channels=self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)

    def spatial_pool(self, x1, x2):
        batch, channel, height, width = x1.size()
        # [N, C, H * W]
        x1 = x1.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        x1 = x1.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x2)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(x1, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x1, x2):
        # [N, C, 1, 1]
        context1 = self.spatial_pool(x1, x2)
        context2 = self.spatial_pool(x2, x1)

        out1 = x1 * torch.sigmoid(self.channel_mul_conv1(context1))
        out2 = x2 * torch.sigmoid(self.channel_mul_conv2(context2))
        out = torch.cat([out1, out2], dim=1)

        avgout = torch.mean(out, dim=1, keepdim=True)
        maxout, _ = torch.max(out, dim=1, keepdim=True)
        mask = self.conv2d(torch.cat([avgout, maxout], dim=1))
        mask = self.sigmoid(mask)
        out = self.conv(out) * mask

        return out

@MODELS.register_module()
class TwoStreamTwoStageCMAFusionDetector(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.backbone = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)
        # self.fusion_module = CrossModalGlobalContextBlock()
        self.fusion_module = nn.ModuleList([CrossModalAttentionBlock(256),
                                            CrossModalAttentionBlock(512),
                                            CrossModalAttentionBlock(1024),
                                            CrossModalAttentionBlock(2048,)])
        if neck is not None:
            self.neck = MODELS.build(neck)
            # self.neck_lwir = MODELS.build(neck)
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        # if self.with_neck:
        #     x = self.neck(x)
        return x
    def extract_visfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        return x
    def extract_lwirfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_lwir(img)
        # if self.with_neck:
        #     x = self.neck_lwir(x)
        return x
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()

        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        x_vis = self.extract_visfeat(img_vis)
        x_lwir = self.extract_lwirfeat(img_lwir)
        x = []
        # 两个流合成一个
        for i in range(len(x_vis)):
            x.append(self.fusion_module[i](x_vis[i] , x_lwir[i]))
        x = tuple(x)
        x = self.neck(x)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        x_vis = self.extract_visfeat(img_vis)
        x_lwir = self.extract_lwirfeat(img_lwir)
        x = []
        # 两个流合成一个
        for i in range(len(x_vis)):
            x.append(self.fusion_module[i](x_vis[i] , x_lwir[i]))
        x = tuple(x)
        x = self.neck(x)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'

        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']

        x_vis = self.extract_visfeat(img_vis)
        x_lwir = self.extract_lwirfeat(img_lwir)

        x = []
        # 两个流合成一个
        for i in range(len(x_vis)):
            x.append(self.fusion_module[i](x_vis[i] , x_lwir[i]))
        x = tuple(x)
        x = self.neck(x)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
