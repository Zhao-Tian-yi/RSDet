# from ..builder import DETECTORS
import os
from .two_stream_two_stage_simple_fusion import TwoStreamTwoStageSimpleFusionDetector
from mmdet.registry import MODELS
from .faster_rcnn import FasterRCNN
from .two_stream_faster_rcnn import TwoStreamFasterRCNN
from mmengine.config import ConfigDict
import copy
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from .base import BaseDetector

import cv2
import datetime
import numpy as np
import torch.nn.functional as F

@MODELS.register_module()
class RSDet_14th(TwoStreamFasterRCNN):

    def __init__(self,
                 backbone: ConfigDict,
                 Gmask: ConfigDict,
                 Gcommon: ConfigDict,
                 FeaFusion: ConfigDict,
                 # FMOE: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 # neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            backbone=backbone,
            # neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.backbone_vis = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)

        self.Gmask=MODELS.build(Gmask)
        self.Gcommon=MODELS.build(Gcommon)
        self.FeaFusion=MODELS.build(FeaFusion)
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
        return x
    def extract_feat_vis(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_vis(img)
        return x
    def extract_feat_lwir(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_lwir(img)
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

        mask_vis, mask_lwir = self.Gmask(img_vis, img_lwir)
        # # EXTRACT_unique_feature

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)
        fre_m_vis = torch.fft.fftshift(fre_m_vis)
        fre_p_vis = torch.angle(vis_fre)
        masked_fre_m_vis = fre_m_vis * mask_vis
        masked_fre_m_vis = torch.fft.ifftshift(masked_fre_m_vis)
        fre_vis = masked_fre_m_vis * torch.e ** (1j * fre_p_vis)
        img_vis_unique = torch.real(torch.fft.ifft2(fre_vis))

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        fre_p_lwir = torch.angle(lwir_fre)
        masked_fre_m_lwir = fre_m_lwir * mask_lwir
        masked_fre_m_lwir = torch.fft.ifftshift(masked_fre_m_lwir)
        fre_lwir = masked_fre_m_lwir * torch.e ** (1j * fre_p_lwir)

        img_lwir_unique = torch.real(torch.fft.ifft2(fre_lwir))
        x_common, MI_loss = self.Gcommon(img_vis_unique, img_lwir_unique)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)

        x,_,_ = self.FeaFusion(x_vis, x_lwir, x_common,img_vis,img_lwir)

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
        losses = dict()

        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']
        mask_vis, mask_lwir = self.Gmask(img_vis, img_lwir)
        # # EXTRACT_unique_feature

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)  # 幅度谱，求模得到
        fre_m_vis = torch.fft.fftshift(fre_m_vis)
        fre_p_vis = torch.angle(vis_fre)  # 相位谱，求相角得到
        masked_fre_m_vis = fre_m_vis * mask_vis
        masked_fre_m_vis = torch.fft.ifftshift(masked_fre_m_vis)
        fre_vis = masked_fre_m_vis * torch.e ** (1j * fre_p_vis)
        img_vis_unique = torch.real(torch.fft.ifft2(fre_vis))

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        fre_p_lwir = torch.angle(lwir_fre)
        masked_fre_m_lwir = fre_m_lwir * mask_lwir
        masked_fre_m_lwir = torch.fft.ifftshift(masked_fre_m_lwir)
        fre_lwir = masked_fre_m_lwir * torch.e ** (1j * fre_p_lwir)

        img_lwir_unique = torch.real(torch.fft.ifft2(fre_lwir))
        x_common,_ = self.Gcommon(img_vis_unique, img_lwir_unique)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)

        x,MI_loss_vis, MI_loss_lwir = self.FeaFusion(x_vis, x_lwir, x_common,img_vis,img_lwir)
        MI_loss_vis = {'loss_MI_vis': 0.1 * MI_loss_vis}
        losses.update(MI_loss_vis)
        MI_loss_lwir = {'loss_MI_lwir': 0.1 * MI_loss_lwir}
        losses.update(MI_loss_lwir)
        # print(MI_loss_lwir,MI_loss_vis)
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
        mask_vis, mask_lwir = self.Gmask(img_vis, img_lwir)

        vis_fre = torch.fft.fft2(img_vis)
        fre_m_vis = torch.abs(vis_fre)
        fre_m_vis = torch.fft.fftshift(fre_m_vis)
        fre_p_vis = torch.angle(vis_fre)
        masked_fre_m_vis = fre_m_vis * mask_vis
        masked_fre_m_vis = torch.fft.ifftshift(masked_fre_m_vis)
        fre_vis = masked_fre_m_vis * torch.e ** (1j * fre_p_vis)
        img_vis_unique = torch.real(torch.fft.ifft2(fre_vis))

        lwir_fre = torch.fft.fft2(img_lwir)
        fre_m_lwir = torch.abs(lwir_fre)
        fre_m_lwir = torch.fft.fftshift(fre_m_lwir)
        fre_p_lwir = torch.angle(lwir_fre)
        masked_fre_m_lwir = fre_m_lwir * mask_lwir
        masked_fre_m_lwir = torch.fft.ifftshift(masked_fre_m_lwir)
        fre_lwir = masked_fre_m_lwir * torch.e ** (1j * fre_p_lwir)

        img_lwir_unique = torch.real(torch.fft.ifft2(fre_lwir))
        x_common, MI_loss = self.Gcommon(img_vis_unique, img_lwir_unique)
        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)
        x ,_,_= self.FeaFusion(x_vis, x_lwir, x_common, img_vis, img_lwir)

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