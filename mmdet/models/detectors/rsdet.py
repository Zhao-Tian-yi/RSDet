import copy
from typing import List, Union

import torch
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stream_faster_rcnn import TwoStreamFasterRCNN


@MODELS.register_module()
class RSDet(TwoStreamFasterRCNN):
    """Paper-aligned open-source RSDet implementation based on the 15th core."""

    def __init__(self,
                 backbone: ConfigDict,
                 removal_module: ConfigDict,
                 shared_feature_generator: ConfigDict,
                 selection_fusion: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        del neck
        self.backbone = MODELS.build(backbone)
        self.backbone_lwir = MODELS.build(backbone)
        self.removal_module = MODELS.build(removal_module)
        self.shared_feature_generator = MODELS.build(shared_feature_generator)
        self.selection_fusion = MODELS.build(selection_fusion)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            if rpn_head_.get('num_classes', None) != 1:
                rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
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

    def extract_feat_vis(self, img: Tensor):
        return self.backbone(img)

    def extract_feat_lwir(self, img: Tensor):
        return self.backbone_lwir(img)

    @staticmethod
    def _complex_magnitude(freq_tensor: Tensor, eps: float = 1e-12) -> Tensor:
        return torch.sqrt(
            freq_tensor.real.square() + freq_tensor.imag.square() + eps)

    def _apply_frequency_mask(self, image: Tensor, freq_mask: Tensor) -> Tensor:
        freq = torch.fft.fft2(image)
        magnitude = self._complex_magnitude(freq)
        shifted_magnitude = torch.fft.fftshift(magnitude)
        masked_magnitude = torch.fft.ifftshift(shifted_magnitude * freq_mask)
        magnitude_scale = masked_magnitude / (magnitude + 1e-12)
        masked_freq = freq * magnitude_scale
        return torch.real(torch.fft.ifft2(masked_freq))

    def _extract_rsdet_features(self, batch_inputs: Tensor):
        img_vis = batch_inputs['img_vis']
        img_lwir = batch_inputs['img_lwir']
        mask_vis, mask_lwir = self.removal_module(img_vis, img_lwir)

        img_vis_unique = self._apply_frequency_mask(img_vis, mask_vis)
        img_lwir_unique = self._apply_frequency_mask(img_lwir, mask_lwir)

        x_vis = self.extract_feat_vis(img_vis_unique)
        x_lwir = self.extract_feat_lwir(img_lwir_unique)
        x_common = self.shared_feature_generator(img_vis_unique, img_lwir_unique)
        x, mi_loss_vis, mi_loss_lwir = self.selection_fusion(
            x_vis, x_lwir, x_common, img_vis, img_lwir)
        return x, mi_loss_vis, mi_loss_lwir

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        x, _, _ = self._extract_rsdet_features(batch_inputs)
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)
        return (roi_outs, )

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        losses = dict()
        x, mi_loss_vis, mi_loss_lwir = self._extract_rsdet_features(batch_inputs)
        losses.update({'loss_MI_vis': 0.001 * mi_loss_vis})
        losses.update({'loss_MI_lwir': 0.001 * mi_loss_lwir})

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            for key in list(rpn_losses.keys()):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        losses.update(self.roi_head.loss(x, rpn_results_list, batch_data_samples))
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        x, _, _ = self._extract_rsdet_features(batch_inputs)
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)
