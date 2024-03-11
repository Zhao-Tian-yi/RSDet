import pdb

import cv2
import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
# from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
# from mmengine.runner import auto_fp16
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class TwoStreamTwoStageSimpleFusionDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStreamTwoStageSimpleFusionDetector, self).__init__()
        # add by yuanmaoxun
        self.backbone_vis = build_backbone(backbone)
        self.backbone_lwir = build_backbone(backbone)  # 完全相等的两个 backbone

        if neck is not None:
            self.neck_vis = build_neck(neck)
            self.neck_lwir = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck_vis') and hasattr(self, 'neck_lwir') and self.neck_vis is not None and self.neck_lwir is not None

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(TwoStreamTwoStageSimpleFusionDetector, self).init_weights(pretrained)
        # add by yuanmaoxun
        self.backbone_vis.init_weights(pretrained=pretrained)
        print("load vis backbone end!")
        self.backbone_lwir.init_weights(pretrained=pretrained)
        print("load lwir backbone end!")
        if self.with_neck:
            if isinstance(self.neck_vis, nn.Sequential):
                for m in self.neck_vis:
                    m.init_weights()
            else:
                self.neck_vis.init_weights()
            # add by yuan
            if isinstance(self.neck_lwir, nn.Sequential):
                for m in self.neck_lwir:
                    m.init_weights()
            else:
                self.neck_lwir.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    # add by yuanmaoxun
    def extract_visfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_vis(img)
        if self.with_neck:
            x = self.neck_vis(x)
        return x
    def extract_lwirfeat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_lwir(img)
        if self.with_neck:
            x = self.neck_lwir(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_lwir,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add by yuanmaoxun
        vis_x = self.extract_visfeat(img)
        lwir_x = self.extract_lwirfeat(img_lwir)
        x = []
        # 两个流合成一个
        for i in range(len(vis_x)):
            x.append(0.5 * (vis_x[i] + lwir_x[i]))
        x = tuple(x)


        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

            
           
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        #my_test = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=False) 
        #print("in two stage roi_losses:", len(my_test))
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_lwir,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        vis_x = self.extract_visfeat(img)
        lwir_x = self.extract_lwirfeat(img_lwir)
        x = []
        for i in range(len(vis_x)):
            x.append(0.5 * (vis_x[i] + lwir_x[i]))
        x = tuple(x)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_lwir, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        # img = img[0].cpu().numpy().transpose(1,2,0)
        # cv2.imwrite('img.jpg',img)
        # img_lwir = img_lwir[0].cpu().numpy().transpose(1, 2, 0)
        # cv2.imwrite('img_lwir.jpg', img_lwir)
        # pdb.set_trace()  
        vis_x = self.extract_visfeat(img)  # 经过 feture pyramid 的多尺度特征
        lwir_x = self.extract_lwirfeat(img_lwir)

        x = []
        for i in range(len(vis_x)):
            x.append(0.5*(vis_x[i] + lwir_x[i]))
        x = tuple(x)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, imgs_lwir, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        vis_x = self.extract_visfeat(imgs)
        lwir_x = self.extract_lwirfeat(imgs_lwir)
        x = []
        for i in range(len(vis_x)):
            x.append(0.5 * (vis_x[i] + lwir_x[i]))
        x = tuple(x)

        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    # add by yuanmaoxun
    def forward_test(self, imgs, imgs_lwir, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'),(imgs_lwir, 'imgs_lwir'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], imgs_lwir[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, imgs_lwir, img_metas, **kwargs)

    # @auto_fp16(apply_to=('img', 'img_lwir', ))
    def forward(self, img, img_lwir, img_metas, return_loss=True, **kwargs):  # train or predict forward()
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_lwir, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_lwir, img_metas, **kwargs)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(TwoStreamTwoStageSimpleFusionDetector, self).show_result(data, result, **kwargs)