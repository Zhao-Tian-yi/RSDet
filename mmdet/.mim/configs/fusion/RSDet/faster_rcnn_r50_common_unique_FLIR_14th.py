_base_ = [
    # '../_base_/models/two_stream_faster_rcnn_r50_fpn_FLIR.py',
    '../../_base_/datasets/FLIR.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
model = dict(
    type='RSDet_14th',
     data_preprocessor=dict(
        type='PairedDetDataPreprocessor',
        mean=[159.8808906080302, 162.22057018543336, 160.28301196773916],
        std=[56.96897676312916, 59.57937492901139, 63.11906486423505],
        mean_lwir=[136.63746562356317, 136.63746562356317, 136.63746562356317],
        std_lwir=[64.97730349740912, 64.97730349740912, 64.97730349740912],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='/home/yuanmaoxun/.cache/torch/hub/checkpoints/resnet50_cityscape.pth'),
    ),
    Gmask = dict(
        type='UniqueMaskGenerator3',
        imgshape = (512,640),
        keep_low  = True,
        # imgshape = (320, 256),
        patch_num=20,
    ),
    Gcommon = dict(
        type='CommonFeatureGenerator2',
        loss_MI1=dict(type='MutualInfoLoss',
                     input_channels=512,
                     ),
        loss_MI2=dict(type='MutualInfoLoss',
                      input_channels=1024,
                      ),
        strides=[4, 8, 16, 32, 64],  # 特征图感受野
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),  # [256, 512, 1024, 2048],
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='/home/yuanmaoxun/.cache/torch/hub/checkpoints/resnet50_cityscape.pth')
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
    ),
    FeaFusion = dict(
        type='Conv11_Fusion3',
        loss_MI=dict(type='MutualInfoLoss',
                     input_channels=259,
                     ),
        feature_nums=3,
        num_ins=4,
        channel_nums=[256, 512, 1024, 2048],
        scale = [4, 8, 16, 32],
        num_gate = 5 ,
        imgshape =(512, 640),

        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=259,
            num_outs=5),
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=259,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4,8],              # scale*strides决定了框的大小
            ratios=[0.5,1,2.0],
            strides=[4, 8, 16, 32,64]),#特征图感受野
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss',beta=1/9, loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=259,
            featmap_strides=[4, 8, 16, 32,64]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=259,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss',beta=1/9, loss_weight=1.0))),
    # model training and testing settings

    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.99),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=2000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
