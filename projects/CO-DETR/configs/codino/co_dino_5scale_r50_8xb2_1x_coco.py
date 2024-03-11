_base_ = './co_dino_5scale_r50_lsj_8xb2_1x_coco.py'
dataset_type = 'MultispectralDataset'
classes = ('car', 'person', 'bicycle')
device = 'cpu'
model = dict(
    use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))
data_root = '/home/zhangguiwei/KK/Datasets/FLIR_align/'
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadLwirImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(512, 640), keep_ratio=True),
    # dict(
    #     type='RandomChoice',
    #     transforms=[
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ],
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 # The radio of all image in train dataset < 7
    #                 # follow the original implement
    #                 scales=[(400, 4200), (500, 4200), (600, 4200)],
    #                 keep_ratio=True),
    #             dict(
    #                 type='RandomCrop',
    #                 crop_type='absolute_range',
    #                 crop_size=(384, 600),
    #                 allow_negative_crop=True),
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ]
    #     ]),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'img_lwir_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotation_train.json',
        data_prefix=dict(img='train/'),
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

test_pipeline = [
    dict(type='LoadLwirImageFromFile', backend_args=_base_.backend_args),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale=(512, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'img_lwir_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Annotation_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=_base_.backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'Annotation_test.json',
    metric=['bbox'],
    format_only=False,
    backend_args=_base_.backend_args)
test_evaluator = val_evaluator