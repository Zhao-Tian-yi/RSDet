_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/M3FD_RGB.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    # './retinanet_tta.py'
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    # optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
)
train_dataloader = dict(
    batch_size=4,)