_base_ = [
    '../_base_/models/densedepth.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/amy/code/depth-estimation/Monocular-Depth-Estimation-Toolbox'\
                +'/configs/simipu/simclr-r50-imagenet-standard.pth'),
                style='pytorch',
    ),
    decode_head=dict(
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )
