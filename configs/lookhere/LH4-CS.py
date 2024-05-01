_base_ = ['../_base_/datasets/cityscapes_768x768.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_80k.py']
custom_imports = dict(imports=['configs.lookhere'], allow_failed_imports=False)
crop_size = (768, 768)
checkpoint= 'configs/lookhere/weights/qsmllyjn.pth'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[255*0.485, 255*0.456, 255*0.406],
    std=[255*0.229, 255*0.224, 255*0.225],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained = checkpoint,
    backbone=dict(
        type='Alibi',
        img_size=crop_size,
        drop_path_rate=0.1,
        alibi_config=dict(global_slope=2,layer_slopes=[1.5,0.5],\
                                head_directions='corners-triangles-none'),
        ),
    decode_head=dict(
        type='FCNHead',
        in_channels=768,
        channels=768,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=False)
            ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(480, 480)),
    )

train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=4)
val_dataloader = dict(batch_size=1)