_base_ = '../segmenter/segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512.py'

custom_imports = dict(imports=['configs.lookhere'], allow_failed_imports=False)
crop_size = (512, 512)
checkpoint= 'configs/lookhere/weights/kyyowg0l.pth'

data_preprocessor = dict(
    mean=[255*0.485, 255*0.456, 255*0.406],
    std=[255*0.229, 255*0.224, 255*0.225],
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained = checkpoint,
    backbone=dict(
        _delete_=True,
        type='Alibi',
        img_size=crop_size,
        drop_path_rate=0.1,
        alibi_config=dict(global_slope=2,layer_slopes=[1.5,0.5],\
                                head_directions='halves-diagonals-none'),
        ),
    decode_head=dict(
        in_channels=768,
        channels=768),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(200, 200)),
    )

train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=8)
val_dataloader = dict(batch_size=1)
