cfg = dict(
    model_type='bisenetv2_boundary_refine',

    n_cats=3,
    num_aux_heads=4,

    lr_start=2.5e-3,
    weight_decay=5e-4,

    warmup_iters=1000,
    max_iter=80000,
    save_interval=5000,

    dataset='RUGD3Dataset',

    im_root='./datasets/rugd3',
    train_im_anns='./datasets/rugd3/train.txt',
    val_im_anns='./datasets/rugd3/test.txt',

    scales=[0.5, 2.0],
    cropsize=[512, 640],

    eval_crop=[512, 640],
    eval_scales=[1.0],

    ims_per_gpu=16,
    eval_ims_per_gpu=2,

    use_fp16=True,
    use_sync_bn=False,

    respth=(
        './experiments/'
        'rugd3_boundary_refine_b16_seed123_formal'
    ),
)