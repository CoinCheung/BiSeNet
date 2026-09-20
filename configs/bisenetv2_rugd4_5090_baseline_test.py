cfg = dict(
    model_type='bisenetv2',

    n_cats=4,
    num_aux_heads=4,

    # Batch 16 下的迁移学习起始值
    lr_start=2.5e-3,
    weight_decay=5e-4,

    warmup_iters=1000,

    # 冒烟时先改为 300
    # max_iter=300,
    max_iter=80000,

    dataset='RUGD4Dataset',

    im_root='./datasets/rugd4',
    train_im_anns='./datasets/rugd4/train.txt',
    val_im_anns='./datasets/rugd4/test.txt',

    scales=[0.5, 2.0],

    # RK3588 后续也统一使用该输入
    cropsize=[512, 640],

    eval_crop=[512, 640],
    eval_scales=[1.0],

    # RTX 5090 首先尝试 16
    ims_per_gpu=16,
    eval_ims_per_gpu=2,

    use_fp16=True,
    use_sync_bn=False,

    #
    save_interval=5000, 

    # respth='./experiments/rugd4_baseline_b16_seed123',
    respth='./experiments/rugd4_baseline_b16_seed123_formal'
)