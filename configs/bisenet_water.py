cfg = dict(
    # 模型配置
    model_type='bisenetv2',      # 或 'bisenetv1'
    n_cats=2,                     # 水/非水 二分类
    num_aux_heads=2,              # bisenetv2=2, v1=1
    
    # 优化器
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=80000,                 # 总迭代次数
    
    # 数据集
    dataset='WaterDataset',
    im_root='./datasets/water_seg',
    train_im_anns='./datasets/water_seg/train.txt',
    val_im_anns='./datasets/water_seg/val.txt',
    
    # 数据增强（必须用列表！）
    scales=[0.5, 2.0],            # 随机缩放范围 [最小, 最大]
    cropsize=[512, 512],          # 训练裁剪尺寸 [高, 宽]
    
    # 评估配置
    eval_crop=[512, 512],         # 评估裁剪尺寸
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],  # 多尺度评估
    
    # 批次大小（注意是ims_per_gpu不是batch_size！）
    ims_per_gpu=4,                # 每张GPU的图像数（8GB显存建议2-4）
    eval_ims_per_gpu=2,           # 评估时的批次大小
    
    # 其他
    use_fp16=True,                # 混合精度
    use_sync_bn=False,            # Windows必须False
    respth='./res',               # 结果保存路径
)