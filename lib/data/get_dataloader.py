import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler

from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.ade20k import ADE20k
from lib.data.customer_dataset import CustomerDataset
from .water_dataset import WaterDataset



def get_data_loader(cfg, mode='train'):
    if mode == 'train':
        trans_func = T.TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize = cfg.ims_per_gpu
        annpath = cfg.train_im_anns
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = T.TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    # 修改：支持WaterDataset
    if cfg.dataset == 'WaterDataset':
        ds = WaterDataset(
            im_root=cfg.im_root,
            annpath=annpath,
            trans_func=trans_func,
            mode=mode
        )
        # 添加BiSeNet需要的属性
        ds.n_cats = cfg.n_cats
        ds.lb_ignore = 255
    else:
        ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode)

    # 修改：Windows单卡模式，不使用分布式采样
    if dist.is_initialized() and dist.get_world_size() > 1:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not cfg.max_iter is None
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=0,  # Windows必须设为0
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,  # Windows必须设为0
            pin_memory=True,
        )
    return dl