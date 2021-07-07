
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.cityscapes_cv2 import CityScapes
from lib.coco import CocoStuff



class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


def get_data_loader(cfg, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize = cfg.ims_per_gpu
        annpath = cfg.train_im_anns
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode)

    if distributed:
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
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl
