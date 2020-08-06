#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import bisenetv2.transform_cv2 as T
from sampler import RepeatedDistSampler


labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
]


class CityScapes(Dataset):
    '''
    '''
    def __init__(self, datapth, trans_func=None, mode='train'):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func
        self.n_cats = 19
        self.lb_ignore = 255

        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(datapth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(datapth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth, lbpth = self.imgs[fn], self.labels[fn]
        img, label = cv2.imread(impth), cv2.imread(lbpth, 0)
        label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self):
        self.trans_func = T.Compose([
            #  T.RandomResizedCrop([0.375, 1.], [512, 1024]),
            T.RandomResizedCrop([0.25, 2], [512, 1024]),
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
        #  im = cv2.resize(im, (1024, 512))
        return dict(im=im, lb=lb)


def get_data_loader(datapth, ims_per_gpu, max_iter=None, mode='train', distributed=True):
    train_trans_func = TransformationTrain()
    val_trans_func = TransformationVal()
    if mode == 'train':
        trans_func = train_trans_func
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = val_trans_func
        batchsize = ims_per_gpu * 2
        shuffle = False
        drop_last = False

    ds = CityScapes(datapth, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
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



if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
