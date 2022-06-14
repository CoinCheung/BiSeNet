#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch
import torch.nn.functional as F



class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    @torch.no_grad()
    def __call__(self, im_lb):
        '''
        im should be CHW
        lb should be 1HW
        '''
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[-2:] == lb.shape[-2:]

        im, lb = im.unsqueeze(0), lb.unsqueeze(0)
        H, W = im.size()[2:]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = math.ceil(H * scale), math.ceil(W * scale)

        im = F.interpolate(im, (im_h, im_w), mode='bilinear', align_corners=False)
        lb = F.interpolate(lb, (im_h, im_w), mode='nearest')

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = F.pad(im, (pad_w, pad_w, pad_h, pad_h), (0, 0)))
            lb = F.pad(lb, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=255)

        im_h, im_w = im.size()[-2:]
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        im = im[0, :, sh:sh+crop_h, sw:sw+crop_w].clone()
        lb = lb[0, :, sh:sh+crop_h, sw:sw+crop_w].clone()
        return dict(im=im, lb=lb)



class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    @torch.no_grad()
    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[-2:] == lb.shape[-2:]
        return dict(
            im=im.flip(dims=-1),
            lb=lb.flip(dims=-1),
        )



class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    @torch.no_grad()
    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[-2:] == lb.shape[-2:]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return dict(im=im, lb=lb,)

    def adj_saturation(self, im, rate):
        M = torch.tensor([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ], dtype=torch.float32)
        im = torch.einsum('b...,bc->c...', im, M) / 3
        im = im.clip(0, 255).to(torch.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = torch.arange(256) * rate
        table = table.clip(0, 255).to(torch.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = (torch.arange(256) - 74) * rate + 74
        table = table.clip(0, 255).to(torch.uint8)
        return table[im]



class Numpy2Tensor(object):
    '''
        input:
            im should be HWC, type is np.array
            lb should be HW, type is np.array
        output:
            im is CHW, type is torch.tensor
            lb is 1HW, type is torch.tensor
    '''
    @torch.no_grad()
    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = torch.tensor(im).permute(2, 0, 1)
        if not lb is None:
            lb = torch.tensor(lb).unsqueeze(0)
        return dict(im=im, lb=lb)


class ToTensor(object):
    '''
    mean and std should be of the channel order 'rgb'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.mean = self.mean.view(-1, 1, 1)
        self.std = self.std.view(-1, 1, 1)

    @torch.no_grad()
    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        mean = self.mean.to(im.device)
        std = self.std.to(im.device)
        im = im.to(torch.float32).div_(255.)
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = lb.to(torch.int64)
        return dict(im=im, lb=lb)


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    @torch.no_grad()
    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            Numpy2Tensor(),
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    @torch.no_grad()
    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __init__(self, scales, cropsize):
        self.trans_func = Numpy2Tensor()

    @torch.no_grad()
    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb



if __name__ == '__main__':
    pass

