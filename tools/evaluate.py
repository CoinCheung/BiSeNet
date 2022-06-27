#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.logger import setup_logger
from lib.data import get_data_loader


def get_round_size(size, divisor=32):
    return [math.ceil(el / divisor) * divisor for el in size]


class SizePreprocessor(object):

    def __init__(self, shape=None, shortside=None):
        self.shape = shape
        self.shortside = shortside

    def __call__(self, imgs):
        new_size = None
        if not self.shape is None:
            new_size = self.shape
        elif not self.shortside is None:
            h, w = imgs.size()[2:]
            ss = self.shortside
            if h < w: h, w = ss, int(ss / h * w)
            else: h, w = int(ss / w * h), ss
            new_size = h, w

        if not new_size is None:
            imgs = F.interpolate(imgs, size=new_size,
                    mode='bilinear', align_corners=False)
        return imgs


class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, lb_ignore=255, size_processor=None):
        self.scales = scales
        self.flip = flip
        self.ignore_label = lb_ignore
        self.sp = size_processor

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            imgs = self.sp(imgs)
            N, _, H, W = imgs.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            probs = torch.zeros(
                    (N, n_classes, *size),
                    dtype=torch.float32).cuda().detach()
            for scale in self.scales:
                sH, sW = int(scale * H), int(scale * W)
                sH, sW = get_round_size((sH, sW))
                im_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)

                im_sc = im_sc.cuda()
                logits = net(im_sc)[0]
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    im_sc = torch.flip(im_sc, dims=(3, ))
                    logits = net(im_sc)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()



class MscEvalCrop(object):

    def __init__(
        self,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
        size_processor=None
    ):
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.sp = size_processor

        self.cropsize = cropsize if isinstance(cropsize, (list, tuple)) else (cropsize, cropsize)
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.cropsize
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, net, im, n_classes):
        cropH, cropW = self.cropsize
        stride_rate = self.cropstride
        im, indices = self.pad_tensor(im)
        N, C, H, W = im.size()

        strdH = math.ceil(cropH * stride_rate)
        strdW = math.ceil(cropW * stride_rate)
        n_h = math.ceil((H - cropH) / strdH) + 1
        n_w = math.ceil((W - cropW) / strdW) + 1
        prob = torch.zeros(N, n_classes, H, W).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                stH, stW = strdH * i, strdW * j
                endH, endW = min(H, stH + cropH), min(W, stW + cropW)
                stH, stW = endH - cropH, endW - cropW
                chip = im[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob


    def scale_crop_eval(self, net, im, scale, size, n_classes):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, im, n_classes)
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=True)
        return prob


    @torch.no_grad()
    def __call__(self, net, dl, n_classes):

        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))

        for i, (imgs, label) in diter:
            imgs = imgs.cuda()
            imgs = self.sp(imgs)
            label = label.squeeze(1).cuda()
            N, *size = label.size()

            probs = torch.zeros((N, n_classes, *size)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, size, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)

            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)

        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = np.nanmean(ious.detach().cpu().numpy())
        return miou.item()



@torch.no_grad()
def eval_model(cfg, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'
    net.eval()

    is_dist = dist.is_initialized()
    dl = get_data_loader(cfg, mode='val')
    lb_ignore = dl.dataset.lb_ignore

    heads, mious = [], []
    logger = logging.getLogger()

    size_processor = SizePreprocessor(
            cfg.get('eval_start_shape'),
            cfg.get('eval_start_shortside'))

    single_scale = MscEvalV0(
            scales=(1., ),
            flip=False,
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    mIOU = single_scale(net, dl, cfg.n_cats)
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('single mIOU is: %s\n', mIOU)

    single_crop = MscEvalCrop(
        cropsize=cfg.eval_crop,
        cropstride=2. / 3,
        flip=False,
        scales=(1., ),
        lb_ignore=lb_ignore,
        size_processor=size_processor
    )
    mIOU = single_crop(net, dl, cfg.n_cats)
    heads.append('single_scale_crop')
    mious.append(mIOU)
    logger.info('single scale crop mIOU is: %s\n', mIOU)

    ms_flip = MscEvalV0(
            scales=cfg.eval_scales,
            flip=True,
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    mIOU = ms_flip(net, dl, cfg.n_cats)
    heads.append('ms_flip')
    mious.append(mIOU)
    logger.info('ms flip mIOU is: %s\n', mIOU)

    ms_flip_crop = MscEvalCrop(
        cropsize=cfg.eval_crop,
        cropstride=2. / 3,
        flip=True,
        scales=cfg.eval_scales,
        lb_ignore=lb_ignore,
        size_processor=size_processor
    )
    mIOU = ms_flip_crop(net, dl, cfg.n_cats)
    heads.append('ms_flip_crop')
    mious.append(mIOU)
    logger.info('ms crop mIOU is: %s\n', mIOU)

    net.aux_mode = org_aux
    return heads, mious


def evaluate(cfg, weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.n_cats)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'))
    net.cuda()

    #  is_dist = dist.is_initialized()
    #  if is_dist:
    #      local_rank = dist.get_rank()
    #      net = nn.parallel.DistributedDataParallel(
    #          net,
    #          device_ids=[local_rank, ],
    #          output_device=local_rank
    #      )

    ## evaluator
    heads, mious = eval_model(cfg, net)
    logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='model_final.pth',)
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    return parse.parse_args()


def main():
    args = parse_args()
    cfg = set_cfg_from_file(args.config)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-eval'.format(cfg.model_type), cfg.respth)
    evaluate(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
