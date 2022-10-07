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

    def __init__(self, shape=None, shortside=None, longside=None):
        self.shape = shape
        self.shortside = shortside
        self.longside = longside

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
        elif not self.longside is None: # long size limit
            h, w = imgs.size()[2:]
            if max(h, w) > self.longside:
                ls = self.longside
                if h < w: h, w = int(ls / w * h), ls
                else: h, w = ls, int(ls / h * w)
                new_size = h, w

        if not new_size is None:
            imgs = F.interpolate(imgs, size=new_size,
                    mode='bilinear', align_corners=False)
        return imgs



class Metrics(object):

    def __init__(self, n_classes, lb_ignore=255):
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.confusion = torch.zeros((n_classes, n_classes)).cuda().detach()

    @torch.no_grad()
    def update(self, preds, label):
        keep = label != self.lb_ignore
        preds, label = preds[keep], label[keep]
        self.confusion += torch.bincount(
                label * self.n_classes + preds,
                minlength=self.n_classes ** 2
                ).view(self.n_classes, self.n_classes)

    @torch.no_grad()
    def compute_metrics(self,):
        if dist.is_initialized():
            dist.all_reduce(self.confusion, dist.ReduceOp.SUM)

        confusion = self.confusion
        weights = confusion.sum(dim=1) / confusion.sum()
        tps = confusion.diag()
        fps = confusion.sum(dim=0) - tps
        fns = confusion.sum(dim=1) - tps

        # iou and fw miou
        #  ious = confusion.diag() / (confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag() + 1)
        ious = tps / (tps + fps + fns + 1)
        miou = ious.nanmean()
        fw_miou = torch.sum(weights * ious)

        eps = 1e-6
        # macro f1 score
        macro_precision = tps / (tps + fps + 1)
        macro_recall = tps / (tps + fns + 1)
        f1_scores = (2 * macro_precision * macro_recall) / (
                macro_precision + macro_recall + eps)
        macro_f1 = f1_scores.nanmean(dim=0)

        # micro f1 score
        tps_ = tps.sum(dim=0)
        fps_ = fps.sum(dim=0)
        fns_ = fns.sum(dim=0)
        micro_precision = tps_ / (tps_ + fps_ + 1)
        micro_recall = tps_ / (tps_ + fns_ + 1)
        micro_f1 = (2 * micro_precision * micro_recall) / (
                micro_precision + micro_recall + eps)

        metric_dict = dict(
                weights=weights.tolist(),
                ious=ious.tolist(),
                miou=miou.item(),
                fw_miou=fw_miou.item(),
                f1_scores=f1_scores.tolist(),
                macro_f1=macro_f1.item(),
                micro_f1=micro_f1.item(),
                )
        return metric_dict



class MscEvalV0(object):

    def __init__(self, n_classes, scales=(0.5, ), flip=False, lb_ignore=255, size_processor=None):
        self.n_classes = n_classes
        self.scales = scales
        self.flip = flip
        self.ignore_label = lb_ignore
        self.sp = size_processor
        self.metric_observer = Metrics(n_classes, lb_ignore)

    @torch.no_grad()
    def __call__(self, net, dl):
        ## evaluate
        n_classes = self.n_classes
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
            self.metric_observer.update(preds, label)

        metric_dict = self.metric_observer.compute_metrics()
        return metric_dict



class MscEvalCrop(object):

    def __init__(
        self,
        n_classes,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        lb_ignore=255,
        size_processor=None
    ):
        self.n_classes = n_classes
        self.scales = scales
        self.ignore_label = lb_ignore
        self.flip = flip
        self.sp = size_processor
        self.metric_observer = Metrics(n_classes, lb_ignore)

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
    def __call__(self, net, dl):

        n_classes = self.n_classes
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
            self.metric_observer.update(preds, label)

        metric_dict = self.metric_observer.compute_metrics()
        return metric_dict


def print_res_table(tname, heads, weights, metric, cat_metric):
    heads = [tname, 'ratio'] + heads
    lines = []
    for k, v in metric.items():
        line = [k, '-'] + [f'{el:.6f}' for el in v]
        lines.append(line)
    cat_res = [weights,] + cat_metric
    cat_res = [
            [f'cat {idx}',] + [f'{el:.6f}' for el in group]
            for idx,group in enumerate(zip(*cat_res))]
    content = cat_res + lines
    return heads, content


@torch.no_grad()
def eval_model(cfg, net):
    org_aux = net.aux_mode
    net.aux_mode = 'eval'
    net.eval()

    is_dist = dist.is_initialized()
    dl = get_data_loader(cfg, mode='val')
    lb_ignore = dl.dataset.lb_ignore

    heads, mious, fw_mious, cat_ious = [], [], [], []
    f1_scores, macro_f1, micro_f1 = [], [], []
    logger = logging.getLogger()

    size_processor = SizePreprocessor(
            cfg.get('eval_start_shape'),
            cfg.get('eval_start_shortside'),
            cfg.get('eval_start_longside'),
            )

    single_scale = MscEvalV0(
            n_classes=cfg.n_cats,
            scales=(1., ),
            flip=False,
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    logger.info('compute single scale metrics')
    metrics = single_scale(net, dl)
    heads.append('ss')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    single_crop = MscEvalCrop(
            n_classes=cfg.n_cats,
            cropsize=cfg.eval_crop,
            cropstride=2. / 3,
            flip=False,
            scales=(1., ),
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    logger.info('compute single scale crop metrics')
    metrics = single_crop(net, dl)
    heads.append('ssc')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    ms_flip = MscEvalV0(
            n_classes=cfg.n_cats,
            scales=cfg.eval_scales,
            flip=True,
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    logger.info('compute multi scale flip metrics')
    metrics = ms_flip(net, dl)
    heads.append('msf')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    ms_flip_crop = MscEvalCrop(
            n_classes=cfg.n_cats,
            cropsize=cfg.eval_crop,
            cropstride=2. / 3,
            flip=True,
            scales=cfg.eval_scales,
            lb_ignore=lb_ignore,
            size_processor=size_processor
    )
    logger.info('compute multi scale flip crop metrics')
    metrics = ms_flip_crop(net, dl)
    heads.append('msfc')
    mious.append(metrics['miou'])
    fw_mious.append(metrics['fw_miou'])
    cat_ious.append(metrics['ious'])
    f1_scores.append(metrics['f1_scores'])
    macro_f1.append(metrics['macro_f1'])
    micro_f1.append(metrics['micro_f1'])

    weights = metrics['weights']

    metric = dict(mious=mious, fw_mious=fw_mious)
    iou_heads, iou_content = print_res_table('iou', heads,
            weights, metric, cat_ious)
    metric = dict(macro_f1=macro_f1, micro_f1=micro_f1)
    f1_heads, f1_content = print_res_table('f1 score', heads,
            weights, metric, f1_scores)

    net.aux_mode = org_aux
    return iou_heads, iou_content, f1_heads, f1_content


def evaluate(cfg, weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.n_cats)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'))
    net.cuda()

    #  if dist.is_initialized():
    #      local_rank = dist.get_rank()
    #      net = nn.parallel.DistributedDataParallel(
    #          net,
    #          device_ids=[local_rank, ],
    #          output_device=local_rank
    #      )

    ## evaluator
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))


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
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-eval', cfg.respth)
    evaluate(cfg, args.weight_pth)


if __name__ == "__main__":
    main()
