#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import logging
import argparse
import math

from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from bisenetv2.bisenetv2 import BiSeNetV2
from bisenetv2.logger import setup_logger
from bisenetv2.cityscapes_cv2 import get_data_loader




class MscEvalV0(object):

    def __init__(self, ignore_label=255):
        self.ignore_label = ignore_label

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:
            N, _, H, W = label.shape
            label = label.squeeze(1).cuda()
            size = label.size()[-2:]
            imgs = imgs.cuda()
            logits = net(imgs)[0]
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()



def eval_model(net, ims_per_gpu):
    is_dist = dist.is_initialized()
    dl = get_data_loader('./data', ims_per_gpu, mode='val', distributed=is_dist)
    net.eval()

    with torch.no_grad():
        single_scale = MscEvalV0()
        mIOU = single_scale(net, dl, 19)
    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)


def evaluate(weight_pth):
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    net = BiSeNetV2(19)
    net.load_state_dict(torch.load(weight_pth))
    net.cuda()

    is_dist = dist.is_initialized()
    if is_dist:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank
        )

    ## evaluator
    eval_model(net, 2)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank',
                       type=int, default=-1,)
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='model_final.pth',)
    parse.add_argument('--port', dest='port', type=int, default=44553,)
    parse.add_argument('--respth', dest='respth', type=str, default='./res',)
    return parse.parse_args()


def main():
    args = parse_args()
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(args.respth): os.makedirs(args.respth)
    setup_logger('BiSeNetV2-eval', args.respth)
    evaluate(args.weight_pth)



if __name__ == "__main__":
    main()
