#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import *
from model import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import numba


@numba.jit
def compute_iou(pred, lb, lb_ignore=255):
    assert pred.shape == lb.shape

    mask = np.logical_not(lb == lb_ignore)
    clses = set(np.unique(lb).tolist())
    if lb_ignore in clses: clses.remove(lb_ignore)
    ious = []
    for cls in clses:
        ## TODO: use p_val, l_val = pred[mask], lb[mask] instead of always slice
        pmask = pred[mask] == cls
        lmask = lb[mask] == cls
        cross = np.logical_and(pmask, lmask)
        union = np.logical_or(pmask, lmask)
        iou = float(np.sum(cross)) / float(np.sum(union))
        ious.append(iou)
    ious = sum(ious) / len(ious)
    return ious


def eval_model(net):
    ## dataloader
    dsval = CityScapes('./data', mode='val')

    ## evaluate
    ious = []
    for i, (im, lb) in enumerate(tqdm(dsval)):
        im = im.cuda().unsqueeze(0)
        lb = np.squeeze(lb, 0)
        H, W = lb.shape
        with torch.no_grad():
            out_p, out16 ,out32 = net(im)
            probs = F.softmax(out_p, 1).squeeze(0)
        probs = probs.detach().cpu().numpy()
        pred = np.argmax(probs, axis=0)

        iou = compute_iou(pred, lb)
        ious.append(iou)
    mIOU = sum(ious) / len(ious)
    return mIOU


def evaluate():
    respth = './res'
    logger = logging.getLogger()

    ## model
    logger.info('setup and restore model')
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.eval()
    save_pth = osp.join(respth, 'model_final.pth')
    net.load_state_dict(torch.load(save_pth))

    ## dataset
    logger.info('compute the mIOU')
    mIOU = eval_model(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))




if __name__ == "__main__":
    import sys
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    evaluate()
