#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import json
import argparse
import warnings
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, log_msg

# 忽略警告
warnings.filterwarnings('ignore')


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.ce(pred, target) + 0.5 * self.dice(pred, target)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()


args = parse_args()
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    
    if args.finetune_from is not None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
            map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    
    # Windows不支持SyncBatchNorm，跳过
    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    net.cuda()
    net.train()
    
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore)
            for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()
    # 添加字段兼容性处理（保险起见）
    if not hasattr(cfg, 'ims_per_gpu'):
        cfg.ims_per_gpu = getattr(cfg, 'batch_size', 4)

    ## dataset
    dl = get_data_loader(cfg, mode='train')
    
    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)
    
    ## optimizer
    optim = set_optimizer(net)
    
    ## mixed precision training
    scaler = amp.GradScaler()
    
    ## 注意：Windows单卡模式，不使用DistributedDataParallel
    ## net本身就是单卡模型，不需要包装
    
    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()
    
    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    
    ## train loop
    for it, (im, lb) in enumerate(dl):
        im = im.cuda()
        lb = lb.cuda()
        lb = torch.squeeze(lb, 1)
        
        optim.zero_grad()
        with amp.autocast(enabled=cfg.use_fp16):
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
        
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        torch.cuda.synchronize()
        
        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
        
        ## print training log message
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
            msg = log_msg(
                it, cfg.max_iter, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
            logger.info(msg)
        lr_schdr.step()
    
    ## dump the final model and evaluate the result
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    logger.info('\nsave models to {}'.format(save_pth))
    
    # 单卡模式直接保存，不需要判断rank
    state = net.state_dict() if not hasattr(net, 'module') else net.module.state_dict()
    torch.save(state, save_pth)
    
    logger.info('\nevaluating the final model')
    torch.cuda.empty_cache()
    
    # 单卡评估
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
    
    return


def main():
    # ========== Windows单卡模式：完全移除分布式 ==========
    # 设置GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建输出目录
    if not osp.exists(cfg.respth): 
        os.makedirs(cfg.respth)
    
    # 设置日志（移除分布式rank相关）
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    logger = logging.getLogger()
    logger.info(f"Config file: {args.config}")
    logger.info(f"Model type: {cfg.model_type}")
    logger.info(f"Num classes: {cfg.n_cats}")
    logger.info(f"Max iterations: {cfg.max_iter}")
    logger.info(f"Batch size (ims_per_gpu): {cfg.ims_per_gpu}")
    
    # 开始训练
    train()


if __name__ == "__main__":
    main()