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
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
# import torch.cuda.amp as amp
# 适配 PyTorch 2.11
from torch import amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, log_msg



## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

SEED = 123

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.benchmark = True


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='configs/bisenetv2.py',)
    parse.add_argument('--finetune-from', type=str, default=None,)

    parse.add_argument(
        '--resume',
        type=str,
        default=None,
    )
    
    return parse.parse_args()

args = parse_args()
if (
    args.finetune_from is not None
    and args.resume is not None
):
    raise ValueError(
        "--finetune-from and --resume "
        "cannot be used together"
    )
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    # if not args.finetune_from is None:
    #     logger.info(f'load pretrained weights from {args.finetune_from}')
    #     msg = net.load_state_dict(torch.load(args.finetune_from,
    #         map_location='cpu'), strict=False)
    if args.finetune_from is not None:
        logger.info(
            f"load compatible pretrained weights from "
            f"{args.finetune_from}"
        )

        pretrained = torch.load(
            args.finetune_from,
            map_location="cpu",
            weights_only=True,
        )

        model_state = net.state_dict()

        compatible = {
            name: value
            for name, value in pretrained.items()
            if name in model_state
            and value.shape == model_state[name].shape
        }

        skipped = {
            name: tuple(value.shape)
            for name, value in pretrained.items()
            if name not in compatible
        }

        msg = net.load_state_dict(
            compatible,
            strict=False,
        )

        logger.info(
            f"loaded {len(compatible)} tensors"
        )
        logger.info(
            f"skipped {len(skipped)} incompatible tensors"
        )
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7, lb_ignore)
    criteria_aux = [OhemCELoss(0.7, lb_ignore)
            for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
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


def set_model_dist(net):
    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        #  find_unused_parameters=True,
        output_device=local_rank
        )
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters



def train():
    logger = logging.getLogger()

    # Dataset
    dl = get_data_loader(cfg, mode="train")

    # Model
    net, criteria_pre, criteria_aux = set_model(
        dl.dataset.lb_ignore
    )

    # Optimizer
    optim = set_optimizer(net)

    # Mixed precision
    scaler = amp.GradScaler(
        "cuda",
        enabled=cfg.use_fp16,
    )

    # LR scheduler
    lr_schdr = WarmupPolyLrScheduler(
        optim,
        power=0.9,
        max_iter=cfg.max_iter,
        warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1,
        warmup="exp",
        last_epoch=-1,
    )

    start_iter = 0

    # Resume complete training state
    if args.resume is not None:
        logger.info(
            f"resume complete checkpoint from {args.resume}"
        )

        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        # 此时 net 尚未经过 DDP 包装，直接加载即可
        net.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optim.load_state_dict(
            checkpoint["optimizer"]
        )

        # 将 optimizer 中的动量等状态移动到 GPU
        for state in optim.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda(
                        non_blocking=True
                    )

        scaler.load_state_dict(
            checkpoint["scaler"]
        )

        lr_schdr.load_state_dict(
            checkpoint["lr_scheduler"]
        )

        start_iter = int(
            checkpoint["iter"]
        )

        logger.info(
            f"resume succeeded, "
            f"next iteration={start_iter + 1}"
        )

    # DDP 应放到 checkpoint 恢复之后
    net = set_model_dist(net)

    # Meters
    (
        time_meter,
        loss_meter,
        loss_pre_meter,
        loss_aux_meters,
    ) = set_meters()

    # Train loop
    for it, (im, lb) in enumerate(
        dl,
        start=start_iter,
    ):
        if it >= cfg.max_iter:
            break

        im = im.cuda(
            non_blocking=True
        )
        lb = lb.cuda(
            non_blocking=True
        )

        lb = torch.squeeze(lb, 1)

        optim.zero_grad(
            set_to_none=True
        )

        with amp.autocast(
            "cuda",
            enabled=cfg.use_fp16,
        ):
            logits, *logits_aux = net(im)

            loss_pre = criteria_pre(
                logits,
                lb,
            )

            loss_aux = [
                criterion(aux_logits, lb)
                for criterion, aux_logits in zip(
                    criteria_aux,
                    logits_aux,
                )
            ]

            loss = (
                loss_pre
                + sum(loss_aux)
            )

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        lr_schdr.step()

        time_meter.update()
        loss_meter.update(
            loss.item()
        )
        loss_pre_meter.update(
            loss_pre.item()
        )

        for meter, aux_loss in zip(
            loss_aux_meters,
            loss_aux,
        ):
            meter.update(
                aux_loss.item()
            )

        if (it + 1) % 100 == 0:
            lr_values = lr_schdr.get_lr()
            lr = sum(lr_values) / len(lr_values)

            msg = log_msg(
                it,
                cfg.max_iter,
                lr,
                time_meter,
                loss_meter,
                loss_pre_meter,
                loss_aux_meters,
            )

            logger.info(msg)

        if (
            dist.get_rank() == 0
            and (it + 1) % cfg.save_interval == 0
        ):
            checkpoint_path = osp.join(
                cfg.respth,
                f"checkpoint_iter_{it + 1}.pth",
            )

            torch.save(
                {
                    "iter": it + 1,
                    "model": net.module.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "lr_scheduler": lr_schdr.state_dict(),
                },
                checkpoint_path,
            )

            logger.info(
                f"saved checkpoint to "
                f"{checkpoint_path}"
            )

    save_pth = osp.join(
        cfg.respth,
        "model_final.pth",
    )

    if dist.get_rank() == 0:
        logger.info(
            f"\nsave models to {save_pth}"
        )

        torch.save(
            net.module.state_dict(),
            save_pth,
        )

    logger.info(
        "\nevaluating the final model"
    )

    torch.cuda.empty_cache()

    (
        iou_heads,
        iou_content,
        f1_heads,
        f1_content,
    ) = eval_model(
        cfg,
        net.module,
    )

    logger.info(
        "\neval results of f1 score metric:"
    )
    logger.info(
        "\n"
        + tabulate(
            f1_content,
            headers=f1_heads,
            tablefmt="orgtbl",
        )
    )

    logger.info(
        "\neval results of miou metric:"
    )
    logger.info(
        "\n"
        + tabulate(
            iou_content,
            headers=iou_heads,
            tablefmt="orgtbl",
        )
    )

    return


def main():
    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

    dist.init_process_group(
        backend="nccl"
    )

    try:
        os.makedirs(
            cfg.respth,
            exist_ok=True,
        )

        setup_logger(
            (
                f"{cfg.model_type}-"
                f"{cfg.dataset.lower()}-train"
            ),
            cfg.respth,
        )

        train()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
