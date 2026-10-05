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

from lib.boundary_supervision import (
    semantic_to_boundary,
    boundary_bce_dice_loss,
)



## fix all random seeds
#  torch.manual_seed(123)
#  torch.cuda.manual_seed(123)
#  np.random.seed(123)
#  random.seed(123)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')

SEED = int(os.environ.get("EXPERIMENT_SEED", "123"))

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

    loss_meter = AvgMeter(
        'loss'
    )

    loss_pre_meter = AvgMeter(
        'loss_prem'
    )

    loss_aux_meters = [
        AvgMeter(
            'loss_aux{}'.format(i)
        )
        for i in range(
            cfg.num_aux_heads
        )
    ]

    # Experiment B:
    # 用于统计 Boundary Loss
    boundary_loss_meter = AvgMeter(
        'loss_boundary'
    )

    return (
        time_meter,
        loss_meter,
        loss_pre_meter,
        loss_aux_meters,
        boundary_loss_meter,
    )



def train():
    logger = logging.getLogger()

    if dist.get_rank() == 0:
        logger.info(
            "Experiment B Boundary Supervision:"
        )
    
        logger.info(
            f"  boundary_width="
            f"{cfg.boundary_width}"
        )
    
        logger.info(
            f"  boundary_loss_weight="
            f"{cfg.boundary_loss_weight}"
        )
    
        logger.info(
            "  boundary_loss=BCE+Dice"
        )

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
        boundary_loss_meter,
    ) = set_meters()

    amp_skipped_steps = 0

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
        
        # ---------------------------------
        # Boundary GT
        # 不需要梯度，也不需要 AMP
        # ---------------------------------
        
        with torch.no_grad():
            (
                boundary_target,
                boundary_valid,
            ) = semantic_to_boundary(
                lb,
                ignore_index=(
                    dl.dataset.lb_ignore
                ),
                width=cfg.boundary_width,
            )
        # ---------------------------------
        # 只打印第一个 batch 的
        # boundary positive ratio
        # --------------------------------
        if (
            it == start_iter
            and dist.get_rank() == 0
        ):
            logger.info(
                "boundary positive ratio: "
                f"{boundary_target.mean().item():.6f}"
            )
        
        
        # ---------------------------------
        # Forward + semantic losses
        # ---------------------------------
        
        with amp.autocast(
            "cuda",
            enabled=cfg.use_fp16,
        ):
            outputs = net(im)
        
            expected_outputs = (
                cfg.num_aux_heads + 2
            )
        
            if len(outputs) != expected_outputs:
                raise RuntimeError(
                    f"Expected {expected_outputs} outputs, "
                    f"got {len(outputs)}"
                )
        
            # Main semantic output
            logits = outputs[0]
        
            # Auxiliary semantic outputs
            logits_aux = outputs[
                1:
                1 + cfg.num_aux_heads
            ]
        
            # Training-only boundary output
            boundary_logits = outputs[-1]
        
            loss_pre = criteria_pre(
                logits,
                lb,
            )
        
            loss_aux = [
                crit(aux_logits, lb)
                for crit, aux_logits
                in zip(
                    criteria_aux,
                    logits_aux,
                )
            ]
        
        
        # ---------------------------------
        # Boundary shape check
        # ---------------------------------
        
        if (
            boundary_logits.shape[-2:]
            != boundary_target.shape[-2:]
        ):
            raise RuntimeError(
                "Boundary shape mismatch: "
                f"logits={tuple(boundary_logits.shape)}, "
                f"target={tuple(boundary_target.shape)}"
            )
        
        
        # ---------------------------------
        # Boundary Loss
        # 强制使用 FP32
        # ---------------------------------
        
        with amp.autocast(
            "cuda",
            enabled=False,
        ):
            loss_boundary = (
                boundary_bce_dice_loss(
                    boundary_logits.float(),
                    boundary_target.float(),
                    boundary_valid.float(),
                )
            )
        
        
        # ---------------------------------
        # Total Loss
        # ---------------------------------
        
        loss = (
            loss_pre
            + sum(loss_aux)
            + (
                cfg.boundary_loss_weight
                * loss_boundary
            )
        )
        
        
        # ---------------------------------
        # Backward
        # ---------------------------------
        
        scaler.scale(
            loss
        ).backward()
        
        scale_before = (
            scaler.get_scale()
        )
        
        scaler.step(
            optim
        )
        
        scaler.update()
        
        scale_after = (
            scaler.get_scale()
        )
        
        # 如果 scale 下降，
        # 表示本轮发生 overflow，
        # optimizer.step 被跳过。
        #
        # 此时 scheduler 也不应该前进一步。
        if scale_after >= scale_before:
            lr_schdr.step()
        else:
            amp_skipped_steps += 1
        
            if dist.get_rank() == 0:
                logger.warning(
                    f"AMP overflow at iter {it + 1}, "
                    f"scale {scale_before} -> {scale_after}"
                )
        
        
        # ---------------------------------
        # Meters
        # ---------------------------------
        
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
        
        boundary_loss_meter.update(
            loss_boundary.item()
        )
        
        
        # ---------------------------------
        # Logging
        # ---------------------------------
        
        if (it + 1) % 100 == 0:
            lr = lr_schdr.get_lr()
            lr = sum(lr) / len(lr)
        
            msg = log_msg(
                it,
                cfg.max_iter,
                lr,
                time_meter,
                loss_meter,
                loss_pre_meter,
                loss_aux_meters,
            )
        
            boundary_avg, _ = (
                boundary_loss_meter.get()
            )
        
            msg += (
                f", loss_boundary: "
                f"{boundary_avg:.4f}"
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
