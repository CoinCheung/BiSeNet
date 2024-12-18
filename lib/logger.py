#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist


def setup_logger(name, logpth):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile, force=True)
    except Exception:
        for hl in logging.root.handlers: logging.root.removeHandler(hl)
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def log_msg(it, max_iter, lr, time_meter, loss_meter, loss_pre_meter,
        loss_aux_meters):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join([
        f'iter: {it+1}/{max_iter}',
        f'lr: {lr:4f}',
        f'eta: {eta}',
        f'time: {t_intv:.2f}',
        f'loss: {loss_avg:.4f}',
        f'loss_pre: {loss_pre_avg:.4f}',
    ])
    msg += ', ' + loss_aux_avg

    return msg
