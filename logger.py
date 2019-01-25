#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import sys
import logging


## TODO: here log name change to BiSeNet rather than deeplab
def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


