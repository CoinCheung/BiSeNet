import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch
from torch2trt import torch2trt

from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)


parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--fp16', action='store_true')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.trt')
args = parse.parse_args()


cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False

net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
net.load_state_dict(torch.load(args.weight_path), strict=False)
net.cuda()
net.eval()


#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
dummy_input = torch.randn(1, 3, 1024, 2048).cuda()

trt_model = torch2trt(net, [dummy_input, ], fp16_mode=args.fp16, max_workspace=1 << 30)

with open(args.out_pth, 'wb') as fw:
    fw.write(trt_model.engine.serialize())
