import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torchvision
import json
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

import coremltools as ct

torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_city.py',)
parse.add_argument('--weight-path', type=str, default='./lib/models/model_zoo/model_final_v2_city.pth',)
parse.add_argument('--fp16', action='store_true')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.mlmodel')
parse.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
args = parse.parse_args()

cfg = set_cfg_from_file(args.config)

# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
# net.load_state_dict(torch.load(args.weight_path), strict=False)
# net.cuda()
# net.eval()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
print('Loading image:', args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, cfg):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
        self.model.load_state_dict(torch.load(args.weight_path), strict=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)[0]

torch_model = WrappedBiSeNetv2(cfg).eval()
traced_model = torch.jit.trace(torch_model, im)

mlmodel_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input", shape=im.shape)],
)
mlmodel_from_trace.save(args.out_pth)
print(f"Saved the model to {args.out_pth}")