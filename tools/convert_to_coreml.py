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
        default='model.mlpackage')
parse.add_argument('--img-path', dest='img_path', type=str, default='./datasets/custom_images/test.jpg',)
args = parse.parse_args()

cfg = set_cfg_from_file(args.config)

# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
# net.load_state_dict(torch.load(args.weight_path), strict=False)
# net.cuda()
# net.eval()

# prepare data
to_tensor = T.ToTensor(
    # mean=(0.3257, 0.3690, 0.3223), # city, rgb
    # std=(0.2112, 0.2148, 0.2115),
    mean=(0.0, 0.0, 0.0), # placeholder
    std=(1.0, 1.0, 1.0),
)
scale = 1/(0.2125*255.0)
bias = [- 0.3257/(0.2112) , - 0.3690/(0.2148), - 0.3223/(0.2115)]
print('Loading image:', args.img_path)
im = cv2.imread(args.img_path)#[:, :, ::-1]
# Resize
im = cv2.resize(im, (512, 256))
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)#.cuda()

class WrappedBiSeNetv2(nn.Module):
    def __init__(self, cfg):
        super(WrappedBiSeNetv2, self).__init__()
        self.model = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
        self.n_cats = cfg.n_cats
        self.model.load_state_dict(torch.load(args.weight_path), strict=False)
        self.model.eval()

    def forward(self, x):
        res = self.model(x)[0]
        out = torch.argmax(res, dim=1, keepdim=True).float()
        # out = out.float() / 255
        return out

torch_model = WrappedBiSeNetv2(cfg).eval()
traced_model = torch.jit.trace(torch_model, im)

mlmodel_from_trace = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=im.shape, scale=scale, bias=bias)],
    outputs=[ct.ImageType(name="output", color_layout=ct.colorlayout.GRAYSCALE)],
    # compute_precision=ct.precision.FLOAT16
    # minimum_deployment_target=ct.target.iOS13,
    # compute_units=ct.ComputeUnit.CPU_AND_GPU

)
mlmodel_from_trace.save(args.out_pth)
print(f"Saved the model to {args.out_pth}")


# Model list
# model1: model with multiarray input and output, image size (2048 x 922)
# model2: model with multiarray input and output, image size (1024 x 512)
# model3: model with image input and multiarray output, image size (1024 x 512)
# model4: model with image input and multiarray output, image size (1024 x 512)
# model5: model with image input and image output, image size (1024 x 512)
# model6: model with image input and image output, image size (1024 x 512)
# model7: model with image input and image output, image size (1024 x 512), cpu and gpu
# model8: model with image input and image output, image size (1024 x 512), with fp16 (which happens to be default anyway)
# model9: model with image input and image output, image size (512 x 256)
# model10: model with image input and image output, image size (512 x 256), with normalization done within the coreml model. 