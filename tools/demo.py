
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


# palette and mean/std
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
mean = torch.tensor([0.3257, 0.3690, 0.3223], dtype=torch.float32).view(-1, 1, 1)
std = torch.tensor([0.2112, 0.2148, 0.2115], dtype=torch.float32).view(-1, 1, 1)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
im = cv2.imread(args.img_path)
im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
im = torch.from_numpy(im).div_(255).sub_(mean).div_(std).unsqueeze(0).cuda()

# inference
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./res.jpg', pred)
