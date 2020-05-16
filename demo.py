
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2

from fp16.model import BiSeNet


# args
parse = argparse.ArgumentParser()
parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default='./res/model_final.pth',)
parse.add_argument(
        '--img_path',
        dest='img_path',
        type=str,
        default='./pic.jpg',)
args = parse.parse_args()


# define model
net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
im = to_tensor(Image.open(args.img_path).convert('RGB')).unsqueeze(0).cuda()

# inference
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
cv2.imwrite('./res.jpg', out)
