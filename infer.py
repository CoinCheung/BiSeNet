#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import json
import cv2


def infer():
    ## dataset color map
    with open('cityscapes_info.json', 'r') as fr:
        cs_info = json.load(fr)
        colormap = {el['trainId']:el['color'] for el in cs_info}
        colormap[19] = colormap[255]

    ## load model
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.eval()
    save_pth = './res/model_final.pth'
    net.load_state_dict(torch.load(save_pth))

    ## image
    impth = './example.png'
    im = Image.open(impth)
    W, H = im.size
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    im = to_tensor(im)
    im = im.unsqueeze(0).cuda()

    ## inference
    with torch.no_grad():
        out, out16, out32 = net(im)
        probs = F.softmax(out, 1)
    probs = probs.detach().cpu().numpy()
    pred = np.squeeze(np.argmax(probs, axis=1), 0)
    H, W = pred.shape

    ## show infered picture
    out = np.empty((H, W, 3), dtype = np.uint8)
    clses = np.unique(pred).tolist()
    for cls in clses:
        out[pred == cls, :] = colormap[cls]

    cv2.imshow('pred', out)
    cv2.waitKey(0)




if __name__ == "__main__":
    infer()
