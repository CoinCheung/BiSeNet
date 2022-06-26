
import os
import os.path as osp
import argparse
from tqdm import tqdm

import cv2
import numpy as np


parse = argparse.ArgumentParser()
parse.add_argument('--im_root', dest='im_root', type=str, default='./datasets/cityscapes',)
parse.add_argument('--im_anns', dest='im_anns', type=str, default='./datasets/cityscapes/train.txt',)
args = parse.parse_args()


with open(args.im_anns, 'r') as fr:
    lines = fr.read().splitlines()

n_pairs = len(lines)
impaths, lbpaths = [], []
for l in lines:
    impth, lbpth = l.split(',')
    impth = osp.join(args.im_root, impth)
    lbpth = osp.join(args.im_root, lbpth)
    impaths.append(impth)
    lbpaths.append(lbpth)


## shapes
max_shape_area, min_shape_area = [0, 0], [100000, 100000]
max_shape_height, min_shape_height = [0, 0], [100000, 100000]
max_shape_width, min_shape_width = [0, 0], [100000, 100000]
max_lb_val, min_lb_val = -1, 10000000
for impth, lbpth in tqdm(zip(impaths, lbpaths), total=n_pairs):
    im = cv2.imread(impth)[:, :, ::-1]
    lb = cv2.imread(lbpth, 0)
    assert im.shape[:2] == lb.shape

    shape = lb.shape
    area = shape[0] * shape[1]
    if area > max_shape_area[0] * max_shape_area[1]:
        max_shape_area = shape
    if area < min_shape_area[0] * min_shape_area[1]:
        min_shape_area = shape

    if shape[0] > max_shape_height[0]:
        max_shape_height = shape
    if shape[0] < min_shape_height[0]:
        min_shape_height = shape

    if shape[1] > max_shape_width[1]:
        max_shape_width = shape
    if shape[1] < min_shape_width[1]:
        min_shape_width = shape

    max_lb_val = max(max_lb_val, np.max(lb.ravel()))
    min_lb_val = min(min_lb_val, np.min(lb.ravel()))


## label info
lb_minlength = max_lb_val+1-min_lb_val
lb_hist = np.zeros(lb_minlength)
for impth in tqdm(impaths):
    lb = cv2.imread(lbpth, 0).ravel() + min_lb_val
    lb_hist += np.bincount(lb, minlength=lb_minlength)

lb_missing_vals = [ind + min_lb_val
        for ind, el in enumerate(lb_hist.tolist()) if el == 0]
lb_ratios = (lb_hist / lb_hist.sum()).tolist()


## pixel mean/std
rgb_mean = np.zeros(3).astype(np.float32)
n_pixels = 0
for impth in tqdm(impaths):
    im = cv2.imread(impth)[:, :, ::-1].astype(np.float32)
    im = im.reshape(-1, 3)
    n_pixels += im.shape[0]
    rgb_mean += im.sum(axis=0)
rgb_mean = rgb_mean / n_pixels

rgb_std = np.zeros(3).astype(np.float32)
for impth in tqdm(impaths):
    im = cv2.imread(impth)[:, :, ::-1].astype(np.float32)
    im = im.reshape(-1, 3)

    a = (im - rgb_mean.reshape(1, 3)) ** 2
    rgb_std += a.sum(axis=0)
rgb_std = (rgb_std / n_pixels) ** (0.5)


print(f'there are {n_pairs} lines in {args.im_anns}, which means {n_pairs} image/label image pairs')
print('\n')

print('max and min image shapes by area are: ')
print(f'\t{max_shape_area}, {min_shape_area}')
print('max and min image shapes by height are: ')
print(f'\t{max_shape_height}, {min_shape_height}')
print('max and min image shapes by width are: ')
print(f'\t{max_shape_width}, {min_shape_width}')
print('\n')

print(f'label values are within range of ({min_lb_val}, {max_lb_val})')
print('label values that are missing: ')
print('\t', lb_missing_vals)
print('ratios of each label value: ')
print('\t', lb_ratios)
print('\n')

print('pixel mean rgb: ', mean)
print('pixel std rgb: ', std)
