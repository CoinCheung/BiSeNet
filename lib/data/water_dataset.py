import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import os

class WaterDataset(Dataset):
    def __init__(self, im_root, annpath, trans_func=None, mode='train'):
        self.im_root = im_root
        self.mode = mode
        self.trans_func = trans_func
        
        with open(annpath, 'r') as f:
            self.anns = [line.strip().split(',') for line in f.readlines()]
        
        self.n_cats = 2
        self.lb_ignore = 255
        
        print(f'Water {mode} set: {len(self.anns)} samples')
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        impath, lbpath = self.anns[idx]
        
        im = cv2.imread(os.path.join(self.im_root, impath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        lb = cv2.imread(os.path.join(self.im_root, lbpath), cv2.IMREAD_GRAYSCALE)
        lb = np.clip(lb, 0, 1) #确保标签只有0和1
        if self.trans_func is not None:
            result = self.trans_func({'im': im, 'lb': lb})
            im = result['im']  # 从字典取
            lb = result['lb']
        
        im = torch.from_numpy(im.copy()).permute(2, 0, 1).float()
        lb = torch.from_numpy(lb.copy()).long()
        
        return im, lb