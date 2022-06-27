#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class CustomerDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CustomerDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.4, 0.4, 0.4), # rgb
            std=(0.2, 0.2, 0.2),
        )


