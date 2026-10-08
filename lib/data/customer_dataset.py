#!/usr/bin/python
# -*- encoding: utf-8 -*-


from lib.data.base_dataset import BaseDataset


class CustomerDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func, mode='train'):
        super(CustomerDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255



