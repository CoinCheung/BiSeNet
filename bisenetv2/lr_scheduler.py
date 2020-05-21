#!/usr/bin/python
# -*- encoding: utf-8 -*-

import math
from bisect import bisect_right
import torch


class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio


class WarmupExpLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            gamma,
            interval=1,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.gamma = gamma
        self.interval = interval
        super(WarmupExpLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** (real_iter // self.interval)
        return ratio


class WarmupCosineLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            max_iter,
            eta_ratio=0,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super(WarmupCosineLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        return self.eta_ratio + (1 - self.eta_ratio) * (
                1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2


class WarmupStepLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            milestones: list,
            gamma=0.1,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupStepLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        ratio = self.gamma ** bisect_right(self.milestones, real_iter)
        return ratio


if __name__ == "__main__":
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 20000
    lr_scheduler = WarmupPolyLrScheduler(optim, 0.9, max_iter, 200, 0.1, 'linear', -1)
    lrs = []
    for _ in range(max_iter):
        lr = lr_scheduler.get_lr()[0]
        print(lr)
        lrs.append(lr)
        lr_scheduler.step()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    lrs = np.array(lrs)
    n_lrs = len(lrs)
    plt.plot(np.arange(n_lrs), lrs)
    plt.grid()
    plt.show()


