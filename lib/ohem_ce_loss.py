#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


#  import ohem_cpp
#  class OhemCELoss(nn.Module):
#
#      def __init__(self, thresh, ignore_lb=255):
#          super(OhemCELoss, self).__init__()
#          self.score_thresh = thresh
#          self.ignore_lb = ignore_lb
#          self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='mean')
#
#      def forward(self, logits, labels):
#          n_min = labels[labels != self.ignore_lb].numel() // 16
#          labels = ohem_cpp.score_ohem_label(
#                  logits, labels, self.ignore_lb, self.score_thresh, n_min).detach()
#          loss = self.criteria(logits, labels)
#          return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == '__main__':
    pass

