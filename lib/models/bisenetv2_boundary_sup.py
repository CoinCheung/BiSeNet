#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bisenetv2 import (
    BiSeNetV2,
    ConvBNReLU,
    CustomArgMax,
)


class BoundaryHead(nn.Module):
    """
    Training-only auxiliary boundary head.

    Input:
        BGA fused feature
        N x 128 x H/8 x W/8

    Output:
        boundary logits
        N x 1 x H x W
    """

    def __init__(
        self,
        in_channels=128,
        mid_channels=32,
    ):
        super().__init__()

        self.feature = ConvBNReLU(
            in_channels,
            mid_channels,
            ks=3,
            stride=1,
            padding=1,
        )

        self.predict = nn.Conv2d(
            mid_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        x,
        output_size,
    ):
        x = self.feature(x)
        x = self.predict(x)

        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


class BiSeNetV2BoundarySup(
    BiSeNetV2
):

    def __init__(
        self,
        n_classes,
        aux_mode="train",
    ):
        # 完整 baseline
        super().__init__(
            n_classes=n_classes,
            aux_mode=aux_mode,
        )

        # 在 super 后增加，
        # 因此不会改变原 baseline 权重初始化
        self.boundary_head = BoundaryHead(
            in_channels=128,
            mid_channels=32,
        )

        self.init_boundary_head()

    def init_boundary_head(self):
        for module in (
            self.boundary_head.modules()
        ):
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x,
    ):
        output_size = x.shape[-2:]

        # ---------------------------------
        # 原始 BiSeNetV2
        # ---------------------------------

        feat_d = self.detail(x)

        (
            feat2,
            feat3,
            feat4,
            feat5_4,
            feat_s,
        ) = self.segment(x)

        feat_head = self.bga(
            feat_d,
            feat_s,
        )

        logits = self.head(
            feat_head
        )

        # ---------------------------------
        # training
        # ---------------------------------

        if self.aux_mode == "train":

            logits_aux2 = self.aux2(
                feat2
            )

            logits_aux3 = self.aux3(
                feat3
            )

            logits_aux4 = self.aux4(
                feat4
            )

            logits_aux5_4 = (
                self.aux5_4(
                    feat5_4
                )
            )

            boundary_logits = (
                self.boundary_head(
                    feat_head,
                    output_size,
                )
            )

            return (
                logits,
                logits_aux2,
                logits_aux3,
                logits_aux4,
                logits_aux5_4,
                boundary_logits,
            )

        # ---------------------------------
        # evaluation
        # boundary head completely unused
        # ---------------------------------

        elif self.aux_mode == "eval":

            return logits,

        elif self.aux_mode == "pred":

            pred = CustomArgMax.apply(
                logits,
                1,
            )

            return pred

        else:
            raise NotImplementedError
