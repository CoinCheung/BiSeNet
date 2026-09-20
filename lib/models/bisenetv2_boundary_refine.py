#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bisenetv2 import (
    BiSeNetV2,
    CustomArgMax,
)


class LightweightBoundaryRefine(nn.Module):
    """
    Lightweight Boundary Refinement Module (LBRM)

    No explicit boundary supervision is used.

    Input:
        BGA fused feature:
        N x 128 x H/8 x W/8

    Output:
        Refined feature with identical shape.
    """

    def __init__(
        self,
        channels=128,
        bottleneck_channels=32,
        gate_channels=16,
    ):
        super().__init__()

        # ---------------------------------
        # Local contrast extraction
        # ---------------------------------

        self.local_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # ---------------------------------
        # Boundary-aware spatial gate
        #
        # 128 -> 16 -> 1
        # ---------------------------------

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                gate_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                gate_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                gate_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

        # ---------------------------------
        # Lightweight refinement branch
        #
        # 128
        #  ↓ 1x1
        # 32
        #  ↓ depthwise 3x3
        # 32
        #  ↓ 1x1
        # 128
        # ---------------------------------

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                bottleneck_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                bottleneck_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.BatchNorm2d(
                bottleneck_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                bottleneck_channels,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.init_weights()

    def init_weights(
        self,
    ):
        """
        Normal initialization + identity-start
        initialization.

        The final BN gamma of the refinement
        branch is initialized to zero, so the
        module initially behaves as:

            output = input

        This minimizes disturbance to the
        pretrained BiSeNetV2.
        """

        for module in self.modules():

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

        # ---------------------------------
        # Zero-init final BN
        #
        # Initial refinement delta = 0
        # so initial output ~= input.
        # ---------------------------------

        final_bn = self.refine[-1]

        nn.init.zeros_(
            final_bn.weight
        )

        nn.init.zeros_(
            final_bn.bias
        )

    def forward(
        self,
        x,
    ):
        # ---------------------------------
        # Local high-frequency contrast
        # ---------------------------------

        local_context = (
            self.local_pool(x)
        )

        local_contrast = (
            x - local_context
        )

        # ---------------------------------
        # Boundary-aware spatial gate
        #
        # N x 1 x H x W
        # ---------------------------------

        gate = torch.sigmoid(
            self.gate(
                local_contrast
            )
        )

        # ---------------------------------
        # Lightweight feature refinement
        #
        # N x C x H x W
        # ---------------------------------

        delta = self.refine(
            x
        )

        # ---------------------------------
        # Boundary-gated residual
        # ---------------------------------

        out = (
            x
            + gate * delta
        )

        out = self.relu(
            out
        )

        return out


class BiSeNetV2BoundaryRefine(
    BiSeNetV2
):

    def __init__(
        self,
        n_classes,
        aux_mode="train",
    ):
        super().__init__(
            n_classes=n_classes,
            aux_mode=aux_mode,
        )

        self.boundary_refine = (
            LightweightBoundaryRefine(
                channels=128,
                bottleneck_channels=32,
                gate_channels=16,
            )
        )

    def forward(
        self,
        x,
    ):
        # ---------------------------------
        # Original BiSeNetV2 branches
        # ---------------------------------

        feat_d = self.detail(
            x
        )

        (
            feat2,
            feat3,
            feat4,
            feat5_4,
            feat_s,
        ) = self.segment(
            x
        )

        # ---------------------------------
        # Original BGA
        # ---------------------------------

        feat_head = self.bga(
            feat_d,
            feat_s,
        )

        # ---------------------------------
        # Experiment C:
        # Boundary Refinement Only
        # ---------------------------------

        feat_head = (
            self.boundary_refine(
                feat_head
            )
        )

        # ---------------------------------
        # Original segmentation head
        # ---------------------------------

        logits = self.head(
            feat_head
        )

        # ---------------------------------
        # Original auxiliary heads
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

            return (
                logits,
                logits_aux2,
                logits_aux3,
                logits_aux4,
                logits_aux5_4,
            )

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


    def get_params(
        self,
    ):
        """
        Keep original BiSeNetV2 optimizer
        grouping.

        Newly initialized refinement module
        uses the same 10x LR policy as the
        segmentation/auxiliary heads.
        """

        def add_param_to_list(
            module,
            wd_params,
            nowd_params,
        ):
            for param in module.parameters():

                if not param.requires_grad:
                    continue

                if param.dim() == 1:
                    nowd_params.append(
                        param
                    )

                elif param.dim() == 4:
                    wd_params.append(
                        param
                    )

                else:
                    raise RuntimeError(
                        "Unexpected parameter "
                        f"dimension: "
                        f"{param.dim()}"
                    )

        wd_params = []
        nowd_params = []

        lr_mul_wd_params = []
        lr_mul_nowd_params = []

        for name, child in (
            self.named_children()
        ):

            if (
                "head" in name
                or "aux" in name
                or name
                == "boundary_refine"
            ):
                add_param_to_list(
                    child,
                    lr_mul_wd_params,
                    lr_mul_nowd_params,
                )

            else:
                add_param_to_list(
                    child,
                    wd_params,
                    nowd_params,
                )

        return (
            wd_params,
            nowd_params,
            lr_mul_wd_params,
            lr_mul_nowd_params,
        )
