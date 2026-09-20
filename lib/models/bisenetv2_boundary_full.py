#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bisenetv2 import (
    BiSeNetV2,
    CustomArgMax,
)

from .bisenetv2_boundary_refine import (
    LightweightBoundaryRefine,
)


class BoundaryHead(nn.Module):
    """
    Training-only boundary head.

    Input:
        refined BGA feature
        N x 128 x H/8 x W/8

    Output:
        boundary logits
        N x 1 x H x W
    """

    def __init__(
        self,
        in_channels=128,
        hidden_channels=32,
    ):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.predict = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):

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


class BiSeNetV2BoundaryFull(
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

        # ------------------------------
        # Experiment C component
        # ------------------------------

        self.boundary_refine = (
            LightweightBoundaryRefine(
                channels=128,
                bottleneck_channels=32,
                gate_channels=16,
            )
        )

        # ------------------------------
        # Experiment B component
        # training only
        # ------------------------------

        self.boundary_head = BoundaryHead(
            in_channels=128,
            hidden_channels=32,
        )

    def forward(
        self,
        x,
    ):
        input_size = x.shape[2:]

        # ------------------------------
        # Original BiSeNetV2
        # ------------------------------

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

        feat_head = self.bga(
            feat_d,
            feat_s,
        )

        # ------------------------------
        # Experiment C:
        # LBRM
        # ------------------------------

        feat_refined = (
            self.boundary_refine(
                feat_head
            )
        )

        # ------------------------------
        # Main semantic prediction
        # ------------------------------

        logits = self.head(
            feat_refined
        )

        # ------------------------------
        # Training
        # ------------------------------

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

            # Boundary head receives
            # refined feature.
            boundary_logits = (
                self.boundary_head(
                    feat_refined,
                    input_size,
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

        # ------------------------------
        # Evaluation
        #
        # Boundary Head NOT executed.
        # ------------------------------

        elif self.aux_mode == "eval":

            return logits,

        # ------------------------------
        # Prediction
        # ------------------------------

        elif self.aux_mode == "pred":

            pred = CustomArgMax.apply(
                logits,
                1,
            )

            return pred

        else:

            raise NotImplementedError


    def get_params(self):
        """
        Backbone/BGA:
            normal LR

        Semantic heads,
        boundary head,
        boundary refinement:
            10x LR
        """

        def add_params(
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
                or name == "boundary_refine"
            ):

                add_params(
                    child,
                    lr_mul_wd_params,
                    lr_mul_nowd_params,
                )

            else:

                add_params(
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
