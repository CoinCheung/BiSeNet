#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F


def semantic_to_boundary(
    label,
    ignore_index=255,
    width=3,
):
    """
    Args:
        label:
            N x H x W

    Returns:
        boundary:
            N x 1 x H x W
            {0, 1}

        valid_mask:
            N x 1 x H x W
    """

    if label.ndim != 3:
        raise ValueError(
            f"Expected N,H,W label, "
            f"got {label.shape}"
        )

    valid = (
        label != ignore_index
    )

    edge = torch.zeros_like(
        label,
        dtype=torch.bool,
    )

    # -------------------------------
    # horizontal
    # -------------------------------

    difference = (
        label[:, :, 1:]
        != label[:, :, :-1]
    )

    pair_valid = (
        valid[:, :, 1:]
        & valid[:, :, :-1]
    )

    difference &= pair_valid

    edge[:, :, 1:] |= difference
    edge[:, :, :-1] |= difference

    # -------------------------------
    # vertical
    # -------------------------------

    difference = (
        label[:, 1:, :]
        != label[:, :-1, :]
    )

    pair_valid = (
        valid[:, 1:, :]
        & valid[:, :-1, :]
    )

    difference &= pair_valid

    edge[:, 1:, :] |= difference
    edge[:, :-1, :] |= difference

    boundary = (
        edge.float().unsqueeze(1)
    )

    valid_mask = (
        valid.float().unsqueeze(1)
    )

    if width > 1:

        if width % 2 == 0:
            raise ValueError(
                "boundary width must be odd"
            )

        boundary = F.max_pool2d(
            boundary,
            kernel_size=width,
            stride=1,
            padding=width // 2,
        )

    # dilation 可能扩展进入 ignore
    boundary *= valid_mask

    return boundary, valid_mask


def boundary_bce_dice_loss(
    logits,
    target,
    valid_mask,
):
    """
    BCE + Soft Dice.

    注意：
    logits 不要提前 sigmoid。
    """

    # -------------------------------
    # BCE
    # -------------------------------

    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    )

    bce = (
        (bce * valid_mask).sum()
        / (
            valid_mask.sum()
            + 1e-6
        )
    )

    # -------------------------------
    # Dice
    # -------------------------------

    probability = torch.sigmoid(
        logits
    )

    probability = (
        probability * valid_mask
    )

    target = (
        target * valid_mask
    )

    intersection = (
        probability * target
    ).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        probability.sum(
            dim=(1, 2, 3)
        )
        +
        target.sum(
            dim=(1, 2, 3)
        )
    )

    dice = (
        1.0
        -
        (
            2.0 * intersection
            + 1.0
        )
        /
        (
            denominator
            + 1.0
        )
    )

    dice = dice.mean()

    return bce + dice
