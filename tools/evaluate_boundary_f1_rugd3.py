#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys

sys.path.insert(0, ".")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import set_cfg_from_file
from lib.data import get_data_loader
from lib.models import model_factory


def get_round_size(size, divisor=32):
    return [
        math.ceil(value / divisor) * divisor
        for value in size
    ]


def semantic_boundary(
    mask,
    valid,
    mode="all",
):
    """
    mask: H x W int ndarray
    valid: H x W bool ndarray

    mode:
        all:
            任意两个不同有效类别之间的边界

        traversability:
            只计算 class 1 <-> class 2
    """

    edge = np.zeros(
        mask.shape,
        dtype=np.bool_,
    )

    # -----------------------------
    # horizontal neighbours
    # -----------------------------
    left = mask[:, :-1]
    right = mask[:, 1:]

    pair_valid = (
        valid[:, :-1]
        & valid[:, 1:]
    )

    if mode == "all":
        difference = (
            left != right
        ) & pair_valid

    elif mode == "traversability":
        difference = (
            (
                (left == 1)
                & (right == 2)
            )
            |
            (
                (left == 2)
                & (right == 1)
            )
        ) & pair_valid

    else:
        raise ValueError(
            f"Unknown boundary mode: {mode}"
        )

    edge[:, :-1] |= difference
    edge[:, 1:] |= difference

    # -----------------------------
    # vertical neighbours
    # -----------------------------
    top = mask[:-1, :]
    bottom = mask[1:, :]

    pair_valid = (
        valid[:-1, :]
        & valid[1:, :]
    )

    if mode == "all":
        difference = (
            top != bottom
        ) & pair_valid

    else:
        difference = (
            (
                (top == 1)
                & (bottom == 2)
            )
            |
            (
                (top == 2)
                & (bottom == 1)
            )
        ) & pair_valid

    edge[:-1, :] |= difference
    edge[1:, :] |= difference

    # GT ignore 区域不参与
    edge &= valid

    return edge


def dilation(
    edge,
    tolerance,
):
    """
    使用椭圆 kernel 近似 tolerance pixel
    的欧氏距离容差。
    """

    if tolerance <= 0:
        return edge

    size = 2 * tolerance + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    dilated = cv2.dilate(
        edge.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return dilated.astype(bool)


class BoundaryMetric:
    def __init__(
        self,
        tolerance=3,
    ):
        self.tolerance = tolerance

        self.pred_total = 0
        self.gt_total = 0

        self.pred_matched = 0
        self.gt_matched = 0

    def update(
        self,
        pred_edge,
        gt_edge,
    ):
        pred_dilated = dilation(
            pred_edge,
            self.tolerance,
        )

        gt_dilated = dilation(
            gt_edge,
            self.tolerance,
        )

        self.pred_total += int(
            pred_edge.sum()
        )

        self.gt_total += int(
            gt_edge.sum()
        )

        self.pred_matched += int(
            (
                pred_edge
                & gt_dilated
            ).sum()
        )

        self.gt_matched += int(
            (
                gt_edge
                & pred_dilated
            ).sum()
        )

    def compute(self):
        eps = 1e-12

        precision = (
            self.pred_matched
            / max(
                self.pred_total,
                1,
            )
        )

        recall = (
            self.gt_matched
            / max(
                self.gt_total,
                1,
            )
        )

        f1 = (
            2 * precision * recall
            / max(
                precision + recall,
                eps,
            )
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_pixels": self.pred_total,
            "gt_pixels": self.gt_total,
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--weight-path",
        required=True,
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    cfg = set_cfg_from_file(
        args.config
    )

    loader = get_data_loader(
        cfg,
        mode="val",
    )

    model = model_factory[
        cfg.model_type
    ](
        cfg.n_cats
    )

    checkpoint = torch.load(
        args.weight_path,
        map_location="cpu",
        weights_only=False,
    )

    # 支持完整 checkpoint
    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict,
        )
    ):
        checkpoint = checkpoint["model"]

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    if hasattr(
        model,
        "aux_mode",
    ):
        model.aux_mode = "eval"

    model.cuda()
    model.eval()

    metric_all = BoundaryMetric(
        tolerance=args.tolerance
    )

    metric_trav = BoundaryMetric(
        tolerance=args.tolerance
    )

    printed_shape = False

    with torch.no_grad():

        for images, labels in tqdm(
            loader
        ):
            labels = labels.squeeze(1)

            target_size = (
                labels.shape[-2:]
            )

            original_size = (
                images.shape[-2:]
            )

            rounded_size = (
                get_round_size(
                    original_size,
                    divisor=32,
                )
            )

            if not printed_shape:
                print(
                    "\nEvaluation shape:"
                    f" original={original_size},"
                    f" rounded={rounded_size},"
                    f" label={target_size}"
                )
                printed_shape = True

            images = F.interpolate(
                images,
                size=rounded_size,
                mode="bilinear",
                align_corners=True,
            )

            images = images.cuda(
                non_blocking=True
            )

            outputs = model(images)

            if isinstance(
                outputs,
                (tuple, list),
            ):
                logits = outputs[0]
            else:
                logits = outputs

            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=True,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            targets = labels.numpy()

            for prediction, target in zip(
                predictions,
                targets,
            ):
                valid = (
                    target != 255
                )

                # prediction 也只在 GT 有效区域评估
                pred_all = semantic_boundary(
                    prediction,
                    valid,
                    mode="all",
                )

                gt_all = semantic_boundary(
                    target,
                    valid,
                    mode="all",
                )

                metric_all.update(
                    pred_all,
                    gt_all,
                )

                pred_trav = semantic_boundary(
                    prediction,
                    valid,
                    mode="traversability",
                )

                gt_trav = semantic_boundary(
                    target,
                    valid,
                    mode="traversability",
                )

                metric_trav.update(
                    pred_trav,
                    gt_trav,
                )

    all_result = (
        metric_all.compute()
    )

    trav_result = (
        metric_trav.compute()
    )

    print(
        "\n================================"
    )

    print(
        f"Boundary tolerance: "
        f"{args.tolerance} pixels"
    )

    print(
        "================================"
    )

    print("\n[All semantic boundaries]")

    print(
        f"Precision : "
        f"{all_result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{all_result['recall']:.6f}"
    )

    print(
        f"BF1-All   : "
        f"{all_result['f1']:.6f}"
    )

    print(
        "\n[Traversable <-> "
        "Non-traversable boundary]"
    )

    print(
        f"Precision : "
        f"{trav_result['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{trav_result['recall']:.6f}"
    )

    print(
        f"BF1-Trav  : "
        f"{trav_result['f1']:.6f}"
    )


if __name__ == "__main__":
    main()
