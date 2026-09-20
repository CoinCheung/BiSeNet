#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import set_cfg_from_file
from lib.data import get_data_loader
from lib.models import model_factory


CLASS_NAMES = [
    "sky",
    "traversable",
    "non_traversable",
    # "obstacle",
]


def get_round_size(
    size: tuple[int, int],
    divisor: int = 32,
) -> tuple[int, int]:
    """
    将输入高度和宽度向上取整到 divisor 的倍数。

    例如：
        550 × 688 -> 576 × 704
    """
    height, width = size

    rounded_height = math.ceil(height / divisor) * divisor
    rounded_width = math.ceil(width / divisor) * divisor

    return rounded_height, rounded_width


def apply_size_preprocessor(
    images: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    复现 tools/evaluate.py 中 SizePreprocessor 的行为。
    """
    target_size = None

    eval_start_shape = cfg.get("eval_start_shape")
    eval_start_shortside = cfg.get("eval_start_shortside")
    eval_start_longside = cfg.get("eval_start_longside")

    if eval_start_shape is not None:
        target_size = tuple(eval_start_shape)

    elif eval_start_shortside is not None:
        height, width = images.shape[-2:]
        shortside = int(eval_start_shortside)

        if height < width:
            new_height = shortside
            new_width = int(shortside / height * width)
        else:
            new_height = int(shortside / width * height)
            new_width = shortside

        target_size = (new_height, new_width)

    elif eval_start_longside is not None:
        height, width = images.shape[-2:]
        longside = int(eval_start_longside)

        if max(height, width) > longside:
            if height < width:
                new_height = int(longside / width * height)
                new_width = longside
            else:
                new_height = longside
                new_width = int(longside / height * width)

            target_size = (new_height, new_width)

    if target_size is not None:
        images = F.interpolate(
            images,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return images


def update_confusion(
    confusion: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> None:
    """
    confusion 的行是真实类别，列是预测类别。
    """
    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < num_classes)
    )

    target = target[valid].astype(np.int64)
    prediction = prediction[valid].astype(np.int64)

    indices = (
        target * num_classes
        + prediction
    )

    confusion += np.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def print_confusion_results(
    confusion: np.ndarray,
) -> None:
    row_sum = confusion.sum(
        axis=1,
        keepdims=True,
    )

    normalized = (
        confusion
        / np.maximum(row_sum, 1)
    )

    true_positive = np.diag(confusion)
    false_positive = (
        confusion.sum(axis=0)
        - true_positive
    )
    false_negative = (
        confusion.sum(axis=1)
        - true_positive
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + 1
        )
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
            + 1
        )
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
            + 1
        )
    )

    f1 = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12,
        )
    )

    print("\nRaw confusion matrix")
    print("Rows = ground truth")
    print("Columns = prediction")
    print(confusion)

    print("\nNormalized confusion matrix (%)")
    print("Rows = ground truth")
    print("Columns = prediction")

    header = f"{'GT / Pred':20s}"

    for name in CLASS_NAMES:
        header += f"{name:20s}"

    print(header)

    for row_index, class_name in enumerate(CLASS_NAMES):
        line = f"{class_name:20s}"

        for value in normalized[row_index]:
            line += f"{value * 100:19.2f}%"

        print(line)

    print("\nPer-class metrics")

    print(
        f"{'Class':20s}"
        f"{'IoU':>12s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'GT ratio':>12s}"
    )

    total = confusion.sum()

    for index, class_name in enumerate(CLASS_NAMES):
        gt_ratio = (
            confusion[index].sum()
            / max(total, 1)
        )

        print(
            f"{class_name:20s}"
            f"{iou[index]:12.6f}"
            f"{precision[index]:12.6f}"
            f"{recall[index]:12.6f}"
            f"{f1[index]:12.6f}"
            f"{gt_ratio:12.6f}"
        )

    print(f"\nmIoU: {iou.mean():.6f}")
    print(f"Macro F1: {f1.mean():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--weight-path",
        required=True,
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
        weights_only=True,
    )

    # 同时兼容纯 state_dict 和完整 checkpoint
    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
        and isinstance(checkpoint["model"], dict)
    ):
        checkpoint = checkpoint["model"]

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    # 只返回主分割头，不计算四个辅助头
    if hasattr(model, "aux_mode"):
        model.aux_mode = "eval"

    model.cuda()
    model.eval()

    confusion = np.zeros(
        (
            cfg.n_cats,
            cfg.n_cats,
        ),
        dtype=np.int64,
    )

    printed_shape = False

    with torch.no_grad():
        for images, labels in tqdm(loader):
            # 标签保持原始尺寸，用于最终评估
            labels = labels.squeeze(1)
            label_size = labels.shape[-2:]

            # 复现官方评估的起始尺寸预处理
            images = apply_size_preprocessor(
                images,
                cfg,
            )

            original_image_size = (
                images.shape[-2:]
            )

            # 关键修复：向上取整到 32 的倍数
            rounded_height, rounded_width = get_round_size(
                original_image_size,
                divisor=32,
            )

            if not printed_shape:
                print(
                    "\nEvaluation shape:"
                    f" original={original_image_size},"
                    f" rounded="
                    f"({rounded_height}, {rounded_width}),"
                    f" label={label_size}"
                )
                printed_shape = True

            images = F.interpolate(
                images,
                size=(
                    rounded_height,
                    rounded_width,
                ),
                mode="bilinear",
                align_corners=True,
            )

            images = images.cuda(
                non_blocking=True,
            )

            outputs = model(images)

            if isinstance(
                outputs,
                (tuple, list),
            ):
                logits = outputs[0]
            else:
                logits = outputs

            # 模型输出恢复到原始标签尺寸
            logits = F.interpolate(
                logits,
                size=label_size,
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
                update_confusion(
                    confusion=confusion,
                    prediction=prediction,
                    target=target,
                    num_classes=cfg.n_cats,
                    ignore_index=loader.dataset.lb_ignore,
                )

    print_confusion_results(
        confusion
    )


if __name__ == "__main__":
    main()
