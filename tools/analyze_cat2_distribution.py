#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CAT2_CLASSES = {
    "bush",
    "water",
    "rock-bed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets_raw/RUGD"
        ),
    )

    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path(
            "datasets/rugd4/sequence_split.json"
        ),
    )

    return parser.parse_args()


def rgb_to_code(
    red: int,
    green: int,
    blue: int,
) -> np.uint32:
    """将一个 RGB 颜色编码为 24 位整数。"""
    return np.uint32(
        (red << 16)
        | (green << 8)
        | blue
    )


def load_cat2_colors(
    colormap_file: Path,
) -> dict[str, np.uint32]:
    """
    读取：
        class_id class_name R G B

    返回：
        {
            "bush": encoded_color,
            "water": encoded_color,
            "rock-bed": encoded_color,
        }
    """
    result: dict[str, np.uint32] = {}

    for raw_line in colormap_file.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if len(fields) < 5:
            raise ValueError(
                f"Invalid colormap line: {raw_line}"
            )

        class_name = " ".join(fields[1:-3])
        red, green, blue = map(
            int,
            fields[-3:],
        )

        if class_name in CAT2_CLASSES:
            result[class_name] = rgb_to_code(
                red,
                green,
                blue,
            )

    missing = CAT2_CLASSES - set(result)

    if missing:
        raise ValueError(
            "The following classes were not found in "
            f"the colormap: {sorted(missing)}"
        )

    return result


def read_rgb_code_image(
    mask_path: Path,
) -> np.ndarray:
    """
    OpenCV 默认读取 BGR，将其直接编码为与 RGB 颜色相同的整数：
        R << 16 | G << 8 | B
    """
    bgr = cv2.imread(
        str(mask_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            f"Cannot read mask: {mask_path}"
        )

    blue = bgr[:, :, 0].astype(np.uint32)
    green = bgr[:, :, 1].astype(np.uint32)
    red = bgr[:, :, 2].astype(np.uint32)

    return (
        (red << 16)
        | (green << 8)
        | blue
    )


def analyze_sequence(
    annotation_dir: Path,
    cat2_colors: dict[str, np.uint32],
) -> tuple[Counter, int]:
    mask_paths = sorted(
        annotation_dir.glob("*.png")
    )

    if not mask_paths:
        raise RuntimeError(
            f"No PNG masks found in {annotation_dir}"
        )

    result = Counter()

    progress = tqdm(
        mask_paths,
        desc=annotation_dir.name,
        unit="image",
        leave=False,
    )

    for mask_path in progress:
        encoded = read_rgb_code_image(
            mask_path
        )

        for class_name, color_code in cat2_colors.items():
            result[class_name] += int(
                np.count_nonzero(
                    encoded == color_code
                )
            )

    return result, len(mask_paths)


def print_counter(
    title: str,
    counter: Counter,
    image_count: int | None = None,
) -> None:
    total = sum(counter.values())

    print(
        f"\n===== {title} =====",
        flush=True,
    )

    if image_count is not None:
        print(
            f"images: {image_count}",
            flush=True,
        )

    print(
        f"cat2 pixels: {total}",
        flush=True,
    )

    for class_name in sorted(CAT2_CLASSES):
        count = counter[class_name]
        ratio = count / max(total, 1)

        print(
            f"{class_name:15s} "
            f"pixels={count:12d}, "
            f"ratio={ratio:9.4%}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    annotation_root = (
        args.raw_root
        / "RUGD_annotations"
    )

    colormap_file = (
        args.raw_root
        / "RUGD_annotation-colormap.txt"
    )

    if not annotation_root.is_dir():
        raise FileNotFoundError(
            annotation_root
        )

    if not colormap_file.is_file():
        raise FileNotFoundError(
            colormap_file
        )

    if not args.split_file.is_file():
        raise FileNotFoundError(
            args.split_file
        )

    split = json.loads(
        args.split_file.read_text(
            encoding="utf-8"
        )
    )

    cat2_colors = load_cat2_colors(
        colormap_file
    )

    print(
        "CAT2 colors:",
        {
            name: int(code)
            for name, code in cat2_colors.items()
        },
        flush=True,
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_counter = Counter()
        split_image_count = 0

        print(
            f"\n######## {split_name} ########",
            flush=True,
        )

        for sequence in split[split_name]:
            annotation_dir = (
                annotation_root
                / sequence
            )

            sequence_counter, image_count = (
                analyze_sequence(
                    annotation_dir,
                    cat2_colors,
                )
            )

            split_counter.update(
                sequence_counter
            )

            split_image_count += image_count

            print_counter(
                sequence,
                sequence_counter,
                image_count,
            )

        print_counter(
            f"{split_name} TOTAL",
            split_counter,
            split_image_count,
        )


if __name__ == "__main__":
    main()
