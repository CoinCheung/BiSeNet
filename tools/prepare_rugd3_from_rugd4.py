#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VALID_SOURCE_IDS = {0, 1, 2, 3, 255}
VALID_TARGET_IDS = {0, 1, 2, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/rugd4"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/rugd3"),
    )

    return parser.parse_args()


def create_image_symlink(
    source_root: Path,
    output_root: Path,
) -> None:
    source_images = (
        source_root / "images"
    ).resolve()

    output_images = (
        output_root / "images"
    )

    if output_images.is_symlink():
        if output_images.resolve() != source_images:
            raise RuntimeError(
                f"{output_images} points to an unexpected path"
            )
        return

    if output_images.exists():
        raise RuntimeError(
            f"{output_images} already exists and is not a symlink"
        )

    os.symlink(
        source_images,
        output_images,
        target_is_directory=True,
    )


def convert_label(
    source_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    source = cv2.imread(
        str(source_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if source is None:
        raise RuntimeError(
            f"Cannot read label: {source_path}"
        )

    source_ids = set(
        np.unique(source).tolist()
    )

    invalid_source = (
        source_ids - VALID_SOURCE_IDS
    )

    if invalid_source:
        raise ValueError(
            f"{source_path}: invalid source IDs "
            f"{sorted(invalid_source)}"
        )

    output = source.copy()

    # 旧 obstacle 合并到新的 non-traversable
    output[source == 3] = 2

    output_ids = set(
        np.unique(output).tolist()
    )

    invalid_output = (
        output_ids - VALID_TARGET_IDS
    )

    if invalid_output:
        raise ValueError(
            f"{output_path}: invalid target IDs "
            f"{sorted(invalid_output)}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        output,
    ):
        raise RuntimeError(
            f"Cannot write label: {output_path}"
        )

    expected_non_traversable = int(
        np.count_nonzero(
            (source == 2) | (source == 3)
        )
    )

    actual_non_traversable = int(
        np.count_nonzero(output == 2)
    )

    return (
        expected_non_traversable,
        actual_non_traversable,
    )


def main() -> None:
    args = parse_args()

    source_root = args.source_root
    output_root = args.output_root

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_image_symlink(
        source_root,
        output_root,
    )

    statistics = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        source_list = (
            source_root / f"{split}.txt"
        )

        output_list = (
            output_root / f"{split}.txt"
        )

        lines = source_list.read_text(
            encoding="utf-8"
        ).splitlines()

        expected_total = 0
        actual_total = 0

        for line in tqdm(
            lines,
            desc=split,
            unit="image",
        ):
            image_rel, label_rel = (
                item.strip()
                for item in line.split(",", 1)
            )

            source_label = (
                source_root / label_rel
            )

            output_label = (
                output_root / label_rel
            )

            expected, actual = convert_label(
                source_label,
                output_label,
            )

            expected_total += expected
            actual_total += actual

        output_list.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        statistics[split] = {
            "images": len(lines),
            "expected_non_traversable_pixels": (
                expected_total
            ),
            "actual_non_traversable_pixels": (
                actual_total
            ),
            "difference": (
                actual_total - expected_total
            ),
        }

    split_source = (
        source_root / "sequence_split.json"
    )

    if split_source.exists():
        shutil.copy2(
            split_source,
            output_root
            / "sequence_split.json",
        )

    ontology = """classes:
  0: sky
  1: traversable
  2: non_traversable
  255: ignore

mapping_from_rugd4:
  0: 0
  1: 1
  2: 2
  3: 2
  255: 255
"""

    (
        output_root / "ontology.yaml"
    ).write_text(
        ontology,
        encoding="utf-8",
    )

    (
        output_root
        / "conversion_statistics.json"
    ).write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            statistics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
