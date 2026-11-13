#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def split_of(path: Path):
    parts = {p.lower() for p in path.parts}

    if "train" in parts:
        return "train"

    if (
        "valid" in parts
        or "validation" in parts
        or "val" in parts
    ):
        return "val"

    return "unknown"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--mapping",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    mapping_path = Path(args.mapping).resolve()
    out = Path(args.out_dir).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    with mapping_path.open() as f:
        cfg = yaml.safe_load(f)

    source_cfg = cfg["source_classes"]

    index_to_target = {
        int(k): int(v["target"])
        for k, v in source_cfg.items()
    }

    index_to_name = {
        int(k): str(v["name"])
        for k, v in source_cfg.items()
    }

    expected_indices = set(
        index_to_target
    )

    labels = sorted(
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and p.name.lower() == "labels.png"
        )
    )

    if len(labels) != 1076:
        raise RuntimeError(
            f"Expected 1076 masks, found {len(labels)}"
        )

    raw_counts = Counter()

    mapped_counts = {
        "all": Counter(),
        "train": Counter(),
        "val": Counter(),
    }

    raw_split_counts = {
        "all": Counter(),
        "train": Counter(),
        "val": Counter(),
    }

    image_counts = Counter()

    unknown_indices = set()

    split_image_counts = Counter()

    for path in tqdm(
        labels,
        desc="Mapping audit",
    ):

        split = split_of(path)

        if split not in {
            "train",
            "val",
        }:
            raise RuntimeError(
                f"Unknown split for {path}"
            )

        split_image_counts[
            split
        ] += 1

        with Image.open(path) as im:

            if im.mode != "P":
                raise RuntimeError(
                    f"Expected P-mode mask: "
                    f"{path} mode={im.mode}"
                )

            arr = np.asarray(im)

        values, counts = np.unique(
            arr,
            return_counts=True,
        )

        present = set()

        for value, count in zip(
            values,
            counts,
        ):

            src = int(value)
            n = int(count)

            raw_counts[src] += n

            raw_split_counts[
                "all"
            ][src] += n

            raw_split_counts[
                split
            ][src] += n

            if src not in index_to_target:

                unknown_indices.add(src)
                continue

            target = index_to_target[
                src
            ]

            mapped_counts[
                "all"
            ][target] += n

            mapped_counts[
                split
            ][target] += n

            present.add(src)

        for src in present:
            image_counts[src] += 1

    observed_indices = set(
        raw_counts
    )

    if unknown_indices:
        raise RuntimeError(
            "Unknown source indices: "
            f"{sorted(unknown_indices)}"
        )

    if observed_indices != expected_indices:
        raise RuntimeError(
            "Observed raw indices differ "
            "from frozen mapping. "
            f"observed={sorted(observed_indices)} "
            f"expected={sorted(expected_indices)}"
        )

    rows = []

    for subset in [
        "all",
        "train",
        "val",
    ]:

        counts = mapped_counts[
            subset
        ]

        total = sum(
            counts.values()
        )

        ignored = counts.get(
            255,
            0,
        )

        valid = (
            total
            - ignored
        )

        sky = counts.get(
            0,
            0,
        )

        traversable = counts.get(
            1,
            0,
        )

        non_traversable = counts.get(
            2,
            0,
        )

        coverage = (
            valid / total
            if total
            else 0.0
        )

        print()
        print("=" * 80)
        print(subset.upper())
        print("=" * 80)

        print(
            "images:",
            (
                len(labels)
                if subset == "all"
                else split_image_counts[subset]
            )
        )

        print(
            "total pixels:",
            total,
        )

        print(
            "valid pixels:",
            valid,
        )

        print(
            "ignore pixels:",
            ignored,
        )

        print(
            f"coverage: "
            f"{coverage * 100:.6f}%"
        )

        for target, name in [
            (0, "sky"),
            (1, "traversable"),
            (2, "non_traversable"),
        ]:

            n = counts.get(
                target,
                0,
            )

            share_valid = (
                n / valid
                if valid
                else 0.0
            )

            print(
                f"{name:<18} "
                f"{n:>14d} "
                f"{share_valid * 100:>10.6f}%"
            )

        rows.append(
            {
                "subset": subset,
                "images": (
                    len(labels)
                    if subset == "all"
                    else split_image_counts[subset]
                ),
                "total_pixels": total,
                "valid_pixels": valid,
                "ignore_pixels": ignored,
                "coverage_percent":
                    coverage * 100.0,
                "sky_pixels": sky,
                "sky_valid_percent":
                    100.0 * sky / valid,
                "traversable_pixels":
                    traversable,
                "traversable_valid_percent":
                    100.0
                    * traversable
                    / valid,
                "non_traversable_pixels":
                    non_traversable,
                "non_traversable_valid_percent":
                    100.0
                    * non_traversable
                    / valid,
            }
        )

    summary_csv = (
        out /
        "rugd3_mapping_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    source_csv = (
        out /
        "source_class_stats.csv"
    )

    with source_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "raw_index",
                "source_class",
                "target_id",
                "target_class",
                "pixels",
                "images",
            ]
        )

        target_names = {
            0: "sky",
            1: "traversable",
            2: "non_traversable",
            255: "ignore",
        }

        for src in sorted(
            expected_indices
        ):

            target = index_to_target[
                src
            ]

            writer.writerow(
                [
                    src,
                    index_to_name[src],
                    target,
                    target_names[target],
                    raw_counts[src],
                    image_counts[src],
                ]
            )

    summary = {
        "mapping_file":
            str(mapping_path),

        "images_total":
            len(labels),

        "split_images":
            dict(split_image_counts),

        "observed_indices":
            sorted(observed_indices),

        "unknown_indices":
            sorted(unknown_indices),

        "subsets":
            rows,
    }

    (
        out /
        "mapping_audit.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 80)
    print(
        "YCOR -> RUGD3 MAPPING AUDIT PASSED"
    )
    print("=" * 80)

    print(summary_csv)
    print(source_csv)


if __name__ == "__main__":
    main()
