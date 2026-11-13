#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


EXPECTED_TOTAL = 1076
EXPECTED_TRAIN = 931
EXPECTED_VAL = 145


# ------------------------------------------------------------
# Reference palette
#
# This is NOT yet a RUGD3 ontology mapping.
#
# It is the RGB palette commonly used by the public
# YCOR archive and public YCOR conversion tools.
# The audit verifies the actual archive against it.
# ------------------------------------------------------------

REFERENCE_CLASSES = {
    (255, 255, 255):
        "background",

    (40, 80, 0):
        "high_vegetation",

    (128, 255, 0):
        "traversable_grass",

    (178, 176, 153):
        "smooth_trail",

    (255, 0, 0):
        "obstacle",

    (1, 88, 255):
        "sky",

    (156, 76, 30):
        "rough_trail",

    (255, 0, 128):
        "puddle",

    (0, 160, 0):
        "non_traversable_low_vegetation",
}


def sha256_file(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def classify_split(
    sample_dir: Path,
):

    parts = [
        p.lower()
        for p in sample_dir.parts
    ]

    if "train" in parts:
        return "train"

    if (
        "valid" in parts
        or "validation" in parts
        or "val" in parts
    ):
        return "val"

    return "unknown"


def discover_samples(
    root: Path,
):

    rgb_files = sorted(
        root.rglob("rgb.jpg")
    )

    label_files = sorted(
        root.rglob("labels.png")
    )

    samples = []

    orphan_rgb = []

    for rgb in rgb_files:

        label = (
            rgb.parent /
            "labels.png"
        )

        if not label.is_file():

            orphan_rgb.append(
                str(rgb)
            )

            continue

        split = classify_split(
            rgb.parent
        )

        rel_dir = str(
            rgb.parent.relative_to(
                root
            )
        )

        samples.append(
            {
                "key":
                    rel_dir,

                "split":
                    split,

                "rgb":
                    rgb,

                "label":
                    label,
            }
        )

    orphan_labels = []

    for label in label_files:

        rgb = (
            label.parent /
            "rgb.jpg"
        )

        if not rgb.is_file():

            orphan_labels.append(
                str(label)
            )

    return (
        samples,
        rgb_files,
        label_files,
        orphan_rgb,
        orphan_labels,
    )


def encode_rgb(
    arr,
):

    arr = arr.astype(
        np.uint32
    )

    return (
        (arr[..., 0] << 16)
        |
        (arr[..., 1] << 8)
        |
        arr[..., 2]
    )


def decode_color(
    value,
):

    value = int(value)

    return (
        (value >> 16)
        & 255,

        (value >> 8)
        & 255,

        value
        & 255,
    )


def inspect_sample(
    item,
):

    key = item[
        "key"
    ]

    split = item[
        "split"
    ]

    rgb_path = Path(
        item[
            "rgb"
        ]
    )

    label_path = Path(
        item[
            "label"
        ]
    )

    result = {
        "key":
            key,

        "split":
            split,

        "rgb":
            str(rgb_path),

        "label":
            str(label_path),

        "error":
            None,
    }

    try:

        with Image.open(
            rgb_path
        ) as im:

            rgb_size = im.size
            rgb_mode = im.mode

        with Image.open(
            label_path
        ) as lm:

            label_size = lm.size
            label_mode = lm.mode

            # Convert any P/RGB/RGBA label into canonical RGB
            # for exact color auditing.
            label_rgb = np.asarray(
                lm.convert("RGB"),
                dtype=np.uint8,
            )

        encoded = encode_rgb(
            label_rgb
        )

        values, counts = np.unique(
            encoded,
            return_counts=True,
        )

        color_counts = {
            int(v): int(c)
            for v, c
            in zip(
                values,
                counts,
            )
        }

        result.update(
            {
                "rgb_size":
                    list(
                        rgb_size
                    ),

                "label_size":
                    list(
                        label_size
                    ),

                "rgb_mode":
                    rgb_mode,

                "label_mode":
                    label_mode,

                "shape_match":
                    rgb_size
                    == label_size,

                "color_counts":
                    color_counts,

                "rgb_sha256":
                    sha256_file(
                        rgb_path
                    ),

                "label_sha256":
                    sha256_file(
                        label_path
                    ),
            }
        )

    except Exception as exc:

        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 88
    )

    print(
        "YCOR DATASET / ONTOLOGY AUDIT"
    )

    print(
        "=" * 88
    )

    (
        samples,
        rgb_files,
        label_files,
        orphan_rgb,
        orphan_labels,
    ) = discover_samples(
        data_root
    )

    print(
        "rgb.jpg files :",
        len(
            rgb_files
        ),
    )

    print(
        "labels.png    :",
        len(
            label_files
        ),
    )

    print(
        "paired samples:",
        len(
            samples
        ),
    )

    split_counts = Counter(
        x["split"]
        for x in samples
    )

    print(
        "split counts  :",
        dict(
            split_counts
        ),
    )

    print(
        "orphan RGB    :",
        len(
            orphan_rgb
        ),
    )

    print(
        "orphan labels :",
        len(
            orphan_labels
        ),
    )

    # --------------------------------------------------------
    # Check duplicate sample keys
    # --------------------------------------------------------

    key_counts = Counter(
        x["key"]
        for x in samples
    )

    duplicate_keys = {
        k: v
        for k, v
        in key_counts.items()
        if v != 1
    }

    # --------------------------------------------------------
    # Full pixel audit
    # --------------------------------------------------------

    total_color_counts = Counter()

    split_color_counts = {
        "train":
            Counter(),

        "val":
            Counter(),

        "unknown":
            Counter(),
    }

    color_image_counts = Counter()

    split_color_image_counts = {
        "train":
            Counter(),

        "val":
            Counter(),

        "unknown":
            Counter(),
    }

    rgb_sizes = Counter()
    label_sizes = Counter()

    rgb_modes = Counter()
    label_modes = Counter()

    shape_mismatches = []
    read_errors = []

    results = []

    print()
    print(
        "Reading all images/masks..."
    )

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        iterator = executor.map(
            inspect_sample,
            samples,
            chunksize=8,
        )

        for result in tqdm(
            iterator,
            total=len(samples),
        ):

            results.append(
                result
            )

            if (
                result[
                    "error"
                ]
                is not None
            ):

                read_errors.append(
                    result
                )

                continue

            rgb_sizes[
                tuple(
                    result[
                        "rgb_size"
                    ]
                )
            ] += 1

            label_sizes[
                tuple(
                    result[
                        "label_size"
                    ]
                )
            ] += 1

            rgb_modes[
                result[
                    "rgb_mode"
                ]
            ] += 1

            label_modes[
                result[
                    "label_mode"
                ]
            ] += 1

            if not result[
                "shape_match"
            ]:

                shape_mismatches.append(
                    result
                )

            split = result[
                "split"
            ]

            present_colors = set(
                result[
                    "color_counts"
                ].keys()
            )

            total_color_counts.update(
                result[
                    "color_counts"
                ]
            )

            split_color_counts[
                split
            ].update(
                result[
                    "color_counts"
                ]
            )

            for c in (
                present_colors
            ):

                color_image_counts[
                    c
                ] += 1

                split_color_image_counts[
                    split
                ][c] += 1

    # --------------------------------------------------------
    # Actual colors
    # --------------------------------------------------------

    observed_colors = {
        decode_color(v)
        for v
        in total_color_counts
    }

    reference_colors = set(
        REFERENCE_CLASSES.keys()
    )

    unknown_colors = (
        observed_colors
        - reference_colors
    )

    missing_reference_colors = (
        reference_colors
        - observed_colors
    )

    print()
    print(
        "Observed label colors:"
    )

    for c in sorted(
        observed_colors
    ):

        name = (
            REFERENCE_CLASSES.get(
                c,
                "UNKNOWN",
            )
        )

        encoded = (
            (c[0] << 16)
            |
            (c[1] << 8)
            |
            c[2]
        )

        print(
            f"  {c!s:<18s} "
            f"{name:<36s} "
            f"pixels="
            f"{total_color_counts[encoded]}"
        )

    print()
    print(
        "Unknown colors:",
        sorted(
            unknown_colors
        ),
    )

    print(
        "Missing reference colors:",
        sorted(
            missing_reference_colors
        ),
    )

    # --------------------------------------------------------
    # Cross-split exact RGB duplicate audit
    # --------------------------------------------------------

    train_hashes = {}

    val_hashes = {}

    for r in results:

        if r[
            "error"
        ] is not None:
            continue

        h = r[
            "rgb_sha256"
        ]

        if (
            r[
                "split"
            ]
            == "train"
        ):

            train_hashes.setdefault(
                h,
                []
            ).append(
                r[
                    "key"
                ]
            )

        elif (
            r[
                "split"
            ]
            == "val"
        ):

            val_hashes.setdefault(
                h,
                []
            ).append(
                r[
                    "key"
                ]
            )

    common_hashes = (
        set(
            train_hashes
        )
        &
        set(
            val_hashes
        )
    )

    cross_split_duplicates = []

    for h in sorted(
        common_hashes
    ):

        cross_split_duplicates.append(
            {
                "sha256":
                    h,

                "train":
                    train_hashes[
                        h
                    ],

                "val":
                    val_hashes[
                        h
                    ],
            }
        )

    # --------------------------------------------------------
    # CSV sample index
    # --------------------------------------------------------

    sample_csv = (
        out_dir /
        "sample_index.csv"
    )

    with sample_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "split",
                "key",
                "rgb",
                "label",
                "rgb_sha256",
                "label_sha256",
            ]
        )

        for r in sorted(
            results,
            key=lambda x:
                (
                    x[
                        "split"
                    ],
                    x[
                        "key"
                    ],
                ),
        ):

            writer.writerow(
                [
                    r[
                        "split"
                    ],

                    r[
                        "key"
                    ],

                    r[
                        "rgb"
                    ],

                    r[
                        "label"
                    ],

                    r.get(
                        "rgb_sha256",
                        "",
                    ),

                    r.get(
                        "label_sha256",
                        "",
                    ),
                ]
            )

    # --------------------------------------------------------
    # Color statistics CSV
    # --------------------------------------------------------

    stats_csv = (
        out_dir /
        "mask_color_stats.csv"
    )

    total_pixels = sum(
        total_color_counts.values()
    )

    train_pixels = sum(
        split_color_counts[
            "train"
        ].values()
    )

    val_pixels = sum(
        split_color_counts[
            "val"
        ].values()
    )

    with stats_csv.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "r",
                "g",
                "b",
                "reference_name",
                "all_pixels",
                "all_percent",
                "all_images",
                "train_pixels",
                "train_percent",
                "train_images",
                "val_pixels",
                "val_percent",
                "val_images",
            ]
        )

        for c in sorted(
            observed_colors
            |
            reference_colors
        ):

            encoded = (
                (c[0] << 16)
                |
                (c[1] << 8)
                |
                c[2]
            )

            ap = (
                total_color_counts.get(
                    encoded,
                    0,
                )
            )

            tp = (
                split_color_counts[
                    "train"
                ].get(
                    encoded,
                    0,
                )
            )

            vp = (
                split_color_counts[
                    "val"
                ].get(
                    encoded,
                    0,
                )
            )

            writer.writerow(
                [
                    c[0],
                    c[1],
                    c[2],

                    REFERENCE_CLASSES.get(
                        c,
                        "UNKNOWN",
                    ),

                    ap,

                    (
                        100.0
                        * ap
                        / total_pixels
                        if total_pixels
                        else 0.0
                    ),

                    color_image_counts.get(
                        encoded,
                        0,
                    ),

                    tp,

                    (
                        100.0
                        * tp
                        / train_pixels
                        if train_pixels
                        else 0.0
                    ),

                    split_color_image_counts[
                        "train"
                    ].get(
                        encoded,
                        0,
                    ),

                    vp,

                    (
                        100.0
                        * vp
                        / val_pixels
                        if val_pixels
                        else 0.0
                    ),

                    split_color_image_counts[
                        "val"
                    ].get(
                        encoded,
                        0,
                    ),
                ]
            )

    # --------------------------------------------------------
    # Audit JSON
    # --------------------------------------------------------

    audit = {
        "expected": {
            "total":
                EXPECTED_TOTAL,

            "train":
                EXPECTED_TRAIN,

            "val":
                EXPECTED_VAL,
        },

        "data_root":
            str(
                data_root
            ),

        "rgb_file_count":
            len(
                rgb_files
            ),

        "label_file_count":
            len(
                label_files
            ),

        "paired_sample_count":
            len(
                samples
            ),

        "split_counts":
            dict(
                split_counts
            ),

        "orphan_rgb_count":
            len(
                orphan_rgb
            ),

        "orphan_label_count":
            len(
                orphan_labels
            ),

        "orphan_rgb_examples":
            orphan_rgb[:20],

        "orphan_label_examples":
            orphan_labels[:20],

        "duplicate_sample_key_count":
            len(
                duplicate_keys
            ),

        "duplicate_sample_key_examples":
            dict(
                list(
                    duplicate_keys.items()
                )[:20]
            ),

        "read_error_count":
            len(
                read_errors
            ),

        "read_error_examples":
            read_errors[:20],

        "shape_mismatch_count":
            len(
                shape_mismatches
            ),

        "shape_mismatch_examples":
            shape_mismatches[:20],

        "rgb_sizes":
            {
                f"{k[0]}x{k[1]}":
                    v
                for k, v
                in rgb_sizes.items()
            },

        "label_sizes":
            {
                f"{k[0]}x{k[1]}":
                    v
                for k, v
                in label_sizes.items()
            },

        "rgb_modes":
            dict(
                rgb_modes
            ),

        "label_modes":
            dict(
                label_modes
            ),

        "reference_palette":
            {
                f"{k[0]},{k[1]},{k[2]}":
                    v
                for k, v
                in REFERENCE_CLASSES.items()
            },

        "observed_colors":
            [
                list(c)
                for c in sorted(
                    observed_colors
                )
            ],

        "unknown_colors":
            [
                list(c)
                for c in sorted(
                    unknown_colors
                )
            ],

        "missing_reference_colors":
            [
                list(c)
                for c in sorted(
                    missing_reference_colors
                )
            ],

        "total_pixels":
            total_pixels,

        "train_pixels":
            train_pixels,

        "val_pixels":
            val_pixels,

        "cross_split_duplicate_rgb_count":
            len(
                cross_split_duplicates
            ),

        "cross_split_duplicate_rgb_examples":
            cross_split_duplicates[:20],
    }

    audit_json = (
        out_dir /
        "dataset_audit.json"
    )

    audit_json.write_text(
        json.dumps(
            audit,
            indent=2,
        )
        + "\n"
    )

    # --------------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------------

    summary = (
        out_dir /
        "audit_summary.txt"
    )

    lines = [
        "YCOR DATASET / ONTOLOGY AUDIT",
        "=" * 72,

        f"rgb_file_count="
        f"{len(rgb_files)}",

        f"label_file_count="
        f"{len(label_files)}",

        f"paired_sample_count="
        f"{len(samples)}",

        f"split_counts="
        f"{dict(split_counts)}",

        f"orphan_rgb_count="
        f"{len(orphan_rgb)}",

        f"orphan_label_count="
        f"{len(orphan_labels)}",

        f"duplicate_sample_key_count="
        f"{len(duplicate_keys)}",

        f"read_error_count="
        f"{len(read_errors)}",

        f"shape_mismatch_count="
        f"{len(shape_mismatches)}",

        f"unknown_colors="
        f"{sorted(unknown_colors)}",

        f"missing_reference_colors="
        f"{sorted(missing_reference_colors)}",

        f"cross_split_duplicate_rgb_count="
        f"{len(cross_split_duplicates)}",

        f"rgb_sizes="
        f"{dict(rgb_sizes)}",

        f"label_sizes="
        f"{dict(label_sizes)}",

        f"rgb_modes="
        f"{dict(rgb_modes)}",

        f"label_modes="
        f"{dict(label_modes)}",
    ]

    summary.write_text(
        "\n".join(
            lines
        )
        + "\n"
    )

    print()
    print(
        "=" * 88
    )

    print(
        "AUDIT OUTPUT"
    )

    print(
        "=" * 88
    )

    print(
        audit_json
    )

    print(
        stats_csv
    )

    print(
        sample_csv
    )

    print(
        summary
    )


if __name__ == "__main__":
    main()
