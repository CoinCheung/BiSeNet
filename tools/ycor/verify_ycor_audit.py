#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


EXPECTED_TOTAL = 1076
EXPECTED_TRAIN = 931
EXPECTED_VAL = 145


EXPECTED_COLORS = {
    (255, 255, 255),
    (40, 80, 0),
    (128, 255, 0),
    (178, 176, 153),
    (255, 0, 0),
    (1, 88, 255),
    (156, 76, 30),
    (255, 0, 128),
    (0, 160, 0),
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit-json",
        required=True,
    )

    args = parser.parse_args()

    data = json.loads(
        Path(
            args.audit_json
        ).read_text()
    )

    checks = []

    def check(
        name,
        condition,
        detail,
    ):

        checks.append(
            (
                name,
                bool(condition),
                detail,
            )
        )

    check(
        "rgb_file_count",
        data[
            "rgb_file_count"
        ] == EXPECTED_TOTAL,
        (
            f"observed="
            f"{data['rgb_file_count']} "
            f"expected="
            f"{EXPECTED_TOTAL}"
        ),
    )

    check(
        "label_file_count",
        data[
            "label_file_count"
        ] == EXPECTED_TOTAL,
        (
            f"observed="
            f"{data['label_file_count']} "
            f"expected="
            f"{EXPECTED_TOTAL}"
        ),
    )

    check(
        "paired_sample_count",
        data[
            "paired_sample_count"
        ] == EXPECTED_TOTAL,
        (
            f"observed="
            f"{data['paired_sample_count']} "
            f"expected="
            f"{EXPECTED_TOTAL}"
        ),
    )

    split = data[
        "split_counts"
    ]

    check(
        "train_count",
        split.get(
            "train",
            0,
        ) == EXPECTED_TRAIN,
        (
            f"observed="
            f"{split.get('train', 0)} "
            f"expected="
            f"{EXPECTED_TRAIN}"
        ),
    )

    check(
        "val_count",
        split.get(
            "val",
            0,
        ) == EXPECTED_VAL,
        (
            f"observed="
            f"{split.get('val', 0)} "
            f"expected="
            f"{EXPECTED_VAL}"
        ),
    )

    check(
        "unknown_split_count",
        split.get(
            "unknown",
            0,
        ) == 0,
        (
            f"observed="
            f"{split.get('unknown', 0)}"
        ),
    )

    check(
        "no_orphan_rgb",
        data[
            "orphan_rgb_count"
        ] == 0,
        str(
            data[
                "orphan_rgb_count"
            ]
        ),
    )

    check(
        "no_orphan_labels",
        data[
            "orphan_label_count"
        ] == 0,
        str(
            data[
                "orphan_label_count"
            ]
        ),
    )

    check(
        "no_duplicate_sample_keys",
        data[
            "duplicate_sample_key_count"
        ] == 0,
        str(
            data[
                "duplicate_sample_key_count"
            ]
        ),
    )

    check(
        "no_read_errors",
        data[
            "read_error_count"
        ] == 0,
        str(
            data[
                "read_error_count"
            ]
        ),
    )

    check(
        "image_label_shapes_match",
        data[
            "shape_mismatch_count"
        ] == 0,
        str(
            data[
                "shape_mismatch_count"
            ]
        ),
    )

    observed_colors = {
        tuple(x)
        for x
        in data[
            "observed_colors"
        ]
    }

    unknown_colors = {
        tuple(x)
        for x
        in data[
            "unknown_colors"
        ]
    }

    missing_colors = {
        tuple(x)
        for x
        in data[
            "missing_reference_colors"
        ]
    }

    check(
        "no_unknown_mask_colors",
        len(
            unknown_colors
        ) == 0,
        (
            f"unknown="
            f"{sorted(unknown_colors)}"
        ),
    )

    check(
        "all_reference_colors_observed",
        len(
            missing_colors
        ) == 0,
        (
            f"missing="
            f"{sorted(missing_colors)}"
        ),
    )

    check(
        "exact_reference_palette",
        observed_colors
        == EXPECTED_COLORS,
        (
            f"observed="
            f"{sorted(observed_colors)}"
        ),
    )

    check(
        "no_exact_rgb_duplicates_across_split",
        data[
            "cross_split_duplicate_rgb_count"
        ] == 0,
        str(
            data[
                "cross_split_duplicate_rgb_count"
            ]
        ),
    )

    check(
        "nonzero_pixels",
        data[
            "total_pixels"
        ] > 0,
        str(
            data[
                "total_pixels"
            ]
        ),
    )

    print()
    print(
        "=" * 100
    )

    print(
        "YCOR STRICT DATASET / "
        "ONTOLOGY VALIDATION"
    )

    print(
        "=" * 100
    )

    failed = []

    for (
        name,
        passed,
        detail,
    ) in checks:

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{status:<6s} "
            f"{name:<42s} "
            f"{detail}"
        )

        if not passed:
            failed.append(
                name
            )

    print()
    print(
        "Split counts:",
        split,
    )

    print(
        "Observed colors:",
        sorted(
            observed_colors
        ),
    )

    print(
        "Image sizes:",
        data[
            "rgb_sizes"
        ],
    )

    print(
        "Label sizes:",
        data[
            "label_sizes"
        ],
    )

    print(
        "Label modes:",
        data[
            "label_modes"
        ],
    )

    print()

    if failed:

        print(
            "YCOR STRICT AUDIT FAILED"
        )

        print(
            "Failed checks:",
            failed,
        )

        raise SystemExit(
            1
        )

    print(
        "ALL YCOR STRICT AUDIT CHECKS PASSED"
    )


if __name__ == "__main__":
    main()
