#!/usr/bin/env python3

import csv
import re
from pathlib import Path

import numpy as np


OUT = Path(
    "experiments/"
    "rugd3_sequence_eval/"
    "DBGA_seed123"
)

SEQ_ROOT = Path(
    "datasets/"
    "rugd3/"
    "sequence_eval"
)


SPLITS = {
    "val": [
        "trail-4",
        "trail-5",
    ],

    "test": [
        "trail",
        "trail-10",
        "trail-12",
        "trail-14",
    ],
}


CLASS_NAMES = [
    "sky",
    "traversable",
    "non_traversable",
]


def read_nonempty_count(
    path,
):
    return sum(
        1
        for line
        in path.read_text().splitlines()
        if line.strip()
    )


def parse_confusion(
    path,
):

    text = path.read_text(
        errors="replace"
    )

    if "Raw confusion matrix" not in text:
        raise RuntimeError(
            f"No raw confusion matrix: {path}"
        )

    raw_block = (
        text
        .split(
            "Raw confusion matrix",
            1,
        )[1]
        .split(
            "Normalized confusion matrix",
            1,
        )[0]
    )

    nums = [
        int(x)
        for x
        in re.findall(
            r"\d+",
            raw_block,
        )
    ]

    if len(nums) != 9:
        raise RuntimeError(
            f"Expected 9 confusion "
            f"entries in {path}, "
            f"got {len(nums)}: "
            f"{nums}"
        )

    matrix = np.array(
        nums,
        dtype=np.int64,
    ).reshape(
        3,
        3,
    )

    if "Per-class metrics" not in text:
        raise RuntimeError(
            f"No per-class metrics: {path}"
        )

    metric_block = text.split(
        "Per-class metrics",
        1,
    )[1]

    ious = {}

    for name in CLASS_NAMES:

        match = re.search(
            rf"^{re.escape(name)}"
            rf"\s+([0-9.]+)",
            metric_block,
            flags=re.MULTILINE,
        )

        if not match:
            raise RuntimeError(
                f"Cannot parse IoU for "
                f"{name}: {path}"
            )

        ious[name] = float(
            match.group(1)
        )

    match_miou = re.search(
        r"mIoU:\s*([0-9.]+)",
        text,
    )

    match_macro = re.search(
        r"Macro F1:\s*([0-9.]+)",
        text,
    )

    if not match_miou:
        raise RuntimeError(
            f"Cannot parse mIoU: {path}"
        )

    if not match_macro:
        raise RuntimeError(
            f"Cannot parse Macro F1: {path}"
        )

    return {
        "matrix": matrix,

        "miou":
            float(
                match_miou.group(1)
            ),

        "macro_f1":
            float(
                match_macro.group(1)
            ),

        "sky_iou":
            ious["sky"],

        "trav_iou":
            ious["traversable"],

        "nontrav_iou":
            ious[
                "non_traversable"
            ],
    }


def parse_boundary(
    path,
):

    text = path.read_text(
        errors="replace"
    )

    match_all = re.search(
        r"BF1-All\s*:\s*([0-9.]+)",
        text,
    )

    match_trav = re.search(
        r"BF1-Trav\s*:\s*([0-9.]+)",
        text,
    )

    if not match_all:
        raise RuntimeError(
            f"Cannot parse BF1-All: "
            f"{path}"
        )

    if not match_trav:
        raise RuntimeError(
            f"Cannot parse BF1-Trav: "
            f"{path}"
        )

    return {
        "bf1_all":
            float(
                match_all.group(1)
            ),

        "bf1_trav":
            float(
                match_trav.group(1)
            ),
    }


def miou_from_confusion(
    matrix,
):

    matrix = matrix.astype(
        np.float64
    )

    tp = np.diag(
        matrix
    )

    gt = matrix.sum(
        axis=1
    )

    pred = matrix.sum(
        axis=0
    )

    union = (
        gt
        + pred
        - tp
    )

    iou = tp / union

    return (
        float(
            iou.mean()
        ),
        iou,
    )


def main():

    rows = []

    print()
    print(
        "=" * 100
    )

    print(
        "A / BiSeNetV2 "
        "PER-SEQUENCE RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Split':<7}"
        f"{'Sequence':<12}"
        f"{'N':>6}"
        f"{'mIoU':>10}"
        f"{'Sky':>10}"
        f"{'Trav':>10}"
        f"{'NonTrav':>10}"
        f"{'MacroF1':>10}"
        f"{'BF1-All':>10}"
        f"{'BF1-Trav':>11}"
    )

    for split, sequences in (
        SPLITS.items()
    ):

        summed_matrix = np.zeros(
            (
                3,
                3,
            ),
            dtype=np.int64,
        )

        total_images = 0

        for sequence in sequences:

            confusion_path = (
                OUT /
                f"{split}_{sequence}_"
                f"confusion.log"
            )

            boundary_path = (
                OUT /
                f"{split}_{sequence}_"
                f"boundary_tol3.log"
            )

            ann_path = (
                SEQ_ROOT /
                split /
                f"{sequence}.txt"
            )

            conf = parse_confusion(
                confusion_path
            )

            boundary = parse_boundary(
                boundary_path
            )

            n_images = (
                read_nonempty_count(
                    ann_path
                )
            )

            summed_matrix += (
                conf["matrix"]
            )

            total_images += (
                n_images
            )

            row = {
                "split": split,
                "sequence":
                    sequence,
                "images":
                    n_images,
                "miou":
                    conf["miou"],
                "sky_iou":
                    conf["sky_iou"],
                "trav_iou":
                    conf["trav_iou"],
                "nontrav_iou":
                    conf[
                        "nontrav_iou"
                    ],
                "macro_f1":
                    conf["macro_f1"],
                "bf1_all":
                    boundary[
                        "bf1_all"
                    ],
                "bf1_trav":
                    boundary[
                        "bf1_trav"
                    ],
            }

            rows.append(
                row
            )

            print(
                f"{split:<7}"
                f"{sequence:<12}"
                f"{n_images:>6d}"
                f"{conf['miou']*100:>9.4f}%"
                f"{conf['sky_iou']*100:>9.4f}%"
                f"{conf['trav_iou']*100:>9.4f}%"
                f"{conf['nontrav_iou']*100:>9.4f}%"
                f"{conf['macro_f1']*100:>9.4f}%"
                f"{boundary['bf1_all']*100:>9.4f}%"
                f"{boundary['bf1_trav']*100:>10.4f}%"
            )

        # ----------------------------------------------------
        # Exact global validation
        # ----------------------------------------------------

        global_conf_path = (
            OUT /
            f"global_{split}_"
            f"confusion.log"
        )

        global_conf = (
            parse_confusion(
                global_conf_path
            )
        )

        global_matrix = (
            global_conf["matrix"]
        )

        exact_matrix_match = (
            np.array_equal(
                summed_matrix,
                global_matrix,
            )
        )

        # ----------------------------------------------------
        # Numerical-equivalence validation
        #
        # Per-sequence evaluation can change batch boundaries.
        # For example, an odd-sized sequence produces a final
        # batch of size 1, whereas the global split may keep
        # batch size 2 across sequence boundaries.
        #
        # Tiny floating-point differences can therefore flip
        # argmax decisions for a very small number of pixels.
        #
        # We require:
        #   1. GT row sums to match exactly.
        #   2. Prediction disagreement to remain extremely
        #      small.
        #   3. Reconstructed mIoU to match the global result.
        # ----------------------------------------------------

        matrix_diff = (
            summed_matrix
            - global_matrix
        )

        gt_row_sum_match = (
            np.array_equal(
                summed_matrix.sum(
                    axis=1
                ),
                global_matrix.sum(
                    axis=1
                ),
            )
        )

        matrix_l1_diff = int(
            np.abs(
                matrix_diff
            ).sum()
        )

        # If row sums are identical, every changed prediction
        # contributes +1 to one cell and -1 to another cell.
        changed_prediction_pixels = (
            matrix_l1_diff // 2
        )

        total_pixels = int(
            global_matrix.sum()
        )

        changed_pixel_ratio = (
            changed_prediction_pixels
            / total_pixels
        )

        MAX_CHANGED_PRED_PIXELS = 100

        numerical_matrix_match = (
            gt_row_sum_match
            and
            changed_prediction_pixels
            <= MAX_CHANGED_PRED_PIXELS
        )

        aggregated_miou, agg_iou = (
            miou_from_confusion(
                summed_matrix
            )
        )

        global_miou = (
            global_conf[
                "miou"
            ]
        )

        miou_match = (
            abs(
                aggregated_miou
                - global_miou
            )
            < 1e-6
        )

        source_count = (
            read_nonempty_count(
                Path(
                    f"datasets/rugd3/"
                    f"{split}.txt"
                )
            )
        )

        count_match = (
            total_images
            == source_count
        )

        print()
        print(
            "-" * 100
        )

        print(
            f"{split.upper()} "
            f"STRICT VALIDATION"
        )

        print(
            "-" * 100
        )

        print(
            "sequence image total :",
            total_images,
        )

        print(
            "source image total   :",
            source_count,
        )

        print(
            "image count match    :",
            count_match,
        )

        print(
            "confusion exact match:",
            exact_matrix_match,
        )

        print(
            "GT row sums match    :",
            gt_row_sum_match,
        )

        print(
            "matrix L1 difference :",
            matrix_l1_diff,
        )

        print(
            "changed pred pixels  :",
            changed_prediction_pixels,
        )

        print(
            "total valid pixels   :",
            total_pixels,
        )

        print(
            "changed pixel ratio  :",
            f"{changed_pixel_ratio:.12e}",
        )

        print(
            "numerical matrix pass:",
            numerical_matrix_match,
        )

        print(
            "aggregated mIoU      :",
            f"{aggregated_miou:.6f}",
        )

        print(
            "global mIoU          :",
            f"{global_miou:.6f}",
        )

        print(
            "mIoU match (<1e-6)   :",
            miou_match,
        )

        print(
            "Aggregated class IoU :"
        )

        for name, value in zip(
            CLASS_NAMES,
            agg_iou,
        ):
            print(
                f"  {name:<18s}"
                f"{value:.6f}"
            )

        if not count_match:
            raise RuntimeError(
                f"{split}: image "
                f"count mismatch"
            )

        if not numerical_matrix_match:
            print()
            print(
                "Summed sequence "
                "confusion:"
            )
            print(
                summed_matrix
            )

            print()
            print(
                "Global confusion:"
            )
            print(
                global_matrix
            )

            print()
            print(
                "Difference "
                "(summed - global):"
            )
            print(
                matrix_diff
            )

            raise RuntimeError(
                f"{split}: confusion "
                f"matrix discrepancy "
                f"exceeds tolerance"
            )

        if not miou_match:
            raise RuntimeError(
                f"{split}: aggregated "
                f"mIoU mismatch"
            )

        print(
            f"{split.upper()} "
            f"VALIDATION PASSED"
        )

        print()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUT /
        "per_sequence_metrics.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        fieldnames = [
            "split",
            "sequence",
            "images",
            "miou",
            "sky_iou",
            "trav_iou",
            "nontrav_iou",
            "macro_f1",
            "bf1_all",
            "bf1_trav",
        ]

        writer = (
            csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        "=" * 100
    )

    print(
        "ALL STRICT CHECKS PASSED"
    )

    print(
        "CSV:",
        csv_path,
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
