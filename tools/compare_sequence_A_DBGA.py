#!/usr/bin/env python3

import csv
from pathlib import Path


A_FILE = Path(
    "experiments/"
    "rugd3_sequence_eval/"
    "A_seed123/"
    "per_sequence_metrics.csv"
)

D_FILE = Path(
    "experiments/"
    "rugd3_sequence_eval/"
    "DBGA_seed123/"
    "per_sequence_metrics.csv"
)


METRICS = [
    ("miou", "mIoU"),
    ("sky_iou", "Sky-IoU"),
    ("trav_iou", "Trav-IoU"),
    (
        "nontrav_iou",
        "NonTrav-IoU",
    ),
    ("bf1_all", "BF1-All"),
    (
        "bf1_trav",
        "BF1-Trav",
    ),
]


def load(path):

    data = {}

    with path.open() as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                row["split"],
                row["sequence"],
            )

            data[key] = row

    return data


def main():

    a = load(A_FILE)
    d = load(D_FILE)

    if set(a) != set(d):

        print(
            "Only in A:",
            sorted(
                set(a)
                - set(d)
            ),
        )

        print(
            "Only in D-BGA:",
            sorted(
                set(d)
                - set(a)
            ),
        )

        raise RuntimeError(
            "Sequence keys differ"
        )

    print()
    print(
        "=" * 108
    )

    print(
        "A -> D-BGA "
        "PER-SEQUENCE DELTAS"
    )

    print(
        "=" * 108
    )

    print(
        f"{'Split':<7}"
        f"{'Sequence':<12}"
        f"{'N':>6}"
        f"{'ΔmIoU':>12}"
        f"{'ΔSky':>12}"
        f"{'ΔTrav':>12}"
        f"{'ΔNonTrav':>12}"
        f"{'ΔBF1-All':>12}"
        f"{'ΔBF1-Trav':>13}"
    )

    order = [
        ("val", "trail-4"),
        ("val", "trail-5"),
        ("test", "trail"),
        ("test", "trail-10"),
        ("test", "trail-12"),
        ("test", "trail-14"),
    ]

    for key in order:

        ar = a[key]
        dr = d[key]

        if (
            ar["images"]
            != dr["images"]
        ):
            raise RuntimeError(
                f"Image count differs: "
                f"{key}"
            )

        deltas = {}

        for metric, label in METRICS:

            deltas[metric] = (
                float(
                    dr[metric]
                )
                - float(
                    ar[metric]
                )
            )

        print(
            f"{key[0]:<7}"
            f"{key[1]:<12}"
            f"{int(ar['images']):>6d}"
            f"{deltas['miou']*100:>+11.4f}"
            f"{deltas['sky_iou']*100:>+11.4f}"
            f"{deltas['trav_iou']*100:>+11.4f}"
            f"{deltas['nontrav_iou']*100:>+11.4f}"
            f"{deltas['bf1_all']*100:>+11.4f}"
            f"{deltas['bf1_trav']*100:>+12.4f}"
        )

    print()
    print(
        "All deltas are percentage points:"
    )

    print(
        "D-BGA - A"
    )


if __name__ == "__main__":
    main()
