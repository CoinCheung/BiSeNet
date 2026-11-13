#!/usr/bin/env python3

import csv
from pathlib import Path


FILES = {
    "A": Path(
        "experiments/"
        "rugd3_sequence_eval/"
        "A_seed123/"
        "per_sequence_metrics.csv"
    ),

    "D-BGA": Path(
        "experiments/"
        "rugd3_sequence_eval/"
        "DBGA_seed123/"
        "per_sequence_metrics.csv"
    ),

    "PIDNet-S": Path(
        "experiments/"
        "rugd3_sequence_eval/"
        "PIDNET_seed123/"
        "per_sequence_metrics.csv"
    ),
}


ORDER = [
    ("val", "trail-4"),
    ("val", "trail-5"),
    ("test", "trail"),
    ("test", "trail-10"),
    ("test", "trail-12"),
    ("test", "trail-14"),
]


def load(path):

    result = {}

    with path.open() as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                row["split"],
                row["sequence"],
            )

            result[key] = row

    return result


def pct(row, key):
    return (
        float(row[key])
        * 100.0
    )


def main():

    data = {
        name: load(path)
        for name, path
        in FILES.items()
    }

    expected = set(
        ORDER
    )

    for name, rows in data.items():

        if set(rows) != expected:

            raise RuntimeError(
                f"{name}: sequence keys "
                f"do not match expected set"
            )

    print()
    print(
        "=" * 118
    )

    print(
        "A / D-BGA / PIDNet-S "
        "PER-SEQUENCE COMPARISON"
    )

    print(
        "=" * 118
    )

    print(
        f"{'Split':<7}"
        f"{'Sequence':<12}"
        f"{'N':>6}"
        f"{'A mIoU':>11}"
        f"{'DBGA mIoU':>12}"
        f"{'PID mIoU':>11}"
        f"{'A BF1':>11}"
        f"{'DBGA BF1':>11}"
        f"{'PID BF1':>11}"
    )

    for key in ORDER:

        a = data["A"][key]
        d = data["D-BGA"][key]
        p = data["PIDNet-S"][key]

        counts = {
            a["images"],
            d["images"],
            p["images"],
        }

        if len(counts) != 1:
            raise RuntimeError(
                f"Image count mismatch: "
                f"{key}"
            )

        n = int(
            a["images"]
        )

        print(
            f"{key[0]:<7}"
            f"{key[1]:<12}"
            f"{n:>6d}"
            f"{pct(a,'miou'):>10.4f}%"
            f"{pct(d,'miou'):>11.4f}%"
            f"{pct(p,'miou'):>10.4f}%"
            f"{pct(a,'bf1_trav'):>10.4f}%"
            f"{pct(d,'bf1_trav'):>10.4f}%"
            f"{pct(p,'bf1_trav'):>10.4f}%"
        )

    print()
    print(
        "=" * 118
    )

    print(
        "PIDNet-S Val -> Test diagnostic"
    )

    print(
        "=" * 118
    )

    for key in ORDER:

        row = data[
            "PIDNet-S"
        ][key]

        print(
            f"{key[0]:<7}"
            f"{key[1]:<12}"
            f"mIoU="
            f"{pct(row,'miou'):>8.4f}%  "
            f"Trav="
            f"{pct(row,'trav_iou'):>8.4f}%  "
            f"NonTrav="
            f"{pct(row,'nontrav_iou'):>8.4f}%  "
            f"BF1-Trav="
            f"{pct(row,'bf1_trav'):>8.4f}%"
        )


if __name__ == "__main__":
    main()
