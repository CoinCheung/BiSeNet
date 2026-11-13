#!/usr/bin/env python3

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(
    "experiments/external_ycor_zero_shot/results/ycor_all"
)

OUT = Path(
    "experiments/external_ycor_zero_shot/results/summary"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED = {
    ("A", 123),
    ("A", 456),
    ("A", 789),
    ("D-BGA", 123),
    ("D-BGA", 456),
    ("D-BGA", 789),
    ("PIDNet-S", 123),
    ("PIDNet-S", 456),
    ("PIDNet-S", 789),
}


def get_boundary(s, tol, mode):
    return float(
        s["boundary"][str(tol)][mode]["f1"]
    )


files = sorted(
    ROOT.glob("*/metrics.json")
)

if len(files) != 9:
    raise RuntimeError(
        f"Expected exactly 9 metrics.json files, "
        f"found {len(files)}"
    )


records = []

for path in files:

    d = json.loads(
        path.read_text()
    )

    s = d["subsets"]["all"]

    row = {
        "model": d["model"],
        "seed": int(d["seed"]),
        "best_iter": int(d["best_iter"]),

        "sky_iou":
            float(s["iou"][0]),

        "trav_iou":
            float(s["iou"][1]),

        "nontrav_iou":
            float(s["iou"][2]),

        "miou":
            float(s["miou"]),

        "macro_f1":
            float(s["macro_f1"]),

        "bf1_all_2":
            get_boundary(
                s, 2, "all"
            ),

        "bf1_trav_2":
            get_boundary(
                s, 2, "traversability"
            ),

        "bf1_all_3":
            get_boundary(
                s, 3, "all"
            ),

        "bf1_trav_3":
            get_boundary(
                s, 3, "traversability"
            ),

        "bf1_all_5":
            get_boundary(
                s, 5, "all"
            ),

        "bf1_trav_5":
            get_boundary(
                s, 5, "traversability"
            ),

        "coverage":
            float(s["coverage"]),
    }

    records.append(row)


actual = {
    (
        r["model"],
        r["seed"],
    )
    for r in records
}

if actual != EXPECTED:
    raise RuntimeError(
        "Formal result identities do not match "
        "the frozen 9-run design.\n"
        f"actual={sorted(actual)}"
    )


# ------------------------------------------------------------
# Per-seed table
# ------------------------------------------------------------

per_seed_csv = (
    OUT /
    "per_seed_results.csv"
)

with per_seed_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            records[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(records)


# ------------------------------------------------------------
# Three-seed mean ± sample SD
# ------------------------------------------------------------

metrics = [
    "sky_iou",
    "trav_iou",
    "nontrav_iou",
    "miou",
    "macro_f1",
    "bf1_all_2",
    "bf1_trav_2",
    "bf1_all_3",
    "bf1_trav_3",
    "bf1_all_5",
    "bf1_trav_5",
]


summary_rows = []

for model in [
    "A",
    "D-BGA",
    "PIDNet-S",
]:

    rr = [
        r
        for r in records
        if r["model"] == model
    ]

    assert len(rr) == 3

    row = {
        "model": model
    }

    for metric in metrics:

        values = [
            r[metric]
            for r in rr
        ]

        row[
            metric + "_mean"
        ] = statistics.mean(
            values
        )

        row[
            metric + "_sd"
        ] = statistics.stdev(
            values
        )

    summary_rows.append(row)


summary_csv = (
    OUT /
    "three_seed_summary.csv"
)

with summary_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ------------------------------------------------------------
# Paired D-BGA - A deltas
# ------------------------------------------------------------

lookup = {
    (
        r["model"],
        r["seed"],
    ):
        r
    for r in records
}


delta_metrics = [
    "miou",
    "macro_f1",
    "bf1_trav_2",
    "bf1_trav_3",
    "bf1_trav_5",
]

delta_rows = []

for seed in [
    123,
    456,
    789,
]:

    a = lookup[
        ("A", seed)
    ]

    d = lookup[
        ("D-BGA", seed)
    ]

    row = {
        "seed": seed
    }

    for metric in delta_metrics:

        row[
            "delta_" + metric
        ] = (
            d[metric]
            -
            a[metric]
        )

    delta_rows.append(row)


delta_csv = (
    OUT /
    "paired_dbga_minus_a.csv"
)

with delta_csv.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            delta_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        delta_rows
    )


def pct(x):
    return 100.0 * x


print(
    "=" * 100
)

print(
    "YCOR-ALL THREE-SEED SUMMARY"
)

print(
    "=" * 100
)


for row in summary_rows:

    print(
        f"\n{row['model']}"
    )

    print(
        "  mIoU       "
        f"{pct(row['miou_mean']):.4f}"
        " ± "
        f"{pct(row['miou_sd']):.4f}%"
    )

    print(
        "  MacroF1    "
        f"{pct(row['macro_f1_mean']):.4f}"
        " ± "
        f"{pct(row['macro_f1_sd']):.4f}%"
    )

    print(
        "  BF1-Trav@3 "
        f"{pct(row['bf1_trav_3_mean']):.4f}"
        " ± "
        f"{pct(row['bf1_trav_3_sd']):.4f}%"
    )


print()
print(
    "D-BGA - A paired deltas"
)

for row in delta_rows:

    print(
        f"seed{row['seed']}: "
        f"ΔmIoU="
        f"{pct(row['delta_miou']):+.4f} pp, "
        f"ΔMacroF1="
        f"{pct(row['delta_macro_f1']):+.4f} pp, "
        f"ΔBF1-Trav@3="
        f"{pct(row['delta_bf1_trav_3']):+.4f} pp"
    )


print()
print(
    "Mean paired deltas"
)

for metric in [
    "miou",
    "macro_f1",
    "bf1_trav_3",
]:

    values = [
        r[
            "delta_" + metric
        ]
        for r in delta_rows
    ]

    print(
        f"{metric:<14}"
        f"{pct(statistics.mean(values)):+.4f} pp"
    )


print()
print(
    "Written:"
)

print(per_seed_csv)
print(summary_csv)
print(delta_csv)
