#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

export OMP_NUM_THREADS=1

EXP="experiments/rugd3_boundary_full_bga_b16_seed789_formal"
CONFIG="configs/bisenetv2_rugd3_5090_boundary_full_bga_seed789.py"

EXTRACT="$EXP/extracted_models"
VAL_DIR="$EXP/val_checkpoints"

echo "============================================"
echo "D-BGA / seed=789: formal Validation"
echo "============================================"

# --------------------------------------------
# 1. Preflight
# --------------------------------------------

test -f "$CONFIG"
test -s "$EXP/model_final.pth"
test -s "$EXP/train_console.log"

if [[ -e "$EXP/best_model.pth" ]] ||
   [[ -e "$EXP/best_model_metadata.json" ]]; then
    echo "ERROR: A selected best model already exists."
    echo "Check the existing result before rerunning."
    exit 1
fi

for iter in $(seq 5000 5000 80000); do
    test -s "$EXP/checkpoint_iter_${iter}.pth"
done

mkdir -p "$EXTRACT" "$VAL_DIR"

echo "Preflight PASSED: all 16 checkpoints found."

# --------------------------------------------
# 2. Extract model weights
# --------------------------------------------

export EXP

python - <<'PY'
import os
from pathlib import Path
import torch

exp = Path(os.environ["EXP"])
out = exp / "extracted_models"

for iteration in range(5000, 80001, 5000):

    source = exp / f"checkpoint_iter_{iteration}.pth"
    target = out / f"model_iter_{iteration}.pth"

    # Load only checkpoints produced by your own trusted training.
    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint: {source}")

    if "model" not in checkpoint:
        raise KeyError(f"Missing model state: {source}")

    state = checkpoint["model"]

    if not isinstance(state, dict) or not state:
        raise ValueError(f"Empty model state: {source}")

    if "iter" in checkpoint:
        print(
            f"File iteration={iteration}, "
            f"checkpoint iteration={checkpoint['iter']}"
        )

    # Avoid writing to a file with the final name
    # until extraction has completed successfully.
    temp = target.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(target)

    print(f"EXTRACTED: iteration {iteration}")

print("All 16 model states extracted.")
PY

# --------------------------------------------
# 3. Evaluate all 16 checkpoints
# --------------------------------------------

for iter in $(seq 5000 5000 80000); do

    echo
    echo "============================================"
    echo "Validation: iteration $iter"
    echo "============================================"

    WEIGHT="$EXTRACT/model_iter_${iter}.pth"
    LOG="$VAL_DIR/eval_val_${iter}.log"

    python tools/evaluate.py \
        --config "$CONFIG" \
        --weight-path "$WEIGHT" \
        2>&1 | tee "$LOG"

done

# --------------------------------------------
# 4. Select best by Val SS mIoU only
# --------------------------------------------

python - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

import torch

exp = Path(os.environ["EXP"])
log_dir = exp / "val_checkpoints"

results = []

for iteration in range(5000, 80001, 5000):

    path = log_dir / f"eval_val_{iteration}.log"
    content = path.read_text(errors="replace")

    # evaluate.py uses 'mious' as the table row name.
    # Column order: ratio, ss, ssc, msf, msfc.
    matches = re.findall(
        r"^\|\s*mious\s*\|\s*-\s*\|\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        content,
        flags=re.MULTILINE,
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely parse Val SS mIoU: {path}"
        )

    miou = float(matches[0])

    if not 0 <= miou <= 1:
        raise ValueError(
            f"Invalid mIoU at iteration {iteration}: {miou}"
        )

    results.append((iteration, miou))

assert len(results) == 16

# Highest mIoU wins; if tied, choose earlier iteration.
best_iter, best_miou = max(
    results,
    key=lambda x: (x[1], -x[0]),
)

print()
print("============================================")
print("ALL CHECKPOINT VALIDATION RESULTS")
print("============================================")

for iteration, miou in results:
    flag = " <-- BEST" if iteration == best_iter else ""
    print(
        f"{iteration:6d}: "
        f"{miou:.6f} "
        f"({miou * 100:.4f}%){flag}"
    )

print()
print("BEST ITER:", best_iter)
print("BEST mIoU:", f"{best_miou:.6f}")

# Save a reproducible CSV.
csv_path = exp / "val_checkpoint_summary.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["iteration", "val_ss_mIoU", "selected"]
    )

    for iteration, miou in results:
        writer.writerow(
            [
                iteration,
                f"{miou:.6f}",
                int(iteration == best_iter),
            ]
        )

# Select from the original full checkpoint,
# rather than copying a potentially stale extracted file.
source = exp / f"checkpoint_iter_{best_iter}.pth"
target = exp / "best_model.pth"

if target.exists():
    raise FileExistsError(target)

checkpoint = torch.load(
    source,
    map_location="cpu",
    weights_only=False,
)

torch.save(checkpoint["model"], target)

metadata = {
    "experiment": "D-BGA",
    "seed": 789,
    "model_type": "bisenetv2_boundary_full_bga",
    "selection_split": "validation",
    "selection_metric": "single_scale_mIoU",
    "best_iteration": best_iter,
    "best_val_ss_mIoU": best_miou,
    "evaluated_checkpoints": [
        {
            "iteration": iteration,
            "val_ss_mIoU": miou,
        }
        for iteration, miou in results
    ],
}

metadata_path = exp / "best_model_metadata.json"
metadata_path.write_text(
    json.dumps(metadata, indent=2) + "\n"
)

print("Saved:", target)
print("Saved:", metadata_path)
print("Saved:", csv_path)
PY

# --------------------------------------------
# 5. Formal best-checkpoint Validation
# --------------------------------------------

WEIGHT="$EXP/best_model.pth"

echo
echo "============================================"
echo "Best-model mIoU and F1"
echo "============================================"

python tools/evaluate.py \
    --config "$CONFIG" \
    --weight-path "$WEIGHT" \
    2>&1 | tee "$EXP/evaluate_val_best.log"

echo
echo "============================================"
echo "Best-model confusion matrix"
echo "============================================"

python tools/evaluate_confusion_rugd3.py \
    --config "$CONFIG" \
    --weight-path "$WEIGHT" \
    2>&1 | tee "$EXP/confusion_val_best.log"

# --------------------------------------------
# 6. Boundary F1 at 2, 3, 5 pixels
# --------------------------------------------

for tol in 2 3 5; do

    echo
    echo "============================================"
    echo "Boundary F1: tolerance $tol"
    echo "============================================"

    python tools/evaluate_boundary_f1_rugd3.py \
        --config "$CONFIG" \
        --weight-path "$WEIGHT" \
        --tolerance "$tol" \
        2>&1 | tee \
        "$EXP/boundary_val_best_tol${tol}.log"

done

# --------------------------------------------
# 7. Save best weight hash
# --------------------------------------------

sha256sum "$WEIGHT" \
    > "$EXP/best_model_sha256.txt"

# --------------------------------------------
# 8. Produce final result summary
# --------------------------------------------

python - <<'PY'
import json
import os
import re
from pathlib import Path

exp = Path(os.environ["EXP"])

metadata = json.loads(
    (exp / "best_model_metadata.json").read_text()
)

def extract_metric(path, label):
    content = path.read_text(errors="replace")

    matches = re.findall(
        rf"^\s*{re.escape(label)}\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        content,
        flags=re.MULTILINE,
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely parse {label}: {path}"
        )

    return float(matches[0])


bf1 = {}

for tol in (2, 3, 5):
    path = exp / f"boundary_val_best_tol{tol}.log"

    bf1[tol] = {
        "BF1-All": extract_metric(path, "BF1-All"),
        "BF1-Trav": extract_metric(path, "BF1-Trav"),
    }

confusion_path = exp / "confusion_val_best.log"

macro_f1 = extract_metric(
    confusion_path,
    "Macro F1",
)

result = {
    "experiment": "D-BGA",
    "seed": 789,
    "best_iteration": metadata["best_iteration"],
    "val_ss_mIoU": metadata["best_val_ss_mIoU"],
    "val_macro_f1": macro_f1,
    "boundary_f1": bf1,
}

summary_path = exp / "validation_summary.json"
summary_path.write_text(
    json.dumps(result, indent=2) + "\n"
)

print()
print("============================================")
print("FORMAL VALIDATION SUMMARY")
print("============================================")

print(
    "BEST ITER:",
    result["best_iteration"],
)

print(
    "Val SS mIoU:",
    f"{result['val_ss_mIoU']:.6f}",
)

print(
    "Val Macro F1:",
    f"{result['val_macro_f1']:.6f}",
)

for tol in (2, 3, 5):
    print(
        f"BF1-Trav@{tol}:",
        f"{bf1[tol]['BF1-Trav']:.6f}",
    )

print("Saved:", summary_path)
print()
print("D-BGA / seed=789 VALIDATION PASSED")
PY
