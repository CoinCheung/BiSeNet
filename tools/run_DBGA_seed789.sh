#!/usr/bin/env bash
set -euo pipefail

# ============================================
# D-BGA / seed=789 / Formal Training
# ============================================

cd /root/autodl-tmp/projects/BiSeNet

export EXPERIMENT_SEED=789
export OMP_NUM_THREADS=1

CONFIG="configs/bisenetv2_rugd3_5090_boundary_full_bga_seed789.py"
TRAIN_SCRIPT="tools/train_amp_boundary_sup.py"
PRETRAINED="./model_final_v2_city.pth"

EXP="experiments/rugd3_boundary_full_bga_b16_seed789_formal"

echo "============================================"
echo "D-BGA / seed 789 / 80k formal training"
echo "============================================"

# --------------------------------------------
# 1. Check required files
# --------------------------------------------

test -s "$CONFIG"
test -s "$TRAIN_SCRIPT"
test -s "$PRETRAINED"

test -s datasets/rugd3/train.txt
test -s datasets/rugd3/val.txt
test -s datasets/rugd3/test.txt

# Ensure the preceding A / seed 789 experiment
# has already selected its best checkpoint.
test -s \
  experiments/rugd3_baseline_b16_seed789_formal/best_model.pth

# Prevent accidental overwrite of an existing run.
if [[ -f "$EXP/train_console.log" ]] ||
   [[ -f "$EXP/model_final.pth" ]] ||
   compgen -G "$EXP/checkpoint_iter_*.pth" > /dev/null; then

    echo "ERROR: existing training files detected:"
    echo "$EXP"
    echo "Refusing to overwrite the experiment."
    exit 1
fi

# --------------------------------------------
# 2. Verify configuration and seed control
# --------------------------------------------

python - <<'PY'
import os
import runpy
from pathlib import Path

import torch

config_path = (
    "configs/"
    "bisenetv2_rugd3_5090_boundary_full_bga_seed789.py"
)

cfg = runpy.run_path(config_path)["cfg"]

assert cfg["model_type"] == "bisenetv2_boundary_full_bga"
assert cfg["max_iter"] == 80000
assert cfg["save_interval"] == 5000

assert str(cfg["respth"]).endswith(
    "rugd3_boundary_full_bga_b16_seed789_formal"
)

assert os.environ["EXPERIMENT_SEED"] == "789"

trainer = Path(
    "tools/train_amp_boundary_sup.py"
).read_text()

assert "EXPERIMENT_SEED" in trainer, (
    "Training script does not support the seed "
    "environment variable."
)

assert torch.cuda.is_available(), (
    "CUDA GPU is unavailable."
)

print("Model:", cfg["model_type"])
print("Seed:", os.environ["EXPERIMENT_SEED"])
print("Max iterations:", cfg["max_iter"])
print("Save interval:", cfg["save_interval"])
print("Output:", cfg["respth"])
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

print("\nPREFLIGHT PASSED")
PY

# --------------------------------------------
# 3. Save experiment provenance
# --------------------------------------------

mkdir -p "$EXP/code_snapshot"

git rev-parse HEAD \
    > "$EXP/git_commit.txt"

git status --short \
    > "$EXP/git_status.txt"

git diff \
    > "$EXP/source_diff.patch"

cp "$CONFIG" \
    "$EXP/training_config.py"

cp "$TRAIN_SCRIPT" \
    "$EXP/code_snapshot/train_amp_boundary_sup.py"

cp lib/models/bisenetv2_boundary_full_bga.py \
    "$EXP/code_snapshot/"

cp lib/models/__init__.py \
    "$EXP/code_snapshot/models_init.py"

sha256sum \
    "$PRETRAINED" \
    datasets/rugd3/train.txt \
    datasets/rugd3/val.txt \
    datasets/rugd3/test.txt \
    > "$EXP/input_sha256.txt"

python -m pip freeze \
    > "$EXP/python_packages.txt"

{
    echo "experiment=D-BGA"
    echo "seed=789"
    echo "model_type=bisenetv2_boundary_full_bga"
    echo "max_iter=80000"
    echo "save_interval=5000"
    echo "pretrained=$PRETRAINED"
    echo "checkpoint_selection=validation_single_scale_mIoU"
    echo "test_evaluation=deferred"
} > "$EXP/experiment_manifest.txt"

nvidia-smi \
    > "$EXP/nvidia_smi_start.txt"

echo
echo "Saved experiment configuration and metadata."
echo

# --------------------------------------------
# 4. Formal 80k training
# --------------------------------------------

torchrun \
    --standalone \
    --nproc_per_node=1 \
    "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --finetune-from "$PRETRAINED" \
    2>&1 | tee "$EXP/train_console.log"

# --------------------------------------------
# 5. Check training artifacts
# --------------------------------------------

echo
echo "============================================"
echo "Checking training results"
echo "============================================"

test -s "$EXP/model_final.pth"

for iter in $(seq 5000 5000 80000); do

    checkpoint="$EXP/checkpoint_iter_${iter}.pth"

    if [[ ! -s "$checkpoint" ]]; then
        echo "ERROR: missing checkpoint:"
        echo "$checkpoint"
        exit 1
    fi

done

COUNT=$(
    find "$EXP" -maxdepth 1 \
        -name 'checkpoint_iter_*.pth' \
        -type f | wc -l
)

if [[ "$COUNT" -ne 16 ]]; then

    echo "ERROR: expected 16 checkpoints."
    echo "Found: $COUNT"
    exit 1

fi

echo
echo "============================================"
echo "D-BGA / seed 789 TRAINING PASSED"
echo "============================================"

echo "Iterations: 80000"
echo "Checkpoints: $COUNT"
echo "Final model: $EXP/model_final.pth"
echo "Training log: $EXP/train_console.log"

echo
echo "NEXT: evaluate all 16 checkpoints on Validation."
echo "Do not evaluate Test yet."
