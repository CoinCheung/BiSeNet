#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

export EXPERIMENT_SEED=456
export OMP_NUM_THREADS=1

CONFIG="configs/bisenetv2_rugd3_5090_baseline_seed456.py"
WEIGHTS="./model_final_v2_city.pth"

EXP="experiments/rugd3_baseline_b16_seed456_formal"

# Never silently overwrite an existing formal run.
if [[ -f "$EXP/train_console.log" ]] ||
   [[ -f "$EXP/model_final.pth" ]] ||
   compgen -G "$EXP/checkpoint_iter_*.pth" > /dev/null; then
    echo "ERROR: existing experiment files in $EXP"
    echo "Refusing to overwrite the formal experiment."
    exit 1
fi

test -f "$CONFIG"
test -f "$WEIGHTS"
test -f datasets/rugd3/train.txt
test -f datasets/rugd3/val.txt

mkdir -p "$EXP"

# Save provenance before training.
git rev-parse HEAD > "$EXP/git_commit.txt"
git status --short > "$EXP/git_status.txt"
git diff > "$EXP/source_diff.patch"

cp "$CONFIG" "$EXP/training_config.py"

sha256sum \
    "$WEIGHTS" \
    datasets/rugd3/train.txt \
    datasets/rugd3/val.txt \
    datasets/rugd3/test.txt \
    > "$EXP/input_sha256.txt"

{
    echo "model=A"
    echo "seed=456"
    echo "max_iter=80000"
    echo "checkpoint_selection=validation_single_scale_mIoU"
    echo "finetune_from=$WEIGHTS"
    python -c \
      'import torch; print("torch=" + torch.__version__)'
} > "$EXP/experiment_manifest.txt"

nvidia-smi > "$EXP/nvidia_smi_start.txt"

echo "=========================================="
echo "Experiment A / seed 456"
echo "=========================================="

torchrun \
    --standalone \
    --nproc_per_node=1 \
    tools/train_amp_5090.py \
    --config "$CONFIG" \
    --finetune-from "$WEIGHTS" \
    2>&1 | tee "$EXP/train_console.log"

echo
echo "Checking training artifacts..."

test -f "$EXP/model_final.pth"

for iter in $(seq 5000 5000 80000); do
    test -s "$EXP/checkpoint_iter_${iter}.pth"
done

COUNT=$(
    find "$EXP" -maxdepth 1 \
        -name 'checkpoint_iter_*.pth' | wc -l
)

if [[ "$COUNT" -ne 16 ]]; then
    echo "ERROR: expected 16 checkpoints, found $COUNT"
    exit 1
fi

echo "TRAINING PASSED"
echo "Checkpoints: $COUNT"
echo "Final model: $EXP/model_final.pth"
