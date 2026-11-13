#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

export EXPERIMENT_SEED=789
export OMP_NUM_THREADS=1

EXP="experiments/rugd3_boundary_full_bga_b16_seed789_formal"
CONFIG="configs/bisenetv2_rugd3_5090_boundary_full_bga_seed789.py"
PRETRAINED="./model_final_v2_city.pth"

# 1. 检查输入
test -s "$CONFIG"
test -s "$PRETRAINED"
test -s datasets/rugd3/train.txt
test -s datasets/rugd3/val.txt
test -s datasets/rugd3/test.txt

# 2. 防止覆盖已有正式实验
if [[ -f "$EXP/train_console.log" ]] ||
   [[ -f "$EXP/model_final.pth" ]] ||
   compgen -G "$EXP/checkpoint_iter_*.pth" > /dev/null; then
    echo "ERROR: existing experiment detected: $EXP"
    echo "Refusing to overwrite the previous run."
    exit 1
fi

mkdir -p "$EXP"

# 3. 保存实验环境与来源
git rev-parse HEAD > "$EXP/git_commit.txt"
git status --short > "$EXP/git_status.txt"
git diff > "$EXP/source_diff.patch"

cp "$CONFIG" "$EXP/training_config.py"

sha256sum \
    "$PRETRAINED" \
    datasets/rugd3/train.txt \
    datasets/rugd3/val.txt \
    datasets/rugd3/test.txt \
    > "$EXP/input_sha256.txt"

{
    echo "experiment=D-BGA"
    echo "seed=789"
    echo "iterations=80000"
    echo "batch_size=16"
    echo "pretrained=$PRETRAINED"
    echo "checkpoint_selection=Val_SS_mIoU"
    python -c \
      'import torch; print("torch=" + torch.__version__)'
} > "$EXP/experiment_manifest.txt"

nvidia-smi > "$EXP/nvidia_smi_start.txt"

# 4. 正式训练
echo "============================================"
echo "D-BGA / seed 789 / 80000 iterations"
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    tools/train_amp_boundary_sup.py \
    --config "$CONFIG" \
    --finetune-from "$PRETRAINED" \
    2>&1 | tee "$EXP/train_console.log"

# 5. 训练完成检查
test -s "$EXP/model_final.pth"

for iter in $(seq 5000 5000 80000); do
    test -s "$EXP/checkpoint_iter_${iter}.pth"
done

COUNT=$(
    find "$EXP" -maxdepth 1 \
        -name 'checkpoint_iter_*.pth' | wc -l
)

[[ "$COUNT" -eq 16 ]] || {
    echo "ERROR: expected 16 checkpoints, got $COUNT"
    exit 1
}

echo
echo "============================================"
echo "D-BGA / seed 789 TRAINING PASSED"
echo "Checkpoint count: $COUNT"
echo "============================================"
