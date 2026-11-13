#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

export OMP_NUM_THREADS=1

WEIGHT="experiments/rugd3_baseline_b16_seed123_formal/best_model.pth"

VAL_CFG="configs/bisenetv2_rugd3_5090_baseline.py"

TEST_CFG="configs/bisenetv2_rugd3_5090_baseline_test.py"

SEQ_ROOT="datasets/rugd3/sequence_eval"

OUT="experiments/rugd3_sequence_eval/A_seed123"

CFG_OUT="$OUT/configs"

mkdir -p \
    "$OUT" \
    "$CFG_OUT"


echo "============================================================"
echo "A / BiSeNetV2 seed123 sequence evaluation"
echo "============================================================"

echo "Weight: $WEIGHT"
echo


# ------------------------------------------------------------
# 1. Generate and validate sequence annotation lists
# ------------------------------------------------------------

python \
tools/make_rugd3_sequence_lists.py \
2>&1 | tee \
"$OUT/sequence_list_generation.log"


# ------------------------------------------------------------
# 2. Global anchor evaluation
#
# These are rerun using the exact same checkpoint/config
# so sequence confusion matrices can be compared against
# a fresh global result.
# ------------------------------------------------------------

echo
echo "============================================================"
echo "GLOBAL VAL ANCHOR"
echo "============================================================"

python \
tools/evaluate_confusion_rugd3.py \
    --config "$VAL_CFG" \
    --weight-path "$WEIGHT" \
2>&1 | tee \
"$OUT/global_val_confusion.log"


python \
tools/evaluate_boundary_f1_rugd3.py \
    --config "$VAL_CFG" \
    --weight-path "$WEIGHT" \
    --tolerance 3 \
2>&1 | tee \
"$OUT/global_val_boundary_tol3.log"


echo
echo "============================================================"
echo "GLOBAL TEST ANCHOR"
echo "============================================================"

python \
tools/evaluate_confusion_rugd3.py \
    --config "$TEST_CFG" \
    --weight-path "$WEIGHT" \
2>&1 | tee \
"$OUT/global_test_confusion.log"


python \
tools/evaluate_boundary_f1_rugd3.py \
    --config "$TEST_CFG" \
    --weight-path "$WEIGHT" \
    --tolerance 3 \
2>&1 | tee \
"$OUT/global_test_boundary_tol3.log"


# ------------------------------------------------------------
# 3. VAL sequences
# ------------------------------------------------------------

VAL_SEQS=(
    "trail-4"
    "trail-5"
)

for SEQ in "${VAL_SEQS[@]}"; do

    echo
    echo "============================================================"
    echo "VAL / $SEQ"
    echo "============================================================"

    ANN="$SEQ_ROOT/val/$SEQ.txt"

    CFG="$CFG_OUT/val_${SEQ}.py"

    test -s "$ANN"

    python \
    tools/make_sequence_eval_config.py \
        --base "$VAL_CFG" \
        --ann "$ANN" \
        --out "$CFG"

    grep -n \
    'val_im_anns' \
    "$CFG"

    python \
    tools/evaluate_confusion_rugd3.py \
        --config "$CFG" \
        --weight-path "$WEIGHT" \
    2>&1 | tee \
    "$OUT/val_${SEQ}_confusion.log"

    python \
    tools/evaluate_boundary_f1_rugd3.py \
        --config "$CFG" \
        --weight-path "$WEIGHT" \
        --tolerance 3 \
    2>&1 | tee \
    "$OUT/val_${SEQ}_boundary_tol3.log"

done


# ------------------------------------------------------------
# 4. TEST sequences
# ------------------------------------------------------------

TEST_SEQS=(
    "trail"
    "trail-10"
    "trail-12"
    "trail-14"
)

for SEQ in "${TEST_SEQS[@]}"; do

    echo
    echo "============================================================"
    echo "TEST / $SEQ"
    echo "============================================================"

    ANN="$SEQ_ROOT/test/$SEQ.txt"

    CFG="$CFG_OUT/test_${SEQ}.py"

    test -s "$ANN"

    python \
    tools/make_sequence_eval_config.py \
        --base "$TEST_CFG" \
        --ann "$ANN" \
        --out "$CFG"

    grep -n \
    'val_im_anns' \
    "$CFG"

    python \
    tools/evaluate_confusion_rugd3.py \
        --config "$CFG" \
        --weight-path "$WEIGHT" \
    2>&1 | tee \
    "$OUT/test_${SEQ}_confusion.log"

    python \
    tools/evaluate_boundary_f1_rugd3.py \
        --config "$CFG" \
        --weight-path "$WEIGHT" \
        --tolerance 3 \
    2>&1 | tee \
    "$OUT/test_${SEQ}_boundary_tol3.log"

done


echo
echo "============================================================"
echo "A SEQUENCE EVALUATION COMPLETE"
echo "============================================================"
