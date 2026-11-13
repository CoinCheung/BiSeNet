#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

export OMP_NUM_THREADS=1

WEIGHT="experiments/pidnet_s_rugd3_seed123_formal/best_model.pth"

VAL_CFG="configs/pidnet_s_rugd3_5090_external.py"

TEST_CFG="configs/pidnet_s_rugd3_5090_external_test.py"

SEQ_ROOT="datasets/rugd3/sequence_eval"

OUT="experiments/rugd3_sequence_eval/PIDNET_seed123"

CFG_OUT="$OUT/configs"

mkdir -p \
    "$OUT" \
    "$CFG_OUT"


echo "============================================================"
echo "PIDNet-S / seed123 sequence evaluation"
echo "============================================================"

echo "Weight   : $WEIGHT"
echo "Val cfg  : $VAL_CFG"
echo "Test cfg : $TEST_CFG"
echo


# ============================================================
# 1. Input checks
# ============================================================

test -s "$WEIGHT"
test -s "$VAL_CFG"
test -s "$TEST_CFG"

for f in \
    "$SEQ_ROOT/val/trail-4.txt" \
    "$SEQ_ROOT/val/trail-5.txt" \
    "$SEQ_ROOT/test/trail.txt" \
    "$SEQ_ROOT/test/trail-10.txt" \
    "$SEQ_ROOT/test/trail-12.txt" \
    "$SEQ_ROOT/test/trail-14.txt"
do
    test -s "$f"
done

echo "Sequence lists: OK"


# ============================================================
# 2. GLOBAL VAL anchor
# ============================================================

echo
echo "============================================================"
echo "PIDNet-S GLOBAL VAL ANCHOR"
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


# ============================================================
# 3. GLOBAL TEST anchor
# ============================================================

echo
echo "============================================================"
echo "PIDNet-S GLOBAL TEST ANCHOR"
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


# ============================================================
# 4. VAL sequences
# ============================================================

VAL_SEQS=(
    "trail-4"
    "trail-5"
)

for SEQ in "${VAL_SEQS[@]}"; do

    echo
    echo "============================================================"
    echo "PIDNet-S VAL / $SEQ"
    echo "============================================================"

    ANN="$SEQ_ROOT/val/$SEQ.txt"

    CFG="$CFG_OUT/val_${SEQ}.py"

    python \
    tools/make_sequence_eval_config.py \
        --base "$VAL_CFG" \
        --ann "$ANN" \
        --out "$CFG"

    echo "Sequence config:"
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


# ============================================================
# 5. TEST sequences
# ============================================================

TEST_SEQS=(
    "trail"
    "trail-10"
    "trail-12"
    "trail-14"
)

for SEQ in "${TEST_SEQS[@]}"; do

    echo
    echo "============================================================"
    echo "PIDNet-S TEST / $SEQ"
    echo "============================================================"

    ANN="$SEQ_ROOT/test/$SEQ.txt"

    CFG="$CFG_OUT/test_${SEQ}.py"

    python \
    tools/make_sequence_eval_config.py \
        --base "$TEST_CFG" \
        --ann "$ANN" \
        --out "$CFG"

    echo "Sequence config:"
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
echo "PIDNet-S SEQUENCE EVALUATION COMPLETE"
echo "============================================================"
