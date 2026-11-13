#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet
export OMP_NUM_THREADS=1

for seed in 456 789; do
    for model in A DBGA; do

        if [[ "$model" == "A" ]]; then
            EXP="experiments/rugd3_baseline_b16_seed${seed}_formal"
            CONFIG="configs/bisenetv2_rugd3_5090_baseline_test.py"
        else
            EXP="experiments/rugd3_boundary_full_bga_b16_seed${seed}_formal"
            CONFIG="configs/bisenetv2_rugd3_5090_boundary_full_bga_test.py"
        fi

        WEIGHT="$EXP/best_model.pth"

        test -s "$WEIGHT"
        test -s "$EXP/best_model_metadata.json"
        test -s "$CONFIG"

        # Don't silently repeat an official Test evaluation.
        for name in \
            evaluate_test_best.log \
            confusion_test_best.log \
            boundary_test_best_tol3.log
        do
            if [[ -e "$EXP/$name" ]]; then
                echo "Existing Test result: $EXP/$name"
                echo "Inspect it before rerunning."
                exit 1
            fi
        done

        echo "============================================"
        echo "$model / seed $seed — FINAL TEST"
        echo "============================================"

        python tools/evaluate.py \
            --config "$CONFIG" \
            --weight-path "$WEIGHT" \
            2>&1 | tee "$EXP/evaluate_test_best.log"

        python tools/evaluate_confusion_rugd3.py \
            --config "$CONFIG" \
            --weight-path "$WEIGHT" \
            2>&1 | tee "$EXP/confusion_test_best.log"

        python tools/evaluate_boundary_f1_rugd3.py \
            --config "$CONFIG" \
            --weight-path "$WEIGHT" \
            --tolerance 3 \
            2>&1 | tee "$EXP/boundary_test_best_tol3.log"

        echo "$model / seed $seed TEST PASSED"
    done
done
