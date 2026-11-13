#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

EXT="experiments/external_ycor_zero_shot"
PROTOCOL="$EXT/protocol"
RESULTS="$EXT/results/ycor_all"


if [[ ! -f "$PROTOCOL/PROTOCOL_FROZEN.sha256" ]]; then
    echo "ERROR: protocol has not been frozen."
    exit 2
fi


mkdir -p "$RESULTS"


echo "============================================================"
echo "VERIFY FROZEN PROTOCOL"
echo "============================================================"

(
    cd "$PROTOCOL"

    sha256sum -c \
        PROTOCOL_FROZEN.sha256
)


echo
echo "============================================================"
echo "VERIFY CHECKPOINTS"
echo "============================================================"

sha256sum -c \
"$PROTOCOL/checkpoints_sha256.txt"


echo
echo "============================================================"
echo "VERIFY DATASET / ONTOLOGY"
echo "============================================================"

sha256sum -c \
"$PROTOCOL/dataset_protocol_sha256.txt"


echo
echo "============================================================"
echo "VERIFY EVALUATOR CODE"
echo "============================================================"

sha256sum -c \
"$PROTOCOL/evaluator_code_sha256.txt"


echo
echo "============================================================"
echo "VERIFY MODEL CODE"
echo "============================================================"

sha256sum -c \
"$PROTOCOL/model_code_sha256.txt"


# ------------------------------------------------------------
# Refuse to mix with an earlier formal result set.
# ------------------------------------------------------------

if find "$RESULTS" \
    -type f \
    -name metrics.json \
    -print -quit \
    2>/dev/null \
    | grep -q .
then
    echo
    echo "ERROR:"
    echo "Formal metrics.json already exists under:"
    echo "$RESULTS"
    echo
    echo "Refusing to mix or overwrite formal runs."
    exit 3
fi


run_one()
{
    local model="$1"
    local seed="$2"

    local safe_model

    safe_model="$(
        echo "$model" \
        | tr -d '-' \
        | tr '[:upper:]' '[:lower:]'
    )"

    local out

    out="$RESULTS/${safe_model}_seed${seed}"

    mkdir -p "$out"

    echo
    echo "============================================================"
    echo "FORMAL YCOR-ALL ZERO-SHOT"
    echo "MODEL=$model"
    echo "SEED=$seed"
    echo "============================================================"

    python \
    tools/ycor/evaluate_ycor_zero_shot.py \
        --model "$model" \
        --seed "$seed" \
        --batch-size 2 \
        --workers 4 \
        --output-dir "$out" \
        2>&1 | tee \
        "$out/run.log"
}


run_one "A"        123
run_one "A"        456
run_one "A"        789

run_one "D-BGA"    123
run_one "D-BGA"    456
run_one "D-BGA"    789

run_one "PIDNet-S" 123
run_one "PIDNet-S" 456
run_one "PIDNet-S" 789


echo
echo "============================================================"
echo "VERIFY FORMAL RESULT COUNT"
echo "============================================================"

count="$(
    find "$RESULTS" \
        -mindepth 2 \
        -maxdepth 2 \
        -name metrics.json \
        | wc -l
)"

if [[ "$count" -ne 9 ]]; then
    echo "ERROR: expected 9 metrics.json, found $count"
    exit 4
fi

echo "metrics.json count: $count"


echo
echo "============================================================"
echo "SUMMARIZE ALL NINE RUNS"
echo "============================================================"

python \
tools/ycor/summarize_ycor_zero_shot.py \
2>&1 | tee \
"$EXT/results/ycor_all_summary.log"


echo
echo "============================================================"
echo "YCOR FORMAL 9-RUN EXPERIMENT COMPLETE"
echo "============================================================"
