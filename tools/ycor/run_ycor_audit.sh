#!/usr/bin/env bash

set -euo pipefail

cd /root/autodl-tmp/projects/BiSeNet

YCOR_ROOT="${YCOR_ROOT:-/root/autodl-tmp/datasets/YCOR-external}"

DATA_ROOT="$YCOR_ROOT/data"

OUT="experiments/external_ycor_zero_shot/audit"

mkdir -p "$OUT"

test -d "$DATA_ROOT"

echo "============================================================"
echo "YCOR DATASET / ONTOLOGY AUDIT"
echo "============================================================"

python \
tools/ycor/audit_ycor.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT" \
    --workers 8 \
2>&1 | tee \
"$OUT/audit_console.log"


echo
echo "============================================================"
echo "STRICT VALIDATION"
echo "============================================================"

python \
tools/ycor/verify_ycor_audit.py \
    --audit-json \
    "$OUT/dataset_audit.json" \
2>&1 | tee \
"$OUT/strict_validation.log"


echo
echo "Generating SHA256 manifest..."

find "$OUT" \
    -type f \
    ! -name 'AUDIT_SHA256SUMS.txt' \
    -print0 \
| sort -z \
| xargs -0 sha256sum \
> "$OUT/AUDIT_SHA256SUMS.txt"


echo
echo "============================================================"
echo "YCOR AUDIT COMPLETE"
echo "============================================================"

tail -40 \
"$OUT/strict_validation.log"
