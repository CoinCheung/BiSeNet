#!/usr/bin/env bash

set -euo pipefail

ROOT="${YCOR_ROOT:-/root/autodl-tmp/datasets/YCOR-external}"

DOWNLOAD_DIR="$ROOT/downloads"
DATA_DIR="$ROOT/data"
META_DIR="$ROOT/meta"

ZIP="$DOWNLOAD_DIR/yamaha_v0.zip"

OFFICIAL_PAGE="https://theairlab.org/yamaha-offroad-dataset/"
OFFICIAL_BOX="https://cmu.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt"

mkdir -p \
    "$DOWNLOAD_DIR" \
    "$DATA_DIR" \
    "$META_DIR"


echo "============================================================"
echo "YCOR OFFICIAL DOWNLOAD"
echo "============================================================"
echo "ROOT = $ROOT"
echo


# ------------------------------------------------------------
# Save source metadata
# ------------------------------------------------------------

cat > "$META_DIR/source_info.txt" <<EOF
Dataset: Yamaha-CMU Off-Road Dataset (YCOR)

Official page:
$OFFICIAL_PAGE

Official CMU Box share:
$OFFICIAL_BOX

Expected total images:
1076

Expected official split:
train=931
validation=145

License:
CC BY 4.0

Archive commonly distributed as:
yamaha_v0.zip

IMPORTANT:
No RUGD3 ontology mapping has been applied at this stage.
EOF


# ------------------------------------------------------------
# Disk sanity
# ------------------------------------------------------------

AVAILABLE_KB="$(
    df -Pk "$ROOT" \
    | awk 'NR==2 {print $4}'
)"

AVAILABLE_GB="$(
    awk \
        -v kb="$AVAILABLE_KB" \
        'BEGIN {printf "%.2f", kb/1024/1024}'
)"

echo "Available disk: ${AVAILABLE_GB} GiB"

# YCOR is much smaller than RELLIS,
# but keep conservative temporary space.
MIN_KB=$((8 * 1024 * 1024))

if (( AVAILABLE_KB < MIN_KB )); then
    echo "ERROR: less than 8 GiB free."
    exit 1
fi


# ------------------------------------------------------------
# ZIP verifier
# ------------------------------------------------------------

verify_zip()
{
    local f="$1"

    python - "$f" <<'PY'
import sys
import zipfile
from pathlib import Path

p = Path(sys.argv[1])

if not p.is_file():
    raise SystemExit(1)

if p.stat().st_size < 10 * 1024 * 1024:
    print(
        f"File too small: "
        f"{p.stat().st_size} bytes"
    )
    raise SystemExit(1)

if not zipfile.is_zipfile(p):
    print("Not a valid ZIP:", p)
    raise SystemExit(1)

with zipfile.ZipFile(p) as z:
    bad = z.testzip()

    if bad is not None:
        print(
            "Corrupt ZIP member:",
            bad,
        )
        raise SystemExit(1)

print(
    "ZIP VALID:",
    p,
    p.stat().st_size,
)
PY
}


# ------------------------------------------------------------
# Reuse good existing archive
# ------------------------------------------------------------

if [[ -f "$ZIP" ]]; then

    if verify_zip "$ZIP"; then
        echo "Existing archive accepted."
    else
        echo "Existing archive invalid; removing."
        rm -f "$ZIP"
    fi
fi


# ------------------------------------------------------------
# Download from official CMU Box
#
# Box shared links normally accept ?download=1.
# Multiple methods are attempted because Box behavior
# occasionally differs between curl/wget versions.
# ------------------------------------------------------------

if [[ ! -f "$ZIP" ]]; then

    TMP="${ZIP}.part"

    rm -f "$TMP"

    echo
    echo "Downloading from official CMU Box..."

    SUCCESS=0

    if command -v curl >/dev/null 2>&1; then

        echo "Attempt 1: curl + ?download=1"

        if curl \
            -L \
            --fail \
            --retry 5 \
            --retry-delay 3 \
            --retry-all-errors \
            --connect-timeout 30 \
            -A "Mozilla/5.0" \
            -o "$TMP" \
            "${OFFICIAL_BOX}?download=1"
        then
            if verify_zip "$TMP"; then
                SUCCESS=1
            fi
        fi
    fi


    if [[ "$SUCCESS" -eq 0 ]]; then

        rm -f "$TMP"

        if command -v wget >/dev/null 2>&1; then

            echo
            echo "Attempt 2: wget + ?download=1"

            if wget \
                --tries=5 \
                --timeout=30 \
                --content-disposition \
                -O "$TMP" \
                "${OFFICIAL_BOX}?download=1"
            then
                if verify_zip "$TMP"; then
                    SUCCESS=1
                fi
            fi
        fi
    fi


    if [[ "$SUCCESS" -eq 0 ]]; then

        rm -f "$TMP"

        echo
        echo "ERROR: official Box download failed."
        echo
        echo "Open the official page manually if Box"
        echo "requires browser confirmation:"
        echo
        echo "$OFFICIAL_PAGE"
        echo
        echo "After downloading yamaha_v0.zip manually,"
        echo "place it here:"
        echo
        echo "$ZIP"
        echo

        exit 1
    fi

    mv "$TMP" "$ZIP"
fi


# ------------------------------------------------------------
# Final archive integrity check
# ------------------------------------------------------------

verify_zip "$ZIP"

sha256sum "$ZIP" \
    > "$META_DIR/yamaha_v0_zip_sha256.txt"


# ------------------------------------------------------------
# Safe extraction
# ------------------------------------------------------------

MARKER="$META_DIR/extracted.ok"

if [[ ! -f "$MARKER" ]]; then

    echo
    echo "Extracting YCOR..."

    python - \
        "$ZIP" \
        "$DATA_DIR" <<'PY'

import sys
import zipfile
from pathlib import Path

archive = Path(
    sys.argv[1]
)

output = Path(
    sys.argv[2]
)

output.mkdir(
    parents=True,
    exist_ok=True,
)

root = output.resolve()

with zipfile.ZipFile(
    archive
) as z:

    for info in z.infolist():

        dst = (
            output /
            info.filename
        ).resolve()

        if (
            dst != root
            and root not in dst.parents
        ):
            raise RuntimeError(
                "Unsafe ZIP member: "
                + info.filename
            )

    z.extractall(
        output
    )

print(
    "Extraction complete:",
    output,
)
PY

    touch "$MARKER"

else
    echo "Extraction marker exists; skip extraction."
fi


# ------------------------------------------------------------
# Initial structural discovery
# ------------------------------------------------------------

echo
echo "============================================================"
echo "INITIAL STRUCTURE"
echo "============================================================"

echo "rgb.jpg count:"
find "$DATA_DIR" \
    -type f \
    -iname 'rgb.jpg' \
    | wc -l

echo

echo "labels.png count:"
find "$DATA_DIR" \
    -type f \
    -iname 'labels.png' \
    | wc -l

echo

echo "Candidate split directories:"
find "$DATA_DIR" \
    -type d \
    \( \
        -iname train \
        -o -iname valid \
        -o -iname val \
        -o -iname validation \
    \) \
    -print | sort

echo
echo "First sample directories:"

find "$DATA_DIR" \
    -type f \
    -iname 'rgb.jpg' \
    -printf '%h\n' \
    | sort \
    | head -10

echo
echo "Archive SHA256:"
cat "$META_DIR/yamaha_v0_zip_sha256.txt"

echo
echo "============================================================"
echo "YCOR DOWNLOAD STAGE COMPLETE"
echo "============================================================"
