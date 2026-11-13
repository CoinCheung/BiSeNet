#!/usr/bin/env bash

set -euo pipefail

ROOT="${YCOR_ROOT:-/root/autodl-tmp/datasets/YCOR-external}"

ARCHIVE="${YCOR_ARCHIVE:-$ROOT/downloads/yamaha_seg.tar.gz}"

DATA_DIR="$ROOT/data"
META_DIR="$ROOT/meta"

EXPECTED_TOTAL=1076
EXPECTED_TRAIN=931
EXPECTED_VAL=145

mkdir -p \
    "$ROOT" \
    "$ROOT/downloads" \
    "$DATA_DIR" \
    "$META_DIR"


echo "============================================================"
echo "YCOR LOCAL ARCHIVE PREPARATION"
echo "============================================================"

echo "ROOT    : $ROOT"
echo "ARCHIVE : $ARCHIVE"
echo


# ============================================================
# 1. Archive existence
# ============================================================

if [[ ! -s "$ARCHIVE" ]]; then

    echo "ERROR: archive not found or empty:"
    echo "$ARCHIVE"

    exit 1
fi


echo "Archive:"
ls -lh "$ARCHIVE"

echo
file "$ARCHIVE"


# ============================================================
# 2. Gzip / TAR integrity
# ============================================================

echo
echo "Checking gzip integrity..."

gzip -t "$ARCHIVE"

echo "GZIP INTEGRITY: PASS"


echo
echo "Checking tar integrity..."

tar -tzf "$ARCHIVE" \
    >/dev/null

echo "TAR INTEGRITY: PASS"


# ============================================================
# 3. Save archive metadata
# ============================================================

ARCHIVE_SHA="$(
    sha256sum "$ARCHIVE" \
    | awk '{print $1}'
)"

echo "$ARCHIVE_SHA  $ARCHIVE" \
> "$META_DIR/yamaha_seg_tar_sha256.txt"

stat "$ARCHIVE" \
> "$META_DIR/yamaha_seg_tar_stat.txt"

file "$ARCHIVE" \
> "$META_DIR/yamaha_seg_tar_filetype.txt"

tar -tzf "$ARCHIVE" \
> "$META_DIR/yamaha_seg_archive_filelist.txt"


echo
echo "Archive SHA256:"
echo "$ARCHIVE_SHA"


# ============================================================
# 4. Inspect archive structure BEFORE extraction
# ============================================================

RGB_ARCHIVE_COUNT="$(
    tar -tzf "$ARCHIVE" \
    | grep -Ei \
        '(^|/)rgb\.jpg$' \
    | wc -l
)"

LABEL_ARCHIVE_COUNT="$(
    tar -tzf "$ARCHIVE" \
    | grep -Ei \
        '(^|/)labels\.png$' \
    | wc -l
)"

TRAIN_ARCHIVE_COUNT="$(
    tar -tzf "$ARCHIVE" \
    | grep -Ei \
        '(^|/)train/[^/]+/rgb\.jpg$' \
    | wc -l
)"

VAL_ARCHIVE_COUNT="$(
    tar -tzf "$ARCHIVE" \
    | grep -Ei \
        '(^|/)(valid|validation|val)/[^/]+/rgb\.jpg$' \
    | wc -l
)"


echo
echo "============================================================"
echo "ARCHIVE STRUCTURE"
echo "============================================================"

echo "rgb.jpg     : $RGB_ARCHIVE_COUNT"
echo "labels.png  : $LABEL_ARCHIVE_COUNT"
echo "train RGB   : $TRAIN_ARCHIVE_COUNT"
echo "valid RGB   : $VAL_ARCHIVE_COUNT"


if [[ "$RGB_ARCHIVE_COUNT" -ne "$EXPECTED_TOTAL" ]]; then

    echo
    echo "ERROR:"
    echo "Expected $EXPECTED_TOTAL rgb.jpg files,"
    echo "but archive contains $RGB_ARCHIVE_COUNT."

    echo
    echo "First 80 archive entries:"
    head -80 \
        "$META_DIR/yamaha_seg_archive_filelist.txt"

    exit 2
fi


if [[ "$LABEL_ARCHIVE_COUNT" -ne "$EXPECTED_TOTAL" ]]; then

    echo
    echo "ERROR:"
    echo "Expected $EXPECTED_TOTAL labels.png files,"
    echo "but archive contains $LABEL_ARCHIVE_COUNT."

    exit 2
fi


if [[ "$TRAIN_ARCHIVE_COUNT" -ne "$EXPECTED_TRAIN" ]]; then

    echo
    echo "ERROR:"
    echo "Expected $EXPECTED_TRAIN training images,"
    echo "but archive contains $TRAIN_ARCHIVE_COUNT."

    exit 2
fi


if [[ "$VAL_ARCHIVE_COUNT" -ne "$EXPECTED_VAL" ]]; then

    echo
    echo "ERROR:"
    echo "Expected $EXPECTED_VAL validation images,"
    echo "but archive contains $VAL_ARCHIVE_COUNT."

    exit 2
fi


echo
echo "ARCHIVE STRUCTURE: PASS"


# ============================================================
# 5. Extraction-state handling
# ============================================================

HASH_MARKER="$META_DIR/extracted_archive_sha256.txt"

if [[ -s "$HASH_MARKER" ]]; then

    OLD_SHA="$(
        cat "$HASH_MARKER"
    )"

    if [[ "$OLD_SHA" == "$ARCHIVE_SHA" ]]; then

        echo
        echo "Same archive was already extracted."
        echo "Extraction skipped."

        SKIP_EXTRACT=1

    else

        echo
        echo "ERROR:"
        echo "data/ was previously extracted from another archive."
        echo
        echo "Old SHA: $OLD_SHA"
        echo "New SHA: $ARCHIVE_SHA"
        echo
        echo "Do NOT mix two YCOR packages."

        echo
        echo "If you intentionally want to replace it:"
        echo
        echo "FORCE_REEXTRACT=1 $0"

        exit 3
    fi

else

    SKIP_EXTRACT=0

    EXISTING="$(
        find "$DATA_DIR" \
            -mindepth 1 \
            -print \
            -quit
    )"

    if [[ -n "$EXISTING" ]]; then

        if [[ "${FORCE_REEXTRACT:-0}" != "1" ]]; then

            echo
            echo "ERROR:"
            echo "$DATA_DIR already contains files,"
            echo "but no extraction SHA marker exists."

            echo
            echo "To explicitly clear and re-extract:"
            echo
            echo "FORCE_REEXTRACT=1 $0"

            exit 3
        fi

        echo
        echo "FORCE_REEXTRACT=1"
        echo "Clearing existing data directory..."

        find "$DATA_DIR" \
            -mindepth 1 \
            -maxdepth 1 \
            -exec rm -rf {} +
    fi
fi


# ============================================================
# 6. Safe TAR extraction
# ============================================================

if [[ "$SKIP_EXTRACT" -eq 0 ]]; then

    echo
    echo "============================================================"
    echo "SAFE EXTRACTION"
    echo "============================================================"

    python - \
        "$ARCHIVE" \
        "$DATA_DIR" <<'PY'

import sys
import tarfile
from pathlib import Path


archive = Path(
    sys.argv[1]
).resolve()

output = Path(
    sys.argv[2]
).resolve()

output.mkdir(
    parents=True,
    exist_ok=True,
)


def ensure_safe(member):

    name = member.name

    if not name:
        return

    member_path = Path(name)

    if member_path.is_absolute():

        raise RuntimeError(
            f"Absolute archive path rejected: "
            f"{name}"
        )

    target = (
        output /
        member_path
    ).resolve()

    if (
        target != output
        and output not in target.parents
    ):

        raise RuntimeError(
            f"Path traversal rejected: "
            f"{name}"
        )

    if (
        member.issym()
        or member.islnk()
        or member.isdev()
        or member.isfifo()
    ):

        raise RuntimeError(
            f"Special/link TAR member rejected: "
            f"{name}"
        )


with tarfile.open(
    archive,
    "r:*",
) as tf:

    members = tf.getmembers()

    print(
        "Archive members:",
        len(members),
    )

    for member in members:
        ensure_safe(
            member
        )

    tf.extractall(
        path=output,
        members=members,
    )


print(
    "Extraction complete:",
    output,
)
PY

    echo "$ARCHIVE_SHA" \
    > "$HASH_MARKER"
fi


# ============================================================
# 7. Post-extraction structural verification
# ============================================================

echo
echo "============================================================"
echo "POST-EXTRACTION CHECK"
echo "============================================================"


python - \
    "$DATA_DIR" <<'PY'

import sys
from collections import Counter
from pathlib import Path


root = Path(
    sys.argv[1]
).resolve()


rgb_files = sorted(
    p
    for p in root.rglob("*")
    if (
        p.is_file()
        and p.name.lower() == "rgb.jpg"
    )
)

label_files = sorted(
    p
    for p in root.rglob("*")
    if (
        p.is_file()
        and p.name.lower() == "labels.png"
    )
)


def split_of(path):

    parts = {
        x.lower()
        for x in path.parts
    }

    if "train" in parts:
        return "train"

    if (
        "valid" in parts
        or "validation" in parts
        or "val" in parts
    ):
        return "val"

    return "unknown"


rgb_split = Counter(
    split_of(p)
    for p in rgb_files
)


paired = 0

missing_labels = []

for rgb in rgb_files:

    label = (
        rgb.parent /
        "labels.png"
    )

    if label.is_file():
        paired += 1
    else:
        missing_labels.append(
            str(rgb)
        )


print(
    "RGB files       :",
    len(rgb_files),
)

print(
    "Label files     :",
    len(label_files),
)

print(
    "Paired samples  :",
    paired,
)

print(
    "Split counts    :",
    dict(rgb_split),
)

print(
    "Missing labels  :",
    len(missing_labels),
)


EXPECTED_TOTAL = 1076
EXPECTED_TRAIN = 931
EXPECTED_VAL = 145


errors = []

if len(rgb_files) != EXPECTED_TOTAL:
    errors.append(
        "RGB count mismatch"
    )

if len(label_files) != EXPECTED_TOTAL:
    errors.append(
        "label count mismatch"
    )

if paired != EXPECTED_TOTAL:
    errors.append(
        "pair count mismatch"
    )

if rgb_split.get(
    "train",
    0,
) != EXPECTED_TRAIN:

    errors.append(
        "train count mismatch"
    )

if rgb_split.get(
    "val",
    0,
) != EXPECTED_VAL:

    errors.append(
        "val count mismatch"
    )

if rgb_split.get(
    "unknown",
    0,
) != 0:

    errors.append(
        "unknown split samples"
    )


if errors:

    print()
    print(
        "POST-EXTRACTION CHECK: FAILED"
    )

    for e in errors:
        print(
            " -",
            e,
        )

    raise SystemExit(1)


print()
print(
    "POST-EXTRACTION STRUCTURE: PASS"
)
PY


echo
echo "============================================================"
echo "YCOR LOCAL ARCHIVE PREPARATION COMPLETE"
echo "============================================================"
