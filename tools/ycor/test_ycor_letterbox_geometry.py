#!/usr/bin/env python3

from pathlib import Path
from PIL import Image

ORIG_W = 1024
ORIG_H = 544

CANVAS_W = 704
CANVAS_H = 576

scale = min(
    CANVAS_W / ORIG_W,
    CANVAS_H / ORIG_H,
)

new_w = round(
    ORIG_W * scale
)

new_h = round(
    ORIG_H * scale
)

pad_left = (
    CANVAS_W - new_w
) // 2

pad_right = (
    CANVAS_W
    - new_w
    - pad_left
)

pad_top = (
    CANVAS_H - new_h
) // 2

pad_bottom = (
    CANVAS_H
    - new_h
    - pad_top
)

print("original =", (ORIG_W, ORIG_H))
print("scale    =", scale)
print("resize   =", (new_w, new_h))
print(
    "padding  =",
    (
        pad_left,
        pad_top,
        pad_right,
        pad_bottom,
    ),
)
print(
    "canvas   =",
    (
        new_w + pad_left + pad_right,
        new_h + pad_top + pad_bottom,
    ),
)

assert abs(
    scale - 0.6875
) < 1e-12

assert (
    new_w,
    new_h,
) == (
    704,
    374,
)

assert (
    pad_left,
    pad_right,
) == (
    0,
    0,
)

assert (
    pad_top,
    pad_bottom,
) == (
    101,
    101,
)

root = Path(
    "/root/autodl-tmp/datasets/"
    "YCOR-external/data"
)

rgb_files = sorted(
    root.rglob("rgb.jpg")
)

assert len(rgb_files) == 1076

for rgb in rgb_files:

    gt = rgb.parent / "labels.png"

    assert gt.is_file()

    with Image.open(rgb) as im:
        assert im.size == (
            ORIG_W,
            ORIG_H,
        )

    with Image.open(gt) as im:
        assert im.size == (
            ORIG_W,
            ORIG_H,
        )

# Check interpolation semantics on a real sample.

rgb = rgb_files[0]
gt = rgb.parent / "labels.png"

with Image.open(rgb) as im:

    resized = im.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    assert resized.size == (
        704,
        374,
    )

with Image.open(gt) as im:

    resized = im.resize(
        (new_w, new_h),
        Image.Resampling.NEAREST,
    )

    assert resized.size == (
        704,
        374,
    )

print()
print(
    "YCOR LETTERBOX GEOMETRY UNIT TEST: PASS"
)
