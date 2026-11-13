#!/usr/bin/env python3

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(
    "/root/autodl-tmp/datasets/"
    "YCOR-external/data"
)


EXPECTED_INDEX_TO_RGB = {
    0: (255, 255, 255),
    1: (178, 176, 153),
    2: (128, 255, 0),
    3: (156, 76, 30),
    4: (255, 0, 128),
    5: (255, 0, 0),
    6: (0, 160, 0),
    7: (40, 80, 0),
    8: (1, 88, 255),
}


RGB_TO_NAME = {
    (255, 255, 255):
        "background",

    (40, 80, 0):
        "high_vegetation",

    (128, 255, 0):
        "traversable_grass",

    (178, 176, 153):
        "smooth_trail",

    (255, 0, 0):
        "obstacle",

    (1, 88, 255):
        "sky",

    (156, 76, 30):
        "rough_trail",

    (255, 0, 128):
        "puddle",

    (0, 160, 0):
        "non_traversable_low_vegetation",
}


labels = sorted(
    p
    for p in ROOT.rglob("*")
    if (
        p.is_file()
        and p.name.lower() == "labels.png"
    )
)


print("=" * 90)
print("YCOR PALETTE CONSISTENCY AUDIT")
print("=" * 90)

print("label count:", len(labels))


if len(labels) != 1076:
    raise RuntimeError(
        f"Expected 1076 labels, found {len(labels)}"
    )


mode_counts = Counter()
raw_index_image_counts = Counter()

bad_modes = []
bad_raw_indices = []
bad_palette_files = []

palette_signatures = Counter()

total_raw_pixels = Counter()


for i, path in enumerate(labels, start=1):

    with Image.open(path) as im:

        mode_counts[im.mode] += 1

        if im.mode != "P":
            bad_modes.append(
                (
                    str(path),
                    im.mode,
                )
            )
            continue

        raw = np.asarray(im)

        if raw.ndim != 2:
            raise RuntimeError(
                f"Unexpected indexed-mask shape "
                f"{raw.shape}: {path}"
            )

        values, counts = np.unique(
            raw,
            return_counts=True,
        )

        raw_ids = {
            int(v)
            for v in values
        }

        unexpected_ids = (
            raw_ids
            - set(EXPECTED_INDEX_TO_RGB)
        )

        if unexpected_ids:
            bad_raw_indices.append(
                (
                    str(path),
                    sorted(unexpected_ids),
                )
            )

        for value, count in zip(
            values,
            counts,
        ):
            idx = int(value)

            total_raw_pixels[
                idx
            ] += int(count)

            raw_index_image_counts[
                idx
            ] += 1

        palette = im.getpalette()

        if palette is None:
            bad_palette_files.append(
                (
                    str(path),
                    "missing palette",
                )
            )
            continue

        actual = {}

        for idx in EXPECTED_INDEX_TO_RGB:

            base = idx * 3

            rgb = tuple(
                int(x)
                for x in palette[
                    base:
                    base + 3
                ]
            )

            actual[idx] = rgb

        signature = tuple(
            actual[idx]
            for idx in sorted(actual)
        )

        palette_signatures[
            signature
        ] += 1

        mismatches = []

        for idx, expected_rgb in (
            EXPECTED_INDEX_TO_RGB.items()
        ):

            actual_rgb = actual[
                idx
            ]

            if actual_rgb != expected_rgb:

                mismatches.append(
                    (
                        idx,
                        actual_rgb,
                        expected_rgb,
                    )
                )

        if mismatches:

            bad_palette_files.append(
                (
                    str(path),
                    mismatches,
                )
            )


print()
print("PIL modes:")
print(dict(mode_counts))


print()
print("Distinct palette signatures:")
print(len(palette_signatures))

for signature, count in (
    palette_signatures.most_common()
):

    print()
    print(
        "images:",
        count,
    )

    for idx, rgb in enumerate(
        signature
    ):

        name = RGB_TO_NAME.get(
            rgb,
            "UNKNOWN",
        )

        print(
            f"  {idx}: "
            f"{str(rgb):<18} "
            f"{name}"
        )


print()
print("=" * 90)
print("RAW INDEX STATISTICS")
print("=" * 90)

total_pixels = sum(
    total_raw_pixels.values()
)

for idx in sorted(
    total_raw_pixels
):

    pixels = total_raw_pixels[
        idx
    ]

    images = raw_index_image_counts[
        idx
    ]

    rgb = EXPECTED_INDEX_TO_RGB.get(
        idx
    )

    name = RGB_TO_NAME.get(
        rgb,
        "UNKNOWN",
    )

    share = (
        100.0
        * pixels
        / total_pixels
    )

    print(
        f"{idx}: "
        f"{name:<38} "
        f"RGB={rgb!s:<18} "
        f"pixels={pixels:>14d} "
        f"share={share:>9.5f}% "
        f"images={images:>4d}"
    )


print()
print("=" * 90)
print("STRICT CHECKS")
print("=" * 90)


checks = {
    "label_count":
        len(labels) == 1076,

    "all_masks_P_mode":
        mode_counts == {"P": 1076},

    "no_unexpected_raw_indices":
        len(bad_raw_indices) == 0,

    "one_global_palette":
        len(palette_signatures) == 1,

    "expected_palette_mapping":
        len(bad_palette_files) == 0,

    "all_expected_indices_observed":
        set(total_raw_pixels)
        == set(EXPECTED_INDEX_TO_RGB),
}


failed = []

for name, passed in checks.items():

    print(
        f"{'PASS' if passed else 'FAIL':<6} "
        f"{name}"
    )

    if not passed:
        failed.append(name)


if bad_modes:

    print()
    print(
        "Bad modes examples:"
    )

    for item in bad_modes[:20]:
        print(item)


if bad_raw_indices:

    print()
    print(
        "Unexpected raw-index examples:"
    )

    for item in bad_raw_indices[:20]:
        print(item)


if bad_palette_files:

    print()
    print(
        "Palette mismatch examples:"
    )

    for item in bad_palette_files[:20]:
        print(item)


print()

if failed:

    print(
        "YCOR PALETTE AUDIT FAILED"
    )

    print(
        "Failed checks:",
        failed,
    )

    raise SystemExit(1)


print(
    "ALL YCOR PALETTE CHECKS PASSED"
)
