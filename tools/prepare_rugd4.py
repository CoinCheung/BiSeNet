from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


TARGET_GROUPS = {
    0: {
        "sky",
    },
    1: {
        "dirt",
        "sand",
        "grass",
        "asphalt",
        "gravel",
        "mulch",
        "bridge",
        "concrete",
    },
    2: {
        "water",
        "rock-bed",
        "bush",
    },
    3: {
        "tree",
        "pole",
        "vehicle",
        "container/generic-object",
        "building",
        "log",
        "bicycle",
        "person",
        "fence",
        "sign",
        "rock",
        "picnic-table",
    },
    255: {
        "void",
    },
}


def parse_colormap(
    path: Path,
) -> dict[tuple[int, int, int], int]:
    """Parse lines formatted as: id class_name r g b."""
    color_to_target: dict[tuple[int, int, int], int] = {}
    known_names: set[str] = set()

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if len(fields) < 5:
            raise ValueError(
                f"Invalid colormap line: {raw_line}"
            )

        class_name = " ".join(fields[1:-3])
        red, green, blue = map(int, fields[-3:])

        target_id = None

        for candidate_id, names in TARGET_GROUPS.items():
            if class_name in names:
                target_id = candidate_id
                break

        if target_id is None:
            raise ValueError(
                f"Class is not mapped: {class_name}"
            )

        color_to_target[
            (red, green, blue)
        ] = target_id

        known_names.add(class_name)

    expected_names = set().union(
        *TARGET_GROUPS.values()
    )

    missing = expected_names - known_names

    if missing:
        raise ValueError(
            f"Classes missing from colormap: "
            f"{sorted(missing)}"
        )

    return color_to_target


def convert_mask(
    source: Path,
    destination: Path,
    color_to_target: dict[tuple[int, int, int], int],
) -> None:
    rgb = np.asarray(
        Image.open(source).convert("RGB")
    )

    output = np.full(
        rgb.shape[:2],
        255,
        dtype=np.uint8,
    )

    for color, target_id in color_to_target.items():
        matched = np.all(
            rgb == np.asarray(
                color,
                dtype=np.uint8,
            ),
            axis=-1,
        )

        output[matched] = target_id

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        output,
        mode="L",
    ).save(destination)


def create_split(
    sequences: list[str],
    split_file: Path,
    seed: int,
) -> dict[str, list[str]]:
    if split_file.exists():
        return json.loads(
            split_file.read_text(
                encoding="utf-8"
            )
        )

    shuffled = sequences.copy()
    random.Random(seed).shuffle(shuffled)

    sequence_count = len(shuffled)

    val_count = max(
        1,
        round(sequence_count * 0.10),
    )
    test_count = max(
        1,
        round(sequence_count * 0.25),
    )

    split = {
        "val": sorted(
            shuffled[:val_count]
        ),
        "test": sorted(
            shuffled[
                val_count:
                val_count + test_count
            ]
        ),
        "train": sorted(
            shuffled[
                val_count + test_count:
            ]
        ),
    }

    split_file.write_text(
        json.dumps(
            split,
            indent=2,
        ),
        encoding="utf-8",
    )

    return split


def safe_symlink(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() or destination.is_symlink():
        return

    os.symlink(
        source.resolve(),
        destination,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    frames_root = (
        args.raw_root
        / "RUGD_frames-with-annotations"
    )
    masks_root = (
        args.raw_root
        / "RUGD_annotations"
    )
    colormap_path = (
        args.raw_root
        / "RUGD_annotation-colormap.txt"
    )

    for required in (
        frames_root,
        masks_root,
        colormap_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    frame_sequences = {
        path.name
        for path in frames_root.iterdir()
        if path.is_dir()
    }

    mask_sequences = {
        path.name
        for path in masks_root.iterdir()
        if path.is_dir()
    }

    sequences = sorted(
        frame_sequences & mask_sequences
    )

    if not sequences:
        raise RuntimeError(
            "No matching RUGD sequences found"
        )

    split_file = (
        args.output_root
        / "sequence_split.json"
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = create_split(
        sequences,
        split_file,
        args.seed,
    )

    color_to_target = parse_colormap(
        colormap_path
    )

    statistics: dict[str, int] = {}

    for split_name, split_sequences in split.items():
        list_lines: list[str] = []

        for sequence in split_sequences:
            frame_dir = frames_root / sequence
            mask_dir = masks_root / sequence

            for frame_path in sorted(
                frame_dir.glob("*.png")
            ):
                mask_path = (
                    mask_dir
                    / frame_path.name
                )

                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing mask: {mask_path}"
                    )

                relative_image = (
                    Path("images")
                    / split_name
                    / sequence
                    / frame_path.name
                )

                relative_label = (
                    Path("labels")
                    / split_name
                    / sequence
                    / frame_path.name
                )

                output_image = (
                    args.output_root
                    / relative_image
                )

                output_label = (
                    args.output_root
                    / relative_label
                )

                safe_symlink(
                    frame_path,
                    output_image,
                )

                convert_mask(
                    mask_path,
                    output_label,
                    color_to_target,
                )

                list_lines.append(
                    f"{relative_image},"
                    f"{relative_label}"
                )

        list_path = (
            args.output_root
            / f"{split_name}.txt"
        )

        list_path.write_text(
            "\n".join(list_lines) + "\n",
            encoding="utf-8",
        )

        statistics[split_name] = len(
            list_lines
        )

    print(
        json.dumps(
            {
                "sequences": split,
                "images": statistics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()