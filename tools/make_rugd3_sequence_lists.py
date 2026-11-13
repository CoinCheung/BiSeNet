#!/usr/bin/env python3

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SPLITS = {
    "val": {
        "source": Path(
            "datasets/rugd3/val.txt"
        ),
        "sequences": [
            "trail-4",
            "trail-5",
        ],
    },

    "test": {
        "source": Path(
            "datasets/rugd3/test.txt"
        ),
        "sequences": [
            "trail",
            "trail-10",
            "trail-12",
            "trail-14",
        ],
    },
}


OUT_ROOT = Path(
    "datasets/rugd3/sequence_eval"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def path_has_sequence(
    token,
    sequence,
):
    """
    Match sequence as an exact path component.

    Important:
      trail must NOT match trail-10.
    """

    token = (
        token
        .strip()
        .strip("'\";,")
        .replace("\\", "/")
    )

    pattern = (
        rf"(^|/)"
        rf"{re.escape(sequence)}"
        rf"(/|$)"
    )

    return (
        re.search(
            pattern,
            token,
        )
        is not None
    )


def detect_sequence(
    line,
    sequences,
):

    tokens = re.split(
        r"\s+",
        line.strip(),
    )

    found = set()

    for token in tokens:

        for sequence in sequences:

            if path_has_sequence(
                token,
                sequence,
            ):
                found.add(
                    sequence
                )

    if len(found) != 1:

        raise RuntimeError(
            "Could not uniquely determine "
            "sequence.\n"
            f"line={line}\n"
            f"matches={sorted(found)}"
        )

    return next(
        iter(found)
    )


def main():

    manifest = {}

    for split, spec in SPLITS.items():

        source = spec["source"]

        sequences = spec[
            "sequences"
        ]

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        lines = [
            line.rstrip("\n")
            for line
            in source.read_text().splitlines()
            if line.strip()
        ]

        buckets = {
            sequence: []
            for sequence in sequences
        }

        for line in lines:

            sequence = (
                detect_sequence(
                    line,
                    sequences,
                )
            )

            buckets[
                sequence
            ].append(
                line
            )

        out_dir = (
            OUT_ROOT /
            split
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_lines = []

        split_info = {
            "source": str(source),
            "source_sha256":
                sha256_file(source),
            "source_count":
                len(lines),
            "sequences": {},
        }

        print()
        print(
            "=" * 60
        )
        print(
            f"{split.upper()} "
            f"sequence split"
        )
        print(
            "=" * 60
        )

        for sequence in sequences:

            dst = (
                out_dir /
                f"{sequence}.txt"
            )

            seq_lines = (
                buckets[
                    sequence
                ]
            )

            dst.write_text(
                "\n".join(
                    seq_lines
                )
                + "\n"
            )

            generated_lines.extend(
                seq_lines
            )

            split_info[
                "sequences"
            ][sequence] = {
                "count":
                    len(seq_lines),
                "file":
                    str(dst),
                "sha256":
                    sha256_file(dst),
            }

            print(
                f"{sequence:<12s}: "
                f"{len(seq_lines)}"
            )

        # Exact multiset equality:
        # generated sequence lists must
        # partition the original split.
        if Counter(
            generated_lines
        ) != Counter(
            lines
        ):
            raise RuntimeError(
                f"{split}: generated "
                f"sequence lists do not "
                f"exactly partition source"
            )

        generated_count = sum(
            len(v)
            for v
            in buckets.values()
        )

        if (
            generated_count
            != len(lines)
        ):
            raise RuntimeError(
                f"{split}: count mismatch"
            )

        print(
            "TOTAL       :",
            generated_count,
        )

        print(
            "SOURCE      :",
            len(lines),
        )

        print(
            "PARTITION   : PASSED"
        )

        manifest[
            split
        ] = split_info

    manifest_path = (
        OUT_ROOT /
        "sequence_split_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "RUGD3 SEQUENCE LIST "
        "GENERATION PASSED"
    )


if __name__ == "__main__":
    main()
