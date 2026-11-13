#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        required=True,
    )

    parser.add_argument(
        "--ann",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    base = Path(
        args.base
    )

    out = Path(
        args.out
    )

    ann = args.ann

    if not base.is_file():
        raise FileNotFoundError(
            base
        )

    if not Path(
        ann
    ).is_file():
        raise FileNotFoundError(
            ann
        )

    text = base.read_text()

    pattern = re.compile(
        r"(val_im_anns\s*=\s*)"
        r"['\"][^'\"]+['\"]"
    )

    replacement = (
        r"\1"
        + repr(ann)
    )

    new_text, count = (
        pattern.subn(
            replacement,
            text,
        )
    )

    if count != 1:
        raise RuntimeError(
            "Expected exactly one "
            "val_im_anns replacement, "
            f"got {count}"
        )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        new_text
    )

    print(
        "Base config :",
        base,
    )

    print(
        "Annotation  :",
        ann,
    )

    print(
        "Output      :",
        out,
    )


if __name__ == "__main__":
    main()
