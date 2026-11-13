#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    ".",
)

from lib.models import model_factory


MANIFEST = Path(
    "experiments/"
    "external_ycor_zero_shot/"
    "protocol/checkpoints_manifest.json"
)

records = json.loads(
    MANIFEST.read_text()
)

pid_records = [
    r
    for r in records
    if r["model"]
    == "PIDNet-S"
]

assert len(
    pid_records
) == 3


allowed_prefixes = (
    "seghead_p.",
    "seghead_d.",
)


for record in pid_records:

    seed = int(
        record["seed"]
    )

    path = Path(
        record["eval_path"]
    )

    model = model_factory[
        "pidnet_s_external"
    ](
        3
    )

    target = (
        model.state_dict()
    )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(obj, dict)
        and "model" in obj
        and isinstance(
            obj["model"],
            dict,
        )
    ):
        state = obj["model"]
    else:
        state = obj

    target_keys = set(
        target
    )

    source_keys = set(
        state
    )

    missing = sorted(
        target_keys
        - source_keys
    )

    unexpected = sorted(
        source_keys
        - target_keys
    )

    allowed_extra = [
        k
        for k in unexpected
        if k.startswith(
            allowed_prefixes
        )
    ]

    illegal_extra = [
        k
        for k in unexpected
        if not k.startswith(
            allowed_prefixes
        )
    ]

    shape_mismatch = []

    for key in sorted(
        target_keys
        & source_keys
    ):

        if (
            tuple(
                target[key].shape
            )
            !=
            tuple(
                state[key].shape
            )
        ):

            shape_mismatch.append(
                (
                    key,
                    tuple(
                        state[key].shape
                    ),
                    tuple(
                        target[key].shape
                    ),
                )
            )

    print(
        "=" * 80
    )

    print(
        f"PIDNet-S seed{seed}"
    )

    print(
        "checkpoint keys :",
        len(source_keys),
    )

    print(
        "target keys     :",
        len(target_keys),
    )

    print(
        "training extras :",
        len(allowed_extra),
    )

    print(
        "missing         :",
        len(missing),
    )

    print(
        "illegal extras  :",
        len(illegal_extra),
    )

    print(
        "shape mismatch  :",
        len(shape_mismatch),
    )

    print(
        "extra prefixes  :",
        sorted(
            {
                k.split(
                    ".",
                    1,
                )[0]
                for k in allowed_extra
            }
        ),
    )

    if missing:
        print(
            "MISSING:",
            missing,
        )

    if illegal_extra:
        print(
            "ILLEGAL:",
            illegal_extra,
        )

    if shape_mismatch:
        print(
            "SHAPE:",
            shape_mismatch,
        )

    assert not missing
    assert not illegal_extra
    assert not shape_mismatch

    filtered = {
        k: v
        for k, v in state.items()
        if k in target_keys
    }

    assert (
        set(filtered)
        == target_keys
    )

    model.load_state_dict(
        filtered,
        strict=True,
    )

    print(
        "STRICT FILTERED LOAD: PASS"
    )


print()
print(
    "ALL 3 PIDNET INFERENCE-STATE AUDITS PASSED"
)
