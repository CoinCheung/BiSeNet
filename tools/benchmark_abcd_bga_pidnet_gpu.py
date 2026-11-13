#!/usr/bin/env python3

import csv
import gc
import statistics
import time
from pathlib import Path

import torch

from lib.models import model_factory


# ============================================================
# Fixed benchmark protocol
# ============================================================

N_CATS = 3

INPUT_SHAPE = (
    1,
    3,
    576,
    704,
)

WARMUP = 100
RUNS = 500
REPEATS = 3

DEVICE = "cuda"

OUTPUT_CSV = Path(
    "experiments/benchmark/"
    "abcd_bga_pidnet_gpu_benchmark.csv"
)


MODELS = [
    {
        "name": "A",
        "model_type": "bisenetv2",
        "weight": (
            "experiments/"
            "rugd3_baseline_b16_seed123_formal/"
            "best_model.pth"
        ),
    },
    {
        "name": "C",
        "model_type": "bisenetv2_boundary_refine",
        "weight": (
            "experiments/"
            "rugd3_boundary_refine_b16_seed123_formal/"
            "best_model.pth"
        ),
    },
    {
        "name": "D",
        "model_type": "bisenetv2_boundary_full",
        "weight": (
            "experiments/"
            "rugd3_boundary_full_b16_seed123_formal/"
            "best_model.pth"
        ),
    },
    {
        "name": "D-BGA",
        "model_type": "bisenetv2_boundary_full_bga",
        "weight": (
            "experiments/"
            "rugd3_boundary_full_bga_b16_seed123_formal/"
            "best_model.pth"
        ),
    },
    {
        "name": "PIDNet-S",
        "model_type": "pidnet_s_external",
        "weight": (
            "experiments/"
            "pidnet_s_rugd3_seed123_formal/"
            "best_model.pth"
        ),
    },
]


# ============================================================
# Utilities
# ============================================================

def unwrap_state_dict(obj):
    """
    Support:
        plain state_dict
        {'model': state_dict}
        {'state_dict': state_dict}

    Also remove a global 'module.' prefix if present.
    """

    if isinstance(obj, dict):

        if (
            "model" in obj
            and isinstance(obj["model"], dict)
        ):
            obj = obj["model"]

        elif (
            "state_dict" in obj
            and isinstance(obj["state_dict"], dict)
        ):
            obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(
            "Unsupported checkpoint format"
        )

    keys = list(obj.keys())

    if (
        keys
        and all(
            k.startswith("module.")
            for k in keys
        )
    ):
        obj = {
            k[len("module."):]: v
            for k, v in obj.items()
        }

    return obj


def build_model(
    model_type,
    weight_path,
):
    """
    Build the inference/eval model and load only parameters
    that are actually present in that model.

    Formal training checkpoints may contain training-only
    modules such as:
        aux2 / aux3 / aux4 / aux5_4
        boundary_head
        PIDNet seghead_p / seghead_d

    Those modules are intentionally absent from the eval
    graph and therefore must not cause strict-load failure.

    Critical rule:
        every parameter required by the inference model
        MUST exist and have the correct shape.
    """

    if model_type not in model_factory:
        raise KeyError(
            f"Unknown model_type: {model_type}"
        )

    model = model_factory[model_type](
        N_CATS,
        aux_mode="eval",
    )

    checkpoint = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=False,
    )

    state = unwrap_state_dict(
        checkpoint
    )

    target = model.state_dict()

    compatible = {}
    skipped = []

    for key, value in state.items():

        if (
            key in target
            and target[key].shape == value.shape
        ):
            compatible[key] = value

        else:
            skipped.append(key)

    missing = [
        key
        for key in target
        if key not in compatible
    ]

    if missing:
        print()
        print(
            "ERROR: inference model has "
            "missing checkpoint tensors:"
        )

        for key in missing:
            print("  MISSING:", key)

        raise RuntimeError(
            f"{model_type}: "
            f"{len(missing)} required inference "
            f"tensors are missing"
        )

    result = model.load_state_dict(
        compatible,
        strict=True,
    )

    if (
        result.missing_keys
        or result.unexpected_keys
    ):
        raise RuntimeError(
            f"{model_type}: state-dict load "
            f"still inconsistent:\n"
            f"missing={result.missing_keys}\n"
            f"unexpected={result.unexpected_keys}"
        )

    # --------------------------------------------------------
    # Audit ignored training-only tensors
    # --------------------------------------------------------

    skipped_groups = {}

    for key in skipped:

        prefix = key.split(".", 1)[0]

        skipped_groups[prefix] = (
            skipped_groups.get(
                prefix,
                0,
            )
            + 1
        )

    print(
        "Checkpoint tensors:",
        len(state),
    )

    print(
        "Loaded inference tensors:",
        len(compatible),
    )

    print(
        "Ignored checkpoint tensors:",
        len(skipped),
    )

    if skipped_groups:

        print(
            "Ignored tensor groups:"
        )

        for prefix in sorted(
            skipped_groups
        ):
            print(
                f"  {prefix}: "
                f"{skipped_groups[prefix]}"
            )

    print(
        "Missing inference tensors:",
        len(missing),
    )

    # Safety audit:
    # these modules must NEVER be skipped because they are
    # part of the deployed inference graph.
    forbidden_skipped_prefixes = {
        "boundary_refine",
        "bga",
        "head",
        "detail",
        "segment",
    }

    bad_skipped = []

    for key in skipped:

        prefix = key.split(".", 1)[0]

        if prefix in forbidden_skipped_prefixes:
            bad_skipped.append(key)

    if bad_skipped:

        print()
        print(
            "ERROR: inference-path tensors "
            "were skipped:"
        )

        for key in bad_skipped:
            print(
                "  BAD SKIP:",
                key,
            )

        raise RuntimeError(
            f"{model_type}: inference-path "
            f"parameters were skipped"
        )


    model = model.to(
        DEVICE
    ).eval()

    return model

def get_main_output(output):
    """
    All models should finally expose semantic logits.

    BiSeNet evaluator models and PIDNet adapter normally
    return a tuple/list whose first element is the logits.
    """

    if isinstance(
        output,
        (tuple, list),
    ):
        output = output[0]

    if not torch.is_tensor(
        output
    ):
        raise TypeError(
            f"Unexpected model output: "
            f"{type(output)}"
        )

    return output


def count_total_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_trainable_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_reachable_params(
    model,
    x,
):
    """
    Count parameters belonging to modules that are actually
    executed by one eval forward.

    This excludes instantiated but unused auxiliary heads
    and training-only boundary heads.
    """

    param_numel = {
        id(p): p.numel()
        for p in model.parameters()
    }

    executed_param_ids = set()

    handles = []

    def hook_fn(
        module,
        inputs,
        output,
    ):
        for p in module.parameters(
            recurse=False
        ):
            executed_param_ids.add(
                id(p)
            )

    for module in model.modules():

        handles.append(
            module.register_forward_hook(
                hook_fn
            )
        )

    with torch.inference_mode():
        output = model(x)
        logits = get_main_output(
            output
        )

    for handle in handles:
        handle.remove()

    reachable = sum(
        param_numel[pid]
        for pid in executed_param_ids
        if pid in param_numel
    )

    return reachable, logits


def count_special_calls(
    model,
    x,
):
    """
    Verify that LBRM / training-only Boundary Head execute
    as expected in eval mode.

    Expected:
      A       refine=0 boundary=0
      C       refine=1 boundary=0
      D       refine=1 boundary=0
      D-BGA   refine=1 boundary=0
      PIDNet  refine=0 boundary=0
    """

    counts = {
        "refine": 0,
        "boundary": 0,
    }

    handles = []

    if hasattr(
        model,
        "boundary_refine",
    ):
        handles.append(
            model.boundary_refine
            .register_forward_hook(
                lambda m, i, o:
                counts.__setitem__(
                    "refine",
                    counts["refine"] + 1,
                )
            )
        )

    if hasattr(
        model,
        "boundary_head",
    ):
        handles.append(
            model.boundary_head
            .register_forward_hook(
                lambda m, i, o:
                counts.__setitem__(
                    "boundary",
                    counts["boundary"] + 1,
                )
            )
        )

    with torch.inference_mode():
        output = model(x)
        logits = get_main_output(
            output
        )

    for handle in handles:
        handle.remove()

    return counts, logits


def benchmark_model(
    model,
    x,
):
    """
    CUDA-event measurement.

    Each repeat measures RUNS forwards as one block,
    then divides total GPU elapsed time by RUNS.

    Return the median of REPEATS average-forward latencies.
    """

    # Warmup
    with torch.inference_mode():

        for _ in range(
            WARMUP
        ):
            _ = model(x)

    torch.cuda.synchronize()

    latencies_ms = []

    for repeat_idx in range(
        REPEATS
    ):
        start = torch.cuda.Event(
            enable_timing=True
        )

        end = torch.cuda.Event(
            enable_timing=True
        )

        torch.cuda.synchronize()

        start.record()

        with torch.inference_mode():

            for _ in range(
                RUNS
            ):
                _ = model(x)

        end.record()

        torch.cuda.synchronize()

        total_ms = start.elapsed_time(
            end
        )

        avg_ms = (
            total_ms
            / RUNS
        )

        latencies_ms.append(
            avg_ms
        )

    median_ms = statistics.median(
        latencies_ms
    )

    fps = (
        1000.0
        / median_ms
    )

    return (
        latencies_ms,
        median_ms,
        fps,
    )


# ============================================================
# Main
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(
        123
    )

    torch.cuda.manual_seed_all(
        123
    )

    # Keep benchmark behavior stable and close to the
    # previous RTX5090 measurement.
    torch.backends.cudnn.benchmark = True

    device_name = (
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU:",
        device_name,
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA runtime:",
        torch.version.cuda,
    )

    print(
        "Input:",
        INPUT_SHAPE,
    )

    print(
        "Warmup:",
        WARMUP,
    )

    print(
        "Runs:",
        RUNS,
    )

    print(
        "Repeats:",
        REPEATS,
    )

    print(
        "Precision: FP32 "
        "(no autocast)"
    )

    print(
        "cudnn.benchmark:",
        torch.backends.cudnn.benchmark,
    )

    print(
        "cudnn.allow_tf32:",
        torch.backends.cudnn.allow_tf32,
    )

    print(
        "cuda.matmul.allow_tf32:",
        torch.backends.cuda.matmul.allow_tf32,
    )

    print()

    # Same fixed random input for all models.
    x = torch.randn(
        *INPUT_SHAPE,
        device=DEVICE,
        dtype=torch.float32,
    )

    results = []

    for spec in MODELS:

        name = spec["name"]
        model_type = spec["model_type"]
        weight_path = Path(
            spec["weight"]
        )

        if not weight_path.is_file():
            raise FileNotFoundError(
                weight_path
            )

        print(
            "=" * 72
        )

        print(
            f"Benchmark: {name}"
        )

        print(
            f"Model: {model_type}"
        )

        print(
            f"Weights: {weight_path}"
        )

        model = build_model(
            model_type,
            weight_path,
        )

        total_params = (
            count_total_params(
                model
            )
        )

        trainable_params = (
            count_trainable_params(
                model
            )
        )

        reachable_params, logits = (
            count_reachable_params(
                model,
                x,
            )
        )

        if tuple(
            logits.shape
        ) != (
            1,
            3,
            576,
            704,
        ):
            raise RuntimeError(
                f"{name}: bad output shape: "
                f"{tuple(logits.shape)}"
            )

        special_calls, logits2 = (
            count_special_calls(
                model,
                x,
            )
        )

        if tuple(
            logits2.shape
        ) != (
            1,
            3,
            576,
            704,
        ):
            raise RuntimeError(
                f"{name}: bad output shape "
                f"during hook test: "
                f"{tuple(logits2.shape)}"
            )

        print(
            "Output shape:",
            tuple(logits.shape),
        )

        print(
            "Forward-hook counts:",
            special_calls,
        )

        print(
            "Total params:",
            f"{total_params:,}",
        )

        print(
            "Trainable params:",
            f"{trainable_params:,}",
        )

        print(
            "Inference-reachable params:",
            f"{reachable_params:,}",
        )

        unreachable = (
            total_params
            - reachable_params
        )

        print(
            "Not reached in eval forward:",
            f"{unreachable:,}",
        )

        repeated_ms, latency_ms, fps = (
            benchmark_model(
                model,
                x,
            )
        )

        print(
            "Repeated latency (ms):",
            [
                round(v, 4)
                for v in repeated_ms
            ],
        )

        print(
            f"{name}: "
            f"{latency_ms:.4f} ms, "
            f"{fps:.2f} FPS"
        )

        results.append({
            "name": name,
            "model_type": model_type,
            "weight": str(
                weight_path
            ),
            "total_params": (
                total_params
            ),
            "trainable_params": (
                trainable_params
            ),
            "reachable_params": (
                reachable_params
            ),
            "latency_ms": (
                latency_ms
            ),
            "fps": fps,
            "repeat_ms": (
                repeated_ms
            ),
        })

        del model

        gc.collect()

        torch.cuda.empty_cache()

        torch.cuda.synchronize()

        # Small idle interval reduces cross-model thermal carryover.
        time.sleep(
            1.0
        )

    # --------------------------------------------------------
    # Relative latency
    # --------------------------------------------------------

    baseline_ms = results[0][
        "latency_ms"
    ]

    print()
    print(
        "=" * 92
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 92
    )

    print(
        f"{'Model':<12}"
        f"{'Total Params':>16}"
        f"{'Reachable Params':>20}"
        f"{'Latency (ms)':>16}"
        f"{'FPS':>12}"
        f"{'vs A':>12}"
    )

    for r in results:

        delta = (
            r["latency_ms"]
            / baseline_ms
            - 1.0
        ) * 100.0

        r["vs_a_pct"] = delta

        print(
            f"{r['name']:<12}"
            f"{r['total_params']:>16,}"
            f"{r['reachable_params']:>20,}"
            f"{r['latency_ms']:>16.4f}"
            f"{r['fps']:>12.2f}"
            f"{delta:>+11.2f}%"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([
            "model",
            "model_type",
            "weight",
            "total_params",
            "trainable_params",
            "inference_reachable_params",
            "latency_ms_median",
            "fps",
            "latency_vs_A_percent",
            "repeat_1_ms",
            "repeat_2_ms",
            "repeat_3_ms",
        ])

        for r in results:

            writer.writerow([
                r["name"],
                r["model_type"],
                r["weight"],
                r["total_params"],
                r["trainable_params"],
                r["reachable_params"],
                f"{r['latency_ms']:.6f}",
                f"{r['fps']:.6f}",
                f"{r['vs_a_pct']:.6f}",
                *[
                    f"{v:.6f}"
                    for v in r["repeat_ms"]
                ],
            ])

    print()
    print(
        "CSV saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
