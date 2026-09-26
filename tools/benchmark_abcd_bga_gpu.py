#!/usr/bin/env python3

import sys
import statistics
from pathlib import Path

sys.path.insert(0, ".")

import torch
from lib.models import model_factory


# ==========================================
# Benchmark settings
# ==========================================

DEVICE = "cuda:0"
INPUT_SHAPE = (1, 3, 576, 704)

WARMUP = 100
RUNS = 500
REPEATS = 3

N_CLASSES = 3


# ==========================================
# Model configurations
# ==========================================

MODELS = [
    (
        "A",
        "bisenetv2",
        "experiments/"
        "rugd3_baseline_b16_seed123_formal/"
        "best_model.pth",
    ),
    (
        "C",
        "bisenetv2_boundary_refine",
        "experiments/"
        "rugd3_boundary_refine_b16_seed123_formal/"
        "best_model.pth",
    ),
    (
        "D",
        "bisenetv2_boundary_full",
        "experiments/"
        "rugd3_boundary_full_b16_seed123_formal/"
        "best_model.pth",
    ),
    (
        "D-BGA",
        "bisenetv2_boundary_full_bga",
        "experiments/"
        "rugd3_boundary_full_bga_b16_seed123_formal/"
        "best_model.pth",
    ),
]


def load_model(model_type, weight_path):
    """Load the formal best checkpoint."""

    path = Path(weight_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    # Instantiate with default training architecture so
    # that auxiliary-head weights can load strictly.
    model = model_factory[model_type](N_CLASSES)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    # Also accept a complete checkpoint if supplied.
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    model.load_state_dict(
        state,
        strict=True,
    )

    # IMPORTANT:
    # eval() and aux_mode="eval" are different.
    model.aux_mode = "eval"

    model = model.to(DEVICE)
    model.eval()

    return model


def verify_inference_path(model, x):
    """Verify output shape and training-only head usage."""

    counts = {
        "refine": 0,
        "boundary": 0,
    }

    handles = []

    if hasattr(model, "boundary_refine"):
        handles.append(
            model.boundary_refine.register_forward_hook(
                lambda m, i, o: counts.__setitem__(
                    "refine",
                    counts["refine"] + 1,
                )
            )
        )

    if hasattr(model, "boundary_head"):
        handles.append(
            model.boundary_head.register_forward_hook(
                lambda m, i, o: counts.__setitem__(
                    "boundary",
                    counts["boundary"] + 1,
                )
            )
        )

    try:
        with torch.inference_mode():
            outputs = model(x)

        assert isinstance(outputs, (tuple, list))
        assert len(outputs) == 1

        assert outputs[0].shape == (
            x.shape[0],
            N_CLASSES,
            x.shape[2],
            x.shape[3],
        )

        assert counts["boundary"] == 0, (
            "Boundary Head must not execute in eval."
        )

        if hasattr(model, "boundary_refine"):
            assert counts["refine"] == 1, (
                "LBRM must execute exactly once."
            )

    finally:
        for handle in handles:
            handle.remove()

    return counts


def benchmark(model, x):
    """Measure CUDA forward latency with CUDA events."""

    timings = []

    with torch.inference_mode():

        # Warmup includes fixed-shape cuDNN preparation.
        for _ in range(WARMUP):
            model(x)

        torch.cuda.synchronize()

        for _ in range(REPEATS):

            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            torch.cuda.synchronize()

            start.record()

            for _ in range(RUNS):
                model(x)

            end.record()
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)

            timings.append(
                elapsed_ms / RUNS
            )

    # Median across three repeated measurements.
    latency_ms = statistics.median(timings)

    return latency_ms, timings


def main():

    assert torch.cuda.is_available(), (
        "CUDA GPU is required."
    )

    torch.cuda.set_device(0)

    torch.backends.cudnn.benchmark = True

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Input:", INPUT_SHAPE)
    print("Warmup:", WARMUP)
    print("Runs:", RUNS)
    print("Repeats:", REPEATS)
    print("Precision: FP32 (no autocast)")
    print()

    # All models use exactly the same input tensor.
    torch.manual_seed(123)

    x = torch.randn(
        INPUT_SHAPE,
        device=DEVICE,
        dtype=torch.float32,
    )

    results = {}

    for label, model_type, weight_path in MODELS:

        print("=" * 55)
        print("Benchmark:", label)
        print("Model:", model_type)
        print("Weights:", weight_path)

        model = load_model(
            model_type,
            weight_path,
        )

        counts = verify_inference_path(
            model,
            x,
        )

        print("Forward-hook counts:", counts)

        latency, repetitions = benchmark(
            model,
            x,
        )

        fps = 1000.0 / latency

        results[label] = latency

        print(
            "Repeated latency (ms):",
            [round(v, 4) for v in repetitions],
        )

        print(
            f"{label}: "
            f"{latency:.4f} ms, "
            f"{fps:.2f} FPS"
        )

        del model
        torch.cuda.empty_cache()

    print()
    print("=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    print(
        f"{'Model':<12}"
        f"{'Latency (ms)':>16}"
        f"{'FPS':>14}"
        f"{'vs A':>14}"
    )

    baseline = results["A"]

    for label, _, _ in MODELS:

        latency = results[label]

        fps = 1000.0 / latency

        increase = (
            (latency / baseline - 1.0)
            * 100.0
        )

        print(
            f"{label:<12}"
            f"{latency:>16.4f}"
            f"{fps:>14.2f}"
            f"{increase:>+13.2f}%"
        )

    print()
    print(
        "D-BGA vs C:",
        f"{(results['D-BGA'] / results['C'] - 1) * 100:+.2f}%"
    )

    print(
        "D-BGA vs D:",
        f"{(results['D-BGA'] / results['D'] - 1) * 100:+.2f}%"
    )


if __name__ == "__main__":
    main()
