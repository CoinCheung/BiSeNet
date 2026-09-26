#!/usr/bin/env python3

import sys
sys.path.insert(0, ".")

import time

import torch

from lib.models import model_factory


torch.backends.cudnn.benchmark = True

WARMUP = 100
RUNS = 500
INPUT_H = 576
INPUT_W = 704


def benchmark(model_name):
    torch.cuda.empty_cache()

    model = model_factory[
        model_name
    ](3).cuda()

    model.aux_mode = "eval"
    model.eval()

    x = torch.randn(
        1,
        3,
        INPUT_H,
        INPUT_W,
        device="cuda",
    )

    # ----------------------------
    # Warmup
    # ----------------------------
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)

    torch.cuda.synchronize()

    # ----------------------------
    # Benchmark
    # ----------------------------
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(RUNS):
            model(x)

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    latency_ms = (
        elapsed
        / RUNS
        * 1000.0
    )

    fps = (
        1000.0
        / latency_ms
    )

    print(
        f"{model_name:30s}"
        f" latency={latency_ms:8.3f} ms"
        f" FPS={fps:8.2f}"
    )

    del model
    del x

    torch.cuda.empty_cache()

    return latency_ms


print(
    f"Input: 1x3x{INPUT_H}x{INPUT_W}"
)

print(
    f"Warmup={WARMUP}, Runs={RUNS}"
)

print()

baseline_latency = benchmark(
    "bisenetv2"
)

refine_latency = benchmark(
    "bisenetv2_boundary_refine"
)

d_latency = benchmark(
    "bisenetv2_boundary_full"
)

extra_latency = (
    refine_latency
    - baseline_latency
)

increase = (
    (
        refine_latency
        / baseline_latency
    )
    - 1.0
) * 100.0

print()
print(
    f"Extra latency   : "
    f"{extra_latency:.3f} ms"
)

print(
    f"Latency increase: "
    f"{increase:.2f}%"
)
