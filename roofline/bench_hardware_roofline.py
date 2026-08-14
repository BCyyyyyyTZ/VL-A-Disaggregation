#!/usr/bin/env python3
"""Measure a single-GPU PyTorch hardware roofline baseline.

The compute roof uses dense BF16/FP16 tensor-core GEMM throughput.  The memory
roof uses device-to-device copy bandwidth and counts one read plus one write.
The output JSON is consumed by plot_roofline.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(ordered)),
        "p50": float(ordered[len(ordered) // 2]),
        "max": float(max(ordered)),
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _event_ms(torch: Any, fn, *, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def measure_gemm(torch: Any, *, dtype: Any, sizes: list[int], warmup: int, repeats: int) -> list[dict[str, Any]]:
    results = []
    for size in sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        out = torch.empty((size, size), device="cuda", dtype=dtype)

        def run_once() -> None:
            torch.mm(a, b, out=out)

        samples_ms = _event_ms(torch, run_once, warmup=warmup, repeats=repeats)
        flops = 2.0 * size * size * size
        tflops_samples = [flops / (ms / 1000.0) / 1.0e12 for ms in samples_ms]
        results.append(
            {
                "size": size,
                "flops": flops,
                "samples_ms": samples_ms,
                "tflops": _summary(tflops_samples),
            }
        )
        del a, b, out
        torch.cuda.empty_cache()
    return results


def measure_copy(torch: Any, *, sizes_mib: list[int], warmup: int, repeats: int) -> list[dict[str, Any]]:
    results = []
    dtype = torch.float32
    element_size = torch.empty((), dtype=dtype).element_size()
    for size_mib in sizes_mib:
        size_bytes = int(size_mib) * 1024 * 1024
        numel = size_bytes // element_size
        src = torch.empty((numel,), device="cuda", dtype=dtype)
        dst = torch.empty_like(src)

        def run_once() -> None:
            dst.copy_(src)

        samples_ms = _event_ms(torch, run_once, warmup=warmup, repeats=repeats)
        bytes_moved = 2.0 * numel * element_size
        tbps_samples = [bytes_moved / (ms / 1000.0) / 1.0e12 for ms in samples_ms]
        results.append(
            {
                "size_mib": size_mib,
                "bytes_moved": bytes_moved,
                "samples_ms": samples_ms,
                "tbps": _summary(tbps_samples),
            }
        )
        del src, dst
        torch.cuda.empty_cache()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="roofline/logs/hardware_roofline.json")
    parser.add_argument("--gpu", default="7", help="CUDA_VISIBLE_DEVICES value for this run")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--gemm-sizes", default="2048,4096,8192")
    parser.add_argument("--copy-sizes-mib", default="256,512,1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    dtype = _torch_dtype(torch, args.dtype)
    started = time.perf_counter()
    gemm = measure_gemm(
        torch,
        dtype=dtype,
        sizes=_parse_csv_ints(args.gemm_sizes),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    copy = measure_copy(
        torch,
        sizes_mib=_parse_csv_ints(args.copy_sizes_mib),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    peak_tflops = max(item["tflops"]["max"] for item in gemm)
    peak_tbps = max(item["tbps"]["max"] for item in copy)

    payload = {
        "status": "pass",
        "kind": "pytorch_hardware_roofline",
        "gpu": torch.cuda.get_device_name(0),
        "visible_device": str(args.gpu),
        "dtype": args.dtype,
        "elapsed_s": time.perf_counter() - started,
        "peak_tflops": peak_tflops,
        "peak_bandwidth_tbps": peak_tbps,
        "ridge_point_flop_per_byte": peak_tflops / peak_tbps,
        "gemm": gemm,
        "copy": copy,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
