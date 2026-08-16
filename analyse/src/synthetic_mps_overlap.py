#!/usr/bin/env python3
"""Synthetic JAX/PyTorch MPS overlap comparison.

This does not model OpenPI numerically.  It is a scheduling control: a long
GEMM loop and a short GEMM loop are run solo and concurrently for JAX or
PyTorch.  Compare short-work slowdown across frameworks to see whether
JAX/XLA two-client MPS scheduling is less friendly than PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * p
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        w = rank - lo
        return ordered[lo] * (1 - w) + ordered[hi] * w

    return {"avg_ms": float(statistics.fmean(ordered)), "p50_ms": float(pct(0.5)), "p95_ms": float(pct(0.95))}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jax_worker(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    size = args.long_size if args.role == "long" else args.short_size
    inner = args.long_inner if args.role == "long" else args.short_inner
    a = jnp.ones((size, size), dtype=jnp.bfloat16)
    b = jnp.ones((size, size), dtype=jnp.bfloat16)

    @jax.jit
    def run_once(x):
        y = x
        for _ in range(inner):
            y = y @ b
        return y

    y = run_once(a)
    y.block_until_ready()
    for _ in range(args.warmup):
        y = run_once(a)
        y.block_until_ready()
    samples = []
    started = time.perf_counter()
    iters = 0
    while iters < args.repeats or (args.duration_s > 0 and time.perf_counter() - started < args.duration_s):
        t0 = time.perf_counter()
        y = run_once(a)
        y.block_until_ready()
        samples.append((time.perf_counter() - t0) * 1000)
        iters += 1
        if args.duration_s <= 0 and iters >= args.repeats:
            break
    _write(Path(args.result_path), {"status": "pass", "framework": "jax", "role": args.role, "samples_ms": samples, "latency": _summary(samples), "iterations": iters})


def _torch_worker(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_device(0)
    size = args.long_size if args.role == "long" else args.short_size
    inner = args.long_inner if args.role == "long" else args.short_inner
    a = torch.ones((size, size), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((size, size), device="cuda", dtype=torch.bfloat16)

    def run_once():
        y = a
        for _ in range(inner):
            y = y @ b
        return y

    for _ in range(args.warmup):
        y = run_once()
        torch.cuda.synchronize()
    samples = []
    started = time.perf_counter()
    iters = 0
    while iters < args.repeats or (args.duration_s > 0 and time.perf_counter() - started < args.duration_s):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = run_once()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
        iters += 1
        if args.duration_s <= 0 and iters >= args.repeats:
            break
    _write(Path(args.result_path), {"status": "pass", "framework": "torch", "role": args.role, "samples_ms": samples, "latency": _summary(samples), "iterations": iters})


def _worker(args: argparse.Namespace) -> None:
    try:
        if args.framework == "jax":
            _jax_worker(args)
        elif args.framework == "torch":
            _torch_worker(args)
        else:
            raise ValueError(args.framework)
    except Exception as exc:
        _write(Path(args.result_path), {"status": "failed", "framework": args.framework, "role": args.role, "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"})
        raise


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("JAXTYPING_DISABLE", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def _run_pair(args: argparse.Namespace, *, framework: str, mode: str, output_dir: Path) -> dict[str, Any]:
    case = f"{framework}_{mode}"
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    if mode == "solo-long":
        roles = ["long"]
    elif mode == "solo-short":
        roles = ["short"]
    else:
        roles = ["long", "short"]
    procs = []
    for role in roles:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--framework",
            framework,
            "--role",
            role,
            "--gpu",
            args.gpu,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--duration-s",
            str(args.duration_s if mode == "concurrent" else 0.0),
            "--long-size",
            str(args.long_size),
            "--short-size",
            str(args.short_size),
            "--long-inner",
            str(args.long_inner),
            "--short-inner",
            str(args.short_inner),
            "--result-path",
            str(case_dir / f"{role}.json"),
        ]
        procs.append((role, subprocess.Popen(cmd, cwd=REPO_ROOT, env=_base_env(args), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)))
    outputs = {}
    status = "pass"
    for role, proc in procs:
        out, err = proc.communicate(timeout=args.timeout_s)
        (case_dir / f"{role}.stdout.log").write_text(out or "", encoding="utf-8")
        (case_dir / f"{role}.stderr.log").write_text(err or "", encoding="utf-8")
        result = json.loads((case_dir / f"{role}.json").read_text(encoding="utf-8"))
        outputs[role] = result
        if proc.returncode != 0 or result.get("status") != "pass":
            status = "failed"
    payload = {"status": status, "framework": framework, "mode": mode, "results": outputs}
    _write(case_dir / "result.json", payload)
    return payload


def _summarize(output_dir: Path) -> dict[str, Any]:
    rows = []
    for framework in ("jax", "torch"):
        solo_long = json.loads((output_dir / f"{framework}_solo-long" / "result.json").read_text(encoding="utf-8"))
        solo_short = json.loads((output_dir / f"{framework}_solo-short" / "result.json").read_text(encoding="utf-8"))
        conc = json.loads((output_dir / f"{framework}_concurrent" / "result.json").read_text(encoding="utf-8"))
        long_solo = solo_long["results"]["long"]["latency"]["avg_ms"]
        short_solo = solo_short["results"]["short"]["latency"]["avg_ms"]
        long_conc = conc["results"]["long"]["latency"]["avg_ms"]
        short_conc = conc["results"]["short"]["latency"]["avg_ms"]
        rows.append(
            {
                "framework": framework,
                "long_solo_ms": long_solo,
                "short_solo_ms": short_solo,
                "long_concurrent_ms": long_conc,
                "short_concurrent_ms": short_conc,
                "long_slowdown": long_conc / long_solo if long_solo else 0,
                "short_slowdown": short_conc / short_solo if short_solo else 0,
            }
        )
    summary = {"rows": rows}
    _write(output_dir / "synthetic_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analyse/summary/synthetic_mps")
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--framework", choices=("jax", "torch"), default="jax")
    parser.add_argument("--role", choices=("long", "short"), default="long")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--long-size", type=int, default=8192)
    parser.add_argument("--short-size", type=int, default=2048)
    parser.add_argument("--long-inner", type=int, default=2)
    parser.add_argument("--short-inner", type=int, default=1)
    parser.add_argument("--result-path", default="")
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker(args)
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for framework in ("jax", "torch"):
        for mode in ("solo-long", "solo-short", "concurrent"):
            records.append(_run_pair(args, framework=framework, mode=mode, output_dir=output_dir))
    summary = _summarize(output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
