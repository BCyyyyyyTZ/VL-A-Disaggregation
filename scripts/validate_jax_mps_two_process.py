#!/usr/bin/env python3
"""Validate JAX/XLA two-process scheduling behavior under CUDA MPS.

This is a synthetic benchmark for the VA-split hypothesis:

* a "long" JAX process repeatedly runs a larger compiled matmul graph;
* a "short" JAX process measures a smaller compiled graph;
* the parent compares short-graph latency alone vs while the long process is
  continuously submitting work through a separate XLA client.

The benchmark intentionally does not load OpenPI.  It isolates the scheduler
question from model restore, IPC, prefix slabs, and request batching.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


@dataclass(frozen=True)
class Scenario:
    name: str
    short_sm: int | None
    long_sm: int | None


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    def percentile(pct: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * pct
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    return {
        "mean_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(percentile(0.50)),
        "p90_ms": float(percentile(0.90)),
        "p95_ms": float(percentile(0.95)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _parse_scenarios(value: str) -> list[Scenario]:
    scenarios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, spec = item.split(":", 1)
        short_s, long_s = spec.split("/", 1)
        short_sm = None if short_s == "none" else int(short_s)
        long_sm = None if long_s == "none" else int(long_s)
        if short_sm is None and long_sm is None:
            raise ValueError(f"Scenario must enable short or long worker: {item!r}")
        scenarios.append(Scenario(name=name, short_sm=short_sm, long_sm=long_sm))
    return scenarios


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", default="logs/tests/jax_mps_two_process")
    parser.add_argument("--mps-pipe-dir", default=None)
    parser.add_argument("--mps-log-dir", default=None)
    parser.add_argument("--short-size", type=int, default=1536)
    parser.add_argument("--long-size", type=int, default=4096)
    parser.add_argument("--short-iters", type=int, default=1)
    parser.add_argument("--long-iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=40)
    parser.add_argument("--long-duration-s", type=float, default=12.0)
    parser.add_argument(
        "--scenarios",
        default="solo100:100/none,concurrent100:100/100,concurrent80:80/80,short20_long80:20/80",
        help="Comma-separated name:short_sm/long_sm entries; long_sm can be 'none'.",
    )
    parser.add_argument("--worker", choices=["short", "long"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve_gpu_uuid(gpu_id: int) -> str:
    completed = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    uuid = completed.stdout.strip().splitlines()[0].strip()
    if not uuid:
        raise RuntimeError(f"Failed to resolve GPU UUID for GPU {gpu_id}")
    return uuid


def _start_mps(cuda_visible_device: str, pipe_dir: Path, log_dir: Path) -> None:
    pipe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_device,
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
        }
    )
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def _stop_mps(pipe_dir: Path, log_dir: Path) -> None:
    env = os.environ.copy()
    env.update({"CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir), "CUDA_MPS_LOG_DIRECTORY": str(log_dir)})
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env, check=False, capture_output=True)


def _worker_env(base_env: dict[str, str], *, gpu_uuid: str, pipe_dir: Path, log_dir: Path, sm: int) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(sm)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-jax-mps-two-process")
    return env


def _make_fn(size: int, iters: int):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def fn(x):
        y = x
        for _ in range(iters):
            y = jnp.tanh((y @ x) / float(size))
        return y

    return fn


def _block(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _run_short_worker(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    fn = _make_fn(args.short_size, args.short_iters)
    x = jnp.ones((args.short_size, args.short_size), dtype=jnp.float32)
    _block(x)
    for _ in range(args.warmup):
        _block(fn(x))
    samples = []
    for _ in range(args.measure):
        start = time.perf_counter()
        _block(fn(x))
        samples.append((time.perf_counter() - start) * 1000.0)
    payload = {
        "role": "short",
        "samples_ms": samples,
        "summary": _summary(samples),
        "devices": [str(device) for device in jax.devices()],
    }
    Path(args.result_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_long_worker(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    fn = _make_fn(args.long_size, args.long_iters)
    x = jnp.ones((args.long_size, args.long_size), dtype=jnp.float32)
    _block(x)
    for _ in range(args.warmup):
        _block(fn(x))
    ready_path = Path(args.result_path).with_suffix(".ready")
    ready_path.write_text("ready\n", encoding="utf-8")
    samples = []
    deadline = time.perf_counter() + float(args.long_duration_s)
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        _block(fn(x))
        samples.append((time.perf_counter() - start) * 1000.0)
    payload = {
        "role": "long",
        "samples_ms": samples,
        "summary": _summary(samples),
        "devices": [str(device) for device in jax.devices()],
    }
    Path(args.result_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> None:
    if args.worker == "short":
        _run_short_worker(args)
    elif args.worker == "long":
        _run_long_worker(args)
    else:
        raise ValueError("--worker must be set for worker mode")


def _run_scenario(args: argparse.Namespace, scenario: Scenario, *, gpu_uuid: str, pipe_dir: Path, log_dir: Path) -> dict:
    out_dir = Path(args.output_dir)
    scenario_dir = out_dir / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[1] / "src"),
            str(Path(__file__).resolve().parents[1] / "packages" / "openpi-client" / "src"),
            base_env.get("PYTHONPATH", ""),
        ]
    )

    long_proc = None
    long_result = scenario_dir / "long.json"
    if scenario.long_sm is not None:
        long_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "long",
            "--result-path",
            str(long_result),
            "--long-size",
            str(args.long_size),
            "--long-iters",
            str(args.long_iters),
            "--warmup",
            str(args.warmup),
            "--long-duration-s",
            str(args.long_duration_s),
        ]
        long_proc = subprocess.Popen(
            long_cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=_worker_env(base_env, gpu_uuid=gpu_uuid, pipe_dir=pipe_dir, log_dir=log_dir, sm=scenario.long_sm),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ready_path = long_result.with_suffix(".ready")
        deadline = time.perf_counter() + 120.0
        while not ready_path.exists():
            if long_proc.poll() is not None:
                stdout, stderr = long_proc.communicate(timeout=1)
                raise RuntimeError(f"Long worker exited before ready: rc={long_proc.returncode}\n{stdout}\n{stderr}")
            if time.perf_counter() > deadline:
                long_proc.terminate()
                raise TimeoutError("Timed out waiting for long worker warmup")
            time.sleep(0.1)

    completed_returncode = None
    short_payload = None
    if scenario.short_sm is not None:
        short_result = scenario_dir / "short.json"
        short_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "short",
            "--result-path",
            str(short_result),
            "--short-size",
            str(args.short_size),
            "--short-iters",
            str(args.short_iters),
            "--warmup",
            str(args.warmup),
            "--measure",
            str(args.measure),
        ]
        completed = subprocess.run(
            short_cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=_worker_env(base_env, gpu_uuid=gpu_uuid, pipe_dir=pipe_dir, log_dir=log_dir, sm=scenario.short_sm),
            text=True,
            capture_output=True,
            check=False,
        )
        completed_returncode = completed.returncode
        (scenario_dir / "short.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (scenario_dir / "short.stderr.log").write_text(completed.stderr, encoding="utf-8")
        short_payload = json.loads(short_result.read_text(encoding="utf-8")) if short_result.exists() else None

    long_payload = None
    if long_proc is not None:
        try:
            stdout, stderr = long_proc.communicate(timeout=max(5.0, args.long_duration_s + 10.0))
        except subprocess.TimeoutExpired:
            long_proc.terminate()
            stdout, stderr = long_proc.communicate(timeout=10)
        (scenario_dir / "long.stdout.log").write_text(stdout, encoding="utf-8")
        (scenario_dir / "long.stderr.log").write_text(stderr, encoding="utf-8")
        if long_result.exists():
            long_payload = json.loads(long_result.read_text(encoding="utf-8"))

    return {
        "name": scenario.name,
        "short_sm": scenario.short_sm,
        "long_sm": scenario.long_sm,
        "short_returncode": completed_returncode,
        "short": short_payload,
        "long": long_payload,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        _run_worker(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_uuid = _resolve_gpu_uuid(args.gpu_id)
    pipe_dir = Path(args.mps_pipe_dir) if args.mps_pipe_dir else Path(tempfile.mkdtemp(prefix="jax-mps-two-process-pipe-", dir="/tmp"))
    log_dir = Path(args.mps_log_dir) if args.mps_log_dir else output_dir / "mps-log"

    records = []
    started = time.perf_counter()
    mps_started = False
    try:
        _start_mps(gpu_uuid, pipe_dir, log_dir)
        mps_started = True
        for scenario in _parse_scenarios(args.scenarios):
            records.append(_run_scenario(args, scenario, gpu_uuid=gpu_uuid, pipe_dir=pipe_dir, log_dir=log_dir))
            (output_dir / "summary.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    finally:
        if mps_started:
            _stop_mps(pipe_dir, log_dir)

    solo = next((r for r in records if r["long_sm"] is None and r.get("short")), None)
    if solo:
        solo_mean = float(solo["short"]["summary"]["mean_ms"])
        for record in records:
            if record.get("short"):
                mean = float(record["short"]["summary"]["mean_ms"])
                record["short_slowdown_vs_solo_mean"] = mean / solo_mean if solo_mean else None

    payload = {
        "args": vars(args),
        "gpu_uuid": gpu_uuid,
        "elapsed_s": time.perf_counter() - started,
        "records": records,
    }
    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
