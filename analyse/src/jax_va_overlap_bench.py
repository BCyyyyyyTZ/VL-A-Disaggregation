#!/usr/bin/env python3
"""Controlled OpenPI0.5 JAX VLM/AE overlap experiments.

This benchmark is intentionally lower level than full serving.  It measures
compiled JAX VLM and AE helper calls in:

* solo-vlm
* solo-ae
* two-process concurrent VLM + AE under MPS
* single-process serial control
* single-process threaded control

Each worker emits JAX device memory profile snapshots through
``jax.profiler.save_device_memory_profile`` when available.  These snapshots are
the requested SnapshotAPI-style memory artifacts; no nvidia-smi memory numbers
are used for analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "/home/miliang/model/openpi-assets/checkpoints/pi05_libero"
DEFAULT_OUTPUT = REPO_ROOT / "analyse" / "summary" / "jax_va_overlap"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "analyse" / "summary" / "memory_snapshots"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * p
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        w = rank - lo
        return ordered[lo] * (1 - w) + ordered[hi] * w

    return {
        "avg_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(pct(0.50)),
        "p95_ms": float(pct(0.95)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


class TraceRange:
    def __init__(self, name: str):
        self.name = name
        self._jax_ctx = None
        self._nvtx_ctx = None

    def __enter__(self):
        try:
            import jax

            self._jax_ctx = jax.profiler.TraceAnnotation(self.name)
            self._jax_ctx.__enter__()
        except Exception:
            self._jax_ctx = None
        try:
            import nvtx  # type: ignore

            self._nvtx_ctx = nvtx.annotate(self.name)
            self._nvtx_ctx.__enter__()
        except Exception:
            self._nvtx_ctx = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._nvtx_ctx is not None:
            self._nvtx_ctx.__exit__(exc_type, exc, tb)
        if self._jax_ctx is not None:
            self._jax_ctx.__exit__(exc_type, exc, tb)
        return False


def _save_memory_snapshot(snapshot_dir: Path, label: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write a JAX device memory profile snapshot and sidecar metadata."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    profile_path = snapshot_dir / f"{safe_label}.prof"
    meta_path = snapshot_dir / f"{safe_label}.json"
    payload = {"label": label, "profile_path": str(profile_path), "metadata": metadata}
    try:
        import jax

        jax.profiler.save_device_memory_profile(str(profile_path))
        payload["status"] = "pass"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_and_compile(
    *,
    batch_size: int,
    ae_batch_size: int,
    num_steps: int,
    config_name: str,
    checkpoint_dir: str,
    compile_role: str,
):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from openpi.models.jax_split_types import JaxDenoiseState
    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.training import config as _config

    train_config = _config.get_config(config_name)
    model = _load_jax_model(train_config, checkpoint_dir)
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model)

    vlm_observation = observation_factory(batch_size)
    ae_observation = observation_factory(ae_batch_size)
    ae_noise = noise_factory(ae_batch_size)
    dt = jnp.full((ae_batch_size,), -1.0 / float(num_steps), dtype=jnp.float32)
    denoise_state0 = JaxDenoiseState(
        x_t=ae_noise,
        step_idx=jnp.zeros((ae_batch_size,), dtype=jnp.int32),
        num_steps=num_steps,
        dt=dt,
    )
    _block_until_ready((vlm_observation, ae_observation, ae_noise, denoise_state0))

    graphdef, state = nnx.split(model)

    def vlm_fn(module_state, obs):
        module = nnx.merge(graphdef, module_state)
        return module.build_prefix_feature(None, obs)

    def ae_fn(module_state, prefix_batch, state_batch):
        module = nnx.merge(graphdef, module_state)
        return module.denoise_one_batch(prefix_batch, state_batch)

    compiled_vlm = None
    if compile_role in {"vlm", "both"}:
        compiled_vlm = jax.jit(vlm_fn).lower(state, vlm_observation).compile()

    ae_prefix = None
    compiled_ae = None
    if compile_role in {"ae", "both"}:
        prefix_vlm = jax.jit(vlm_fn).lower(state, ae_observation).compile()
        ae_prefix = prefix_vlm(state, ae_observation)
        _block_until_ready(ae_prefix)
        compiled_ae = jax.jit(ae_fn).lower(state, ae_prefix, denoise_state0).compile()
    return {
        "jax": jax,
        "jnp": jnp,
        "JaxDenoiseState": JaxDenoiseState,
        "state": state,
        "compiled_vlm": compiled_vlm,
        "compiled_ae": compiled_ae,
        "vlm_observation": vlm_observation,
        "ae_prefix": ae_prefix,
        "ae_noise": ae_noise,
        "num_steps": num_steps,
        "ae_batch_size": ae_batch_size,
        "model_meta": {
            "action_horizon": int(model.action_horizon),
            "action_dim": int(model.action_dim),
            "max_token_len": int(model.max_token_len),
            "devices": [str(device) for device in jax.devices()],
        },
    }


def _run_vlm_once(ctx: dict[str, Any]) -> Any:
    if ctx["compiled_vlm"] is None:
        raise RuntimeError("VLM executable was not compiled for this worker")
    with TraceRange("VLM_COMPILED_CALL"):
        prefix = ctx["compiled_vlm"](ctx["state"], ctx["vlm_observation"])
        _block_until_ready(prefix)
    return prefix


def _run_ae_once(ctx: dict[str, Any]) -> Any:
    if ctx["compiled_ae"] is None or ctx["ae_prefix"] is None:
        raise RuntimeError("AE executable/prefix was not compiled for this worker")
    jnp = ctx["jnp"]
    JaxDenoiseState = ctx["JaxDenoiseState"]
    x_t = ctx["ae_noise"]
    batch_size = int(ctx["ae_batch_size"])
    steps = int(ctx["num_steps"])
    dt = jnp.full((batch_size,), -1.0 / float(steps), dtype=jnp.float32)
    with TraceRange("AE_5_STEPS" if steps == 5 else f"AE_{steps}_STEPS"):
        for step in range(steps):
            with TraceRange(f"AE_STEP_{step}"):
                step_state = JaxDenoiseState(
                    x_t=x_t,
                    step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
                    num_steps=steps,
                    dt=dt,
                )
                v_t = ctx["compiled_ae"](ctx["state"], ctx["ae_prefix"], step_state)
                x_t = x_t + dt.reshape((-1,) + (1,) * (v_t.ndim - 1)) * v_t
        _block_until_ready(x_t)
    return x_t


def _timed_loop(fn, *, repeats: int, duration_s: float | None, trace_prefix: str) -> dict[str, Any]:
    samples: list[float] = []
    started = time.perf_counter()
    iters = 0
    while True:
        if duration_s is None:
            if iters >= repeats:
                break
        elif iters > 0 and (time.perf_counter() - started) >= duration_s:
            break
        with TraceRange(f"{trace_prefix}_ITER"):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        iters += 1
    elapsed = time.perf_counter() - started
    return {
        "samples_ms": samples,
        "latency": _summary(samples),
        "iterations": iters,
        "elapsed_s": elapsed,
        "throughput_iter_per_s": float(iters / elapsed) if elapsed > 0 else 0.0,
    }


def _worker_main(args: argparse.Namespace) -> None:
    os.environ.setdefault("JAXTYPING_DISABLE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.perf_counter()
    result_path = Path(args.result_path)
    snapshot_dir = Path(args.snapshot_dir)
    try:
        ctx = _load_and_compile(
            batch_size=args.batch_size,
            ae_batch_size=args.ae_batch_size,
            num_steps=args.num_steps,
            config_name=args.config,
            checkpoint_dir=args.checkpoint_dir,
            compile_role=args.role,
        )
        snapshots = [
            _save_memory_snapshot(
                snapshot_dir,
                f"{args.case_id}_{args.role}_after_compile",
                {"case_id": args.case_id, "role": args.role, "batch_size": args.batch_size, "ae_batch_size": args.ae_batch_size},
            )
        ]
        for _ in range(args.warmup):
            if args.role == "vlm":
                _run_vlm_once(ctx)
            elif args.role == "ae":
                _run_ae_once(ctx)
            else:
                raise ValueError(f"Unsupported worker role: {args.role}")
        snapshots.append(
            _save_memory_snapshot(
                snapshot_dir,
                f"{args.case_id}_{args.role}_after_warmup",
                {"case_id": args.case_id, "role": args.role, "batch_size": args.batch_size, "ae_batch_size": args.ae_batch_size},
            )
        )

        ready_path = Path(args.ready_path)
        start_path = Path(args.start_path)
        if args.ready_path:
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text("ready\n", encoding="utf-8")
        if args.start_path:
            while not start_path.exists():
                time.sleep(0.01)

        if args.role == "vlm":
            measured = _timed_loop(
                lambda: _run_vlm_once(ctx),
                repeats=args.repeats,
                duration_s=args.duration_s if args.duration_s > 0 else None,
                trace_prefix="VLM",
            )
        else:
            measured = _timed_loop(
                lambda: _run_ae_once(ctx),
                repeats=args.repeats,
                duration_s=args.duration_s if args.duration_s > 0 else None,
                trace_prefix="AE",
            )
        snapshots.append(
            _save_memory_snapshot(
                snapshot_dir,
                f"{args.case_id}_{args.role}_after_measure",
                {"case_id": args.case_id, "role": args.role, "batch_size": args.batch_size, "ae_batch_size": args.ae_batch_size},
            )
        )
        _write_json(
            result_path,
            {
                "status": "pass",
                "case_id": args.case_id,
                "role": args.role,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "ae_batch_size": args.ae_batch_size,
                "num_steps": args.num_steps,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "duration_s": args.duration_s,
                "elapsed_s": time.perf_counter() - started,
                "measured": measured,
                "snapshots": snapshots,
                "model": ctx["model_meta"],
            },
        )
    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "failed",
                "case_id": args.case_id,
                "role": args.role,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "ae_batch_size": args.ae_batch_size,
                "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise


def _run_solo(args: argparse.Namespace, *, role: str, batch_size: int, ae_batch_size: int, output_dir: Path) -> dict[str, Any]:
    case_id = f"solo_{role}_b{batch_size}_ae{ae_batch_size}"
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    env = _base_worker_env(args)
    cmd = _worker_cmd(
        args,
        case_id=case_id,
        role=role,
        mode=f"solo-{role}",
        batch_size=batch_size,
        ae_batch_size=ae_batch_size,
        result_path=case_dir / "result.json",
        snapshot_dir=Path(args.snapshot_dir),
    )
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    payload = json.loads((case_dir / "result.json").read_text(encoding="utf-8")) if (case_dir / "result.json").exists() else {}
    return {"case_id": case_id, "status": payload.get("status", "failed"), "returncode": completed.returncode, "result_path": str(case_dir / "result.json")}


def _run_concurrent(args: argparse.Namespace, *, batch_size: int, ae_batch_size: int, output_dir: Path) -> dict[str, Any]:
    case_id = f"concurrent_b{batch_size}_ae{ae_batch_size}"
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    ready_vlm = case_dir / "vlm.ready"
    ready_ae = case_dir / "ae.ready"
    start = case_dir / "start"
    for path in (ready_vlm, ready_ae, start):
        if path.exists():
            path.unlink()
    env = _base_worker_env(args)
    vlm_cmd = _worker_cmd(
        args,
        case_id=case_id,
        role="vlm",
        mode="concurrent",
        batch_size=batch_size,
        ae_batch_size=ae_batch_size,
        result_path=case_dir / "vlm_result.json",
        snapshot_dir=Path(args.snapshot_dir),
        ready_path=ready_vlm,
        start_path=start,
        duration_s=args.concurrent_duration_s,
    )
    ae_cmd = _worker_cmd(
        args,
        case_id=case_id,
        role="ae",
        mode="concurrent",
        batch_size=batch_size,
        ae_batch_size=ae_batch_size,
        result_path=case_dir / "ae_result.json",
        snapshot_dir=Path(args.snapshot_dir),
        ready_path=ready_ae,
        start_path=start,
        duration_s=args.concurrent_duration_s,
    )
    vlm_proc = subprocess.Popen(vlm_cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ae_proc = subprocess.Popen(ae_cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + args.case_timeout_s
    while time.time() < deadline:
        if ready_vlm.exists() and ready_ae.exists():
            break
        if vlm_proc.poll() is not None or ae_proc.poll() is not None:
            break
        time.sleep(0.05)
    start.write_text("start\n", encoding="utf-8")
    vlm_out, vlm_err = vlm_proc.communicate(timeout=max(args.case_timeout_s, 1))
    ae_out, ae_err = ae_proc.communicate(timeout=max(args.case_timeout_s, 1))
    (case_dir / "vlm_stdout.log").write_text(vlm_out or "", encoding="utf-8")
    (case_dir / "vlm_stderr.log").write_text(vlm_err or "", encoding="utf-8")
    (case_dir / "ae_stdout.log").write_text(ae_out or "", encoding="utf-8")
    (case_dir / "ae_stderr.log").write_text(ae_err or "", encoding="utf-8")
    vlm_result = _read_json_or_empty(case_dir / "vlm_result.json")
    ae_result = _read_json_or_empty(case_dir / "ae_result.json")
    status = "pass" if vlm_proc.returncode == 0 and ae_proc.returncode == 0 and vlm_result.get("status") == "pass" and ae_result.get("status") == "pass" else "failed"
    combined = {
        "status": status,
        "case_id": case_id,
        "mode": "concurrent",
        "batch_size": batch_size,
        "ae_batch_size": ae_batch_size,
        "vlm": vlm_result,
        "ae": ae_result,
        "returncodes": {"vlm": vlm_proc.returncode, "ae": ae_proc.returncode},
    }
    _write_json(case_dir / "result.json", combined)
    return {"case_id": case_id, "status": status, "returncode": 0 if status == "pass" else 1, "result_path": str(case_dir / "result.json")}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_single_process(args: argparse.Namespace, *, mode: str, batch_size: int, ae_batch_size: int, output_dir: Path) -> dict[str, Any]:
    case_id = f"{mode}_b{batch_size}_ae{ae_batch_size}"
    case_dir = output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    env = _base_worker_env(args)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-worker",
        "--mode",
        mode,
        "--case-id",
        case_id,
        "--batch-size",
        str(batch_size),
        "--ae-batch-size",
        str(ae_batch_size),
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--num-steps",
        str(args.num_steps),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--concurrent-duration-s",
        str(args.concurrent_duration_s),
        "--result-path",
        str(case_dir / "result.json"),
        "--snapshot-dir",
        args.snapshot_dir,
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    payload = _read_json_or_empty(case_dir / "result.json")
    return {"case_id": case_id, "status": payload.get("status", "failed"), "returncode": completed.returncode, "result_path": str(case_dir / "result.json")}


def _single_worker_main(args: argparse.Namespace) -> None:
    os.environ.setdefault("JAXTYPING_DISABLE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.perf_counter()
    try:
        ctx = _load_and_compile(
            batch_size=args.batch_size,
            ae_batch_size=args.ae_batch_size,
            num_steps=args.num_steps,
            config_name=args.config,
            checkpoint_dir=args.checkpoint_dir,
            compile_role="both",
        )
        snapshots = [
            _save_memory_snapshot(Path(args.snapshot_dir), f"{args.case_id}_after_compile", {"case_id": args.case_id, "mode": args.mode})
        ]
        for _ in range(args.warmup):
            _run_vlm_once(ctx)
            _run_ae_once(ctx)
        snapshots.append(
            _save_memory_snapshot(Path(args.snapshot_dir), f"{args.case_id}_after_warmup", {"case_id": args.case_id, "mode": args.mode})
        )
        if args.mode == "single-serial":
            samples = []
            for _ in range(args.repeats):
                with TraceRange("SERIAL_ITER"):
                    t0 = time.perf_counter()
                    _run_vlm_once(ctx)
                    _run_ae_once(ctx)
                    samples.append((time.perf_counter() - t0) * 1000.0)
            measured = {"combined": {"samples_ms": samples, "latency": _summary(samples), "iterations": len(samples)}}
        elif args.mode == "single-threads":
            results: dict[str, Any] = {}

            def run_vlm_thread():
                results["vlm"] = _timed_loop(
                    lambda: _run_vlm_once(ctx),
                    repeats=args.repeats,
                    duration_s=args.concurrent_duration_s if args.concurrent_duration_s > 0 else None,
                    trace_prefix="THREAD_VLM",
                )

            def run_ae_thread():
                results["ae"] = _timed_loop(
                    lambda: _run_ae_once(ctx),
                    repeats=args.repeats,
                    duration_s=args.concurrent_duration_s if args.concurrent_duration_s > 0 else None,
                    trace_prefix="THREAD_AE",
                )

            with TraceRange("SINGLE_PROCESS_THREADS"):
                t0 = time.perf_counter()
                t1 = threading.Thread(target=run_vlm_thread, name="vlm-thread")
                t2 = threading.Thread(target=run_ae_thread, name="ae-thread")
                t1.start()
                t2.start()
                t1.join()
                t2.join()
                wall_s = time.perf_counter() - t0
            measured = {**results, "wall_s": wall_s}
        else:
            raise ValueError(f"Unsupported single-worker mode: {args.mode}")
        snapshots.append(
            _save_memory_snapshot(Path(args.snapshot_dir), f"{args.case_id}_after_measure", {"case_id": args.case_id, "mode": args.mode})
        )
        _write_json(
            Path(args.result_path),
            {
                "status": "pass",
                "case_id": args.case_id,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "ae_batch_size": args.ae_batch_size,
                "num_steps": args.num_steps,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "duration_s": args.concurrent_duration_s,
                "elapsed_s": time.perf_counter() - started,
                "measured": measured,
                "snapshots": snapshots,
                "model": ctx["model_meta"],
            },
        )
    except Exception as exc:
        _write_json(
            Path(args.result_path),
            {
                "status": "failed",
                "case_id": args.case_id,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "ae_batch_size": args.ae_batch_size,
                "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise


def _base_worker_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("JAXTYPING_DISABLE", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    pythonpath = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "openpi-client" / "src"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(item for item in pythonpath if item)
    return env


def _worker_cmd(
    args: argparse.Namespace,
    *,
    case_id: str,
    role: str,
    mode: str,
    batch_size: int,
    ae_batch_size: int,
    result_path: Path,
    snapshot_dir: Path,
    ready_path: Path | None = None,
    start_path: Path | None = None,
    duration_s: float | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--role",
        role,
        "--case-id",
        case_id,
        "--batch-size",
        str(batch_size),
        "--ae-batch-size",
        str(ae_batch_size),
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--num-steps",
        str(args.num_steps),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--duration-s",
        str(duration_s if duration_s is not None else 0.0),
        "--result-path",
        str(result_path),
        "--snapshot-dir",
        str(snapshot_dir),
        "--ready-path",
        str(ready_path or ""),
        "--start-path",
        str(start_path or ""),
    ]


def _run_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for batch_size in _parse_csv_ints(args.batch_sizes):
        ae_batch_size = int(args.ae_batch_size or batch_size)
        if args.experiment in {"all", "solo"}:
            records.append(_run_solo(args, role="vlm", batch_size=batch_size, ae_batch_size=ae_batch_size, output_dir=output_dir))
            records.append(_run_solo(args, role="ae", batch_size=batch_size, ae_batch_size=ae_batch_size, output_dir=output_dir))
        if args.experiment in {"all", "concurrent"}:
            records.append(_run_concurrent(args, batch_size=batch_size, ae_batch_size=ae_batch_size, output_dir=output_dir))
        if args.experiment in {"all", "single"}:
            records.append(_run_single_process(args, mode="single-serial", batch_size=batch_size, ae_batch_size=ae_batch_size, output_dir=output_dir))
            records.append(_run_single_process(args, mode="single-threads", batch_size=batch_size, ae_batch_size=ae_batch_size, output_dir=output_dir))
        _write_json(output_dir / "run_records.json", {"records": records})
        print(json.dumps(records[-1], indent=2), flush=True)
    _write_json(output_dir / "run_records.json", {"records": records})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ae-batch-size", type=int, default=0, help="0 means use --batch-size for each case")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--concurrent-duration-s", type=float, default=12.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument("--experiment", choices=("all", "solo", "concurrent", "single"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("vlm", "ae"), default="vlm", help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--ready-path", default="")
    parser.add_argument("--start-path", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.ae_batch_size == 0:
        args.ae_batch_size = args.batch_size
    if args.worker:
        _worker_main(args)
        return
    if args.single_worker:
        _single_worker_main(args)
        return
    _run_all(args)


if __name__ == "__main__":
    main()
