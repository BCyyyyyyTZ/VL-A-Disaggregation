#!/usr/bin/env python3
"""JAX OpenPI stage-level benchmark under single-GPU MPS SM quotas.

The parent process optionally starts one NVIDIA MPS control daemon for one
physical GPU, then runs one worker process per MPS percentage and batch size.
The worker loads a JAX OpenPI checkpoint once per case and reports wall-clock
latency for:

* ``vlm``: prefix/VLM cache construction via ``build_prefix_feature``.
* ``action_head``: denoising steps via ``denoise_one_batch``.

Unlike the PyTorch benchmark, JAX does not expose CUDA events here.  The worker
uses ``block_until_ready`` at stage boundaries, so metrics are synchronous device
wall-clock timings with XLA dispatch overhead included.
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
import time
import traceback
from typing import Any


DEFAULT_CONFIG = "pi05_libero"
DEFAULT_CHECKPOINT_DIR = "/mnt/tianze/models/pi05_libero"
DEFAULT_OUTPUT_DIR = "/mnt/tianze/VL-A-Disaggregation/logs/JAX/openpi_jax_stage_mps_profile"


@dataclass(frozen=True)
class StageProfileCase:
    case_id: str
    mps_sm: int
    batch_size: int
    gpu_id: int
    cuda_visible_device: str | None = None


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _validate_mps_percentage(value: int) -> int:
    if value < 1 or value > 100:
        raise ValueError(f"MPS SM percentage must be in [1, 100], got {value}")
    return value


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)

    def percentile(pct: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * pct
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "avg_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(percentile(0.50)),
        "p95_ms": float(percentile(0.95)),
    }


def _metric_avg(metrics: dict[str, Any] | None, key: str) -> float:
    if not metrics:
        return 0.0
    value = metrics.get(key) or {}
    return float(value.get("avg_ms", 0.0))


def build_stage_cases(args: argparse.Namespace) -> list[StageProfileCase]:
    cases: list[StageProfileCase] = []
    cuda_visible_device = getattr(args, "cuda_visible_device", None)
    for mps_sm in sorted(set(_parse_int_csv(args.mps_sm))):
        _validate_mps_percentage(mps_sm)
        for batch_size in sorted(set(_parse_int_csv(args.batch_sizes))):
            if batch_size <= 0:
                raise ValueError("--batch-sizes values must be positive")
            cases.append(
                StageProfileCase(
                    case_id=f"mps-sm{mps_sm}-bs{batch_size}",
                    mps_sm=mps_sm,
                    batch_size=batch_size,
                    gpu_id=int(args.gpu_id),
                    cuda_visible_device=cuda_visible_device,
                )
            )
    return cases


def build_worker_env(
    *,
    base_env: dict[str, str],
    case: StageProfileCase,
    mps_pipe_dir: str,
    mps_log_dir: str,
) -> dict[str, str]:
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = str(case.cuda_visible_device or case.gpu_id)
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(_validate_mps_percentage(case.mps_sm))
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_pipe_dir)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_log_dir)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-jax-stage-profile")
    return env


def write_summary(output_dir: Path, records: list[dict]) -> dict:
    counts = {
        "total": len(records),
        "pass": sum(1 for record in records if record.get("status") == "pass"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
    }
    summary = {"counts": counts, "cases": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# JAX OpenPI Stage MPS Profile Summary",
        "",
        f"- Total: {counts['total']}",
        f"- Pass: {counts['pass']}",
        f"- Failed: {counts['failed']}",
        "",
        "| case_id | status | mps_sm | batch_size | total_ms(avg) | "
        "vlm_ms(avg) | action_head_ms(avg) | other_ms(avg) | gpu_id |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record.get("metrics")
        lines.append(
            "| {case_id} | {status} | {mps_sm} | {batch_size} | {total:.3f} | "
            "{vlm:.3f} | {action_head:.3f} | {other:.3f} | {gpu_id} |".format(
                case_id=record.get("case_id", ""),
                status=record.get("status", ""),
                mps_sm=record.get("mps_sm", ""),
                batch_size=record.get("batch_size", ""),
                total=_metric_avg(metrics, "total"),
                vlm=_metric_avg(metrics, "vlm"),
                action_head=_metric_avg(metrics, "action_head"),
                other=_metric_avg(metrics, "other"),
                gpu_id=record.get("gpu_id", ""),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _load_existing_passed_case(output_dir: Path, case: StageProfileCase) -> dict | None:
    report_path = output_dir / "cases" / case.case_id / "case_report.json"
    if not report_path.exists():
        return None
    record = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "pass"
        or int(record.get("mps_sm", -1)) != case.mps_sm
        or int(record.get("batch_size", -1)) != case.batch_size
        or int(record.get("gpu_id", -1)) != case.gpu_id
    ):
        return None
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile JAX OpenPI VLM and action-head time under MPS",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="OpenPI training config name")
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="JAX OpenPI checkpoint directory containing params/",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for reports")
    parser.add_argument("--gpu-id", type=int, default=0, help="Physical GPU id to use")
    parser.add_argument(
        "--mps-sm",
        default="20,40,60,80,100",
        help="Comma-separated CUDA_MPS_ACTIVE_THREAD_PERCENTAGE values",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,8,16,24",
        help="Comma-separated synthetic observation batch sizes",
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Denoising steps per inference")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Untimed iterations after model load")
    parser.add_argument("--measure-steps", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=None,
        help="Optional timeout per case",
    )
    parser.add_argument(
        "--mps-pipe-dir",
        default=None,
        help="CUDA MPS pipe directory. Defaults to a run-specific /tmp path.",
    )
    parser.add_argument(
        "--mps-log-dir",
        default=None,
        help="CUDA MPS log directory. Defaults to a run-specific /tmp path.",
    )
    parser.add_argument(
        "--no-manage-mps",
        action="store_false",
        dest="manage_mps",
        help="Do not start/stop an MPS daemon; assume one is already running.",
    )
    parser.add_argument(
        "--no-jax-jit",
        action="store_true",
        help="Do not jit build_prefix_feature / denoise_one_batch.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing passing case reports and run only missing/failed cases.",
    )
    parser.set_defaults(manage_mps=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mps-sm-current", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _worker_result_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "cases" / case_id / "result.json"


def _write_worker_result(output_dir: Path, case_id: str, payload: dict) -> None:
    path = _worker_result_path(output_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_worker(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    try:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-jax-stage-profile")

        import jax
        import jax.numpy as jnp

        from openpi.models.jax_split_types import JaxDenoiseState
        from openpi.policies.jax_va_split_policy import _load_jax_model
        from openpi.serving.va_split_jax.compile import JaxCompileConfig
        from openpi.serving.va_split_jax.compile import make_model_noise_factory
        from openpi.serving.va_split_jax.compile import make_model_observation_factory
        from openpi.serving.va_split_jax.compile import maybe_jit_split_model
        from openpi.training import config as _config

        if not jax.devices("gpu"):
            raise RuntimeError("No JAX GPU device is available in the worker process")

        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        num_steps = int(args.num_steps)
        if num_steps <= 0:
            raise ValueError("--num-steps must be positive")

        train_config = _config.get_config(args.config)
        model = _load_jax_model(train_config, args.checkpoint_dir)
        compile_config = JaxCompileConfig(
            enabled=not args.no_jax_jit,
            warmup_enabled=True,
            warmup_max_batch_size=batch_size,
            num_steps=num_steps,
        )
        model = maybe_jit_split_model(model, compile_config)

        observation_factory = make_model_observation_factory(model)
        noise_factory = make_model_noise_factory(model)
        observation = observation_factory(batch_size)
        noise = noise_factory(batch_size)
        _block_until_ready(observation)
        _block_until_ready(noise)

        total_steps = int(args.warmup_steps) + int(args.measure_steps)
        samples = {"total": [], "vlm": [], "action_head": [], "other": []}
        for step_idx in range(total_steps):
            result = _profile_once(
                model=model,
                observation=observation,
                noise=noise,
                num_steps=num_steps,
                denoise_state_cls=JaxDenoiseState,
                jnp=jnp,
            )
            if step_idx >= int(args.warmup_steps):
                for key, value in result.items():
                    samples[key].append(value)

        payload = {
            "status": "pass",
            "case_id": str(args.case_id),
            "mps_sm": int(args.mps_sm_current),
            "batch_size": batch_size,
            "gpu_id": int(args.gpu_id),
            "warmup_steps": int(args.warmup_steps),
            "measure_steps": int(args.measure_steps),
            "elapsed_s": time.perf_counter() - started,
            "metrics": {key: _summary(values) for key, values in samples.items()},
            "model": {
                "config": str(args.config),
                "checkpoint_dir": str(args.checkpoint_dir),
                "num_steps": num_steps,
                "action_horizon": int(getattr(model, "action_horizon")),
                "action_dim": int(getattr(model, "action_dim")),
                "max_token_len": int(getattr(model, "max_token_len")),
                "jax_jit": not args.no_jax_jit,
                "devices": [str(device) for device in jax.devices()],
            },
            "timing_kind": "synchronous_jax_wall_clock_ms",
        }
        _write_worker_result(output_dir, str(args.case_id), payload)
    except Exception as exc:
        _write_worker_result(
            output_dir,
            str(args.case_id),
            {
                "status": "failed",
                "case_id": str(args.case_id),
                "mps_sm": args.mps_sm_current,
                "batch_size": args.batch_size,
                "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            },
        )
        raise


def _profile_once(
    *,
    model: Any,
    observation: Any,
    noise: Any,
    num_steps: int,
    denoise_state_cls: Any,
    jnp: Any,
) -> dict[str, float]:
    total_start = time.perf_counter()

    vlm_start = time.perf_counter()
    prefix = model.build_prefix_feature(None, observation)
    _block_until_ready(prefix)
    vlm_ms = (time.perf_counter() - vlm_start) * 1000.0

    batch_size = int(noise.shape[0])
    dt = jnp.full((batch_size,), -1.0 / float(num_steps), dtype=jnp.float32)
    x_t = noise
    action_start = time.perf_counter()
    for step in range(num_steps):
        denoise_state = denoise_state_cls(
            x_t=x_t,
            step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
            num_steps=num_steps,
            dt=dt,
        )
        v_t = model.denoise_one_batch(prefix, denoise_state)
        dt_b = dt.reshape((-1,) + (1,) * (v_t.ndim - 1))
        x_t = x_t + dt_b * v_t
    _block_until_ready(x_t)
    action_ms = (time.perf_counter() - action_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "total": total_ms,
        "vlm": vlm_ms,
        "action_head": action_ms,
        "other": max(total_ms - vlm_ms - action_ms, 0.0),
    }


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _resolve_gpu_uuid(gpu_id: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    uuid = completed.stdout.strip().splitlines()[0].strip()
    if not uuid:
        raise RuntimeError(f"Failed to resolve UUID for GPU {gpu_id}")
    return uuid


def _start_mps_daemon(cuda_visible_device: str, pipe_dir: str, log_dir: str) -> None:
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_device,
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
    )
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def _stop_mps_daemon(pipe_dir: str, log_dir: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
        }
    )
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        check=False,
        capture_output=True,
    )


def _run_case(args: argparse.Namespace, case: StageProfileCase) -> dict:
    output_dir = Path(args.output_dir)
    case_dir = output_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    base_env = os.environ.copy()
    python_path_entries = [
        str(repo_root / "src"),
        str(repo_root / "packages" / "openpi-client" / "src"),
    ]
    existing_python_path = base_env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    base_env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    env = build_worker_env(
        base_env=base_env,
        case=case,
        mps_pipe_dir=str(args.mps_pipe_dir),
        mps_log_dir=str(args.mps_log_dir),
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-dir",
        args.output_dir,
        "--gpu-id",
        str(case.gpu_id),
        "--case-id",
        case.case_id,
        "--mps-sm-current",
        str(case.mps_sm),
        "--batch-size",
        str(case.batch_size),
        "--num-steps",
        str(args.num_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measure-steps",
        str(args.measure_steps),
    ]
    if args.no_jax_jit:
        cmd.append("--no-jax-jit")

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        env=env,
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
        timeout=args.case_timeout_s,
    )
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    result_path = _worker_result_path(output_dir, case.case_id)
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    status = "pass" if completed.returncode == 0 and payload.get("status") == "pass" else "failed"
    record = {
        "case_id": case.case_id,
        "status": status,
        "mps_sm": case.mps_sm,
        "batch_size": case.batch_size,
        "gpu_id": case.gpu_id,
        "elapsed_s": time.perf_counter() - started,
        "returncode": completed.returncode,
        "metrics": payload.get("metrics"),
        "model": payload.get("model"),
        "error_message": None,
    }
    if status != "pass":
        record["error_message"] = payload.get("error_message") or completed.stderr[-4000:]
    (case_dir / "case_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def _resolve_mps_dirs(args: argparse.Namespace) -> None:
    if args.mps_pipe_dir is None:
        args.mps_pipe_dir = f"/tmp/openpi-jax-stage-mps-{os.getuid()}-{os.getpid()}/pipe"
    if args.mps_log_dir is None:
        args.mps_log_dir = f"/tmp/openpi-jax-stage-mps-{os.getuid()}-{os.getpid()}/log"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        _run_worker(args)
        return

    _resolve_mps_dirs(args)
    output_dir = Path(args.output_dir)
    records: list[dict] = []
    mps_started = False
    try:
        if args.manage_mps:
            args.cuda_visible_device = _resolve_gpu_uuid(int(args.gpu_id))
            _start_mps_daemon(
                str(args.cuda_visible_device),
                str(args.mps_pipe_dir),
                str(args.mps_log_dir),
            )
            mps_started = True
        else:
            args.cuda_visible_device = str(args.gpu_id)

        for case in build_stage_cases(args):
            record = None
            if args.skip_existing:
                record = _load_existing_passed_case(output_dir, case)
            if record is None:
                record = _run_case(args, case)
            records.append(record)
            write_summary(output_dir, records)
    finally:
        if mps_started:
            _stop_mps_daemon(str(args.mps_pipe_dir), str(args.mps_log_dir))

    print(json.dumps(write_summary(output_dir, records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
