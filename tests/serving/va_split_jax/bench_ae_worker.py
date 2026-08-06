#!/usr/bin/env python3
"""Standalone JAX VA-split AE worker benchmark.

This intentionally lives under tests/ so optimization experiments can stay close
to the AE worker unit tests while writing results outside pytest output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", default="3")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        default="/mnt/tianze/models/pi05_libero",
    )
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-prefix-slots", type=int, default=24)
    parser.add_argument("--max-ae-batch-size", type=int, default=999)
    parser.add_argument("--warmup-max-batch-size", type=int, default=24)
    parser.add_argument("--batch-sizes", default="1,8,24")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--log-dir", default="logs/tests/AE-test")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-ae-test")

    import jax

    from openpi.models.jax_split_types import JaxPrefixSlotHandle
    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.ae_process import JaxAEWorker
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.serving.va_split_jax.compile import make_prefix_feature_template
    from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
    from openpi.serving.va_split_jax.compile import prune_split_model_for_role
    from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
    from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
    from openpi.serving.va_split_jax.types import JaxPrefixReady
    from openpi.training import config as _config

    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if max(batch_sizes) > args.max_prefix_slots:
        raise ValueError("Largest batch size cannot exceed --max-prefix-slots")

    train_config = _config.get_config(args.config)
    model = _load_jax_model(train_config, args.checkpoint_dir)
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model)
    template = make_prefix_feature_template(model, observation_factory)

    compile_config = JaxCompileConfig(
        enabled=not args.no_compile,
        warmup_enabled=not args.no_warmup,
        warmup_max_batch_size=args.warmup_max_batch_size,
        num_steps=args.num_steps,
    )
    model = prune_split_model_for_role(model, role="ae")
    model = maybe_jit_ae_model(model, compile_config)

    backend = make_default_device_slab_backend()
    pool = JaxVlmPrefixCacheLanePool(max_lanes=args.max_prefix_slots, backend=backend)
    worker = JaxAEWorker(
        model=model,
        max_batch_size=args.max_ae_batch_size,
        max_prefix_slots=args.max_prefix_slots,
        backend=backend,
        compile_config=compile_config,
        noise_factory=noise_factory,
        owned_pool=pool,
    )
    slab_ready, credits = worker.initialize_owned_pool(template)
    del slab_ready, credits

    runs: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for repeat in range(args.repeats):
            runs.append(
                _run_one_batch(
                    worker,
                    pool,
                    template,
                    noise_factory,
                    prefix_ready_cls=JaxPrefixReady,
                    slot_handle_cls=JaxPrefixSlotHandle,
                    batch_size=batch_size,
                    num_steps=args.num_steps,
                    repeat=repeat,
                )
            )

    summary = _summarize_runs(runs)
    payload = {
        "args": vars(args),
        "devices": [str(device) for device in jax.devices()],
        "jax_warmup_batches": worker.ipc_warmup_batches,
        "runs": runs,
        "summary": summary,
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"ae_worker_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def _run_one_batch(
    worker,
    pool,
    template,
    noise_factory,
    *,
    prefix_ready_cls,
    slot_handle_cls,
    batch_size: int,
    num_steps: int,
    repeat: int,
) -> dict[str, Any]:
    noise = noise_factory(batch_size)
    _block_until_ready(noise)
    for lane_id in range(batch_size):
        pool.write_lane(lane_id, template)
        worker.add_prefix(
            prefix_ready_cls(
                request_id=f"b{batch_size}-r{repeat}-{lane_id}",
                slot_handle=slot_handle_cls(
                    slot_id=lane_id,
                    batch_rows=1,
                    prefix_shape_tree=None,
                    prefix_dtype_tree=None,
                ),
                num_steps=num_steps,
                sample_kwargs={"noise": noise[lane_id : lane_id + 1]},
                timing={},
            )
        )

    start = time.perf_counter()
    results = []
    while worker.active:
        step_results, _ = worker.step_once()
        results.extend(step_results)
    action_ms = (time.perf_counter() - start) * 1000.0
    for result in results:
        _block_until_ready(result.actions)

    timings = [dict(result.timing or {}) for result in results]
    return {
        "batch_size": batch_size,
        "repeat": repeat,
        "action_ms": action_ms,
        "timing_mean": _mean_timing(timings),
    }


def _mean_timing(timings: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for timing in timings for key in timing})
    return {
        key: statistics.fmean(float(timing[key]) for timing in timings if key in timing)
        for key in keys
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for batch_size in sorted({int(run["batch_size"]) for run in runs}):
        selected = [run for run in runs if int(run["batch_size"]) == batch_size]
        timing_keys = sorted({key for run in selected for key in run["timing_mean"]})
        summary[str(batch_size)] = {
            "action_ms_mean": statistics.fmean(float(run["action_ms"]) for run in selected),
            "timing_mean": {
                key: statistics.fmean(float(run["timing_mean"][key]) for run in selected if key in run["timing_mean"])
                for key in timing_keys
            },
        }
    return summary


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


if __name__ == "__main__":
    main()
