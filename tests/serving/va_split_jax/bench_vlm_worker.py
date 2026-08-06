#!/usr/bin/env python3
"""Standalone JAX VA-split VLM worker benchmark.

Runs direct prefix forward and the production VLM worker path side by side:
request observation stacking, host-to-JAX staging, unchecked Observation
construction, prefix forward, and prefix slab writes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", default="3")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default="/mnt/tianze/models/pi05_libero")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-vlm-batch-size", type=int, default=8)
    parser.add_argument("--max-prefix-slots", type=int, default=24)
    parser.add_argument("--warmup-max-batch-size", type=int, default=24)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bench-warmup-repeats", type=int, default=1)
    parser.add_argument("--log-dir", default="logs/tests/VLM-test")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-vlm-test")

    import jax

    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.serving.va_split_jax.compile import make_prefix_feature_template
    from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
    from openpi.serving.va_split_jax.compile import prune_split_model_for_role
    from openpi.serving.va_split_jax.compile import warmup_vlm_ae_slab_writes
    from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
    from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
    from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
    from openpi.serving.va_split_jax.types import JaxLaneCredits
    from openpi.serving.va_split_jax.types import JaxReleaseFeature
    from openpi.serving.va_split_jax.types import JaxRequestEnvelope
    from openpi.serving.va_split_jax.vlm_process import JaxVLMWorker
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
    model = prune_split_model_for_role(model, role="vlm")
    model = maybe_jit_vlm_model(model, compile_config)

    backend = make_default_device_slab_backend()
    pool = JaxVlmPrefixCacheLanePool(max_lanes=args.max_prefix_slots, backend=backend)
    pool.initialize_from_feature(template)
    worker = JaxVLMWorker(
        model=model,
        max_live_features=args.max_prefix_slots,
        backend=backend,
        shared_pool=pool,
    )
    worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=tuple(range(args.max_prefix_slots))))

    warmup_batches = 0.0
    if compile_config.warmup_enabled:
        stats = warmup_vlm_prefix_model(
            model=model,
            observation_factory=observation_factory,
            max_vlm_batch_size=args.max_vlm_batch_size,
            config=compile_config,
        )
        warmup_batches += float(stats["jax_warmup_batches"])
        stats = warmup_vlm_ae_slab_writes(
            model=model,
            observation_factory=observation_factory,
            backend=backend,
            slab_tree=pool.local_slab_tree(),
            max_lanes=args.max_prefix_slots,
            max_vlm_batch_size=args.max_vlm_batch_size,
            config=compile_config,
        )
        warmup_batches += float(stats["jax_warmup_batches"])

    runs: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for warmup_repeat in range(args.bench_warmup_repeats):
            _run_one_batch(
                model=model,
                worker=worker,
                observation_factory=observation_factory,
                noise_factory=noise_factory,
                request_cls=JaxRequestEnvelope,
                release_cls=JaxReleaseFeature,
                batch_size=batch_size,
                num_steps=args.num_steps,
                repeat=-(warmup_repeat + 1),
            )
        for repeat in range(args.repeats):
            runs.append(
                _run_one_batch(
                    model=model,
                    worker=worker,
                    observation_factory=observation_factory,
                    noise_factory=noise_factory,
                    request_cls=JaxRequestEnvelope,
                    release_cls=JaxReleaseFeature,
                    batch_size=batch_size,
                    num_steps=args.num_steps,
                    repeat=repeat,
                )
            )

    summary = _summarize_runs(runs)
    payload = {
        "args": vars(args),
        "devices": [str(device) for device in jax.devices()],
        "jax_warmup_batches": warmup_batches,
        "runs": runs,
        "summary": summary,
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"vlm_worker_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def _run_one_batch(
    *,
    model,
    worker,
    observation_factory,
    noise_factory,
    request_cls,
    release_cls,
    batch_size: int,
    num_steps: int,
    repeat: int,
) -> dict[str, Any]:
    observation = observation_factory(batch_size)
    _block_until_ready(observation)
    direct_start = time.perf_counter()
    prefix = model.build_prefix_feature(None, observation)
    _block_until_ready(prefix)
    direct_prefix_ms = (time.perf_counter() - direct_start) * 1000.0

    observation_dict = _observation_to_dict(observation)
    noise = np.asarray(noise_factory(batch_size))
    requests = [
        request_cls(
            request_id=f"b{batch_size}-r{repeat}-{row}",
            observation=_slice_observation_row_to_numpy(observation_dict, row),
            sample_kwargs={"num_steps": num_steps, "noise": noise[row : row + 1].copy()},
            enqueue_ns=time.monotonic_ns(),
        )
        for row in range(batch_size)
    ]

    worker_start = time.perf_counter()
    ready = worker.handle_batch(requests)
    worker_total_ms = (time.perf_counter() - worker_start) * 1000.0
    for item in ready:
        worker.release(release_cls(request_id=item.request_id, slot_id=item.slot_handle.slot_id))

    return {
        "batch_size": batch_size,
        "repeat": repeat,
        "direct_prefix_ms": direct_prefix_ms,
        "worker_total_ms": worker_total_ms,
        "worker_timing_mean": _mean_timing([dict(item.timing or {}) for item in ready]),
    }


def _observation_to_dict(observation) -> dict[str, Any]:
    return {
        "image": observation.images,
        "image_mask": observation.image_masks,
        "state": observation.state,
        "tokenized_prompt": observation.tokenized_prompt,
        "tokenized_prompt_mask": observation.tokenized_prompt_mask,
        "token_ar_mask": observation.token_ar_mask,
        "token_loss_mask": observation.token_loss_mask,
    }


def _slice_observation_row_to_numpy(value: Any, row: int) -> Any:
    if isinstance(value, dict):
        return {key: _slice_observation_row_to_numpy(item, row) for key, item in value.items()}
    if value is None:
        return None
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return np.asarray(value[row : row + 1]).copy()
    return value


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
        timing_keys = sorted({key for run in selected for key in run["worker_timing_mean"]})
        summary[str(batch_size)] = {
            "direct_prefix_ms_mean": statistics.fmean(float(run["direct_prefix_ms"]) for run in selected),
            "direct_prefix_ms_p50": statistics.median(float(run["direct_prefix_ms"]) for run in selected),
            "worker_total_ms_mean": statistics.fmean(float(run["worker_total_ms"]) for run in selected),
            "worker_total_ms_p50": statistics.median(float(run["worker_total_ms"]) for run in selected),
            "worker_timing_mean": {
                key: statistics.fmean(
                    float(run["worker_timing_mean"][key]) for run in selected if key in run["worker_timing_mean"]
                )
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
