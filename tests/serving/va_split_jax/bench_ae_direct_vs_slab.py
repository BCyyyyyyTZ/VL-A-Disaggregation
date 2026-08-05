#!/usr/bin/env python3
"""Compare JAX AE denoise with direct prefixes vs split slab-view prefixes."""
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
    parser.add_argument("--checkpoint-dir", default="/data1/miliang/models/pi05_libero")
    parser.add_argument("--batch-sizes", default="1,8,24")
    parser.add_argument("--max-prefix-slots", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--log-dir", default="logs/tests/AE-test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-ae-test")

    import jax
    import jax.numpy as jnp

    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
    from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
    from openpi.serving.va_split_jax.compile import planned_warmup_batches
    from openpi.serving.va_split_jax.compile import prune_split_model_for_role
    from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
    from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
    from openpi.training import config as _config

    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    if not batch_sizes or min(batch_sizes) <= 0:
        raise ValueError("--batch-sizes must be a non-empty comma-separated list of positive integers")
    if max(batch_sizes) > args.max_prefix_slots:
        raise ValueError("largest --batch-sizes entry cannot exceed --max-prefix-slots")

    train_config = _config.get_config(args.config)
    print("[ae-compare] loading model", flush=True)
    model = _load_jax_model(train_config, args.checkpoint_dir)
    print("[ae-compare] model loaded", flush=True)
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model)

    cfg = JaxCompileConfig(
        enabled=True,
        warmup_enabled=True,
        warmup_max_batch_size=args.max_prefix_slots,
        num_steps=args.num_steps,
    )
    model = maybe_jit_vlm_model(model, cfg)

    direct_prefix_by_size = {}
    for batch_size in batch_sizes:
        print(f"[ae-compare] building direct prefix B={batch_size}", flush=True)
        prefix = model.build_prefix_feature(None, observation_factory(batch_size))
        _block_until_ready(prefix)
        direct_prefix_by_size[batch_size] = prefix

    print("[ae-compare] pruning/jitting AE model", flush=True)
    model = prune_split_model_for_role(model, role="ae")
    model = maybe_jit_ae_model(model, cfg)
    backend = make_default_device_slab_backend()

    slab_prefix_by_size = {}
    slab_prefix_block_ms_by_size = {}
    slab_pools = []
    for batch_size, direct_prefix in direct_prefix_by_size.items():
        print(f"[ae-compare] building slab prefix B={batch_size}", flush=True)
        pool = JaxVlmPrefixCacheLanePool(max_lanes=args.max_prefix_slots, backend=backend)
        slab_pools.append(pool)
        for row in range(batch_size):
            pool.put_lane(f"b{batch_size}-{row}", _prefix_feature_row_view(direct_prefix, row))
        prefix_view = pool.view_prefix_batch(batch_size)
        start = time.perf_counter()
        _block_until_ready(prefix_view)
        slab_prefix_block_ms_by_size[batch_size] = (time.perf_counter() - start) * 1000.0
        slab_prefix_by_size[batch_size] = prefix_view

    print("[ae-compare] warming direct denoise", flush=True)
    _warmup_denoise(model, noise_factory, direct_prefix_by_size, args.num_steps, args.max_prefix_slots)
    print("[ae-compare] warming slab denoise", flush=True)
    _warmup_denoise(model, noise_factory, slab_prefix_by_size, args.num_steps, args.max_prefix_slots)

    runs = []
    for batch_size in batch_sizes:
        for repeat in range(args.repeats):
            print(f"[ae-compare] timed B={batch_size} repeat={repeat}", flush=True)
            noise = noise_factory(batch_size)
            _block_until_ready(noise)
            direct_prefix = direct_prefix_by_size[batch_size]
            slab_prefix = slab_prefix_by_size[batch_size]
            direct_sync = _time_sync_steps(model, direct_prefix, noise, args.num_steps)
            slab_sync = _time_sync_steps(model, slab_prefix, noise, args.num_steps)
            direct_async = _time_async_chain(model, direct_prefix, noise, args.num_steps)
            slab_async = _time_async_chain(model, slab_prefix, noise, args.num_steps)
            runs.append(
                {
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "direct_sync_step_ms": direct_sync,
                    "slab_sync_step_ms": slab_sync,
                    "direct_async_total_ms": direct_async,
                    "slab_async_total_ms": slab_async,
                    "slab_prefix_view_block_ms": slab_prefix_block_ms_by_size[batch_size],
                }
            )

    summary = _summarize(runs)
    payload = {
        "args": vars(args),
        "devices": [str(device) for device in jax.devices()],
        "runs": runs,
        "summary": summary,
    }
    del slab_pools
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"ae_direct_vs_slab_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def _warmup_denoise(model: Any, noise_factory, prefix_by_size: dict[int, Any], num_steps: int, max_size: int) -> None:
    from openpi.serving.va_split_jax.compile import planned_warmup_batches

    for batch_size, repeats in planned_warmup_batches(max_batch_size=max_size, warmup_max_batch_size=max_size):
        prefix = prefix_by_size.get(batch_size)
        if prefix is None:
            continue
        for _ in range(repeats):
            noise = noise_factory(batch_size)
            _time_sync_steps(model, prefix, noise, num_steps)


def _time_sync_steps(model: Any, prefix: Any, noise: Any, num_steps: int) -> float:
    import jax.numpy as jnp

    from openpi.models.jax_split_types import JaxDenoiseState

    x_t = noise
    batch_size = int(noise.shape[0])
    dt = jnp.full((batch_size,), -1.0 / float(num_steps), dtype=jnp.float32)
    times = []
    for step in range(num_steps):
        state = JaxDenoiseState(
            x_t=x_t,
            step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
            num_steps=num_steps,
            dt=dt,
        )
        start = time.perf_counter()
        v_t = model.denoise_one_batch(prefix, state)
        x_t = x_t + dt.reshape((-1,) + (1,) * (v_t.ndim - 1)) * v_t
        _block_until_ready(x_t)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.fmean(times)


def _time_async_chain(model: Any, prefix: Any, noise: Any, num_steps: int) -> float:
    import jax.numpy as jnp

    from openpi.models.jax_split_types import JaxDenoiseState

    x_t = noise
    batch_size = int(noise.shape[0])
    dt = jnp.full((batch_size,), -1.0 / float(num_steps), dtype=jnp.float32)
    start = time.perf_counter()
    for step in range(num_steps):
        state = JaxDenoiseState(
            x_t=x_t,
            step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
            num_steps=num_steps,
            dt=dt,
        )
        v_t = model.denoise_one_batch(prefix, state)
        x_t = x_t + dt.reshape((-1,) + (1,) * (v_t.ndim - 1)) * v_t
    _block_until_ready(x_t)
    return (time.perf_counter() - start) * 1000.0


def _prefix_feature_row_view(feature: Any, row: int) -> Any:
    import jax

    from openpi.models.jax_split_types import JaxPrefixFeature

    return JaxPrefixFeature(
        past_key_values=jax.tree.map(lambda value: jax.lax.dynamic_slice_in_dim(value, row, 1, axis=1), feature.past_key_values),
        prefix_pad_masks=feature.prefix_pad_masks[row : row + 1],
        state=feature.state[row : row + 1] if feature.state is not None else None,
    )


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for batch_size in sorted({int(run["batch_size"]) for run in runs}):
        selected = [run for run in runs if int(run["batch_size"]) == batch_size]
        summary[str(batch_size)] = {
            key: statistics.fmean(float(run[key]) for run in selected)
            for key in (
                "direct_sync_step_ms",
                "slab_sync_step_ms",
                "direct_async_total_ms",
                "slab_async_total_ms",
                "slab_prefix_view_block_ms",
            )
        }
    return summary


if __name__ == "__main__":
    main()
