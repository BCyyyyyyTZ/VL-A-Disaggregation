#!/usr/bin/env python3
"""Diagnose AE denoise latency after local-slab warmup (direct vs slab views)."""
from __future__ import annotations

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
# Emit compile events so we can see cache misses during timed calls.
os.environ.setdefault("JAX_LOG_COMPILES", "1")

import jax
import jax.numpy as jnp

from openpi.policies.jax_va_split_policy import _load_jax_model
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import make_model_noise_factory
from openpi.serving.va_split_jax.compile import make_model_observation_factory
from openpi.serving.va_split_jax.compile import maybe_jit_split_model
from openpi.serving.va_split_jax.compile import runtime_aligned_denoise_state
from openpi.serving.va_split_jax.compile import warmup_ae_denoise_model
from openpi.serving.va_split_jax.compile import _build_slab_prefix_batch
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.training import config as _config


def _block(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_denoise(model, prefix, state, *, repeats: int = 3) -> list[float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        v_t = model.denoise_one_batch(prefix, state)
        _block(v_t)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _time_step_once_like(model, prefix, xs, dts, step_idxs, num_steps: int) -> float:
    """Mirror ae_process.step_once denoise + per-row update syncs."""
    t0 = time.perf_counter()
    x_t = jnp.concatenate(xs, axis=0)
    step_idx = jnp.asarray(step_idxs, dtype=jnp.int32)
    dt = jnp.stack([jnp.asarray(d) for d in dts])
    state = runtime_aligned_denoise_state(batch_size=x_t.shape[0], noise=x_t, num_steps=num_steps)
    # overwrite with runtime-constructed vectors (same as step_once)
    from openpi.models.jax_split_types import JaxDenoiseState

    state = JaxDenoiseState(x_t=x_t, step_idx=step_idx, num_steps=num_steps, dt=dt)
    v_t = model.denoise_one_batch(prefix, state)
    v_t.block_until_ready()
    for row, (x, d) in enumerate(zip(xs, dts)):
        new_x = x + d * v_t[row : row + 1]
        new_x.block_until_ready()
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    print("jax devices:", jax.devices())
    train_config = _config.get_config("pi05_libero")
    ckpt = "/data/miliang/huggingface/hub/openpi-assets/checkpoints/pi05_libero"
    model = _load_jax_model(train_config, ckpt)
    cfg = JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=24, num_steps=5)
    model = maybe_jit_split_model(model, cfg)
    obs_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model)
    backend = make_default_device_slab_backend()
    print("backend:", type(backend).__name__)

    print("\n=== warmup AE local-slab path B=1..24 ===")
    t0 = time.perf_counter()
    stats = warmup_ae_denoise_model(
        model=model,
        observation_factory=obs_factory,
        noise_factory=noise_factory,
        max_ae_batch_size=999,
        max_prefix_slots=24,
        config=cfg,
        backend=backend,
    )
    print(f"warmup done in {(time.perf_counter()-t0):.1f}s stats={stats}")

    print("\n=== timed denoise after warmup (JAX_LOG_COMPILES=1) ===")
    for batch_size in (1, 8, 18, 24):
        noise = noise_factory(batch_size)
        state = runtime_aligned_denoise_state(batch_size=batch_size, noise=noise, num_steps=5)

        direct_prefix = model.build_prefix_feature(None, obs_factory(batch_size))
        _block(direct_prefix)
        direct_ms = _time_denoise(model, direct_prefix, state, repeats=3)

        slab_prefix = _build_slab_prefix_batch(
            model=model,
            observation_factory=obs_factory,
            batch_size=batch_size,
            max_prefix_slots=24,
            backend=backend,
        )
        slab_ms = _time_denoise(model, slab_prefix, state, repeats=3)

        xs = [noise[i : i + 1] for i in range(batch_size)]
        dts = [jnp.asarray(-1.0 / 5, dtype=jnp.float32) for _ in range(batch_size)]
        step_idxs = [0 for _ in range(batch_size)]
        step_once_ms = [
            _time_step_once_like(model, slab_prefix, xs, dts, step_idxs, 5) for _ in range(3)
        ]

        print(
            f"B={batch_size}: direct={[round(x,1) for x in direct_ms]} "
            f"local_slab={[round(x,1) for x in slab_ms]} "
            f"step_once_like={[round(x,1) for x in step_once_ms]}"
        )


if __name__ == "__main__":
    main()
