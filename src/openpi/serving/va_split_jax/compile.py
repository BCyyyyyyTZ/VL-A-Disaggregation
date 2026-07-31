from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from openpi.shared import nnx_utils


JAX_COMPILE_WARMUP_BATCH_PLAN: tuple[tuple[int, int], ...] = (
    (1, 2),
    (4, 1),
    (8, 2),
    (16, 1),
    (20, 2),
    (24, 1),
    (32, 1),
)


@dataclass(frozen=True, slots=True)
class JaxCompileConfig:
    enabled: bool = True
    warmup_enabled: bool = True
    warmup_max_batch_size: int = 32


def maybe_jit_split_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled:
        return model
    model.build_prefix_feature = nnx_utils.module_jit(model.build_prefix_feature)
    model.denoise_one_batch = nnx_utils.module_jit(model.denoise_one_batch)
    return model


def maybe_jit_monolithic_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled:
        return model
    model.sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    return model


def planned_warmup_batches(*, max_batch_size: int, warmup_max_batch_size: int) -> tuple[tuple[int, int], ...]:
    cap = min(max_batch_size, warmup_max_batch_size)
    return tuple((batch_size, repeats) for batch_size, repeats in JAX_COMPILE_WARMUP_BATCH_PLAN if batch_size <= cap)


def warmup_split_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_vlm_batch_size: int,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    num_steps: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    if not config.warmup_enabled:
        return {"jax_warmup_batches": 0.0}
    batches = planned_warmup_batches(
        max_batch_size=max(max_vlm_batch_size, max_ae_batch_size, max_prefix_slots),
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    if not config.enabled:
        batches = ((1, 1),)
    warmed = 0
    for batch_size, repeats in batches:
        if batch_size > max_prefix_slots:
            continue
        for _ in range(repeats):
            obs = observation_factory(batch_size)
            noise = noise_factory(batch_size)
            prefix = model.build_prefix_feature(None, obs)
            denoise_state = model.init_denoise_state(jax.random.key(0), batch_size, noise, num_steps)
            v_t = model.denoise_one_batch(prefix, denoise_state)
            _block_until_ready(v_t)
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


def warmup_monolithic_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_batch_size: int,
    num_steps: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    if not config.warmup_enabled:
        return {"jax_warmup_batches": 0.0}
    batches = planned_warmup_batches(
        max_batch_size=max_batch_size,
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    if not config.enabled:
        batches = ((1, 1),)
    warmed = 0
    for batch_size, repeats in batches:
        for _ in range(repeats):
            obs = observation_factory(batch_size)
            noise = noise_factory(batch_size)
            actions = model.sample_actions(jax.random.key(0), obs, noise=noise, num_steps=num_steps)
            actions.block_until_ready()
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
