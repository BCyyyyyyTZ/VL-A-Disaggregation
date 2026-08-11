from __future__ import annotations

import contextlib
from dataclasses import dataclass
import gc
from typing import Any

import jax
import jax.numpy as jnp

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.shared import nnx_utils

DEFAULT_WARMUP_MAX_BATCH_SIZE = 20
_VLM_UNUSED_PI0_ATTRS = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)
_PALIGEMMA_LLM_EXPERT_INDEX = 0
_ACTION_LLM_EXPERT_INDEX = 1


@dataclass(frozen=True, slots=True)
class JaxCompileConfig:
    enabled: bool = True
    warmup_enabled: bool = True
    compile_ae: bool = True
    compile_vlm: bool = True
    # Cover every batch size in [1, warmup_max_batch_size] (clamped by runtime capacity).
    # Split warmup defaults cap at 20 to keep peak memory lower than prefix capacity.
    warmup_max_batch_size: int = DEFAULT_WARMUP_MAX_BATCH_SIZE
    num_steps: int = 10


def maybe_jit_split_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled:
        return model
    if config.compile_vlm:
        model.build_prefix_feature = nnx_utils.module_jit(model.build_prefix_feature)
    if config.compile_ae:
        model.denoise_one_batch = nnx_utils.module_jit(model.denoise_one_batch)
    return model


def maybe_jit_vlm_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled or not config.compile_vlm:
        return model
    model.build_prefix_feature = nnx_utils.module_jit(model.build_prefix_feature)
    return model


def maybe_jit_ae_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled or not config.compile_ae:
        return model
    model.denoise_one_batch = nnx_utils.module_jit(model.denoise_one_batch)
    return model


def maybe_jit_monolithic_model(model: Any, config: JaxCompileConfig) -> Any:
    """JIT baseline as two small graphs (VLM prefix + AE denoise), matching split compile.

    Timed monolithic serving can still call ``sample_actions`` when component timing is
    disabled; that path remains a separate full-graph jit in :class:`Policy`.
    """
    return maybe_jit_split_model(model, config)


def prune_split_model_for_role(model: Any, *, role: str) -> Any:
    """Drop role-unused JAX split modules before freezing NNX state for jit."""
    if role == "vlm":
        _delete_attrs(model, _VLM_UNUSED_PI0_ATTRS)
        _prune_fused_paligemma_llm(model, expert_index=_ACTION_LLM_EXPERT_INDEX)
    elif role == "ae":
        paligemma = getattr(model, "PaliGemma", None)
        if paligemma is not None:
            _delete_child(paligemma, "img")
        _prune_fused_paligemma_llm(model, expert_index=_PALIGEMMA_LLM_EXPERT_INDEX)
    else:
        raise ValueError(f"Unsupported split model role: {role!r}")
    gc.collect()
    with contextlib.suppress(Exception):
        jax.clear_caches()
    return model


def _delete_attrs(value: Any, names: tuple[str, ...]) -> None:
    for name in names:
        _delete_child(value, name)


def _prune_fused_paligemma_llm(model: Any, *, expert_index: int) -> None:
    paligemma = getattr(model, "PaliGemma", None)
    llm = None if paligemma is None else _get_child(paligemma, "llm")
    if llm is None:
        return

    if expert_index == _PALIGEMMA_LLM_EXPERT_INDEX:
        _delete_child(llm, "embedder")

    suffix = "" if expert_index == _PALIGEMMA_LLM_EXPERT_INDEX else f"_{expert_index}"
    _delete_child(llm, f"final_norm{suffix}")

    layers = _get_child(llm, "layers")
    if layers is None:
        return
    _delete_child(layers, f"pre_attention_norm{suffix}")
    _delete_child(layers, f"pre_ffw_norm{suffix}")
    _delete_child(layers, f"mlp{suffix}")

    attn = _get_child(layers, "attn")
    if attn is None:
        return
    for base_name in ("qkv_einsum", "q_einsum", "kv_einsum", "attn_vec_einsum"):
        _delete_child(attn, f"{base_name}{suffix}")


def _get_child(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _delete_child(value: Any, name: str) -> None:
    if isinstance(value, dict):
        value.pop(name, None)
        return
    _discard_linen_attribute(value, name)
    if hasattr(value, name):
        delattr(value, name)


def _discard_linen_attribute(value: Any, name: str) -> None:
    linen_attributes = getattr(value, "linen_attributes", None)
    if linen_attributes is None or name not in linen_attributes:
        return
    value.linen_attributes = tuple(attr for attr in linen_attributes if attr != name)


def planned_warmup_batches(*, max_batch_size: int, warmup_max_batch_size: int) -> tuple[tuple[int, int], ...]:
    """Return full-coverage warmup (batch_size, repeats) for every B in 1..cap.

    B=1 is executed twice to stabilize the first XLA compile; larger B once each.
    """
    if max_batch_size <= 0 or warmup_max_batch_size <= 0:
        raise ValueError("max_batch_size and warmup_max_batch_size must be positive")
    cap = min(max_batch_size, warmup_max_batch_size)
    return ((1, 2), *[(batch_size, 1) for batch_size in range(2, cap + 1)])


def runtime_aligned_denoise_state(
    *,
    batch_size: int,
    noise: jax.Array,
    num_steps: int,
    step_idx: int = 0,
) -> JaxDenoiseState:
    """Build JaxDenoiseState with AE step_once-compatible (B,) step_idx/dt layouts."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return JaxDenoiseState(
        x_t=noise,
        step_idx=jnp.full((batch_size,), step_idx, dtype=jnp.int32),
        num_steps=num_steps,
        dt=jnp.full((batch_size,), -1.0 / float(num_steps), dtype=jnp.float32),
    )


def make_model_observation_factory(model: Any):
    """Construct Observation batches matching a loaded Pi0-like model."""
    action_dim = int(model.action_dim)
    max_token_len = int(model.max_token_len)

    def factory(batch_size: int):
        from openpi.models import model as _model

        image = jnp.ones((batch_size, *_model.IMAGE_RESOLUTION, 3), dtype=jnp.float32)
        return _model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
                "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
                "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            },
            state=jnp.ones((batch_size, action_dim), dtype=jnp.float32),
            tokenized_prompt=jnp.ones((batch_size, max_token_len), dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((batch_size, max_token_len), dtype=bool),
        )

    return factory


def make_model_noise_factory(model: Any):
    action_horizon = int(model.action_horizon)
    action_dim = int(model.action_dim)

    def factory(batch_size: int):
        return jnp.zeros((batch_size, action_horizon, action_dim), dtype=jnp.float32)

    return factory


def make_prefix_feature_template(model: Any, observation_factory) -> JaxPrefixFeature:
    """Create a single-row prefix template without executing prefix forward kernels."""

    shape_feature = jax.eval_shape(lambda: model.build_prefix_feature(None, observation_factory(1)))
    return JaxPrefixFeature(
        past_key_values=jax.tree.map(_zeros_from_shape_dtype, shape_feature.past_key_values),
        prefix_pad_masks=jnp.ones(shape_feature.prefix_pad_masks.shape, dtype=shape_feature.prefix_pad_masks.dtype),
        state=None if shape_feature.state is None else _zeros_from_shape_dtype(shape_feature.state),
    )


def make_pi0_prefix_feature_template(model_config: Any) -> JaxPrefixFeature:
    """Create a Pi0 prefix template from config metadata without running vision forward."""

    from openpi.models import gemma as _gemma
    from openpi.models import siglip as _siglip
    from openpi.models.pi0_config import Pi0Config

    if not isinstance(model_config, Pi0Config):
        raise TypeError(f"make_pi0_prefix_feature_template only supports Pi0Config, got {type(model_config)!r}")

    observation_spec, _ = model_config.inputs_spec(batch_size=1)
    image_spec = next(iter(observation_spec.images.values()))
    image_height = int(image_spec.shape[1])
    image_width = int(image_spec.shape[2])
    patch_h, patch_w = _siglip.decode_variant("So400m/14")["patch_size"]
    image_tokens_per_image = (image_height // patch_h) * (image_width // patch_w)
    prefix_len = len(observation_spec.images) * image_tokens_per_image + int(observation_spec.tokenized_prompt.shape[-1])

    paligemma_config = _gemma.get_config(model_config.paligemma_variant)
    kv_dtype = jnp.dtype(model_config.dtype)
    kv_shape = (
        paligemma_config.depth,
        1,
        prefix_len,
        paligemma_config.num_kv_heads,
        paligemma_config.head_dim,
    )
    past_key_values = (
        jnp.zeros(kv_shape, dtype=kv_dtype),
        jnp.zeros(kv_shape, dtype=kv_dtype),
    )
    return JaxPrefixFeature(
        past_key_values=past_key_values,
        prefix_pad_masks=jnp.ones((1, prefix_len), dtype=jnp.bool_),
        state=jnp.zeros(observation_spec.state.shape, dtype=observation_spec.state.dtype),
    )


def _zeros_from_shape_dtype(value: Any) -> jax.Array:
    return jnp.zeros(value.shape, dtype=value.dtype)


def warmup_vlm_prefix_model(
    *,
    model: Any,
    observation_factory,
    max_vlm_batch_size: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    if not config.warmup_enabled or not config.compile_vlm:
        return {"jax_warmup_batches": 0.0}
    batches = planned_warmup_batches(
        max_batch_size=max_vlm_batch_size,
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    if not config.enabled:
        batches = ((1, 1),)
    warmed = 0
    for batch_size, repeats in batches:
        for _ in range(repeats):
            obs = observation_factory(batch_size)
            prefix = model.build_prefix_feature(None, obs)
            _block_until_ready(prefix)
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


def warmup_vlm_ae_slab_writes(
    *,
    model: Any,
    observation_factory,
    backend: DeviceSlabBackend,
    slab_tree: dict[str, Any],
    max_lanes: int,
    max_vlm_batch_size: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    """Warm VLM write-through into AE-owned slabs."""
    from openpi.serving.va_split_jax.prefix_cache_pool import write_feature_batch_to_slab_tree
    from openpi.serving.va_split_jax.prefix_cache_pool import write_feature_to_slab_tree

    if not config.warmup_enabled or not config.compile_vlm:
        return {"jax_warmup_batches": 0.0}
    if max_vlm_batch_size <= 0:
        raise ValueError("max_vlm_batch_size must be positive")
    batches = planned_warmup_batches(
        max_batch_size=max_lanes,
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    if not config.enabled:
        batches = ((1, 1),)
    warmed = 0
    writable = slab_tree
    for batch_size, repeats in batches:
        if batch_size > max_lanes:
            continue
        for _ in range(repeats):
            filled = 0
            while filled < batch_size:
                chunk = min(max_vlm_batch_size, batch_size - filled)
                feature = model.build_prefix_feature(None, observation_factory(chunk))
                _block_until_ready(feature)
                lane_ids = tuple(range(filled, filled + chunk))
                if chunk > 1:
                    writable = write_feature_batch_to_slab_tree(
                        backend,
                        writable,
                        lane_ids,
                        feature,
                    )
                else:
                    writable = write_feature_to_slab_tree(
                        backend,
                        writable,
                        lane_ids[0],
                        _prefix_feature_row_view(feature, 0),
                    )
                filled += chunk
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


def warmup_vlm_prefix_lane_pool(
    *,
    model: Any,
    observation_factory,
    prefix_pool: JaxVlmPrefixCacheLanePool,
    max_vlm_batch_size: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    """Deprecated alias: warm writes via shared AE-owned pool storage."""
    return warmup_vlm_ae_slab_writes(
        model=model,
        observation_factory=observation_factory,
        backend=prefix_pool._backend,  # noqa: SLF001
        slab_tree=prefix_pool.local_slab_tree(),
        max_lanes=prefix_pool.max_lanes,
        max_vlm_batch_size=max_vlm_batch_size,
        config=config,
    )


def _fill_prefix_pool_rows(
    *,
    model: Any,
    observation_factory,
    prefix_pool: JaxVlmPrefixCacheLanePool,
    num_rows: int,
    max_vlm_batch_size: int,
    request_id_prefix: str,
) -> None:
    filled = 0
    while filled < num_rows:
        chunk = min(max_vlm_batch_size, num_rows - filled)
        feature = model.build_prefix_feature(None, observation_factory(chunk))
        _block_until_ready(feature)
        for row in range(chunk):
            prefix_pool.put_lane(
                f"{request_id_prefix}-{filled + row}",
                _prefix_feature_row_view(feature, row),
            )
        filled += chunk


def warmup_ae_denoise_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    config: JaxCompileConfig,
    backend: DeviceSlabBackend | None = None,
) -> dict[str, float]:
    """Warm AE denoise with direct build_prefix_feature prefixes (pre-IPC).

    Production AE reads CUDA-IPC mapped slabs, whose JAX sharding/cache key differs
    from locally created slab views. Direct-prefix warmup covers the common
    UnspecifiedValue path; call :func:`warmup_ae_denoise_on_mapped_slabs` after IPC
    map for the true runtime cache key.
    """
    del backend  # retained for call-site compatibility
    if not config.warmup_enabled or not config.compile_ae:
        return {"jax_warmup_batches": 0.0}
    if max_ae_batch_size <= 0 or max_prefix_slots <= 0:
        raise ValueError("max_ae_batch_size and max_prefix_slots must be positive")
    batches = planned_warmup_batches(
        max_batch_size=max_prefix_slots,
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
            prefix_batch = model.build_prefix_feature(None, obs)
            _block_until_ready(prefix_batch)
            noise = noise_factory(batch_size)
            denoise_state = runtime_aligned_denoise_state(
                batch_size=batch_size,
                noise=noise,
                num_steps=config.num_steps,
            )
            v_t = model.denoise_one_batch(prefix_batch, denoise_state)
            _block_until_ready(v_t)
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


def warmup_ae_denoise_on_mapped_slabs(
    *,
    model: Any,
    noise_factory,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    config: JaxCompileConfig,
    make_prefix_batch,
) -> dict[str, float]:
    """Warm denoise_one_batch using production mapped-slab prefix slices.

    ``make_prefix_batch(slot_ids)`` should return the same prefix views AE uses
    in :meth:`JaxAEWorker.step_once` (IPC-opened or local mapped slabs).
    """
    if not config.warmup_enabled or not config.compile_ae:
        return {"jax_warmup_batches": 0.0}
    if max_ae_batch_size <= 0 or max_prefix_slots <= 0:
        raise ValueError("max_ae_batch_size and max_prefix_slots must be positive")
    batches = planned_warmup_batches(
        max_batch_size=max_prefix_slots,
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    if not config.enabled:
        batches = ((1, 1),)
    warmed = 0
    # Match AE step_once's dense state layout: rows live in one batched device array
    # across denoise steps instead of being rebuilt from per-request slices.
    warmup_steps = min(max(config.num_steps, 2), 5)
    for batch_size, repeats in batches:
        if batch_size > max_prefix_slots:
            continue
        slot_ids = tuple(range(batch_size))
        for _ in range(repeats):
            prefix_batch = make_prefix_batch(slot_ids)
            x_t = noise_factory(batch_size)
            dt = jnp.full((batch_size,), -1.0 / float(config.num_steps), dtype=jnp.float32)
            for step in range(warmup_steps):
                step_idx = jnp.full((batch_size,), step, dtype=jnp.int32)
                denoise_state = JaxDenoiseState(
                    x_t=x_t, step_idx=step_idx, num_steps=config.num_steps, dt=dt
                )
                v_t = model.denoise_one_batch(prefix_batch, denoise_state)
                dt_b = dt.reshape((-1,) + (1,) * (v_t.ndim - 1))
                x_t = x_t + dt_b * v_t
                _block_until_ready(x_t)
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}


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
    backend: DeviceSlabBackend | None = None,
) -> dict[str, float]:
    """Warm both VLM and AE paths in-process (tests / single-process helpers)."""
    config = JaxCompileConfig(
        enabled=config.enabled,
        warmup_enabled=config.warmup_enabled,
        compile_ae=config.compile_ae,
        compile_vlm=config.compile_vlm,
        warmup_max_batch_size=config.warmup_max_batch_size,
        num_steps=num_steps,
    )
    vlm_stats = warmup_vlm_prefix_model(
        model=model,
        observation_factory=observation_factory,
        max_vlm_batch_size=max_vlm_batch_size,
        config=config,
    )
    ae_stats = warmup_ae_denoise_model(
        model=model,
        observation_factory=observation_factory,
        noise_factory=noise_factory,
        max_ae_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
        config=config,
        backend=backend,
    )
    return {
        "jax_warmup_batches": float(vlm_stats["jax_warmup_batches"] + ae_stats["jax_warmup_batches"]),
    }


def warmup_monolithic_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_batch_size: int,
    num_steps: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    """Warm baseline via two small graphs (VLM prefix + AE denoise), not one ``sample_actions`` graph."""
    return warmup_split_model(
        model=model,
        observation_factory=observation_factory,
        noise_factory=noise_factory,
        max_vlm_batch_size=max_batch_size,
        max_ae_batch_size=max_batch_size,
        max_prefix_slots=max_batch_size,
        num_steps=num_steps,
        config=config,
    )


def _build_slab_prefix_batch(
    *,
    model: Any,
    observation_factory,
    batch_size: int,
    max_prefix_slots: int,
    backend: DeviceSlabBackend,
) -> JaxPrefixFeature:
    obs = observation_factory(batch_size)
    feature = model.build_prefix_feature(None, obs)
    _block_until_ready(feature)
    pool = JaxVlmPrefixCacheLanePool(max_lanes=max_prefix_slots, backend=backend)
    for row in range(batch_size):
        pool.put_lane(f"warmup-{batch_size}-{row}", _prefix_feature_row_view(feature, row))
    return pool.view_prefix_batch(batch_size)


def _prefix_feature_row_view(feature: JaxPrefixFeature, row: int) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=_row_view_tree(feature.past_key_values, row, axis=1),
        prefix_pad_masks=feature.prefix_pad_masks[row : row + 1],
        state=feature.state[row : row + 1] if feature.state is not None else None,
    )


def _row_view_tree(value: Any, row: int, *, axis: int) -> Any:
    if isinstance(value, jax.Array):
        return jax.lax.dynamic_slice_in_dim(value, row, 1, axis=axis)
    if isinstance(value, tuple):
        return tuple(_row_view_tree(item, row, axis=axis) for item in value)
    if isinstance(value, list):
        return [_row_view_tree(item, row, axis=axis) for item in value]
    if isinstance(value, dict):
        return {key: _row_view_tree(item, row, axis=axis) for key, item in value.items()}
    return value


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
