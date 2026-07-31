from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True, slots=True)
class JaxPrefixFeature:
    past_key_values: Any
    prefix_pad_masks: jax.Array
    state: jax.Array | None


@dataclass(frozen=True, slots=True)
class JaxDenoiseState:
    x_t: jax.Array
    step_idx: jax.Array
    num_steps: int | jax.Array
    dt: jax.Array


jax.tree_util.register_dataclass(
    JaxDenoiseState,
    data_fields=["x_t", "step_idx", "num_steps", "dt"],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class JaxPrefixSlotHandle:
    slot_id: int
    batch_rows: int
    prefix_shape_tree: Any
    prefix_dtype_tree: Any


@dataclass(frozen=True, slots=True)
class JaxPrefixBatchViewHandle:
    request_ids: tuple[str, ...]
    slot_ids: tuple[int, ...]
    batch_rows: int
    prefix_shape_tree: Any
    prefix_dtype_tree: Any
    view_handle_tree: Any
