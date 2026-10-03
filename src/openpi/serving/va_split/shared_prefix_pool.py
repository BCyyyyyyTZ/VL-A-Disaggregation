"""AE-owned CUDA IPC prefix lane pool for PyTorch VA-split.

Mirrors the realtime-vla / JAX design:
- AE allocates dense lane slabs once and exports them via CUDA IPC.
- VLM attaches once, then writes prefix rows in-place.
- Control-plane Queue messages carry ``lane_id`` metadata, not KV tensors.

This replaces the previous per-request ``PrefixFeature`` CUDA-IPC open cost
(~tens of ms) with a one-time attach plus a GPU→GPU ``copy_`` (~sub-ms, matching
JAX device-slab transfer p50 ≈ 0.3 ms).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.prefix_cache_pool import PrefixCacheLanePool


@dataclass(frozen=True, slots=True)
class PrefixPoolReady:
    """One-shot bootstrap: base CUDA slabs (not DynamicCache copies) + events."""

    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
    prefix_pad_masks: torch.Tensor
    state: torch.Tensor | None
    max_lanes: int
    ready_event_handles: tuple[bytes, ...]
    free_lane_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LaneCredits:
    lane_ids: tuple[int, ...]


class SharedPrefixLanePool:
    """Cross-process view over an AE-owned :class:`PrefixCacheLanePool`."""

    def __init__(
        self,
        *,
        pool: PrefixCacheLanePool,
        ready_events: list[torch.cuda.Event],
        owner: bool,
        free_lane_ids: Sequence[int] | None = None,
    ):
        self.pool = pool
        self.ready_events = ready_events
        self.owner = owner
        self._free_credits: deque[int] = deque(int(x) for x in (free_lane_ids or ()))
        self._request_to_lane: dict[str, int] = {}

    @property
    def max_lanes(self) -> int:
        return self.pool.max_lanes

    @property
    def credit_count(self) -> int:
        return len(self._free_credits)

    def grant_credits(self, lane_ids: Sequence[int]) -> None:
        for lane_id in lane_ids:
            self._validate_lane_id(int(lane_id))
            self._free_credits.append(int(lane_id))

    def acquire_credit(self) -> int | None:
        if not self._free_credits:
            return None
        return self._free_credits.popleft()

    def release_credit(self, lane_id: int) -> None:
        self._validate_lane_id(lane_id)
        self._free_credits.append(int(lane_id))

    def bind_request(self, request_id: str, lane_id: int) -> None:
        self._request_to_lane[request_id] = int(lane_id)

    def take_request_lane(self, request_id: str) -> int | None:
        return self._request_to_lane.pop(request_id, None)

    def write_feature(self, lane_id: int, feature: PrefixFeature, *, record_ready: bool = True) -> None:
        self.pool.put_lane(lane_id, feature)
        if record_ready:
            self.record_ready((lane_id,))

    def record_ready(self, lane_ids: Sequence[int]) -> None:
        if not self.ready_events:
            return
        stream = torch.cuda.current_stream()
        for lane_id in lane_ids:
            self.ready_events[int(lane_id)].record(stream)

    def wait_ready_on_stream(self, lane_ids: Sequence[int]) -> None:
        """Device-side wait: AE stream stalls until VLM's write is visible.

        Does not drain the host (unlike ``Event.wait`` / ``cuda.synchronize``).
        """
        if not self.ready_events:
            return
        stream = torch.cuda.current_stream()
        for lane_id in lane_ids:
            stream.wait_event(self.ready_events[int(lane_id)])

    def view_lane_feature(self, lane_id: int) -> PrefixFeature:
        """Return a single-lane PrefixFeature view (batch dim = 1)."""
        batch = self.pool.view_prefix_batch(lane_id + 1)
        return PrefixFeature(
            past_key_values=_narrow_past_to_lane(batch.past_key_values, lane_id),
            prefix_pad_masks=batch.prefix_pad_masks.narrow(0, lane_id, 1),
            state=None if batch.state is None else batch.state.narrow(0, lane_id, 1),
        )

    def export_ready(self) -> PrefixPoolReady:
        if not self.owner:
            raise RuntimeError("only the owning AE process can export PrefixPoolReady")
        if self.pool._past_pool is None or self.pool._prefix_pad_masks is None:
            raise RuntimeError("cannot export an uninitialized prefix pool")
        slabs = self.pool.export_layer_slabs()
        return PrefixPoolReady(
            layer_keys=tuple(key for key, _value in slabs),
            layer_values=tuple(value for _key, value in slabs),
            prefix_pad_masks=self.pool._prefix_pad_masks,
            state=self.pool._state_pool,
            max_lanes=self.max_lanes,
            ready_event_handles=tuple(event.ipc_handle() for event in self.ready_events),
            free_lane_ids=tuple(self._free_credits),
        )

    @classmethod
    def create_owned_from_feature(cls, feature: PrefixFeature, *, max_lanes: int) -> SharedPrefixLanePool:
        pool = PrefixCacheLanePool(max_lanes=max_lanes)
        # Initialize slab shapes from the template feature without consuming lane 0 state.
        pool.put_lane(0, feature)
        events = _create_ipc_events(max_lanes, device=_feature_device(feature))
        shared = cls(pool=pool, ready_events=events, owner=True, free_lane_ids=range(max_lanes))
        return shared

    @classmethod
    def attach_shared(cls, ready: PrefixPoolReady) -> SharedPrefixLanePool:
        pool = PrefixCacheLanePool(max_lanes=ready.max_lanes)
        # Wrap the exported base storages. Do not rebuild via DynamicCache.update,
        # which would allocate a private copy and break in-place VLM writes.
        pool._past_pool = _DynamicCachePoolFromShared(ready.layer_keys, ready.layer_values, ready.max_lanes)
        pool._prefix_pad_masks = ready.prefix_pad_masks
        pool._state_pool = ready.state
        pool._has_state = ready.state is not None
        device = ready.prefix_pad_masks.device
        events = _attach_ipc_events(ready.ready_event_handles, device=device)
        return cls(
            pool=pool,
            ready_events=events,
            owner=False,
            free_lane_ids=ready.free_lane_ids,
        )

    def _validate_lane_id(self, lane_id: int) -> None:
        if lane_id < 0 or lane_id >= self.max_lanes:
            raise ValueError(f"lane_id {lane_id} outside shared prefix pool capacity {self.max_lanes}")


class _DynamicCachePoolFromShared:
    """Past-pool adapter that reuses already-shared layer slabs."""

    def __init__(self, layer_keys: Sequence[torch.Tensor], layer_values: Sequence[torch.Tensor], max_lanes: int):
        if len(layer_keys) == 0 or len(layer_keys) != len(layer_values):
            raise ValueError("Shared prefix slab must contain one key/value pair per layer")
        self._layers = []
        for key, value in zip(layer_keys, layer_values, strict=True):
            if int(key.shape[0]) != max_lanes or int(value.shape[0]) != max_lanes:
                raise ValueError(
                    f"Shared DynamicCache lane dim {key.shape[0]} does not match max_lanes={max_lanes}"
                )
            self._layers.append(_SharedLayer(key, value))

    def put_lane(self, lane_id: int, value: Any) -> None:
        if not isinstance(value, DynamicCache):
            raise ValueError(f"Expected DynamicCache prefix payload, got {type(value)}")
        for layer, (key, cache_value) in zip(self._layers, ((value[i][0], value[i][1]) for i in range(len(value))), strict=True):
            layer.put_lane(lane_id, key, cache_value)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        for layer in self._layers:
            layer.move_lane(src_lane, dst_lane)

    def gather(self, index: torch.Tensor) -> DynamicCache:
        cache = DynamicCache()
        for layer_idx, layer in enumerate(self._layers):
            key, value = layer.gather(index)
            cache.update(key, value, layer_idx=layer_idx)
        return cache

    def view_batch(self, batch_size: int) -> DynamicCache:
        cache = DynamicCache()
        for layer_idx, layer in enumerate(self._layers):
            key, value = layer.view_batch(batch_size)
            cache.update(key, value, layer_idx=layer_idx)
        return cache

    def validate(self, value: Any) -> None:
        if not isinstance(value, DynamicCache):
            raise ValueError(f"Expected DynamicCache prefix payload, got {type(value)}")
        if len(value) != len(self._layers):
            raise ValueError(f"DynamicCache layer count changed from {len(self._layers)} to {len(value)}")


class _SharedLayer:
    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        self.key = key
        self.value = value

    def put_lane(self, lane_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.key.narrow(0, lane_id, 1).copy_(key, non_blocking=True)
        self.value.narrow(0, lane_id, 1).copy_(value, non_blocking=True)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        self.key.narrow(0, dst_lane, 1).copy_(self.key.narrow(0, src_lane, 1), non_blocking=True)
        self.value.narrow(0, dst_lane, 1).copy_(self.value.narrow(0, src_lane, 1), non_blocking=True)

    def gather(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key.index_select(0, index), self.value.index_select(0, index)

    def view_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key.narrow(0, 0, batch_size), self.value.narrow(0, 0, batch_size)


def _create_ipc_events(count: int, *, device: torch.device) -> list[torch.cuda.Event]:
    if device.type != "cuda":
        return []
    return [
        torch.cuda.Event(enable_timing=False, blocking=False, interprocess=True) for _ in range(count)
    ]


def _attach_ipc_events(handles: Sequence[bytes], *, device: torch.device) -> list[torch.cuda.Event]:
    if device.type != "cuda" or not handles:
        return []
    return [torch.cuda.Event.from_ipc_handle(device, handle) for handle in handles]


def _feature_device(feature: PrefixFeature) -> torch.device:
    return feature.prefix_pad_masks.device


def _narrow_past_to_lane(past: Any, lane_id: int) -> Any:
    if isinstance(past, DynamicCache):
        cache = DynamicCache()
        for layer_idx in range(len(past)):
            key, value = past[layer_idx]
            cache.update(key.narrow(0, lane_id, 1), value.narrow(0, lane_id, 1), layer_idx=layer_idx)
        return cache
    return past
