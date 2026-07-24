from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_split_types import PrefixFeature


class PrefixCacheLanePool:
    """Preallocated prefix feature lanes for AE denoise batching.

    The pool is initialized lazily from the first real prefix feature because
    prefix length and KV tensor shapes come from the model/preprocessing path.
    After that, denoise steps consume `narrow` views over the dense lane window.
    """

    def __init__(self, *, max_lanes: int):
        if max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        self.max_lanes = max_lanes
        self._past_pool: _PastPool | None = None
        self._prefix_pad_masks: torch.Tensor | None = None
        self._state_pool: torch.Tensor | None = None
        self._has_state: bool | None = None

    def put_lane(self, lane_id: int, feature: PrefixFeature) -> None:
        self._validate_lane_id(lane_id)
        _validate_single_row_feature(feature)
        self._ensure_initialized(feature)
        assert self._past_pool is not None
        assert self._prefix_pad_masks is not None

        self._past_pool.put_lane(lane_id, feature.past_key_values)
        self._prefix_pad_masks.narrow(0, lane_id, 1).copy_(feature.prefix_pad_masks, non_blocking=True)
        if self._has_state:
            assert self._state_pool is not None
            assert feature.state is not None
            self._state_pool.narrow(0, lane_id, 1).copy_(feature.state, non_blocking=True)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        self._validate_lane_id(src_lane)
        self._validate_lane_id(dst_lane)
        if src_lane == dst_lane:
            return
        if self._past_pool is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot move lanes before the prefix pool is initialized")
        self._past_pool.move_lane(src_lane, dst_lane)
        self._prefix_pad_masks.narrow(0, dst_lane, 1).copy_(
            self._prefix_pad_masks.narrow(0, src_lane, 1),
            non_blocking=True,
        )
        if self._state_pool is not None:
            self._state_pool.narrow(0, dst_lane, 1).copy_(self._state_pool.narrow(0, src_lane, 1), non_blocking=True)

    def view_prefix_batch(self, batch_size: int) -> PrefixFeature:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.max_lanes:
            raise ValueError(f"batch_size {batch_size} exceeds max lanes {self.max_lanes}")
        if self._past_pool is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot view prefix batch before the prefix pool is initialized")
        return PrefixFeature(
            past_key_values=self._past_pool.view_batch(batch_size),
            prefix_pad_masks=self._prefix_pad_masks.narrow(0, 0, batch_size),
            state=self._state_pool.narrow(0, 0, batch_size) if self._state_pool is not None else None,
        )

    def _ensure_initialized(self, feature: PrefixFeature) -> None:
        if self._past_pool is None:
            self._past_pool = _make_past_pool(feature.past_key_values, self.max_lanes)
        else:
            self._past_pool.validate(feature.past_key_values)

        if self._prefix_pad_masks is None:
            self._prefix_pad_masks = _make_tensor_pool(feature.prefix_pad_masks, self.max_lanes)
        else:
            _validate_tensor_compatible(self._prefix_pad_masks, feature.prefix_pad_masks)

        has_state = feature.state is not None
        if self._has_state is None:
            self._has_state = has_state
            if has_state:
                assert feature.state is not None
                self._state_pool = _make_tensor_pool(feature.state, self.max_lanes)
        elif self._has_state != has_state:
            raise ValueError("Cannot mix prefix features with and without state in one lane pool")

        if has_state:
            assert self._state_pool is not None
            assert feature.state is not None
            _validate_tensor_compatible(self._state_pool, feature.state)

    def _validate_lane_id(self, lane_id: int) -> None:
        if lane_id < 0 or lane_id >= self.max_lanes:
            raise ValueError(f"lane_id {lane_id} outside prefix lane pool capacity {self.max_lanes}")


class _PastPool:
    def put_lane(self, lane_id: int, value: Any) -> None:
        raise NotImplementedError

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        raise NotImplementedError

    def view_batch(self, batch_size: int) -> Any:
        raise NotImplementedError

    def validate(self, value: Any) -> None:
        raise NotImplementedError


class _DynamicCachePool(_PastPool):
    def __init__(self, cache: DynamicCache, max_lanes: int):
        if len(cache) == 0:
            raise ValueError("Cannot initialize prefix lane pool from an empty DynamicCache")
        self._layers = [_LayerPool(key, value, max_lanes) for key, value in _iter_dynamic_cache(cache)]

    def put_lane(self, lane_id: int, value: Any) -> None:
        if not isinstance(value, DynamicCache):
            raise ValueError(f"Expected DynamicCache prefix payload, got {type(value)}")
        self.validate(value)
        for layer_pool, (key, cache_value) in zip(self._layers, _iter_dynamic_cache(value), strict=True):
            layer_pool.put_lane(lane_id, key, cache_value)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        for layer_pool in self._layers:
            layer_pool.move_lane(src_lane, dst_lane)

    def view_batch(self, batch_size: int) -> DynamicCache:
        cache = DynamicCache()
        for layer_idx, layer_pool in enumerate(self._layers):
            key, value = layer_pool.view_batch(batch_size)
            cache.update(key, value, layer_idx=layer_idx)
        return cache

    def validate(self, value: Any) -> None:
        if not isinstance(value, DynamicCache):
            raise ValueError(f"Expected DynamicCache prefix payload, got {type(value)}")
        layers = list(_iter_dynamic_cache(value))
        if len(layers) != len(self._layers):
            raise ValueError(f"DynamicCache layer count changed from {len(self._layers)} to {len(layers)}")
        for layer_pool, (key, cache_value) in zip(self._layers, layers, strict=True):
            layer_pool.validate(key, cache_value)


class _LayerPool:
    def __init__(self, key: torch.Tensor, value: torch.Tensor, max_lanes: int):
        _validate_single_row_tensor(key, name="cache key")
        _validate_single_row_tensor(value, name="cache value")
        self.key = _make_tensor_pool(key, max_lanes)
        self.value = _make_tensor_pool(value, max_lanes)

    def put_lane(self, lane_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self.validate(key, value)
        self.key.narrow(0, lane_id, 1).copy_(key, non_blocking=True)
        self.value.narrow(0, lane_id, 1).copy_(value, non_blocking=True)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        self.key.narrow(0, dst_lane, 1).copy_(self.key.narrow(0, src_lane, 1), non_blocking=True)
        self.value.narrow(0, dst_lane, 1).copy_(self.value.narrow(0, src_lane, 1), non_blocking=True)

    def view_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key.narrow(0, 0, batch_size), self.value.narrow(0, 0, batch_size)

    def validate(self, key: torch.Tensor, value: torch.Tensor) -> None:
        _validate_tensor_compatible(self.key, key, name="cache key")
        _validate_tensor_compatible(self.value, value, name="cache value")


class _TensorTreePool(_PastPool):
    def __init__(self, value: Any, max_lanes: int):
        self._tree = _make_tensor_tree_pool(value, max_lanes)

    def put_lane(self, lane_id: int, value: Any) -> None:
        _copy_tensor_tree_to_lane(self._tree, value, lane_id)

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        _move_tensor_tree_lane(self._tree, src_lane, dst_lane)

    def view_batch(self, batch_size: int) -> Any:
        return _view_tensor_tree_batch(self._tree, batch_size)

    def validate(self, value: Any) -> None:
        _validate_tensor_tree(self._tree, value)


class _FallbackPastPool(_PastPool):
    """Compatibility path for tests or non-tensor payloads.

    Real Pi0/Pi0.5 prefix caches use DynamicCache and therefore take the
    preallocated slab path above.
    """

    def __init__(self, value: Any, max_lanes: int):
        self._values = [None for _ in range(max_lanes)]
        self._value_type = type(value)

    def put_lane(self, lane_id: int, value: Any) -> None:
        self._values[lane_id] = value

    def move_lane(self, src_lane: int, dst_lane: int) -> None:
        self._values[dst_lane] = self._values[src_lane]

    def view_batch(self, batch_size: int) -> Any:
        values = self._values[:batch_size]
        return values[0] if values else None

    def validate(self, value: Any) -> None:
        if not isinstance(value, self._value_type):
            raise ValueError(f"Prefix payload type changed from {self._value_type} to {type(value)}")


def _make_past_pool(value: Any, max_lanes: int) -> _PastPool:
    if isinstance(value, DynamicCache):
        return _DynamicCachePool(value, max_lanes)
    if _is_tensor_tree(value):
        return _TensorTreePool(value, max_lanes)
    return _FallbackPastPool(value, max_lanes)


def _iter_dynamic_cache(cache: DynamicCache):
    for layer_idx in range(len(cache)):
        yield cache[layer_idx]


def _make_tensor_pool(tensor: torch.Tensor, max_lanes: int) -> torch.Tensor:
    _validate_single_row_tensor(tensor)
    return torch.empty(
        (max_lanes, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _validate_single_row_feature(feature: PrefixFeature) -> None:
    _validate_single_row_tensor(feature.prefix_pad_masks, name="prefix_pad_masks")
    if feature.state is not None:
        _validate_single_row_tensor(feature.state, name="state")


def _validate_single_row_tensor(tensor: torch.Tensor, *, name: str = "tensor") -> None:
    if tensor.ndim == 0 or int(tensor.shape[0]) != 1:
        raise ValueError(f"{name} must have a leading batch dimension of 1")


def _validate_tensor_compatible(pool: torch.Tensor, tensor: torch.Tensor, *, name: str = "tensor") -> None:
    _validate_single_row_tensor(tensor, name=name)
    if tuple(pool.shape[1:]) != tuple(tensor.shape[1:]):
        raise ValueError(f"{name} shape changed from {tuple(pool.shape[1:])} to {tuple(tensor.shape[1:])}")
    if pool.dtype != tensor.dtype:
        raise ValueError(f"{name} dtype changed from {pool.dtype} to {tensor.dtype}")
    if pool.device != tensor.device:
        raise ValueError(f"{name} device changed from {pool.device} to {tensor.device}")


def _is_tensor_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, tuple | list):
        return all(_is_tensor_tree(item) for item in value)
    return False


def _make_tensor_tree_pool(value: Any, max_lanes: int) -> Any:
    if torch.is_tensor(value):
        return _make_tensor_pool(value, max_lanes)
    if isinstance(value, tuple):
        return tuple(_make_tensor_tree_pool(item, max_lanes) for item in value)
    if isinstance(value, list):
        return [_make_tensor_tree_pool(item, max_lanes) for item in value]
    raise TypeError(f"Unsupported tensor tree node: {type(value)}")


def _copy_tensor_tree_to_lane(pool: Any, value: Any, lane_id: int) -> None:
    if torch.is_tensor(pool):
        _validate_tensor_compatible(pool, value)
        pool.narrow(0, lane_id, 1).copy_(value, non_blocking=True)
        return
    if isinstance(pool, tuple):
        for child_pool, child_value in zip(pool, value, strict=True):
            _copy_tensor_tree_to_lane(child_pool, child_value, lane_id)
        return
    if isinstance(pool, list):
        for child_pool, child_value in zip(pool, value, strict=True):
            _copy_tensor_tree_to_lane(child_pool, child_value, lane_id)
        return
    raise TypeError(f"Unsupported tensor tree pool node: {type(pool)}")


def _move_tensor_tree_lane(pool: Any, src_lane: int, dst_lane: int) -> None:
    if torch.is_tensor(pool):
        pool.narrow(0, dst_lane, 1).copy_(pool.narrow(0, src_lane, 1), non_blocking=True)
        return
    if isinstance(pool, tuple | list):
        for child_pool in pool:
            _move_tensor_tree_lane(child_pool, src_lane, dst_lane)
        return
    raise TypeError(f"Unsupported tensor tree pool node: {type(pool)}")


def _view_tensor_tree_batch(pool: Any, batch_size: int) -> Any:
    if torch.is_tensor(pool):
        return pool.narrow(0, 0, batch_size)
    if isinstance(pool, tuple):
        return tuple(_view_tensor_tree_batch(child_pool, batch_size) for child_pool in pool)
    if isinstance(pool, list):
        return [_view_tensor_tree_batch(child_pool, batch_size) for child_pool in pool]
    raise TypeError(f"Unsupported tensor tree pool node: {type(pool)}")


def _validate_tensor_tree(pool: Any, value: Any) -> None:
    if torch.is_tensor(pool):
        _validate_tensor_compatible(pool, value)
        return
    if isinstance(pool, tuple):
        if not isinstance(value, tuple) or len(pool) != len(value):
            raise ValueError("Prefix tensor tuple structure changed")
        for child_pool, child_value in zip(pool, value, strict=True):
            _validate_tensor_tree(child_pool, child_value)
        return
    if isinstance(pool, list):
        if not isinstance(value, list) or len(pool) != len(value):
            raise ValueError("Prefix tensor list structure changed")
        for child_pool, child_value in zip(pool, value, strict=True):
            _validate_tensor_tree(child_pool, child_value)
        return
    raise TypeError(f"Unsupported tensor tree pool node: {type(pool)}")
