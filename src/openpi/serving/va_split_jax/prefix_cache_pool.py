from __future__ import annotations

from typing import Any

import jax

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import DeviceSlab
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec


class JaxVlmPrefixCacheLanePool:
    def __init__(self, *, max_lanes: int, backend: DeviceSlabBackend):
        if max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        self.max_lanes = max_lanes
        self._backend = backend
        self._past_slabs: Any | None = None
        self._prefix_pad_masks: DeviceSlab | None = None
        self._state: DeviceSlab | None = None
        self._has_state: bool | None = None
        self._request_to_lane: dict[str, int] = {}
        self._lane_to_request: list[str | None] = [None for _ in range(max_lanes)]
        self._active_count = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    def put_lane(self, request_id: str, feature: JaxPrefixFeature) -> int:
        if request_id in self._request_to_lane:
            raise ValueError(f"request_id {request_id!r} already exists in prefix lane pool")
        if self._active_count >= self.max_lanes:
            raise RuntimeError(f"VLM prefix lane pool is full ({self.max_lanes} active requests)")
        _validate_single_row_feature(feature)
        lane_id = self._active_count
        self._ensure_initialized(feature)
        self._past_slabs = _copy_tree_lane(self._backend, self._past_slabs, feature.past_key_values, lane_id)
        assert self._prefix_pad_masks is not None
        self._prefix_pad_masks = self._backend.copy_lane_from_array(
            self._prefix_pad_masks, lane_id, feature.prefix_pad_masks
        )
        if self._has_state:
            assert self._state is not None
            assert feature.state is not None
            self._state = self._backend.copy_lane_from_array(self._state, lane_id, feature.state)
        self._request_to_lane[request_id] = lane_id
        self._lane_to_request[lane_id] = request_id
        self._active_count += 1
        return lane_id

    def release_lane(self, request_id: str) -> tuple[int, int, str] | None:
        lane_id = self._request_to_lane.pop(request_id, None)
        if lane_id is None:
            return None
        last_lane = self._active_count - 1
        moved_request_id = self._lane_to_request[last_lane]
        self._lane_to_request[lane_id] = None
        if lane_id != last_lane:
            if moved_request_id is None:
                raise RuntimeError(f"Cannot compact empty VLM lane {last_lane}")
            self._move_lane(last_lane, lane_id)
            self._lane_to_request[lane_id] = moved_request_id
            self._request_to_lane[moved_request_id] = lane_id
        self._lane_to_request[last_lane] = None
        self._active_count -= 1
        if lane_id != last_lane and moved_request_id is not None:
            return last_lane, lane_id, moved_request_id
        return None

    def export_batch_view(self, request_ids: tuple[str, ...]) -> JaxPrefixFeature:
        if not request_ids:
            raise ValueError("request_ids must be non-empty")
        lane_ids = tuple(self._request_to_lane[request_id] for request_id in request_ids)
        if lane_ids != tuple(range(len(request_ids))):
            raise ValueError(
                "The first implementation exports only the dense prefix of the VLM lane pool; "
                "AE batch selection must use the request ids currently occupying lanes [0, batch_size)."
            )
        return self.view_prefix_batch(len(request_ids))

    def export_slab_handle_tree(self) -> dict[str, Any]:
        if self._past_slabs is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot export prefix slab handles before initialization")
        return _export_slab_handle_tree(self._past_slabs, self._prefix_pad_masks, self._state)

    def local_slab_tree(self) -> dict[str, Any]:
        if self._past_slabs is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot access prefix slab tree before initialization")
        return {
            "past_key_values": self._past_slabs,
            "prefix_pad_masks": self._prefix_pad_masks,
            "state": self._state,
        }

    def view_prefix_batch(self, batch_size: int) -> JaxPrefixFeature:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._active_count:
            raise ValueError(f"batch_size {batch_size} exceeds active lanes {self._active_count}")
        if self._past_slabs is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot view prefix batch before initialization")
        return JaxPrefixFeature(
            past_key_values=_view_tree_batch(self._backend, self._past_slabs, batch_size),
            prefix_pad_masks=self._backend.view_batch(self._prefix_pad_masks, batch_size),
            state=self._backend.view_batch(self._state, batch_size) if self._state is not None else None,
        )

    def _move_lane(self, src_lane: int, dst_lane: int) -> None:
        batch = self.view_prefix_batch(src_lane + 1)
        row = _row_view_tree(batch.past_key_values, src_lane, axis=1)
        assert self._prefix_pad_masks is not None
        mask_row = self._backend.view_batch(self._prefix_pad_masks, src_lane + 1)[src_lane : src_lane + 1]
        state_row = None
        if self._state is not None:
            state_row = self._backend.view_batch(self._state, src_lane + 1)[src_lane : src_lane + 1]
        assert self._past_slabs is not None
        self._past_slabs = _copy_tree_lane(self._backend, self._past_slabs, row, dst_lane)
        self._prefix_pad_masks = self._backend.copy_lane_from_array(self._prefix_pad_masks, dst_lane, mask_row)
        if self._state is not None:
            assert state_row is not None
            self._state = self._backend.copy_lane_from_array(self._state, dst_lane, state_row)

    def _ensure_initialized(self, feature: JaxPrefixFeature) -> None:
        if self._past_slabs is None:
            self._past_slabs = _make_tree_slabs(
                self._backend, "past", feature.past_key_values, self.max_lanes, lane_axis=1
            )
        else:
            _validate_tree_compatible(self._past_slabs, feature.past_key_values)
        if self._prefix_pad_masks is None:
            self._prefix_pad_masks = _make_slab(
                self._backend, "prefix_pad_masks", feature.prefix_pad_masks, self.max_lanes
            )
        else:
            _validate_slab_compatible(self._prefix_pad_masks, feature.prefix_pad_masks)
        has_state = feature.state is not None
        if self._has_state is None:
            self._has_state = has_state
            if has_state:
                assert feature.state is not None
                self._state = _make_slab(self._backend, "state", feature.state, self.max_lanes)
        elif self._has_state != has_state:
            raise ValueError("Cannot mix prefix features with and without state")
        if has_state:
            assert self._state is not None
            assert feature.state is not None
            _validate_slab_compatible(self._state, feature.state)


def _make_slab(
    backend: DeviceSlabBackend,
    name: str,
    value: jax.Array,
    max_lanes: int,
    *,
    lane_axis: int = 0,
) -> DeviceSlab:
    return backend.create_slab(
        DeviceSlabSpec(
            name=name,
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            max_lanes=max_lanes,
            lane_axis=lane_axis,
        )
    )


def _make_tree_slabs(
    backend: DeviceSlabBackend,
    prefix: str,
    value: Any,
    max_lanes: int,
    *,
    lane_axis: int,
) -> Any:
    if isinstance(value, jax.Array):
        return _make_slab(backend, prefix, value, max_lanes, lane_axis=lane_axis)
    if isinstance(value, tuple):
        return tuple(
            _make_tree_slabs(backend, f"{prefix}.{idx}", item, max_lanes, lane_axis=lane_axis)
            for idx, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _make_tree_slabs(backend, f"{prefix}.{idx}", item, max_lanes, lane_axis=lane_axis)
            for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _make_tree_slabs(backend, f"{prefix}.{key}", item, max_lanes, lane_axis=lane_axis)
            for key, item in value.items()
        }
    raise TypeError(f"Unsupported JAX prefix tree node: {type(value)}")


def _copy_tree_lane(backend: DeviceSlabBackend, slabs: Any, value: Any, lane_id: int) -> Any:
    if isinstance(slabs, DeviceSlab):
        return backend.copy_lane_from_array(slabs, lane_id, value)
    if isinstance(slabs, tuple):
        return tuple(_copy_tree_lane(backend, slab, item, lane_id) for slab, item in zip(slabs, value, strict=True))
    if isinstance(slabs, list):
        return [_copy_tree_lane(backend, slab, item, lane_id) for slab, item in zip(slabs, value, strict=True)]
    if isinstance(slabs, dict):
        return {key: _copy_tree_lane(backend, slabs[key], value[key], lane_id) for key in slabs}
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _view_tree_batch(backend: DeviceSlabBackend, slabs: Any, batch_size: int) -> Any:
    if isinstance(slabs, DeviceSlab):
        return backend.view_batch(slabs, batch_size)
    if isinstance(slabs, tuple):
        return tuple(_view_tree_batch(backend, item, batch_size) for item in slabs)
    if isinstance(slabs, list):
        return [_view_tree_batch(backend, item, batch_size) for item in slabs]
    if isinstance(slabs, dict):
        return {key: _view_tree_batch(backend, item, batch_size) for key, item in slabs.items()}
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _export_slab_handle_tree(
    past_slabs: Any,
    prefix_pad_masks: DeviceSlab | None,
    state: DeviceSlab | None,
) -> dict[str, Any]:
    if prefix_pad_masks is None:
        raise RuntimeError("Cannot export prefix slab handles before initialization")
    return {
        "past_key_values": _export_slab_tree_handles(past_slabs),
        "prefix_pad_masks": prefix_pad_masks.handle,
        "state": state.handle if state is not None else None,
    }


def _export_slab_tree_handles(slabs: Any) -> Any:
    if isinstance(slabs, DeviceSlab):
        return slabs.handle
    if isinstance(slabs, tuple):
        return tuple(_export_slab_tree_handles(item) for item in slabs)
    if isinstance(slabs, list):
        return [_export_slab_tree_handles(item) for item in slabs]
    if isinstance(slabs, dict):
        return {key: _export_slab_tree_handles(item) for key, item in slabs.items()}
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _row_view_tree(value: Any, row: int, *, axis: int) -> Any:
    if isinstance(value, jax.Array):
        return jax.lax.dynamic_slice_in_dim(value, row, 1, axis=axis)
    if isinstance(value, tuple):
        return tuple(_row_view_tree(item, row, axis=axis) for item in value)
    if isinstance(value, list):
        return [_row_view_tree(item, row, axis=axis) for item in value]
    if isinstance(value, dict):
        return {key: _row_view_tree(item, row, axis=axis) for key, item in value.items()}
    raise TypeError(f"Unsupported JAX prefix tree node: {type(value)}")


def _validate_single_row_feature(feature: JaxPrefixFeature) -> None:
    _validate_single_row_tree(feature.past_key_values)
    if feature.prefix_pad_masks.shape[0] != 1:
        raise ValueError(f"prefix_pad_masks must have batch size 1, got {feature.prefix_pad_masks.shape}")
    if feature.state is not None and feature.state.shape[0] != 1:
        raise ValueError(f"state must have batch size 1, got {feature.state.shape}")


def _validate_single_row_tree(value: Any) -> None:
    if isinstance(value, jax.Array):
        if value.ndim < 2 or value.shape[1] != 1:
            raise ValueError(f"prefix tree array must have batch size 1 on axis 1, got {value.shape}")
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_single_row_tree(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_single_row_tree(item)
        return
    raise TypeError(f"Unsupported JAX prefix tree node: {type(value)}")


def _validate_tree_compatible(slabs: Any, value: Any) -> None:
    if isinstance(slabs, DeviceSlab):
        _validate_slab_compatible(slabs, value)
        return
    if isinstance(slabs, tuple):
        if not isinstance(value, tuple) or len(slabs) != len(value):
            raise ValueError("prefix tuple tree structure changed")
        for slab, item in zip(slabs, value, strict=True):
            _validate_tree_compatible(slab, item)
        return
    if isinstance(slabs, list):
        if not isinstance(value, list) or len(slabs) != len(value):
            raise ValueError("prefix list tree structure changed")
        for slab, item in zip(slabs, value, strict=True):
            _validate_tree_compatible(slab, item)
        return
    if isinstance(slabs, dict):
        if not isinstance(value, dict) or slabs.keys() != value.keys():
            raise ValueError("prefix dict tree structure changed")
        for key in slabs:
            _validate_tree_compatible(slabs[key], value[key])
        return
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _validate_slab_compatible(slab: DeviceSlab, value: jax.Array) -> None:
    if not isinstance(value, jax.Array):
        raise TypeError(f"expected jax.Array, got {type(value)}")
    if tuple(value.shape) != tuple(slab.spec.shape):
        raise ValueError(f"feature shape changed from {slab.spec.shape} to {tuple(value.shape)}")
    if str(value.dtype) != slab.spec.dtype:
        raise ValueError(f"feature dtype changed from {slab.spec.dtype} to {value.dtype}")
