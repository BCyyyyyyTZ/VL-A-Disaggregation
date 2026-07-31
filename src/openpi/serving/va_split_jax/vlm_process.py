from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import replace
import queue
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.models.jax_split_types import JaxPrefixSlabHandleTree
from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxBatchRequestEnvelope
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxSlotMoved
from openpi.serving.va_split_jax.types import JaxWorkerError


class JaxVLMWorker:
    """Builds JAX prefix features and stores rows in the VLM-owned device lane pool."""

    def __init__(self, *, model: Any, max_live_features: int, prefix_pool: JaxVlmPrefixCacheLanePool):
        if max_live_features <= 0:
            raise ValueError("max_live_features must be positive")
        self._model = model
        self._max_live_features = max_live_features
        self._prefix_pool = prefix_pool

    @property
    def available_live_feature_slots(self) -> int:
        return self._max_live_features - self._prefix_pool.active_count

    @property
    def max_live_features(self) -> int:
        return self._max_live_features

    @property
    def active_count(self) -> int:
        return self._prefix_pool.active_count

    def has_live_feature_capacity(self, batch_size: int) -> bool:
        return batch_size <= self.available_live_feature_slots

    def export_slab_handle_tree(self) -> JaxPrefixSlabHandleTree:
        slab_tree = self._prefix_pool.export_slab_handle_tree()
        return JaxPrefixSlabHandleTree(
            max_lanes=self._prefix_pool.max_lanes,
            prefix_shape_tree=_shape_tree_from_slab_handles(slab_tree),
            prefix_dtype_tree=_dtype_tree_from_slab_handles(slab_tree),
            slab_handle_tree=slab_tree,
        )

    def handle_request(self, request: JaxRequestEnvelope) -> JaxPrefixReady:
        return self.handle_batch([request])[0]

    def handle_batch(self, requests: list[JaxRequestEnvelope]) -> list[JaxPrefixReady]:
        if not requests:
            raise ValueError("JaxVLMWorker.handle_batch requires at least one request")
        if not self.has_live_feature_capacity(len(requests)):
            raise RuntimeError(f"VLM prefix lane pool is full ({self._prefix_pool.active_count} active requests)")
        request_ids = tuple(request.request_id for request in requests)
        sample_kwargs = _stack_request_sample_kwargs(requests)
        observation = _model.Observation.from_dict(_to_jax_tree(_stack_request_observations([r.observation for r in requests])))
        return self._handle_batched_observation(
            request_ids=request_ids,
            observation=observation,
            sample_kwargs=sample_kwargs,
            enqueue_ns_by_row=tuple(request.enqueue_ns for request in requests),
            dequeue_ns_by_row=tuple(request.dequeue_ns for request in requests),
            dequeue_start_ns_by_row=tuple(request.dequeue_start_ns for request in requests),
        )

    def handle_batch_request(self, request: JaxBatchRequestEnvelope) -> list[JaxPrefixReady]:
        if not self.has_live_feature_capacity(len(request.request_ids)):
            raise RuntimeError(f"VLM prefix lane pool is full ({self._prefix_pool.active_count} active requests)")
        observation = _model.Observation.from_dict(_to_jax_tree(request.observation))
        return self._handle_batched_observation(
            request_ids=request.request_ids,
            observation=observation,
            sample_kwargs=dict(request.sample_kwargs),
            enqueue_ns_by_row=tuple(request.enqueue_ns for _ in request.request_ids),
            dequeue_ns_by_row=tuple(request.dequeue_ns for _ in request.request_ids),
            dequeue_start_ns_by_row=tuple(request.dequeue_start_ns for _ in request.request_ids),
        )

    def release(self, release: JaxReleaseFeature) -> JaxSlotMoved | None:
        moved = self._prefix_pool.release_lane(release.request_id)
        if moved is None:
            return None
        old_slot_id, new_slot_id, moved_request_id = moved
        return JaxSlotMoved(request_id=moved_request_id, old_slot_id=old_slot_id, new_slot_id=new_slot_id)

    def _handle_batched_observation(
        self,
        *,
        request_ids: tuple[str, ...],
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        enqueue_ns_by_row: tuple[int, ...],
        dequeue_ns_by_row: tuple[int | None, ...],
        dequeue_start_ns_by_row: tuple[int | None, ...],
    ) -> list[JaxPrefixReady]:
        if (
            len(enqueue_ns_by_row) != len(request_ids)
            or len(dequeue_ns_by_row) != len(request_ids)
            or len(dequeue_start_ns_by_row) != len(request_ids)
        ):
            raise ValueError("enqueue/dequeue timing must have one entry per request id")
        start_ns = time.monotonic_ns()
        feature = self._model.build_prefix_feature(None, observation)
        _block_prefix_feature(feature)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        batch_size = int(feature.prefix_pad_masks.shape[0])
        if batch_size != len(request_ids):
            raise RuntimeError(f"VLM prefix batch size {batch_size} does not match {len(request_ids)} request ids")

        num_steps = int(sample_kwargs.get("num_steps", 10))
        ready: list[JaxPrefixReady] = []
        for row, request_id in enumerate(request_ids):
            row_feature = _prefix_feature_row_view(feature, row)
            slot_id = self._prefix_pool.put_lane(request_id, row_feature)
            enqueue_ns = enqueue_ns_by_row[row]
            dequeue_ns = dequeue_ns_by_row[row] or enqueue_ns
            queue_wait_ms, transfer_ms = _vlm_request_queue_timings(
                enqueue_ns=enqueue_ns,
                dequeue_start_ns=dequeue_start_ns_by_row[row],
                dequeue_ns=dequeue_ns,
            )
            row_kwargs = _sample_kwargs_for_row(sample_kwargs, row, batch_size)
            ready.append(
                JaxPrefixReady(
                    request_id=request_id,
                    slot_handle=JaxPrefixSlotHandle(
                        slot_id=slot_id,
                        batch_rows=1,
                        prefix_shape_tree=_prefix_feature_shape_tree(row_feature),
                        prefix_dtype_tree=_prefix_feature_dtype_tree(row_feature),
                    ),
                    num_steps=num_steps,
                    sample_kwargs=row_kwargs,
                    timing={
                        "vlm_prefix_forward_ms": elapsed_ms,
                        "vlm_effective_batch": float(batch_size),
                        "vlm_request_queue_wait_ms": queue_wait_ms,
                        "vlm_request_transfer_ms": transfer_ms,
                        "vlm_queue_wait_ms": max(0.0, (start_ns - dequeue_ns) / 1_000_000),
                    },
                )
            )
        return ready


class JaxVLMProcess:
    def __init__(
        self,
        *,
        model: Any,
        request_queue,
        prefix_queue,
        release_queue,
        prefix_pool: JaxVlmPrefixCacheLanePool,
        max_batch_size: int = 8,
        max_wait_ms: float = 2.0,
        max_live_features: int | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        live_features = max_live_features if max_live_features is not None else prefix_pool.max_lanes
        self.worker = JaxVLMWorker(model=model, max_live_features=live_features, prefix_pool=prefix_pool)
        self._request_queue = request_queue
        self._prefix_queue = prefix_queue
        self._release_queue = release_queue
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._backlog: deque[Any] = deque()
        self._sent_slab_ready = False

    def run(self) -> None:
        while True:
            self._drain_releases()
            self._prefetch_request_backlog(max_messages=self._max_batch_size)
            try:
                message = self._next_request_message(timeout=0.01)
            except queue.Empty:
                continue
            if isinstance(message, JaxShutdown):
                self._prefix_queue.put(message)
                return
            if isinstance(message, JaxBatchRequestEnvelope):
                if not self._defer_until_live_feature_capacity(message, len(message.request_ids)):
                    continue
                try:
                    self._put_prefix_batch(self.worker.handle_batch_request(message))
                except Exception as exc:  # pragma: no cover
                    for request_id in message.request_ids:
                        self._prefix_queue.put(
                            JaxWorkerError(request_id=request_id, error=str(exc), traceback=traceback.format_exc())
                        )
                continue
            if not isinstance(message, JaxRequestEnvelope):
                self._prefix_queue.put(JaxWorkerError(request_id=None, error=f"Unexpected VLM message: {type(message)}"))
                continue
            if not self._defer_until_live_feature_capacity(message, 1):
                continue
            requests = [message]
            shutdown_after_batch = False
            try:
                requests, shutdown_after_batch = self._collect_fcfs_batch(message)
                self._put_prefix_batch(self.worker.handle_batch(requests))
            except Exception as exc:  # pragma: no cover
                for request in requests:
                    self._prefix_queue.put(
                        JaxWorkerError(request_id=request.request_id, error=str(exc), traceback=traceback.format_exc())
                    )
            if shutdown_after_batch:
                self._prefix_queue.put(JaxShutdown())
                return

    def _drain_releases(self) -> None:
        while True:
            try:
                message = self._release_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, JaxReleaseFeature):
                moved = self.worker.release(message)
                if moved is not None:
                    self._prefix_queue.put(moved)

    def _put_prefix_batch(self, ready_messages: list[JaxPrefixReady]) -> None:
        if ready_messages and not self._sent_slab_ready:
            self._prefix_queue.put(
                JaxPrefixSlabReady(
                    slab=self.worker.export_slab_handle_tree(),
                    timing={"_prefix_enqueue_ns": float(time.monotonic_ns())},
                )
            )
            self._sent_slab_ready = True
        for ready in ready_messages:
            timing = dict(ready.timing or {})
            timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
            self._prefix_queue.put(replace(ready, timing=timing))

    def _collect_fcfs_batch(self, first_request: JaxRequestEnvelope) -> tuple[list[JaxRequestEnvelope], bool]:
        requests = [first_request]
        compatibility_key = _request_compatibility_key(first_request)
        shutdown_after_batch = False
        max_batch_size = min(self._max_batch_size, self.worker.available_live_feature_slots)
        self._prefetch_request_backlog(max_messages=max_batch_size - len(requests))
        deadline_ns = time.monotonic_ns() + int(self._max_wait_ms * 1_000_000)

        while len(requests) < max_batch_size:
            try:
                message = self._next_fcfs_candidate(deadline_ns)
            except queue.Empty:
                break
            if isinstance(message, JaxShutdown):
                shutdown_after_batch = True
                break
            if not isinstance(message, JaxRequestEnvelope):
                self._backlog.appendleft(message)
                break
            if _request_compatibility_key(message) != compatibility_key:
                self._backlog.appendleft(message)
                break
            requests.append(message)
        return requests, shutdown_after_batch

    def _prefetch_request_backlog(self, *, max_messages: int) -> None:
        for _ in range(max(0, max_messages)):
            try:
                message, get_start_ns, get_end_ns = timed_queue_get(self._request_queue, block=False)
                self._backlog.append(_mark_dequeued_message(message, get_start_ns=get_start_ns, get_end_ns=get_end_ns))
            except queue.Empty:
                return

    def _next_fcfs_candidate(self, deadline_ns: int) -> Any:
        if self._max_wait_ms == 0:
            return self._next_request_message_nowait()
        remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining_s <= 0:
            raise queue.Empty
        return self._next_request_message(timeout=remaining_s)

    def _next_request_message(self, *, timeout: float) -> Any:
        if self._backlog:
            return self._backlog.popleft()
        message, get_start_ns, get_end_ns = timed_queue_get(self._request_queue, timeout=timeout)
        return _mark_dequeued_message(message, get_start_ns=get_start_ns, get_end_ns=get_end_ns)

    def _next_request_message_nowait(self) -> Any:
        if self._backlog:
            return self._backlog.popleft()
        message, get_start_ns, get_end_ns = timed_queue_get(self._request_queue, block=False)
        return _mark_dequeued_message(message, get_start_ns=get_start_ns, get_end_ns=get_end_ns)

    def _defer_until_live_feature_capacity(self, message: Any, batch_size: int) -> bool:
        available = self.worker.available_live_feature_slots
        if batch_size <= available:
            return True
        max_live_features = self.worker.max_live_features
        if batch_size > max_live_features:
            request_ids = message.request_ids if isinstance(message, JaxBatchRequestEnvelope) else (message.request_id,)
            for request_id in request_ids:
                self._prefix_queue.put(
                    JaxWorkerError(
                        request_id=request_id,
                        error=f"Prefix batch size {batch_size} exceeds VLM live feature capacity {max_live_features}",
                    )
                )
            return False
        self._backlog.appendleft(message)
        time.sleep(0.001)
        return False


def _mark_dequeued_message(message: Any, *, get_start_ns: int, get_end_ns: int) -> Any:
    if isinstance(message, JaxRequestEnvelope | JaxBatchRequestEnvelope):
        return replace(message, dequeue_start_ns=get_start_ns, dequeue_ns=get_end_ns)
    return message


def _vlm_request_queue_timings(
    *,
    enqueue_ns: int,
    dequeue_start_ns: int | None,
    dequeue_ns: int,
) -> tuple[float, float]:
    if dequeue_start_ns is None:
        return max(0.0, (dequeue_ns - enqueue_ns) / 1_000_000), 0.0
    return (
        max(0.0, (dequeue_start_ns - enqueue_ns) / 1_000_000),
        max(0.0, (dequeue_ns - dequeue_start_ns) / 1_000_000),
    )


def _stack_request_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return _cat_tree(observations)


def _stack_request_sample_kwargs(requests: list[JaxRequestEnvelope]) -> dict[str, Any]:
    first_kwargs = requests[0].sample_kwargs
    if any(set(request.sample_kwargs) != set(first_kwargs) for request in requests):
        raise ValueError("Cannot batch requests with different sample kwarg keys")
    stacked = {}
    for key in first_kwargs:
        values = [request.sample_kwargs[key] for request in requests]
        first = values[0]
        if _is_array(first):
            if key == "noise" and first.ndim == 3:
                stacked[key] = jnp.concatenate([jnp.asarray(value) for value in values], axis=0)
            else:
                if any(not np.array_equal(np.asarray(value), np.asarray(first)) for value in values):
                    raise ValueError(f"Cannot batch requests with different tensor sample kwarg {key!r}")
                stacked[key] = jnp.asarray(first)
        else:
            if any(value != first for value in values):
                raise ValueError(f"Cannot batch requests with different sample kwarg {key!r}")
            stacked[key] = first
    return stacked


def _cat_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _cat_tree([value[key] for value in values]) for key in first}
    if _is_array(first):
        return jnp.concatenate([jnp.asarray(value) for value in values], axis=0)
    if isinstance(first, tuple):
        return tuple(_cat_tree([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_cat_tree([value[index] for value in values]) for index in range(len(first))]
    return first


def _to_jax_tree(value: Any) -> Any:
    if _is_array(value):
        return jnp.asarray(value)
    if isinstance(value, dict):
        return {key: _to_jax_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jax_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_jax_tree(item) for item in value)
    return value


def _prefix_feature_row_view(feature: JaxPrefixFeature, row: int) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=_row_view_tree(feature.past_key_values, row),
        prefix_pad_masks=feature.prefix_pad_masks[row : row + 1],
        state=feature.state[row : row + 1] if feature.state is not None else None,
    )


def _row_view_tree(value: Any, row: int) -> Any:
    if isinstance(value, jax.Array):
        return value[row : row + 1]
    if isinstance(value, tuple):
        return tuple(_row_view_tree(item, row) for item in value)
    if isinstance(value, list):
        return [_row_view_tree(item, row) for item in value]
    if isinstance(value, dict):
        return {key: _row_view_tree(item, row) for key, item in value.items()}
    return value


def _sample_kwargs_for_row(sample_kwargs: dict[str, Any], row: int, batch_size: int) -> dict[str, Any]:
    row_kwargs = dict(sample_kwargs)
    noise = row_kwargs.get("noise")
    if _is_array(noise) and noise.ndim == 3 and int(noise.shape[0]) == batch_size:
        row_kwargs["noise"] = noise[row : row + 1]
    return row_kwargs


def _request_compatibility_key(request: JaxRequestEnvelope) -> Hashable:
    return (
        _tree_compatibility_key(request.observation),
        _sample_kwargs_compatibility_key(request.sample_kwargs),
    )


def _tree_compatibility_key(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple((key, _tree_compatibility_key(value[key])) for key in sorted(value))
    if _is_array(value):
        shape = tuple(value.shape[1:]) if value.ndim > 0 else tuple(value.shape)
        return ("array", shape, str(value.dtype))
    if isinstance(value, tuple):
        return tuple(_tree_compatibility_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_tree_compatibility_key(item) for item in value)
    return (type(value).__name__, repr(value))


def _sample_kwargs_compatibility_key(sample_kwargs: dict[str, Any]) -> Hashable:
    key_items = []
    for key in sorted(sample_kwargs):
        value = sample_kwargs[key]
        if _is_array(value):
            shape = tuple(value.shape[1:]) if key == "noise" and value.ndim == 3 else tuple(value.shape)
            key_items.append((key, "array", shape, str(value.dtype)))
        else:
            key_items.append((key, type(value).__name__, repr(value)))
    return tuple(key_items)


def _prefix_feature_shape_tree(feature: JaxPrefixFeature) -> dict[str, Any]:
    return {
        "past_key_values": _shape_tree(feature.past_key_values),
        "prefix_pad_masks": tuple(feature.prefix_pad_masks.shape),
        "state": tuple(feature.state.shape) if feature.state is not None else None,
    }


def _prefix_feature_dtype_tree(feature: JaxPrefixFeature) -> dict[str, Any]:
    return {
        "past_key_values": _dtype_tree(feature.past_key_values),
        "prefix_pad_masks": str(feature.prefix_pad_masks.dtype),
        "state": str(feature.state.dtype) if feature.state is not None else None,
    }


def _shape_tree(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return tuple(_shape_tree(item) for item in value)
    if isinstance(value, list):
        return [_shape_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    return None


def _dtype_tree(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return str(value.dtype)
    if isinstance(value, tuple):
        return tuple(_dtype_tree(item) for item in value)
    if isinstance(value, list):
        return [_dtype_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _dtype_tree(item) for key, item in value.items()}
    return None


def _shape_tree_from_slab_handles(value: Any) -> Any:
    if hasattr(value, "spec"):
        return value.spec.shape
    if isinstance(value, tuple):
        return tuple(_shape_tree_from_slab_handles(item) for item in value)
    if isinstance(value, list):
        return [_shape_tree_from_slab_handles(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_tree_from_slab_handles(item) for key, item in value.items()}
    return None


def _dtype_tree_from_slab_handles(value: Any) -> Any:
    if hasattr(value, "spec"):
        return value.spec.dtype
    if isinstance(value, tuple):
        return tuple(_dtype_tree_from_slab_handles(item) for item in value)
    if isinstance(value, list):
        return [_dtype_tree_from_slab_handles(item) for item in value]
    if isinstance(value, dict):
        return {key: _dtype_tree_from_slab_handles(item) for key, item in value.items()}
    return None


def _block_prefix_feature(feature: JaxPrefixFeature) -> None:
    for leaf in jax.tree_util.tree_leaves((feature.past_key_values, feature.prefix_pad_masks, feature.state)):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _is_array(value: Any) -> bool:
    return isinstance(value, jax.Array | np.ndarray)
