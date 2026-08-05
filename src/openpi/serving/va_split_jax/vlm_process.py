from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import replace
import heapq
import queue
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.device_slab import DeviceSlab
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabHandle
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.prefix_cache_pool import write_feature_batch_to_slab_tree
from openpi.serving.va_split_jax.prefix_cache_pool import write_feature_to_slab_tree
from openpi.serving.va_split_jax import timeline_log
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxBatchRequestEnvelope
from openpi.serving.va_split_jax.types import JaxLaneCredits
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxWorkerError


_BACKLOG_DRAIN_PER_ROW_MS = 2.0
_BACKLOG_DRAIN_MAX_MS = 25.0


class JaxVLMWorker:
    """Builds prefix features and write-through into AE-owned slab lanes (credit gated)."""

    def __init__(
        self,
        *,
        model: Any,
        max_live_features: int,
        backend: DeviceSlabBackend | None = None,
        shared_pool: JaxVlmPrefixCacheLanePool | None = None,
    ):
        if max_live_features <= 0:
            raise ValueError("max_live_features must be positive")
        self._model = model
        self._max_live_features = max_live_features
        self._backend = backend or make_default_device_slab_backend()
        self._shared_pool = shared_pool
        self._writable_slab_tree: dict[str, Any] | None = None
        self._lane_credits: list[int] = []
        self._outstanding: set[str] = set()

    @property
    def available_live_feature_slots(self) -> int:
        return len(self._lane_credits)

    @property
    def max_live_features(self) -> int:
        return self._max_live_features

    @property
    def active_count(self) -> int:
        return len(self._outstanding)

    @property
    def has_ae_slab(self) -> bool:
        return self._writable_slab_tree is not None or self._shared_pool is not None

    def attach_ae_slab(
        self,
        ready: JaxPrefixSlabReady,
        credits: JaxLaneCredits,
        *,
        shared_pool: JaxVlmPrefixCacheLanePool | None = None,
    ) -> None:
        if shared_pool is not None:
            self._shared_pool = shared_pool
            self._writable_slab_tree = shared_pool.local_slab_tree()
        else:
            self._writable_slab_tree = _open_slab_tree(self._backend, ready.slab.slab_handle_tree)
        self.grant_lane_credits(credits)

    def attach_shared_pool(self, pool: JaxVlmPrefixCacheLanePool, credits: JaxLaneCredits) -> None:
        self._shared_pool = pool
        self._writable_slab_tree = pool.local_slab_tree()
        self.grant_lane_credits(credits)

    def grant_lane_credits(self, credits: JaxLaneCredits | JaxReleaseFeature) -> None:
        if isinstance(credits, JaxReleaseFeature):
            lane_ids = (credits.slot_id,)
        else:
            lane_ids = credits.lane_ids
        owned = set(self._lane_credits)
        for lane_id in lane_ids:
            lid = int(lane_id)
            if lid in owned:
                continue
            heapq.heappush(self._lane_credits, lid)
            owned.add(lid)

    def has_live_feature_capacity(self, batch_size: int) -> bool:
        return batch_size <= self.available_live_feature_slots

    def handle_request(self, request: JaxRequestEnvelope) -> JaxPrefixReady:
        return self.handle_batch([request])[0]

    def handle_batch(self, requests: list[JaxRequestEnvelope]) -> list[JaxPrefixReady]:
        if not requests:
            raise ValueError("JaxVLMWorker.handle_batch requires at least one request")
        if not self.has_live_feature_capacity(len(requests)):
            raise RuntimeError(f"VLM lane credits exhausted ({self.available_live_feature_slots} available)")
        request_ids = tuple(request.request_id for request in requests)
        stage_start_ns = time.monotonic_ns()
        sample_kwargs_start_ns = stage_start_ns
        sample_kwargs = _stack_request_sample_kwargs(requests)
        sample_kwargs_stage_ms = (time.monotonic_ns() - sample_kwargs_start_ns) / 1_000_000
        observation_stack_start_ns = time.monotonic_ns()
        observation_tree = _stack_request_observations([r.observation for r in requests])
        observation_stack_ms = (time.monotonic_ns() - observation_stack_start_ns) / 1_000_000
        to_jax_start_ns = time.monotonic_ns()
        jax_observation_tree = _to_jax_tree(observation_tree)
        to_jax_tree_ms = (time.monotonic_ns() - to_jax_start_ns) / 1_000_000
        observation_from_dict_start_ns = time.monotonic_ns()
        observation, observation_timing = _fast_observation_from_dict(jax_observation_tree)
        observation_from_dict_ms = (time.monotonic_ns() - observation_from_dict_start_ns) / 1_000_000
        input_stage_ms = (time.monotonic_ns() - stage_start_ns) / 1_000_000
        return self._handle_batched_observation(
            request_ids=request_ids,
            observation=observation,
            sample_kwargs=sample_kwargs,
            enqueue_ns_by_row=tuple(request.enqueue_ns for request in requests),
            dequeue_ns_by_row=tuple(request.dequeue_ns for request in requests),
            dequeue_start_ns_by_row=tuple(request.dequeue_start_ns for request in requests),
            input_stage_start_ns=stage_start_ns,
            input_stage_ms=input_stage_ms,
            input_stage_timing={
                "vlm_sample_kwargs_stage_ms": sample_kwargs_stage_ms,
                "vlm_observation_stack_ms": observation_stack_ms,
                "vlm_to_jax_tree_ms": to_jax_tree_ms,
                "vlm_observation_from_dict_ms": observation_from_dict_ms,
                **observation_timing,
            },
        )

    def handle_batch_request(self, request: JaxBatchRequestEnvelope) -> list[JaxPrefixReady]:
        if not self.has_live_feature_capacity(len(request.request_ids)):
            raise RuntimeError(f"VLM lane credits exhausted ({self.available_live_feature_slots} available)")
        stage_start_ns = time.monotonic_ns()
        to_jax_start_ns = stage_start_ns
        jax_observation_tree = _to_jax_tree(request.observation)
        to_jax_tree_ms = (time.monotonic_ns() - to_jax_start_ns) / 1_000_000
        observation_from_dict_start_ns = time.monotonic_ns()
        observation, observation_timing = _fast_observation_from_dict(jax_observation_tree)
        observation_from_dict_ms = (time.monotonic_ns() - observation_from_dict_start_ns) / 1_000_000
        input_stage_ms = (time.monotonic_ns() - stage_start_ns) / 1_000_000
        return self._handle_batched_observation(
            request_ids=request.request_ids,
            observation=observation,
            sample_kwargs=dict(request.sample_kwargs),
            enqueue_ns_by_row=tuple(request.enqueue_ns for _ in request.request_ids),
            dequeue_ns_by_row=tuple(request.dequeue_ns for _ in request.request_ids),
            dequeue_start_ns_by_row=tuple(request.dequeue_start_ns for _ in request.request_ids),
            input_stage_start_ns=stage_start_ns,
            input_stage_ms=input_stage_ms,
            input_stage_timing={
                "vlm_sample_kwargs_stage_ms": 0.0,
                "vlm_observation_stack_ms": 0.0,
                "vlm_to_jax_tree_ms": to_jax_tree_ms,
                "vlm_observation_from_dict_ms": observation_from_dict_ms,
                **observation_timing,
            },
        )

    def release(self, release: JaxReleaseFeature) -> None:
        self._outstanding.discard(release.request_id)
        self.grant_lane_credits(release)

    def _take_credit(self) -> int:
        if not self._lane_credits:
            raise RuntimeError("No VLM lane credits available")
        return heapq.heappop(self._lane_credits)

    def _write_feature_to_lane(self, lane_id: int, feature: JaxPrefixFeature) -> None:
        if self._shared_pool is not None:
            self._shared_pool.write_lane(lane_id, feature)
            return
        if self._writable_slab_tree is None:
            raise RuntimeError("VLM has not attached AE-owned prefix slabs")
        self._writable_slab_tree = write_feature_to_slab_tree(
            self._backend, self._writable_slab_tree, lane_id, feature
        )

    def _write_feature_to_lanes(self, lane_ids: tuple[int, ...], feature: JaxPrefixFeature) -> bool:
        if len(lane_ids) > 1 and _is_contiguous_lane_span(lane_ids):
            if self._shared_pool is not None:
                self._shared_pool.write_lanes(lane_ids, feature)
            else:
                if self._writable_slab_tree is None:
                    raise RuntimeError("VLM has not attached AE-owned prefix slabs")
                self._writable_slab_tree = write_feature_batch_to_slab_tree(
                    self._backend, self._writable_slab_tree, lane_ids, feature
                )
            return True

        for row, lane_id in enumerate(lane_ids):
            self._write_feature_to_lane(lane_id, _prefix_feature_row_view(feature, row))
        return False

    def _handle_batched_observation(
        self,
        *,
        request_ids: tuple[str, ...],
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
        enqueue_ns_by_row: tuple[int, ...],
        dequeue_ns_by_row: tuple[int | None, ...],
        dequeue_start_ns_by_row: tuple[int | None, ...],
        input_stage_start_ns: int,
        input_stage_ms: float,
        input_stage_timing: dict[str, float],
    ) -> list[JaxPrefixReady]:
        if (
            len(enqueue_ns_by_row) != len(request_ids)
            or len(dequeue_ns_by_row) != len(request_ids)
            or len(dequeue_start_ns_by_row) != len(request_ids)
        ):
            raise ValueError("enqueue/dequeue timing must have one entry per request id")
        if not self.has_ae_slab:
            raise RuntimeError("VLM cannot write prefixes before AE slab export")
        start_ns = time.monotonic_ns()
        timeline_log.emit("vlm_prefix_fwd_begin", batch=len(request_ids), reqs=list(request_ids))
        feature = self._model.build_prefix_feature(None, observation)
        _block_prefix_feature(feature)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        timeline_log.emit(
            "vlm_prefix_fwd_end",
            batch=len(request_ids),
            reqs=list(request_ids),
            ms=round(elapsed_ms, 3),
        )
        batch_size = int(feature.prefix_pad_masks.shape[0])
        if batch_size != len(request_ids):
            raise RuntimeError(f"VLM prefix batch size {batch_size} does not match {len(request_ids)} request ids")

        num_steps = int(sample_kwargs.get("num_steps", 10))
        write_start_ns = time.monotonic_ns()
        timeline_log.emit("vlm_slab_write_begin", batch=batch_size, reqs=list(request_ids))
        slot_ids = tuple(self._take_credit() for _ in range(batch_size))
        used_batch_slab_write = self._write_feature_to_lanes(slot_ids, feature)
        write_ms = (time.monotonic_ns() - write_start_ns) / 1_000_000
        per_row_write_ms = write_ms / max(batch_size, 1)
        row_prefix_shape_tree = _prefix_feature_single_row_shape_tree(feature)
        prefix_dtype_tree = _prefix_feature_dtype_tree(feature)
        ready: list[JaxPrefixReady] = []
        for row, request_id in enumerate(request_ids):
            slot_id = slot_ids[row]
            self._outstanding.add(request_id)
            enqueue_ns = enqueue_ns_by_row[row]
            dequeue_ns = dequeue_ns_by_row[row] or enqueue_ns
            queue_wait_ms, transfer_ms = _vlm_request_queue_timings(
                enqueue_ns=enqueue_ns,
                dequeue_start_ns=dequeue_start_ns_by_row[row],
                dequeue_ns=dequeue_ns,
            )
            batch_wait_ms = max(0.0, (input_stage_start_ns - dequeue_ns) / 1_000_000)
            row_kwargs = _sample_kwargs_for_row(sample_kwargs, row, batch_size)
            ready.append(
                JaxPrefixReady(
                    request_id=request_id,
                    slot_handle=JaxPrefixSlotHandle(
                        slot_id=slot_id,
                        batch_rows=1,
                        prefix_shape_tree=row_prefix_shape_tree,
                        prefix_dtype_tree=prefix_dtype_tree,
                    ),
                    num_steps=num_steps,
                    sample_kwargs=row_kwargs,
                    timing={
                        "vlm_prefix_forward_ms": elapsed_ms,
                        "vlm_effective_batch": float(batch_size),
                        "vlm_request_queue_wait_ms": queue_wait_ms,
                        "vlm_request_transfer_ms": transfer_ms,
                        "vlm_queue_wait_ms": batch_wait_ms,
                        "vlm_batch_wait_ms": batch_wait_ms,
                        "vlm_input_stage_ms": input_stage_ms,
                        **input_stage_timing,
                        "vlm_slab_write_ms": per_row_write_ms,
                        "vlm_slab_write_total_ms": write_ms,
                        "vlm_slab_write_contiguous_batch": 1.0 if used_batch_slab_write else 0.0,
                        "vlm_slab_write_rows": float(batch_size),
                    },
                )
            )
        timeline_log.emit(
            "vlm_slab_write_end",
            batch=batch_size,
            reqs=list(request_ids),
            ms=round(write_ms, 3),
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
        max_batch_size: int = 8,
        max_wait_ms: float = 2.0,
        max_live_features: int = 8,
        backend: DeviceSlabBackend | None = None,
        shared_pool: JaxVlmPrefixCacheLanePool | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.worker = JaxVLMWorker(
            model=model,
            max_live_features=max_live_features,
            backend=backend,
            shared_pool=shared_pool,
        )
        self._request_queue = request_queue
        self._prefix_queue = prefix_queue
        self._release_queue = release_queue
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._backlog: deque[Any] = deque()
        self._ae_export_ready = self.worker.has_ae_slab and self.worker.available_live_feature_slots > 0

    def wait_for_ae_export(self, *, timeout_s: float | None = None) -> None:
        """Block until AE publishes PrefixSlabReady + LaneCredits on the release queue."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        saw_slab = self.worker.has_ae_slab
        saw_credits = self.worker.available_live_feature_slots > 0
        while not (saw_slab and saw_credits):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for AE prefix slab export / lane credits")
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                if remaining is None:
                    message = self._release_queue.get()
                else:
                    message = self._release_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if isinstance(message, JaxPrefixSlabReady):
                # Credits may arrive as a separate message; attach slab first with empty credits.
                if not self.worker.has_ae_slab:
                    self.worker.attach_ae_slab(message, JaxLaneCredits(lane_ids=()))
                saw_slab = True
                continue
            if isinstance(message, JaxLaneCredits):
                self.worker.grant_lane_credits(message)
                saw_credits = self.worker.available_live_feature_slots > 0
                continue
            if isinstance(message, JaxReleaseFeature):
                self.worker.release(message)
                saw_credits = self.worker.available_live_feature_slots > 0
                continue
            if isinstance(message, JaxShutdown):
                raise SystemExit
            # Unexpected control messages are ignored during bootstrap.
        self._ae_export_ready = True

    def run(self) -> None:
        timeline_log.configure("vlm")
        timeline_log.emit("vlm_run_begin")
        if not self._ae_export_ready:
            self.wait_for_ae_export()
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
                # Drain releases again right before consuming credits for a batch so we
                # never reuse a lane that AE has not yet freed (Result can race ahead).
                self._drain_releases()
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
            self._drain_releases()
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
                self.worker.release(message)
            elif isinstance(message, JaxLaneCredits):
                self.worker.grant_lane_credits(message)
            elif isinstance(message, JaxPrefixSlabReady):
                if not self.worker.has_ae_slab:
                    self.worker.attach_ae_slab(message, JaxLaneCredits(lane_ids=()))

    def _put_prefix_batch(self, ready_messages: list[JaxPrefixReady]) -> None:
        for ready in ready_messages:
            timing = dict(ready.timing or {})
            timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
            self._prefix_queue.put(replace(ready, timing=timing))

    def _collect_fcfs_batch(self, first_request: JaxRequestEnvelope) -> tuple[list[JaxRequestEnvelope], bool]:
        requests = [first_request]
        compatibility_key = _request_compatibility_key(first_request)
        shutdown_after_batch = False
        collect_start_ns = time.monotonic_ns()
        deadline_ns = self._fcfs_collect_deadline_ns(first_request, collect_start_ns=collect_start_ns)
        self._drain_releases()

        while len(requests) < self._current_fcfs_batch_limit():
            self._drain_releases()
            self._prefetch_request_backlog(max_messages=self._current_fcfs_batch_limit() - len(requests))
            if len(requests) >= self._current_fcfs_batch_limit():
                break
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

    def _current_fcfs_batch_limit(self) -> int:
        return min(self._max_batch_size, self.worker.available_live_feature_slots)

    def _fcfs_collect_deadline_ns(self, first_request: JaxRequestEnvelope, *, collect_start_ns: int) -> int:
        wait_ns = int(self._max_wait_ms * 1_000_000)
        deadline_ns = collect_start_ns + wait_ns
        if wait_ns <= 0:
            return deadline_ns
        if not _request_waited_past_fcfs_window(first_request, wait_ns=wait_ns):
            return deadline_ns

        # Under load, multiprocessing.Queue can have a large logical backlog in
        # the producer-side feeder while get_nowait() briefly reports empty.
        # Once the head request has already exceeded the FCFS window, spend a
        # bounded drain budget to pull those already-late requests into a batch.
        drain_budget_ms = min(
            _BACKLOG_DRAIN_MAX_MS,
            max(self._max_wait_ms, _BACKLOG_DRAIN_PER_ROW_MS) * max(self._max_batch_size, 1),
        )
        return max(deadline_ns, collect_start_ns + int(drain_budget_ms * 1_000_000))

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


def _open_slab_tree(backend: DeviceSlabBackend, value: Any) -> Any:
    if isinstance(value, DeviceSlabHandle):
        return backend.open_slab(value)
    if isinstance(value, DeviceSlab):
        return value
    if isinstance(value, tuple):
        return tuple(_open_slab_tree(backend, item) for item in value)
    if isinstance(value, list):
        return [_open_slab_tree(backend, item) for item in value]
    if isinstance(value, dict):
        return {key: _open_slab_tree(backend, item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"Unsupported prefix slab handle tree node: {type(value)}")


def _mark_dequeued_message(message: Any, *, get_start_ns: int, get_end_ns: int) -> Any:
    if isinstance(message, JaxRequestEnvelope | JaxBatchRequestEnvelope):
        return replace(message, dequeue_start_ns=get_start_ns, dequeue_ns=get_end_ns)
    return message


def _request_waited_past_fcfs_window(request: JaxRequestEnvelope, *, wait_ns: int) -> bool:
    dequeue_ns = request.dequeue_ns
    if dequeue_ns is None:
        return False
    dequeue_start_ns = request.dequeue_start_ns
    effective_get_start_ns = dequeue_ns if dequeue_start_ns is None else max(dequeue_start_ns, request.enqueue_ns)
    return effective_get_start_ns - request.enqueue_ns > wait_ns


def _vlm_request_queue_timings(
    *,
    enqueue_ns: int,
    dequeue_start_ns: int | None,
    dequeue_ns: int,
) -> tuple[float, float]:
    if dequeue_start_ns is None:
        return max(0.0, (dequeue_ns - enqueue_ns) / 1_000_000), 0.0
    effective_get_start_ns = max(dequeue_start_ns, enqueue_ns)
    return (
        max(0.0, (effective_get_start_ns - enqueue_ns) / 1_000_000),
        max(0.0, (dequeue_ns - effective_get_start_ns) / 1_000_000),
    )


def _stack_request_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(observations) == 1:
        return observations[0]
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
                stacked[key] = tuple(np.asarray(value).copy() for value in values)
            else:
                if any(not np.array_equal(np.asarray(value), np.asarray(first)) for value in values):
                    raise ValueError(f"Cannot batch requests with different tensor sample kwarg {key!r}")
                stacked[key] = np.asarray(first).copy()
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
        return _concat_model_input(values)
    if isinstance(first, tuple):
        return tuple(_cat_tree([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_cat_tree([value[index] for value in values]) for index in range(len(first))]
    return first


def _to_jax_tree(value: Any) -> Any:
    if _is_array(value):
        return _asarray_model_input(value)
    if isinstance(value, dict):
        return {key: _to_jax_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jax_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_jax_tree(item) for item in value)
    return value


def _fast_observation_from_dict(data: dict[str, Any]) -> tuple[_model.Observation, dict[str, float]]:
    """Construct trusted VLM-worker observations without the generic dataclass typecheck hot path."""
    if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
        raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

    normalize_start_ns = time.monotonic_ns()
    images: dict[str, Any] = {}
    uint8_images = 0
    float32_images = 0
    other_images = 0
    for key, value in data["image"].items():
        dtype = _numpy_dtype(value)
        if dtype == np.dtype(np.uint8):
            uint8_images += 1
            value = value.astype(jnp.float32) / 255.0 * 2.0 - 1.0
            _block_jax_tree(value)
        elif dtype == np.dtype(np.float32):
            float32_images += 1
        else:
            other_images += 1
        images[key] = value
    normalize_ms = (time.monotonic_ns() - normalize_start_ns) / 1_000_000

    construct_start_ns = time.monotonic_ns()
    observation = _construct_observation_unchecked(
        images=images,
        image_masks=data["image_mask"],
        state=data["state"],
        tokenized_prompt=data.get("tokenized_prompt"),
        tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        token_ar_mask=data.get("token_ar_mask"),
        token_loss_mask=data.get("token_loss_mask"),
    )
    construct_ms = (time.monotonic_ns() - construct_start_ns) / 1_000_000
    return observation, {
        "vlm_observation_uint8_images": float(uint8_images),
        "vlm_observation_float32_images": float(float32_images),
        "vlm_observation_other_images": float(other_images),
        "vlm_observation_uint8_normalize_ms": normalize_ms,
        "vlm_observation_construct_ms": construct_ms,
    }


def _construct_observation_unchecked(
    *,
    images: dict[str, Any],
    image_masks: dict[str, Any],
    state: Any,
    tokenized_prompt: Any | None,
    tokenized_prompt_mask: Any | None,
    token_ar_mask: Any | None,
    token_loss_mask: Any | None,
) -> _model.Observation:
    observation = object.__new__(_model.Observation)
    object.__setattr__(observation, "images", images)
    object.__setattr__(observation, "image_masks", image_masks)
    object.__setattr__(observation, "state", state)
    object.__setattr__(observation, "tokenized_prompt", tokenized_prompt)
    object.__setattr__(observation, "tokenized_prompt_mask", tokenized_prompt_mask)
    object.__setattr__(observation, "token_ar_mask", token_ar_mask)
    object.__setattr__(observation, "token_loss_mask", token_loss_mask)
    return observation


def _numpy_dtype(value: Any) -> np.dtype | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    try:
        return np.dtype(dtype)
    except TypeError:
        return None


def _asarray_model_input(value: Any) -> jax.Array:
    if isinstance(value, jax.Array):
        return value
    return jnp.asarray(np.asarray(value))


def _concat_model_input(values: list[Any]) -> jax.Array:
    if any(isinstance(value, jax.Array) for value in values):
        return jnp.concatenate([jnp.asarray(value) for value in values], axis=0)
    arrays = [np.asarray(value) for value in values]
    return jnp.asarray(np.concatenate(arrays, axis=0))


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


def _is_contiguous_lane_span(lane_ids: tuple[int, ...]) -> bool:
    if not lane_ids:
        return False
    start = int(lane_ids[0])
    return tuple(int(lane_id) for lane_id in lane_ids) == tuple(range(start, start + len(lane_ids)))


def _sample_kwargs_for_row(sample_kwargs: dict[str, Any], row: int, batch_size: int) -> dict[str, Any]:
    row_kwargs = dict(sample_kwargs)
    noise = row_kwargs.get("noise")
    if isinstance(noise, tuple) and len(noise) == batch_size and all(_is_array(value) for value in noise):
        row_kwargs["noise"] = np.asarray(noise[row]).copy()
        return row_kwargs
    if _is_array(noise) and noise.ndim == 3 and int(noise.shape[0]) == batch_size:
        row_kwargs["noise"] = np.asarray(noise[row : row + 1]).copy()
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


def _prefix_feature_single_row_shape_tree(feature: JaxPrefixFeature) -> dict[str, Any]:
    return {
        "past_key_values": _single_row_shape_tree(feature.past_key_values, axis=1),
        "prefix_pad_masks": _single_row_shape(feature.prefix_pad_masks, axis=0),
        "state": _single_row_shape(feature.state, axis=0) if feature.state is not None else None,
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


def _single_row_shape_tree(value: Any, *, axis: int) -> Any:
    if isinstance(value, jax.Array):
        return _single_row_shape(value, axis=axis)
    if isinstance(value, tuple):
        return tuple(_single_row_shape_tree(item, axis=axis) for item in value)
    if isinstance(value, list):
        return [_single_row_shape_tree(item, axis=axis) for item in value]
    if isinstance(value, dict):
        return {key: _single_row_shape_tree(item, axis=axis) for key, item in value.items()}
    return None


def _single_row_shape(value: jax.Array, *, axis: int) -> tuple[int, ...]:
    shape = list(value.shape)
    normalized_axis = axis if axis >= 0 else axis + len(shape)
    shape[normalized_axis] = 1
    return tuple(shape)


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


def _block_prefix_feature(feature: JaxPrefixFeature) -> None:
    for leaf in jax.tree_util.tree_leaves((feature.past_key_values, feature.prefix_pad_masks, feature.state)):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _block_jax_tree(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _is_array(value: Any) -> bool:
    return isinstance(value, jax.Array | np.ndarray)
