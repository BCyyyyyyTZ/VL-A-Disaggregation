from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from dataclasses import replace
import queue
import time
import traceback
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from openpi.models import model as _model
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.timing import synchronize_cuda_if_needed
from openpi.serving.va_split.timing import timed_queue_get
from openpi.serving.va_split.types import BatchRequestEnvelope
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError

_LATE_FCFS_TARGET_BATCH_SIZE = 6
_LATE_FCFS_DEEP_BATCH_TARGET_SIZE = 7
_LATE_FCFS_SHORT_DRAIN_MS = 4.0
_LATE_FCFS_FULL_BATCH_WAIT_MULTIPLIER = 4.0
_LATE_FCFS_FULL_BATCH_MIN_WAIT_MS = 100.0
_LATE_FCFS_FULL_BATCH_MAX_NUM_STEPS = 5
_BACKLOG_DRAIN_PER_ROW_MS = 2.0
_BACKLOG_DRAIN_MAX_MS = 25.0


@dataclass
class BatchLiveFeature:
    feature: PrefixFeature
    remaining_request_ids: set[str]


class VLMWorker:
    """Builds prefix features and keeps producer-side tensor references alive."""

    def __init__(
        self,
        model: Any,
        device: str,
        max_live_features: int | None = None,
        *,
        enable_component_timing: bool = True,
    ):
        if max_live_features is not None and max_live_features <= 0:
            raise ValueError("max_live_features must be positive")
        self._model = model
        self._device = device
        self._max_live_features = max_live_features
        self._enable_component_timing = enable_component_timing
        self.live_features: dict[str, PrefixFeature] = {}
        self.live_batches: dict[str, BatchLiveFeature] = {}
        self._request_to_batch: dict[str, str] = {}

    @property
    def available_live_feature_slots(self) -> int | None:
        if self._max_live_features is None:
            return None
        return self._max_live_features - len(self.live_features)

    @property
    def max_live_features(self) -> int | None:
        return self._max_live_features

    def has_live_feature_capacity(self, batch_size: int) -> bool:
        available = self.available_live_feature_slots
        return available is None or batch_size <= available

    def handle_request(self, request: RequestEnvelope) -> PrefixReady:
        return self.handle_batch([request])[0]

    def handle_batch(self, requests: list[RequestEnvelope]) -> list[PrefixReady]:
        if not requests:
            raise ValueError("VLMWorker.handle_batch requires at least one request")
        if not self.has_live_feature_capacity(len(requests)):
            raise RuntimeError(f"VLM live prefix feature slots are full ({len(self.live_features)} active features)")
        request_ids = tuple(request.request_id for request in requests)
        sample_kwargs = _stack_request_sample_kwargs(requests)
        observation = _stack_request_observations([request.observation for request in requests])
        return self._handle_batched_observation(
            batch_id=f"batch-{request_ids[0]}",
            request_ids=request_ids,
            observation=observation,
            sample_kwargs=sample_kwargs,
            enqueue_ns_by_row=tuple(request.enqueue_ns for request in requests),
            dequeue_ns_by_row=tuple(request.dequeue_ns for request in requests),
            dequeue_start_ns_by_row=tuple(request.dequeue_start_ns for request in requests),
        )

    def handle_batch_request(self, request: BatchRequestEnvelope) -> list[PrefixReady]:
        if not self.has_live_feature_capacity(len(request.request_ids)):
            raise RuntimeError(f"VLM live prefix feature slots are full ({len(self.live_features)} active features)")
        return self._handle_batched_observation(
            batch_id=request.batch_id,
            request_ids=request.request_ids,
            observation=request.observation,
            sample_kwargs=dict(request.sample_kwargs),
            enqueue_ns_by_row=_batch_enqueue_ns_by_row(request),
            dequeue_ns_by_row=_batch_dequeue_ns_by_row(request),
            dequeue_start_ns_by_row=_batch_dequeue_start_ns_by_row(request),
        )

    def _handle_batched_observation(
        self,
        *,
        batch_id: str,
        request_ids: tuple[str, ...],
        observation: dict[str, Any],
        sample_kwargs: dict[str, Any],
        enqueue_ns_by_row: tuple[int, ...],
        dequeue_ns_by_row: tuple[int | None, ...],
        dequeue_start_ns_by_row: tuple[int | None, ...],
    ) -> list[PrefixReady]:
        if (
            len(enqueue_ns_by_row) != len(request_ids)
            or len(dequeue_ns_by_row) != len(request_ids)
            or len(dequeue_start_ns_by_row) != len(request_ids)
        ):
            raise ValueError("enqueue/dequeue timing must have one entry per request id")
        if self._enable_component_timing:
            synchronize_cuda_if_needed(self._device)
        start_ns = time.monotonic_ns()
        observation = _model.Observation.from_dict(_move_tensors_to_device(dict(observation), self._device))
        feature = self._model.build_prefix_feature(self._device, observation)
        synchronize_cuda_if_needed(self._device)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        batch_size = int(feature.prefix_pad_masks.shape[0])
        if batch_size != len(request_ids):
            raise RuntimeError(f"VLM prefix batch size {batch_size} does not match {len(request_ids)} request ids")
        self.live_batches[batch_id] = BatchLiveFeature(feature=feature, remaining_request_ids=set(request_ids))
        num_steps = int(sample_kwargs.get("num_steps", 10))
        ready = []
        for row, request_id in enumerate(request_ids):
            enqueue_ns = enqueue_ns_by_row[row]
            dequeue_ns = dequeue_ns_by_row[row] or enqueue_ns
            dequeue_start_ns = dequeue_start_ns_by_row[row]
            queue_wait_ms, transfer_ms = _vlm_request_queue_timings(
                enqueue_ns=enqueue_ns,
                dequeue_start_ns=dequeue_start_ns,
                dequeue_ns=dequeue_ns,
            )
            self.live_features[request_id] = feature
            self._request_to_batch[request_id] = batch_id
            row_kwargs = _sample_kwargs_for_row(sample_kwargs, row, batch_size)
            timing = {"vlm_effective_batch": float(batch_size)}
            if self._enable_component_timing:
                timing.update(
                    {
                        "vlm_prefix_forward_ms": elapsed_ms,
                        "vlm_request_queue_wait_ms": queue_wait_ms,
                        "vlm_request_transfer_ms": transfer_ms,
                        "vlm_queue_wait_ms": max(0.0, (start_ns - dequeue_ns) / 1_000_000),
                    }
                )
            ready.append(
                PrefixReady(
                    request_id=request_id,
                    feature=_prefix_feature_row_view(feature, row),
                    num_steps=num_steps,
                    sample_kwargs=row_kwargs,
                    timing=timing,
                )
            )
        return ready

    def release(self, release: ReleaseFeature) -> None:
        self.live_features.pop(release.request_id, None)
        batch_id = self._request_to_batch.pop(release.request_id, None)
        if batch_id is None:
            return
        live_batch = self.live_batches.get(batch_id)
        if live_batch is None:
            return
        live_batch.remaining_request_ids.discard(release.request_id)
        if not live_batch.remaining_request_ids:
            del self.live_batches[batch_id]


class VLMProcess:
    """Queue loop for the VLM worker.

    This class is intentionally small so it can run either in a child process or
    in-process tests. The first implementation forwards per-request CUDA tensors
    through PyTorch multiprocessing, relying on producer-side `live_features`.
    """

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        request_queue,
        prefix_queue,
        release_queue,
        max_batch_size: int = 8,
        max_wait_ms: float = 2.0,
        max_live_features: int | None = None,
        enable_component_timing: bool = True,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.worker = VLMWorker(
            model=model,
            device=device,
            max_live_features=max_live_features,
            enable_component_timing=enable_component_timing,
        )
        self._request_queue = request_queue
        self._prefix_queue = prefix_queue
        self._release_queue = release_queue
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._enable_component_timing = enable_component_timing
        self._backlog: deque[Any] = deque()

    def run(self) -> None:
        while True:
            self._drain_releases()
            self._prefetch_request_backlog(max_messages=self._max_batch_size)
            try:
                message = self._next_request_message(timeout=0.01)
            except queue.Empty:
                continue
            if isinstance(message, Shutdown):
                self._prefix_queue.put(message)
                return
            if isinstance(message, BatchRequestEnvelope):
                message = self._trim_batch_request_to_live_feature_capacity(message)
                if message is None:
                    continue
                try:
                    for ready in self.worker.handle_batch_request(message):
                        self._put_prefix_ready(ready)
                except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                    for request_id in message.request_ids:
                        self._prefix_queue.put(
                            WorkerError(
                                request_id=request_id,
                                error=str(exc),
                                traceback=traceback.format_exc(),
                            )
                        )
                continue
            if not isinstance(message, RequestEnvelope):
                self._prefix_queue.put(WorkerError(request_id=None, error=f"Unexpected VLM message: {type(message)}"))
                continue
            if not self._defer_until_live_feature_capacity(message, 1):
                continue
            requests = [message]
            shutdown_after_batch = False
            try:
                requests, shutdown_after_batch = self._collect_fcfs_batch(message)
                for ready in self.worker.handle_batch(requests):
                    self._put_prefix_ready(ready)
            except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                for request in requests if "requests" in locals() else [message]:
                    self._prefix_queue.put(
                        WorkerError(
                            request_id=request.request_id,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                    )
            if shutdown_after_batch:
                self._prefix_queue.put(Shutdown())
                return

    def _drain_releases(self) -> None:
        while True:
            try:
                message = self._release_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, ReleaseFeature):
                self.worker.release(message)

    def _put_prefix_ready(self, ready: PrefixReady) -> None:
        timing = dict(ready.timing or {})
        if self._enable_component_timing:
            timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
        self._prefix_queue.put(replace(ready, timing=timing))

    def _collect_fcfs_batch(self, first_request: RequestEnvelope) -> tuple[list[RequestEnvelope], bool]:
        requests = [first_request]
        compatibility_key = _request_compatibility_key(first_request)
        shutdown_after_batch = False
        collect_start_ns = time.monotonic_ns()
        deadline_ns = self._fcfs_collect_deadline_ns(first_request, collect_start_ns=collect_start_ns)
        max_batch_size = self._current_fcfs_batch_limit(first_request)
        self._prefetch_request_backlog(max_messages=max_batch_size - len(requests))

        while len(requests) < max_batch_size:
            max_batch_size = self._current_fcfs_batch_limit(first_request)
            if len(requests) >= max_batch_size:
                break
            try:
                message = self._next_fcfs_candidate(deadline_ns)
            except queue.Empty:
                break
            if isinstance(message, Shutdown):
                shutdown_after_batch = True
                break
            if not isinstance(message, RequestEnvelope):
                self._backlog.appendleft(message)
                break
            if _request_compatibility_key(message) != compatibility_key:
                self._backlog.appendleft(message)
                break
            requests.append(message)

        return requests, shutdown_after_batch

    def _current_fcfs_batch_limit(self, first_request: RequestEnvelope | None = None) -> int:
        available_slots = self.worker.available_live_feature_slots
        limit = self._max_batch_size if available_slots is None else min(self._max_batch_size, available_slots)
        if first_request is not None and self._is_late_fcfs_head(first_request):
            target = self._late_fcfs_target_batch_size(first_request)
            limit = min(limit, target)
        return limit

    def _late_fcfs_target_batch_size(self, first_request: RequestEnvelope) -> int:
        if _request_num_steps(first_request) > _LATE_FCFS_FULL_BATCH_MAX_NUM_STEPS:
            return _LATE_FCFS_TARGET_BATCH_SIZE
        if self._is_deep_late_fcfs_head(first_request):
            return _LATE_FCFS_DEEP_BATCH_TARGET_SIZE
        return _LATE_FCFS_TARGET_BATCH_SIZE

    def _is_late_fcfs_head(self, first_request: RequestEnvelope) -> bool:
        wait_ns = int(self._max_wait_ms * 1_000_000)
        return wait_ns > 0 and _request_waited_past_fcfs_window(first_request, wait_ns=wait_ns)

    def _is_deep_late_fcfs_head(self, first_request: RequestEnvelope) -> bool:
        return (
            _request_num_steps(first_request) <= _LATE_FCFS_FULL_BATCH_MAX_NUM_STEPS
            and self._request_waited_past_deep_late_fcfs_window(first_request)
        )

    def _request_waited_past_deep_late_fcfs_window(self, first_request: RequestEnvelope) -> bool:
        wait_ns = int(self._deep_late_fcfs_wait_ms(first_request) * 1_000_000)
        return wait_ns > 0 and _request_waited_past_fcfs_window(first_request, wait_ns=wait_ns)

    def _deep_late_fcfs_wait_ms(self, first_request: RequestEnvelope) -> float:
        wait_ms = max(
            _LATE_FCFS_FULL_BATCH_MIN_WAIT_MS,
            self._max_wait_ms * _LATE_FCFS_FULL_BATCH_WAIT_MULTIPLIER,
        )
        return wait_ms

    def _fcfs_collect_deadline_ns(self, first_request: RequestEnvelope, *, collect_start_ns: int) -> int:
        wait_ns = int(self._max_wait_ms * 1_000_000)
        deadline_ns = collect_start_ns + wait_ns
        if wait_ns <= 0:
            return deadline_ns
        if not _request_waited_past_fcfs_window(first_request, wait_ns=wait_ns):
            return deadline_ns
        if self._request_waited_past_deep_late_fcfs_window(first_request):
            drain_budget_ms = min(
                _BACKLOG_DRAIN_MAX_MS,
                max(self._max_wait_ms, _BACKLOG_DRAIN_PER_ROW_MS) * max(self._max_batch_size, 1),
            )
        else:
            drain_budget_ms = max(self._max_wait_ms, _LATE_FCFS_SHORT_DRAIN_MS)
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
        if available is None or batch_size <= available:
            return True
        max_live_features = self.worker.max_live_features
        if max_live_features is not None and batch_size > max_live_features:
            request_ids = message.request_ids if isinstance(message, BatchRequestEnvelope) else (message.request_id,)
            for request_id in request_ids:
                self._prefix_queue.put(
                    WorkerError(
                        request_id=request_id,
                        error=f"Prefix batch size {batch_size} exceeds VLM live feature capacity {max_live_features}",
                    )
                )
            return False
        self._backlog.appendleft(message)
        time.sleep(0.001)
        return False

    def _trim_batch_request_to_live_feature_capacity(
        self, message: BatchRequestEnvelope
    ) -> BatchRequestEnvelope | None:
        available = self.worker.available_live_feature_slots
        if available is None or len(message.request_ids) <= available:
            return message
        if available <= 0:
            self._backlog.appendleft(message)
            time.sleep(0.001)
            return None
        head, tail = _split_batch_request_envelope(message, rows=int(available))
        self._backlog.appendleft(tail)
        return head


def _mark_dequeued_message(message: Any, *, get_start_ns: int, get_end_ns: int) -> Any:
    if isinstance(message, RequestEnvelope | BatchRequestEnvelope):
        return replace(message, dequeue_start_ns=get_start_ns, dequeue_ns=get_end_ns)
    return message


def _vlm_request_queue_timings(
    *,
    enqueue_ns: int,
    dequeue_start_ns: int | None,
    dequeue_ns: int,
) -> tuple[float, float]:
    if dequeue_start_ns is None:
        # Direct in-process calls have no queue get()/IPC; treat the gap as queue wait.
        return max(0.0, (dequeue_ns - enqueue_ns) / 1_000_000), 0.0
    effective_get_start_ns = max(dequeue_start_ns, enqueue_ns)
    return (
        max(0.0, (effective_get_start_ns - enqueue_ns) / 1_000_000),
        max(0.0, (dequeue_ns - effective_get_start_ns) / 1_000_000),
    )


def _request_waited_past_fcfs_window(request: RequestEnvelope, *, wait_ns: int) -> bool:
    dequeue_ns = request.dequeue_ns
    if dequeue_ns is None:
        return False
    dequeue_start_ns = request.dequeue_start_ns
    effective_get_start_ns = dequeue_ns if dequeue_start_ns is None else max(dequeue_start_ns, request.enqueue_ns)
    return effective_get_start_ns - request.enqueue_ns > wait_ns


def _request_num_steps(request: RequestEnvelope) -> int:
    return int(request.sample_kwargs.get("num_steps", 10))


def _batch_enqueue_ns_by_row(request: BatchRequestEnvelope) -> tuple[int, ...]:
    if request.enqueue_ns_by_row is None:
        return tuple(request.enqueue_ns for _ in request.request_ids)
    if len(request.enqueue_ns_by_row) != len(request.request_ids):
        raise ValueError("BatchRequestEnvelope.enqueue_ns_by_row must match request_ids")
    return tuple(int(value) for value in request.enqueue_ns_by_row)


def _batch_dequeue_ns_by_row(request: BatchRequestEnvelope) -> tuple[int | None, ...]:
    if request.dequeue_ns_by_row is None:
        return tuple(request.dequeue_ns for _ in request.request_ids)
    if len(request.dequeue_ns_by_row) != len(request.request_ids):
        raise ValueError("BatchRequestEnvelope.dequeue_ns_by_row must match request_ids")
    return tuple(None if value is None else int(value) for value in request.dequeue_ns_by_row)


def _batch_dequeue_start_ns_by_row(request: BatchRequestEnvelope) -> tuple[int | None, ...]:
    if request.dequeue_start_ns_by_row is None:
        return tuple(request.dequeue_start_ns for _ in request.request_ids)
    if len(request.dequeue_start_ns_by_row) != len(request.request_ids):
        raise ValueError("BatchRequestEnvelope.dequeue_start_ns_by_row must match request_ids")
    return tuple(None if value is None else int(value) for value in request.dequeue_start_ns_by_row)


def _split_batch_request_envelope(
    request: BatchRequestEnvelope,
    *,
    rows: int,
) -> tuple[BatchRequestEnvelope, BatchRequestEnvelope]:
    if rows <= 0 or rows >= len(request.request_ids):
        raise ValueError("rows must split the batch into non-empty head and tail")
    enqueue_ns_by_row = _batch_enqueue_ns_by_row(request)
    head_ids = request.request_ids[:rows]
    tail_ids = request.request_ids[rows:]
    head_enqueue = enqueue_ns_by_row[:rows]
    tail_enqueue = enqueue_ns_by_row[rows:]
    dequeue_ns_by_row = _batch_dequeue_ns_by_row(request)
    dequeue_start_ns_by_row = _batch_dequeue_start_ns_by_row(request)
    return (
        replace(
            request,
            batch_id=f"{request.batch_id}:head{rows}",
            request_ids=head_ids,
            observation=_slice_batch_tree(request.observation, 0, rows),
            sample_kwargs=_slice_batch_sample_kwargs(request.sample_kwargs, 0, rows),
            enqueue_ns=min(head_enqueue),
            enqueue_ns_by_row=head_enqueue,
            dequeue_ns_by_row=dequeue_ns_by_row[:rows],
            dequeue_start_ns_by_row=dequeue_start_ns_by_row[:rows],
        ),
        replace(
            request,
            batch_id=f"{request.batch_id}:tail{rows}",
            request_ids=tail_ids,
            observation=_slice_batch_tree(request.observation, rows, len(request.request_ids)),
            sample_kwargs=_slice_batch_sample_kwargs(request.sample_kwargs, rows, len(request.request_ids)),
            enqueue_ns=min(tail_enqueue),
            enqueue_ns_by_row=tail_enqueue,
            dequeue_ns_by_row=dequeue_ns_by_row[rows:],
            dequeue_start_ns_by_row=dequeue_start_ns_by_row[rows:],
        ),
    )


def _slice_batch_tree(value: Any, start: int, stop: int) -> Any:
    if isinstance(value, dict):
        return {key: _slice_batch_tree(item, start, stop) for key, item in value.items()}
    if torch.is_tensor(value) and value.ndim > 0:
        return value[start:stop]
    if isinstance(value, tuple):
        return tuple(_slice_batch_tree(item, start, stop) for item in value)
    if isinstance(value, list):
        return [_slice_batch_tree(item, start, stop) for item in value]
    return value


def _slice_batch_sample_kwargs(sample_kwargs: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    row_kwargs = dict(sample_kwargs)
    noise = row_kwargs.get("noise")
    if torch.is_tensor(noise) and noise.ndim == 3:
        row_kwargs["noise"] = noise[start:stop]
    return row_kwargs


def _move_tensors_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value


def _stack_request_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return _cat_tree(observations)


def _stack_request_sample_kwargs(requests: list[RequestEnvelope]) -> dict[str, Any]:
    first_kwargs = requests[0].sample_kwargs
    if any(set(request.sample_kwargs) != set(first_kwargs) for request in requests):
        raise ValueError("Cannot batch requests with different sample kwarg keys")

    stacked = {}
    for key in first_kwargs:
        values = [request.sample_kwargs[key] for request in requests]
        first = values[0]
        if torch.is_tensor(first):
            if key == "noise" and first.ndim == 3:
                stacked[key] = torch.cat(values, dim=0)
            else:
                if any(not torch.equal(value, first) for value in values):
                    raise ValueError(f"Cannot batch requests with different tensor sample kwarg {key!r}")
                stacked[key] = first
        else:
            if any(value != first for value in values):
                raise ValueError(f"Cannot batch requests with different sample kwarg {key!r}")
            stacked[key] = first
    return stacked


def _cat_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _cat_tree([value[key] for value in values]) for key in first}
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, tuple):
        return tuple(_cat_tree([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_cat_tree([value[index] for value in values]) for index in range(len(first))]
    return first


def _prefix_feature_row_view(feature: PrefixFeature, row: int) -> PrefixFeature:
    return PrefixFeature(
        past_key_values=_row_view_tree(feature.past_key_values, row),
        prefix_pad_masks=feature.prefix_pad_masks.narrow(0, row, 1),
        state=feature.state.narrow(0, row, 1) if feature.state is not None else None,
    )


def _row_view_tree(value: Any, row: int) -> Any:
    if isinstance(value, DynamicCache):
        cache = DynamicCache()
        for layer_idx in range(len(value)):
            key, cache_value = value[layer_idx]
            cache.update(key.narrow(0, row, 1), cache_value.narrow(0, row, 1), layer_idx=layer_idx)
        return cache
    if torch.is_tensor(value) and len(value.shape) > 0:
        return value.narrow(0, row, 1)
    if isinstance(value, tuple):
        return tuple(_row_view_tree(item, row) for item in value)
    if isinstance(value, list):
        return [_row_view_tree(item, row) for item in value]
    return value


def _sample_kwargs_for_row(sample_kwargs: dict[str, Any], row: int, batch_size: int) -> dict[str, Any]:
    row_kwargs = dict(sample_kwargs)
    noise = row_kwargs.get("noise")
    if torch.is_tensor(noise) and noise.ndim == 3 and int(noise.shape[0]) == batch_size:
        row_kwargs["noise"] = noise.narrow(0, row, 1)
    return row_kwargs


def _request_compatibility_key(request: RequestEnvelope) -> Hashable:
    return (
        _tree_compatibility_key(request.observation),
        _sample_kwargs_compatibility_key(request.sample_kwargs),
    )


def _tree_compatibility_key(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple((key, _tree_compatibility_key(value[key])) for key in sorted(value))
    if torch.is_tensor(value):
        batchless_shape = tuple(value.shape[1:]) if value.ndim > 0 else tuple(value.shape)
        return ("tensor", batchless_shape, str(value.dtype))
    if isinstance(value, tuple):
        return tuple(_tree_compatibility_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_tree_compatibility_key(item) for item in value)
    return (type(value).__name__, repr(value))


def _sample_kwargs_compatibility_key(sample_kwargs: dict[str, Any]) -> Hashable:
    key_items = []
    for key in sorted(sample_kwargs):
        value = sample_kwargs[key]
        if torch.is_tensor(value):
            shape = tuple(value.shape[1:]) if key == "noise" and value.ndim == 3 else tuple(value.shape)
            key_items.append((key, "tensor", shape, str(value.dtype)))
        else:
            key_items.append((key, type(value).__name__, repr(value)))
    return tuple(key_items)
