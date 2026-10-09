from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from dataclasses import replace
import os
import queue
import time
import traceback
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from openpi.models import model as _model
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.shared_prefix_pool import LaneCredits
from openpi.serving.va_split.shared_prefix_pool import PrefixPoolReady
from openpi.serving.va_split.shared_prefix_pool import SharedPrefixLanePool
from openpi.serving.va_split.timing import CudaEventTimer
from openpi.serving.va_split.timing import timed_queue_get
from openpi.serving.va_split.types import BatchPrefixReady
from openpi.serving.va_split.types import BatchRequestEnvelope
from openpi.serving.va_split.types import PrefixPoolBootstrap
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError


def _diag_cuda_event_timing_enabled() -> bool:
    """When set, resolve CUDA-event elapsed (syncs end event) for wall-vs-GPU checks."""
    return os.environ.get("VA_SPLIT_DIAG_CUDA_EVENT_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}

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
        enable_component_timing: bool = False,
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
        self._last_forward_timer: CudaEventTimer | None = None

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
        batch = self.handle_batch([request])
        timing = batch.timing_by_row[0]
        return PrefixReady(
            request_id=batch.request_ids[0],
            feature=_prefix_feature_row_view(batch.feature, 0),
            num_steps=batch.num_steps,
            sample_kwargs=batch.sample_kwargs_by_row[0],
            timing=None if timing is None else dict(timing),
        )

    def handle_batch(self, requests: list[RequestEnvelope]) -> BatchPrefixReady:
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

    def handle_batch_request(self, request: BatchRequestEnvelope) -> BatchPrefixReady:
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
    ) -> BatchPrefixReady:
        if (
            len(enqueue_ns_by_row) != len(request_ids)
            or len(dequeue_ns_by_row) != len(request_ids)
            or len(dequeue_start_ns_by_row) != len(request_ids)
        ):
            raise ValueError("enqueue/dequeue timing must have one entry per request id")
        timer = CudaEventTimer(self._device) if self._enable_component_timing else None
        self._last_forward_timer = None
        if timer is not None:
            timer.start()
        wall_start_ns = time.monotonic_ns()
        observation = _model.Observation.from_dict(_move_tensors_to_device(dict(observation), self._device))
        feature = self._model.build_prefix_feature(self._device, observation)
        if timer is not None:
            # Record the end event only. Do **not** synchronize here: waiting on the
            # event would host-drain VLM before lane write / AE handoff, blocking
            # overlap. Optional CUDA-event resolve happens after publish when
            # VA_SPLIT_DIAG_CUDA_EVENT_TIMING is set.
            timer.stop()
            self._last_forward_timer = timer
        # No host drain: CUDA IPC / later consumers establish happens-before.
        elapsed_ms = (time.monotonic_ns() - wall_start_ns) / 1_000_000
        batch_size = int(feature.prefix_pad_masks.shape[0])
        if batch_size != len(request_ids):
            raise RuntimeError(f"VLM prefix batch size {batch_size} does not match {len(request_ids)} request ids")
        self.live_batches[batch_id] = BatchLiveFeature(feature=feature, remaining_request_ids=set(request_ids))
        num_steps = int(sample_kwargs.get("num_steps", 10))
        start_ns = wall_start_ns
        sample_kwargs_by_row: list[dict[str, Any]] = []
        timing_by_row: list[dict[str, float] | None] = []
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
            sample_kwargs_by_row.append(_sample_kwargs_for_row(sample_kwargs, row, batch_size))
            if self._enable_component_timing:
                timing_by_row.append(
                    {
                        "vlm_effective_batch": float(batch_size),
                        "vlm_prefix_forward_ms": elapsed_ms,
                        "vlm_request_queue_wait_ms": queue_wait_ms,
                        "vlm_request_transfer_ms": transfer_ms,
                        "vlm_queue_wait_ms": max(0.0, (start_ns - dequeue_ns) / 1_000_000),
                    }
                )
            else:
                timing_by_row.append({"vlm_effective_batch": float(batch_size)})
        return BatchPrefixReady(
            request_ids=request_ids,
            feature=feature,
            num_steps=num_steps,
            sample_kwargs_by_row=tuple(sample_kwargs_by_row),
            timing_by_row=tuple(timing_by_row),
        )

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
        enable_component_timing: bool = False,
        use_shared_prefix_lanes: bool = False,
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
        self._use_shared_prefix_lanes = use_shared_prefix_lanes
        self._shared: SharedPrefixLanePool | None = None
        self._pending_credits: deque[int] = deque()
        self._backlog: deque[Any] = deque()
        self._last_prefix_put_ms = 0.0
        self._last_batch_done_ns: int | None = None
        self._last_inter_batch_gap_ms = 0.0

    def bootstrap_shared_pool_before_ready(self, *, warmup_max_batch: int | None = None) -> None:
        """Export AE slab + attach before announcing ready (Triton-style handshake).

        Builds a one-row template prefix, asks AE to allocate/export the shared
        pool (AE warms denoise shapes first), then optionally warms VLM write
        paths for B=1..warmup_max_batch without handing lanes to AE.
        """
        if not self._use_shared_prefix_lanes:
            return
        if self._shared is not None:
            return
        print("[va_split:vlm] shared-prefix bootstrap starting", flush=True)
        template = self._build_warmup_prefix_feature(batch_size=1)
        self._prefix_queue.put(PrefixPoolBootstrap(feature=_prefix_feature_row_view(template, 0)))
        while self._shared is None:
            self._ingest_control(self._release_queue.get())
        warm_to = int(warmup_max_batch) if warmup_max_batch is not None else self._max_batch_size
        warm_to = max(1, min(warm_to, self._max_batch_size))
        print(f"[va_split:vlm] warmup B=1..{warm_to}", flush=True)
        device = self.worker._device
        for batch_size in range(1, warm_to + 1):
            feature = self._build_warmup_prefix_feature(batch_size=batch_size)
            lane_ids = self._acquire_lane_credits(batch_size)
            assert self._shared is not None
            for row, lane_id in enumerate(lane_ids):
                self._shared.write_feature(lane_id, _prefix_feature_row_view(feature, row))
            # Return credits locally; do not hand lanes to AE during warmup.
            self._shared.grant_credits(lane_ids)
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        print("[va_split:vlm] shared-prefix bootstrap done", flush=True)

    def _build_warmup_prefix_feature(self, *, batch_size: int) -> PrefixFeature:
        model = self.worker._model
        device = self.worker._device
        observation = _make_warmup_observation_dict(model, device, batch_size=batch_size)
        observation = _model.Observation.from_dict(_move_tensors_to_device(observation, device))
        return model.build_prefix_feature(device, observation)

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
                    self._forward_and_publish_shared_or_local(
                        batch_size=len(message.request_ids),
                        forward=lambda: self.worker.handle_batch_request(message),
                    )
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
                batch_requests = requests

                def _forward_batch() -> BatchPrefixReady:
                    return self.worker.handle_batch(batch_requests)

                self._forward_and_publish_shared_or_local(
                    batch_size=len(batch_requests),
                    forward=_forward_batch,
                )
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

    def _forward_and_publish_shared_or_local(self, *, batch_size: int, forward) -> None:
        """Triton order when shared lanes: acquire credit → forward → write → put."""
        gap_ms = 0.0
        if self._enable_component_timing and self._last_batch_done_ns is not None:
            gap_ms = max(0.0, (time.monotonic_ns() - self._last_batch_done_ns) / 1_000_000)
            self._last_inter_batch_gap_ms = gap_ms
        if not self._use_shared_prefix_lanes:
            self._publish_prefix_batch(forward(), inter_batch_gap_ms=gap_ms)
            self._last_batch_done_ns = time.monotonic_ns()
            return
        if self._shared is None:
            # Bootstrap path: need a template feature before the pool exists.
            ready = forward()
            self._publish_prefix_batch(ready, inter_batch_gap_ms=gap_ms)
            self._last_batch_done_ns = time.monotonic_ns()
            return
        credit_acquire_ms = 0.0
        acquire_start_ns = time.monotonic_ns()
        lane_ids = self._acquire_lane_credits(batch_size)
        if self._enable_component_timing:
            credit_acquire_ms = (time.monotonic_ns() - acquire_start_ns) / 1_000_000
        try:
            ready = forward()
        except Exception:
            assert self._shared is not None
            self._shared.grant_credits(lane_ids)
            raise
        self._publish_prefix_batch(
            ready,
            lane_ids=lane_ids,
            credit_acquire_ms=credit_acquire_ms,
            inter_batch_gap_ms=gap_ms,
        )
        self._last_batch_done_ns = time.monotonic_ns()

    def _drain_releases(self) -> None:
        while True:
            try:
                message = self._release_queue.get_nowait()
            except queue.Empty:
                return
            self._ingest_control(message)

    def _ingest_control(self, message: Any) -> None:
        if isinstance(message, ReleaseFeature):
            self.worker.release(message)
            return
        if isinstance(message, LaneCredits):
            if self._shared is None:
                self._pending_credits.extend(int(lane_id) for lane_id in message.lane_ids)
            else:
                self._shared.grant_credits(message.lane_ids)
            return
        if isinstance(message, PrefixPoolReady):
            if self._shared is None:
                self._shared = SharedPrefixLanePool.attach_shared(message)
                if self._pending_credits:
                    self._shared.grant_credits(self._pending_credits)
                    self._pending_credits.clear()
            return
        if isinstance(message, Shutdown):
            raise SystemExit

    def _publish_prefix_batch(
        self,
        ready: BatchPrefixReady,
        *,
        lane_ids: list[int] | None = None,
        credit_acquire_ms: float = 0.0,
        inter_batch_gap_ms: float = 0.0,
    ) -> None:
        if not self._use_shared_prefix_lanes:
            self._put_batch_prefix_ready(ready)
            return
        if ready.feature is None:
            raise RuntimeError("shared prefix lanes require a local prefix feature to write")
        self._ensure_shared_pool(ready)
        if lane_ids is None:
            acquire_start_ns = time.monotonic_ns()
            lane_ids = self._acquire_lane_credits(len(ready.request_ids))
            if self._enable_component_timing:
                credit_acquire_ms = (time.monotonic_ns() - acquire_start_ns) / 1_000_000
        elif len(lane_ids) != len(ready.request_ids):
            raise RuntimeError(
                f"pre-acquired lane credit count {len(lane_ids)} does not match "
                f"{len(ready.request_ids)} request ids"
            )
        assert self._shared is not None
        write_ms = 0.0
        write_start_ns = time.monotonic_ns()
        for row, lane_id in enumerate(lane_ids):
            self._shared.write_feature(lane_id, _prefix_feature_row_view(ready.feature, row))
        if self._enable_component_timing:
            write_ms = (time.monotonic_ns() - write_start_ns) / 1_000_000
        cuda_fwd_ms: float | None = None
        if self._enable_component_timing and _diag_cuda_event_timing_enabled():
            timer = self.worker._last_forward_timer
            if timer is not None:
                # Resolve after lane write so AE can already see recorded ready events;
                # sync only for diagnostic wall-vs-GPU comparison (H3).
                cuda_fwd_ms = float(timer.elapsed_ms())
            self.worker._last_forward_timer = None
        else:
            self.worker._last_forward_timer = None
        timing_by_row = []
        for timing in ready.timing_by_row:
            row_timing = dict(timing or {})
            if self._enable_component_timing:
                row_timing["vlm_credit_acquire_ms"] = float(credit_acquire_ms)
                row_timing["vlm_prefix_write_ms"] = float(write_ms)
                row_timing["vlm_prefix_publish_ms"] = float(self._last_prefix_put_ms)
                row_timing["vlm_inter_batch_gap_ms"] = float(inter_batch_gap_ms)
                if cuda_fwd_ms is not None:
                    row_timing["vlm_prefix_forward_cuda_ms"] = float(cuda_fwd_ms)
                row_timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
            timing_by_row.append(row_timing)
        # Control plane carries lane ids only. KV stays in the AE-owned slab.
        put_start_ns = time.monotonic_ns()
        self._prefix_queue.put(
            replace(
                ready,
                feature=None,
                slot_ids=tuple(int(lane_id) for lane_id in lane_ids),
                timing_by_row=tuple(timing_by_row),
            )
        )
        # put() of lane-id metadata is expected <<1ms; keep a process-local sample for logs.
        if self._enable_component_timing:
            self._last_prefix_put_ms = (time.monotonic_ns() - put_start_ns) / 1_000_000

    def _ensure_shared_pool(self, ready: BatchPrefixReady) -> None:
        if self._shared is not None:
            return
        assert ready.feature is not None
        self._prefix_queue.put(PrefixPoolBootstrap(feature=_prefix_feature_row_view(ready.feature, 0)))
        while self._shared is None:
            self._ingest_control(self._release_queue.get())

    def _acquire_lane_credits(self, count: int) -> list[int]:
        if self._shared is None:
            raise RuntimeError("shared prefix pool is not attached")
        lane_ids: list[int] = []
        while len(lane_ids) < count:
            lane_id = self._shared.acquire_credit()
            if lane_id is None:
                self._ingest_control(self._release_queue.get())
                continue
            lane_ids.append(int(lane_id))
        return lane_ids

    def _put_batch_prefix_ready(self, ready: BatchPrefixReady) -> None:
        timing_by_row = []
        for timing in ready.timing_by_row:
            row_timing = dict(timing or {})
            if self._enable_component_timing:
                row_timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
            timing_by_row.append(row_timing)
        self._prefix_queue.put(replace(ready, timing_by_row=tuple(timing_by_row)))

    def _put_prefix_ready(self, ready: PrefixReady) -> None:
        timing = dict(ready.timing or {})
        if self._enable_component_timing:
            timing["_prefix_enqueue_ns"] = float(time.monotonic_ns())
        self._prefix_queue.put(replace(ready, timing=timing))

    def _collect_fcfs_batch(self, first_request: RequestEnvelope) -> tuple[list[RequestEnvelope], bool]:
        """Collect a VLM batch.

        Already-queued messages (backlog + currently readable queue items) are
        drained first and do **not** consume ``max_vlm_wait_ms``. Only after that
        drain, if the batch is still short, open a short wait window for newly
        arriving requests.
        """
        requests = [first_request]
        compatibility_key = _request_compatibility_key(first_request)
        shutdown_after_batch = False

        # Phase 1: drain everything already waiting. This must not use the wait window.
        while True:
            max_batch_size = self._current_fcfs_batch_limit(first_request)
            if len(requests) >= max_batch_size:
                return requests, shutdown_after_batch
            try:
                message = self._next_request_message_nowait()
            except queue.Empty:
                break
            if isinstance(message, Shutdown):
                return requests, True
            if not isinstance(message, RequestEnvelope):
                self._backlog.appendleft(message)
                break
            if _request_compatibility_key(message) != compatibility_key:
                self._backlog.appendleft(message)
                break
            requests.append(message)

        # Phase 2: optional short wait only for *new* arrivals.
        collect_start_ns = time.monotonic_ns()
        deadline_ns = self._fcfs_collect_deadline_ns(first_request, collect_start_ns=collect_start_ns)
        while len(requests) < self._current_fcfs_batch_limit(first_request):
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


def _make_warmup_observation_dict(model: Any, device: str, *, batch_size: int) -> dict[str, Any]:
    """Minimal Observation dict for shared-pool bootstrap / compile warmup."""
    config = getattr(model, "config", None)
    state_dim = int(getattr(config, "action_dim", 32))
    token_len = int(getattr(config, "max_token_len", 48))
    image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    # Channels-first matches the transformed Observation path (and SigLIP).
    images = {
        key: torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device) for key in image_keys
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in image_keys}
    # right_wrist often unused on libero; keep mask False to match convert paths.
    image_masks["right_wrist_0_rgb"] = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return {
        "image": images,
        "image_mask": image_masks,
        # Real libero transforms commonly emit float64 state; the shared slab
        # locks dtype on first put, so match production here.
        "state": torch.zeros(batch_size, state_dim, dtype=torch.float64, device=device),
        "tokenized_prompt": torch.ones(batch_size, token_len, dtype=torch.int32, device=device),
        "tokenized_prompt_mask": torch.ones(batch_size, token_len, dtype=torch.bool, device=device),
    }


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
