from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import replace
import queue
import time
import traceback
from typing import Any

import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.serving.va_split.prefix_cache_pool import PrefixCacheLanePool
from openpi.serving.va_split.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split.timing import synchronize_cuda_if_needed
from openpi.serving.va_split.timing import timed_queue_get
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError


@dataclass
class AERequestState:
    request_id: str
    lane_id: int
    source_slot_id: int
    x_t: torch.Tensor
    step_idx: int
    num_steps: int
    dt: torch.Tensor
    started_ns: int
    timing: dict[str, float]
    ae_step_ms: list[float]
    ae_batch_sizes: list[int]
    lane_compact_ms: float = 0.0


class AEWorker:
    """Runs step-level continuous batching over active AE requests."""

    def __init__(self, model: Any, device: str, max_batch_size: int, max_prefix_slots: int | None = None):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_prefix_slots is None:
            max_prefix_slots = max_batch_size
        if max_prefix_slots <= 0:
            raise ValueError("max_prefix_slots must be positive")
        self._model = model
        self._device = device
        self._max_batch_size = max_batch_size
        self._max_prefix_slots = max_prefix_slots
        self._prefix_lanes = PrefixCacheLanePool(max_lanes=max_prefix_slots)
        self._lanes: list[AERequestState | None] = [None for _ in range(max_prefix_slots)]
        self._active_count = 0
        self.active: dict[str, AERequestState] = {}

    @property
    def can_accept_prefix(self) -> bool:
        return self._active_count < self._max_prefix_slots

    def add_prefix(self, ready: PrefixReady) -> None:
        if self._active_count >= self._max_prefix_slots:
            raise RuntimeError(f"AE prefix lane pool is full ({self._max_prefix_slots} active requests)")
        sample_kwargs = dict(ready.sample_kwargs)
        noise = sample_kwargs.get("noise")
        if torch.is_tensor(noise):
            noise = noise.to(self._device)
        timing = dict(ready.timing or {})
        prefix_enqueue_ns = timing.pop("_prefix_enqueue_ns", None)
        prefix_get_start_ns = timing.pop("_prefix_get_start_ns", None)
        prefix_get_end_ns = timing.pop("_prefix_get_end_ns", None)
        queue_wait_ms, transfer_ms = queue_wait_and_transfer_ms(
            enqueue_ns=prefix_enqueue_ns,
            get_start_ns=prefix_get_start_ns,
            get_end_ns=prefix_get_end_ns,
        )
        timing["prefix_queue_wait_ms"] = queue_wait_ms
        timing["prefix_transfer_ms"] = transfer_ms
        if prefix_get_end_ns is not None:
            timing["prefix_admit_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_get_end_ns)) / 1_000_000)
        elif prefix_enqueue_ns is not None:
            # Legacy path: no get split available; keep old single-gap accounting as queue wait.
            timing["prefix_admit_wait_ms"] = 0.0
            timing["prefix_queue_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_enqueue_ns)) / 1_000_000)
            timing["prefix_transfer_ms"] = 0.0
        else:
            timing["prefix_admit_wait_ms"] = 0.0
        batch_size = ready.feature.prefix_pad_masks.shape[0]
        denoise_state = self._model.init_denoise_state(self._device, batch_size, noise, ready.num_steps)
        lane_id = self._active_count
        synchronize_cuda_if_needed(self._device)
        ingest_start_ns = time.monotonic_ns()
        self._prefix_lanes.put_lane(lane_id, ready.feature)
        synchronize_cuda_if_needed(self._device)
        timing["prefix_lane_ingest_ms"] = (time.monotonic_ns() - ingest_start_ns) / 1_000_000
        state = AERequestState(
            request_id=ready.request_id,
            lane_id=lane_id,
            source_slot_id=ready.slot_id,
            x_t=denoise_state.x_t,
            step_idx=int(denoise_state.step_idx.item()),
            num_steps=ready.num_steps,
            dt=denoise_state.dt,
            started_ns=time.monotonic_ns(),
            timing=timing,
            ae_step_ms=[],
            ae_batch_sizes=[],
        )
        self.active[ready.request_id] = state
        self._lanes[lane_id] = state
        self._active_count += 1

    def select_ready_lanes(self) -> list[AERequestState]:
        if self._active_count == 0:
            return []
        batch_size = min(self._active_count, self._max_batch_size)
        lanes = self._lanes[:batch_size]
        if any(lane is None for lane in lanes):
            raise RuntimeError("AE lane table is not dense")
        return [lane for lane in lanes if lane is not None]

    def step_once(self) -> tuple[list[ActionResult], list[ReleaseFeature]]:
        batch = self.select_ready_lanes()
        if not batch:
            return [], []

        synchronize_cuda_if_needed(self._device)
        step_start_ns = time.monotonic_ns()
        prefix_batch = self._prefix_lanes.view_prefix_batch(len(batch))
        x_t = torch.cat([request.x_t for request in batch], dim=0)
        step_idx = torch.tensor([request.step_idx for request in batch], device=self._device, dtype=torch.int32)
        dt = torch.stack([request.dt.to(self._device) for request in batch])
        denoise_batch = DenoiseState(x_t=x_t, step_idx=step_idx, num_steps=batch[0].num_steps, dt=dt)
        v_t = self._model.denoise_one_batch(prefix_batch, denoise_batch)

        results: list[ActionResult] = []
        releases: list[ReleaseFeature] = []
        for row, request in enumerate(batch):
            request.x_t = request.x_t + request.dt * v_t[row : row + 1]
            request.step_idx += 1
        synchronize_cuda_if_needed(self._device)
        step_ms = (time.monotonic_ns() - step_start_ns) / 1_000_000

        for request in batch:
            request.ae_step_ms.append(step_ms)
            request.ae_batch_sizes.append(len(batch))
            if request.step_idx == request.num_steps:
                results.append(
                    ActionResult(
                        request_id=request.request_id,
                        actions=request.x_t,
                        timing=_finish_timing(request),
                    )
                )
                releases.append(ReleaseFeature(request_id=request.request_id, slot_id=request.source_slot_id))
        for lane_id in sorted(
            (request.lane_id for request in batch if request.step_idx == request.num_steps), reverse=True
        ):
            self._remove_lane(lane_id)
        return results, releases

    def _remove_lane(self, lane_id: int) -> None:
        request = self._lanes[lane_id]
        if request is None:
            raise RuntimeError(f"Cannot remove empty AE lane {lane_id}")
        del self.active[request.request_id]
        last_lane = self._active_count - 1
        if lane_id != last_lane:
            moved = self._lanes[last_lane]
            if moved is None:
                raise RuntimeError(f"Cannot compact empty AE lane {last_lane}")
            synchronize_cuda_if_needed(self._device)
            compact_start_ns = time.monotonic_ns()
            self._prefix_lanes.move_lane(last_lane, lane_id)
            synchronize_cuda_if_needed(self._device)
            moved.lane_compact_ms += (time.monotonic_ns() - compact_start_ns) / 1_000_000
            moved.lane_id = lane_id
            self._lanes[lane_id] = moved
        self._lanes[last_lane] = None
        self._active_count -= 1

    def clear_active(self) -> list[ReleaseFeature]:
        releases = [
            ReleaseFeature(request_id=request_id, slot_id=request.source_slot_id)
            for request_id, request in self.active.items()
        ]
        self.active.clear()
        self._lanes = [None for _ in range(self._max_prefix_slots)]
        self._active_count = 0
        return releases


class AEProcess:
    """Queue loop for step-level AE continuous batching."""

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        prefix_queue,
        result_queue,
        release_queue,
        max_batch_size: int,
        max_prefix_slots: int | None = None,
    ):
        self.worker = AEWorker(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            max_prefix_slots=max_prefix_slots,
        )
        self._prefix_queue = prefix_queue
        self._result_queue = result_queue
        self._release_queue = release_queue
        self._prefix_backlog: deque[object] = deque()

    def run(self) -> None:
        while True:
            self.drain_prefix_ready(block=not self.worker.active)
            if self.worker.active:
                self.step_active_once()

    def step_active_once(self) -> None:
        try:
            results, releases = self.worker.step_once()
        except Exception as exc:  # pragma: no cover - real model failures are surfaced through queues.
            self._fail_active_requests(error=str(exc), traceback_text=traceback.format_exc())
            return

        for result in results:
            actions = result.actions.detach().cpu() if torch.is_tensor(result.actions) else result.actions
            timing = dict(result.timing or {})
            timing["_ae_result_enqueue_ns"] = float(time.monotonic_ns())
            self._result_queue.put(ActionResult(request_id=result.request_id, actions=actions, timing=timing))
        for release in releases:
            self._release_queue.put(release)

    def drain_prefix_ready(self, *, block: bool) -> None:
        while True:
            try:
                message = self._next_prefix_message(block=block)
            except queue.Empty:
                return
            if isinstance(message, Shutdown):
                self._result_queue.put(message)
                raise SystemExit
            if isinstance(message, WorkerError):
                self._result_queue.put(message)
                continue
            if not isinstance(message, PrefixReady):
                self._result_queue.put(WorkerError(request_id=None, error=f"Unexpected AE message: {type(message)}"))
                continue
            if not self.worker.can_accept_prefix:
                self._prefix_backlog.appendleft(message)
                return
            try:
                self.worker.add_prefix(message)
            except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                self._result_queue.put(
                    WorkerError(
                        request_id=message.request_id,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                self._release_queue.put(ReleaseFeature(request_id=message.request_id, slot_id=message.slot_id))
            if block:
                block = False
            if not self.worker.can_accept_prefix:
                return

    def _next_prefix_message(self, *, block: bool) -> object:
        if self._prefix_backlog:
            return self._prefix_backlog.popleft()
        if not block and not self.worker.can_accept_prefix:
            raise queue.Empty
        message, get_start_ns, get_end_ns = timed_queue_get(self._prefix_queue, block=block)
        return _stamp_prefix_get_timing(message, get_start_ns=get_start_ns, get_end_ns=get_end_ns)

    def _fail_active_requests(self, *, error: str, traceback_text: str) -> None:
        request_ids = list(self.worker.active)
        releases = self.worker.clear_active()
        for request_id in request_ids:
            self._result_queue.put(WorkerError(request_id=request_id, error=error, traceback=traceback_text))
        for release in releases:
            self._release_queue.put(release)


def _stamp_prefix_get_timing(message: object, *, get_start_ns: int, get_end_ns: int) -> object:
    if not isinstance(message, PrefixReady):
        return message
    timing = dict(message.timing or {})
    timing["_prefix_get_start_ns"] = float(get_start_ns)
    timing["_prefix_get_end_ns"] = float(get_end_ns)
    return replace(message, timing=timing)


def _finish_timing(request: AERequestState) -> dict[str, float]:
    timing = dict(request.timing)
    if request.ae_step_ms:
        timing["ae_step_ms"] = sum(request.ae_step_ms) / len(request.ae_step_ms)
        timing["ae_step_total_ms"] = sum(request.ae_step_ms)
    if request.ae_batch_sizes:
        timing["ae_effective_batch"] = sum(request.ae_batch_sizes) / len(request.ae_batch_sizes)
    ingest_ms = float(timing.get("prefix_lane_ingest_ms", 0.0))
    compact_ms = float(request.lane_compact_ms)
    timing["prefix_lane_compact_ms"] = compact_ms
    timing["prefix_lane_overhead_ms"] = ingest_ms + compact_ms
    return timing
