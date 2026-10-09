from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import replace
import os
import queue
import time
import traceback
from typing import Any

import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.prefix_cache_pool import PrefixCacheLanePool
from openpi.serving.va_split.shared_prefix_pool import LaneCredits
from openpi.serving.va_split.shared_prefix_pool import SharedPrefixLanePool
from openpi.serving.va_split.timing import CudaEventTimer
from openpi.serving.va_split.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split.timing import timed_queue_get
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import BatchPrefixReady
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import PrefixPoolBootstrap
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError
from transformers.cache_utils import DynamicCache


def _diag_cuda_event_timing_enabled() -> bool:
    return os.environ.get("VA_SPLIT_DIAG_CUDA_EVENT_TIMING", "").strip().lower() in {"1", "true", "yes", "on"}


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
    ae_step_cuda_ms: list[float]
    ae_batch_sizes: list[int]
    lane_compact_ms: float = 0.0


class AEWorker:
    """Runs step-level continuous batching over active AE requests."""

    def __init__(
        self,
        model: Any,
        device: str,
        max_batch_size: int,
        max_prefix_slots: int | None = None,
        *,
        enable_component_timing: bool = False,
    ):
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
        self._enable_component_timing = enable_component_timing
        self._prefix_lanes = PrefixCacheLanePool(max_lanes=max_prefix_slots)
        self._shared_pool: SharedPrefixLanePool | None = None
        self._lanes: list[AERequestState | None] = [None for _ in range(max_prefix_slots)]
        self._active_count = 0
        self.active: dict[str, AERequestState] = {}
        # Triton-style packed prefix reuse: skip gather when lane layout is unchanged.
        self._packed_lane_ids: list[int] = []
        self._packed_prefix: PrefixFeature | None = None
        self._packed_x_t: torch.Tensor | None = None
        self._packed_dt: torch.Tensor | None = None
        self._packed_step_idx: torch.Tensor | None = None

    @property
    def can_accept_prefix(self) -> bool:
        return self._active_count < self._max_prefix_slots

    @property
    def shared_pool(self) -> SharedPrefixLanePool | None:
        return self._shared_pool

    def bind_shared_pool(self, shared: SharedPrefixLanePool) -> None:
        """AE-owned slab. VLM writes lanes in place; AE gathers by lane id."""
        if shared.max_lanes != self._max_prefix_slots:
            raise ValueError("shared prefix pool capacity must equal max_prefix_slots")
        self._shared_pool = shared
        self._prefix_lanes = shared.pool

    def add_prefix(self, ready: PrefixReady) -> None:
        if self._active_count >= self._max_prefix_slots:
            raise RuntimeError(f"AE prefix lane pool is full ({self._max_prefix_slots} active requests)")
        shared_lane = self._shared_pool is not None and ready.slot_id >= 0 and ready.feature is None
        if ready.feature is None and not shared_lane:
            raise ValueError("PrefixReady.feature is required when shared prefix lanes are not attached")
        sample_kwargs = dict(ready.sample_kwargs)
        noise = sample_kwargs.get("noise")
        if torch.is_tensor(noise):
            noise = noise.to(self._device)
        timing = dict(ready.timing or {})
        prefix_enqueue_ns = timing.pop("_prefix_enqueue_ns", None)
        prefix_get_start_ns = timing.pop("_prefix_get_start_ns", None)
        prefix_get_end_ns = timing.pop("_prefix_get_end_ns", None)
        if self._enable_component_timing:
            queue_wait_ms, transfer_ms = queue_wait_and_transfer_ms(
                enqueue_ns=prefix_enqueue_ns,
                get_start_ns=prefix_get_start_ns,
                get_end_ns=prefix_get_end_ns,
            )
            timing["prefix_queue_wait_ms"] = queue_wait_ms
            timing["prefix_transfer_ms"] = transfer_ms
            if prefix_get_end_ns is not None:
                timing["prefix_admit_wait_ms"] = max(
                    0.0,
                    (time.monotonic_ns() - float(prefix_get_end_ns)) / 1_000_000,
                )
            elif prefix_enqueue_ns is not None:
                # Legacy path: no get split available; keep old single-gap accounting as queue wait.
                timing["prefix_admit_wait_ms"] = 0.0
                timing["prefix_queue_wait_ms"] = max(
                    0.0,
                    (time.monotonic_ns() - float(prefix_enqueue_ns)) / 1_000_000,
                )
                timing["prefix_transfer_ms"] = 0.0
            else:
                timing["prefix_admit_wait_ms"] = 0.0
        if shared_lane:
            lane_id = int(ready.slot_id)
            if self._lanes[lane_id] is not None:
                raise RuntimeError(f"shared prefix lane {lane_id} is already active")
            assert self._shared_pool is not None
            self._shared_pool.wait_ready_on_stream((lane_id,))
            batch_size = 1
            timing["prefix_lane_ingest_ms"] = 0.0
        else:
            assert ready.feature is not None
            batch_size = ready.feature.prefix_pad_masks.shape[0]
            lane_id = self._active_count
            ingest_timer = CudaEventTimer(self._device) if self._enable_component_timing else None
            if ingest_timer is not None:
                ingest_timer.start()
            ingest_wall_ns = time.monotonic_ns()
            self._prefix_lanes.put_lane(lane_id, ready.feature)
            if ingest_timer is not None:
                ingest_timer.stop()
                timing["prefix_lane_ingest_ms"] = (time.monotonic_ns() - ingest_wall_ns) / 1_000_000
        denoise_state = self._model.init_denoise_state(self._device, batch_size, noise, ready.num_steps)
        state = AERequestState(
            request_id=ready.request_id,
            lane_id=lane_id,
            source_slot_id=lane_id if shared_lane else ready.slot_id,
            x_t=denoise_state.x_t,
            # Host int — avoid step_idx.item() CUDA sync on the admit path.
            step_idx=0,
            num_steps=ready.num_steps,
            dt=denoise_state.dt,
            started_ns=time.monotonic_ns(),
            timing=timing,
            ae_step_ms=[],
            ae_step_cuda_ms=[],
            ae_batch_sizes=[],
        )
        self.active[ready.request_id] = state
        self._lanes[lane_id] = state
        self._active_count += 1

    def select_ready_lanes(self) -> list[AERequestState]:
        if self._active_count == 0:
            return []
        if self._shared_pool is not None:
            ordered = list(self.active.values())
            return ordered[: min(len(ordered), self._max_batch_size)]
        batch_size = min(self._active_count, self._max_batch_size)
        lanes = self._lanes[:batch_size]
        if any(lane is None for lane in lanes):
            raise RuntimeError("AE lane table is not dense")
        return [lane for lane in lanes if lane is not None]

    def step_once(self) -> tuple[list[ActionResult], list[ReleaseFeature]]:
        batch = self.select_ready_lanes()
        if not batch:
            return [], []

        step_timer = CudaEventTimer(self._device) if self._enable_component_timing else None
        if step_timer is not None:
            step_timer.start()
        wall_start_ns = time.monotonic_ns()
        lane_ids = [request.lane_id for request in batch]
        if self._shared_pool is not None:
            prefix_batch = self._prefix_lanes.gather_lanes_cached(
                lane_ids,
                cached_lane_ids=self._packed_lane_ids,
                dest=self._packed_prefix,
            )
            self._packed_prefix = prefix_batch
        else:
            prefix_batch = self._prefix_lanes.view_prefix_batch(len(batch))
        x_t, step_idx, dt = self._pack_denoise_inputs(batch)
        denoise_batch = DenoiseState(x_t=x_t, step_idx=step_idx, num_steps=batch[0].num_steps, dt=dt)
        v_t = self._model.denoise_one_batch(prefix_batch, denoise_batch)

        results: list[ActionResult] = []
        releases: list[ReleaseFeature] = []
        for row, request in enumerate(batch):
            request.x_t = request.x_t + request.dt * v_t[row : row + 1]
            request.step_idx += 1
        if step_timer is not None:
            # Record end event only; avoid host-draining the AE stream before the
            # next step / VLM overlap (stream ordering already serializes GPU work).
            step_timer.stop()
        step_ms = (time.monotonic_ns() - wall_start_ns) / 1_000_000 if self._enable_component_timing else 0.0
        step_cuda_ms: float | None = None
        if (
            self._enable_component_timing
            and step_timer is not None
            and _diag_cuda_event_timing_enabled()
        ):
            # Diagnostic only: sync end event to compare wall vs true GPU occupancy (H3).
            step_cuda_ms = float(step_timer.elapsed_ms())

        next_packed: list[int] = []
        for request in batch:
            if self._enable_component_timing:
                request.ae_step_ms.append(step_ms)
                if step_cuda_ms is not None:
                    request.ae_step_cuda_ms.append(step_cuda_ms)
            request.ae_batch_sizes.append(len(batch))
            if request.step_idx == request.num_steps:
                next_packed.append(-1)
                results.append(
                    ActionResult(
                        request_id=request.request_id,
                        actions=request.x_t,
                        timing=_finish_timing(request),
                    )
                )
                releases.append(ReleaseFeature(request_id=request.request_id, slot_id=request.source_slot_id))
            else:
                next_packed.append(int(request.lane_id))
        finished = [request for request in batch if request.step_idx == request.num_steps]
        if self._shared_pool is not None:
            for request in finished:
                self._release_shared_lane(request.lane_id)
            self._packed_lane_ids = next_packed
            if self._active_count == 0:
                self._clear_packed_prefix()
        else:
            for lane_id in sorted((request.lane_id for request in finished), reverse=True):
                self._remove_lane(lane_id)
            self._clear_packed_prefix()
        return results, releases

    def _pack_denoise_inputs(
        self, batch: list[AERequestState]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack per-request denoise tensors, reusing buffers when batch size matches."""
        batch_size = len(batch)
        device = batch[0].x_t.device
        if (
            self._packed_x_t is None
            or self._packed_x_t.shape[0] != batch_size
            or self._packed_x_t.device != device
        ):
            self._packed_x_t = torch.empty(
                (batch_size, *batch[0].x_t.shape[1:]),
                dtype=batch[0].x_t.dtype,
                device=device,
            )
            self._packed_dt = torch.empty(batch_size, dtype=batch[0].dt.dtype, device=device)
            self._packed_step_idx = torch.empty(batch_size, dtype=torch.int32, device=device)
        assert self._packed_x_t is not None
        assert self._packed_dt is not None
        assert self._packed_step_idx is not None
        for row, request in enumerate(batch):
            self._packed_x_t[row].copy_(request.x_t[0], non_blocking=True)
            self._packed_dt[row].copy_(request.dt.to(device=device), non_blocking=True)
            self._packed_step_idx[row] = int(request.step_idx)
        return self._packed_x_t, self._packed_step_idx, self._packed_dt

    def _clear_packed_prefix(self) -> None:
        self._packed_lane_ids = []
        self._packed_prefix = None

    def warmup_shared_shapes(self, *, max_batch_size: int, num_steps: int = 1) -> None:
        """Warm AE denoise shapes against the shared slab before export (Triton-style)."""
        if self._shared_pool is None:
            raise RuntimeError("shared prefix pool must be bound before AE warmup")
        max_batch = max(1, min(int(max_batch_size), self._max_batch_size, self._max_prefix_slots))
        template = self._shared_pool.view_lane_feature(0)
        for batch_size in range(1, max_batch + 1):
            lane_ids = list(range(batch_size))
            for lane_id in lane_ids:
                self._prefix_lanes.put_lane(lane_id, template)
            prefix_batch = self._prefix_lanes.gather_lanes(lane_ids)
            noise = self._model.sample_noise(
                (batch_size, self._model.config.action_horizon, self._model.config.action_dim),
                self._device,
            )
            denoise_state = self._model.init_denoise_state(self._device, batch_size, noise, num_steps)
            for step in range(int(num_steps)):
                step_tensor = torch.full((batch_size,), step, device=self._device, dtype=torch.int32)
                denoise_batch = DenoiseState(
                    x_t=denoise_state.x_t,
                    step_idx=step_tensor,
                    num_steps=num_steps,
                    dt=denoise_state.dt.expand(batch_size) if denoise_state.dt.ndim == 0 else denoise_state.dt,
                )
                denoise_state.x_t = denoise_state.x_t + denoise_state.dt * self._model.denoise_one_batch(
                    prefix_batch, denoise_batch
                )
            if batch_size >= 2 and num_steps >= 2:
                mixed = torch.arange(batch_size, device=self._device, dtype=torch.int32) % num_steps
                denoise_batch = DenoiseState(
                    x_t=denoise_state.x_t,
                    step_idx=mixed,
                    num_steps=num_steps,
                    dt=denoise_state.dt.expand(batch_size) if denoise_state.dt.ndim == 0 else denoise_state.dt,
                )
                self._model.denoise_one_batch(prefix_batch, denoise_batch)
        if torch.device(self._device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _release_shared_lane(self, lane_id: int) -> None:
        request = self._lanes[lane_id]
        if request is None:
            raise RuntimeError(f"Cannot release empty shared AE lane {lane_id}")
        del self.active[request.request_id]
        self._lanes[lane_id] = None
        self._active_count -= 1

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
            compact_timer = CudaEventTimer(self._device) if self._enable_component_timing else None
            if compact_timer is not None:
                compact_timer.start()
            compact_wall_ns = time.monotonic_ns()
            self._prefix_lanes.move_lane(last_lane, lane_id)
            if compact_timer is not None:
                compact_timer.stop()
                moved.lane_compact_ms += (time.monotonic_ns() - compact_wall_ns) / 1_000_000
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
        self._clear_packed_prefix()
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
        enable_component_timing: bool = False,
        use_shared_prefix_lanes: bool = False,
    ):
        self.worker = AEWorker(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            max_prefix_slots=max_prefix_slots,
            enable_component_timing=enable_component_timing,
        )
        self._prefix_queue = prefix_queue
        self._result_queue = result_queue
        self._release_queue = release_queue
        self._prefix_backlog: deque[object] = deque()
        self._enable_component_timing = enable_component_timing
        self._use_shared_prefix_lanes = use_shared_prefix_lanes

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
            copy_start_ns = time.monotonic_ns()
            actions = result.actions
            if torch.is_tensor(actions) and actions.device.type == "cuda":
                # Async D2H; sync only this stream before Queue put (IPC needs host bytes).
                host = torch.empty(actions.shape, dtype=actions.dtype, pin_memory=True)
                host.copy_(actions.detach(), non_blocking=True)
                torch.cuda.current_stream(actions.device).synchronize()
                actions = host
            elif torch.is_tensor(actions):
                actions = actions.detach().cpu()
            timing = dict(result.timing or {})
            if self._enable_component_timing:
                timing["ae_result_cpu_copy_ms"] = (time.monotonic_ns() - copy_start_ns) / 1_000_000
                timing["_ae_result_enqueue_ns"] = float(time.monotonic_ns())
            self._result_queue.put(ActionResult(request_id=result.request_id, actions=actions, timing=timing))
        for release in releases:
            self._release_queue.put(release)
        if self.worker.shared_pool is not None and releases:
            self._release_queue.put(LaneCredits(lane_ids=tuple(int(release.slot_id) for release in releases)))

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
            if isinstance(message, PrefixPoolBootstrap):
                self._install_shared_pool(message)
                continue
            if isinstance(message, BatchPrefixReady):
                ready_messages = _expand_batch_prefix_ready(message)
            elif isinstance(message, PrefixReady):
                ready_messages = [message]
            else:
                self._result_queue.put(WorkerError(request_id=None, error=f"Unexpected AE message: {type(message)}"))
                continue
            for idx, ready in enumerate(ready_messages):
                if not self.worker.can_accept_prefix:
                    for leftover in reversed(ready_messages[idx:]):
                        self._prefix_backlog.appendleft(leftover)
                    return
                try:
                    self.worker.add_prefix(ready)
                except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                    self._result_queue.put(
                        WorkerError(
                            request_id=ready.request_id,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                    )
                    self._release_queue.put(ReleaseFeature(request_id=ready.request_id, slot_id=ready.slot_id))
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


    def _install_shared_pool(self, message: PrefixPoolBootstrap) -> None:
        """Allocate AE-owned slabs from the one-time VLM template and publish them."""
        feature = message.feature
        if int(feature.prefix_pad_masks.shape[0]) != 1:
            raise ValueError("PrefixPoolBootstrap must be a single-row template")
        shared = SharedPrefixLanePool.create_owned_from_feature(feature, max_lanes=self.worker._max_prefix_slots)
        self.worker.bind_shared_pool(shared)
        # Triton: warm B=1..N on the AE slab before export_ready / credits.
        warmup_batch = min(self.worker._max_batch_size, self.worker._max_prefix_slots)
        try:
            self.worker.warmup_shared_shapes(max_batch_size=warmup_batch, num_steps=1)
        except Exception as exc:  # pragma: no cover - warmup is best-effort for fake test models.
            print(f"[va_split:ae] shared-pool warmup skipped: {exc}", flush=True)
        self._release_queue.put(shared.export_ready())


def _expand_batch_prefix_ready(message: BatchPrefixReady) -> list[PrefixReady]:
    """Split one batch payload into per-request rows.

    Shared-lane batches have no feature tensor; each row is a physical lane id.
    """
    if len(message.request_ids) != len(message.sample_kwargs_by_row) or len(message.request_ids) != len(
        message.timing_by_row
    ):
        raise ValueError("BatchPrefixReady row metadata must align with request_ids")
    if message.feature is None:
        slot_ids = message.slot_ids
        if slot_ids is None or len(slot_ids) != len(message.request_ids):
            raise ValueError("Shared-lane BatchPrefixReady requires one slot_id per request")
        return [
            PrefixReady(
                request_id=request_id,
                feature=None,
                num_steps=message.num_steps,
                sample_kwargs=dict(message.sample_kwargs_by_row[row]),
                timing=None if message.timing_by_row[row] is None else dict(message.timing_by_row[row]),
                slot_id=int(slot_ids[row]),
            )
            for row, request_id in enumerate(message.request_ids)
        ]
    batch_size = int(message.feature.prefix_pad_masks.shape[0])
    if batch_size != len(message.request_ids):
        raise ValueError(
            f"BatchPrefixReady feature batch {batch_size} does not match {len(message.request_ids)} request ids"
        )
    slot_ids = message.slot_ids or tuple(-1 for _ in message.request_ids)
    ready_messages: list[PrefixReady] = []
    for row, request_id in enumerate(message.request_ids):
        ready_messages.append(
            PrefixReady(
                request_id=request_id,
                feature=_prefix_feature_row_view(message.feature, row),
                num_steps=message.num_steps,
                sample_kwargs=dict(message.sample_kwargs_by_row[row]),
                timing=None if message.timing_by_row[row] is None else dict(message.timing_by_row[row]),
                slot_id=int(slot_ids[row]),
            )
        )
    return ready_messages


def _prefix_feature_row_view(feature: PrefixFeature, row: int) -> PrefixFeature:
    past = feature.past_key_values
    if isinstance(past, DynamicCache):
        row_past = DynamicCache()
        for layer_idx in range(len(past)):
            key, value = past[layer_idx]
            row_past.update(key[row : row + 1], value[row : row + 1], layer_idx=layer_idx)
    else:
        row_past = past
    return PrefixFeature(
        past_key_values=row_past,
        prefix_pad_masks=feature.prefix_pad_masks[row : row + 1],
        state=None if feature.state is None else feature.state[row : row + 1],
    )


def _stamp_prefix_get_timing(message: object, *, get_start_ns: int, get_end_ns: int) -> object:
    if isinstance(message, BatchPrefixReady):
        timing_by_row = []
        for timing in message.timing_by_row:
            row_timing = dict(timing or {})
            row_timing["_prefix_get_start_ns"] = float(get_start_ns)
            row_timing["_prefix_get_end_ns"] = float(get_end_ns)
            timing_by_row.append(row_timing)
        return replace(message, timing_by_row=tuple(timing_by_row))
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
    if request.ae_step_cuda_ms:
        timing["ae_step_cuda_ms"] = sum(request.ae_step_cuda_ms) / len(request.ae_step_cuda_ms)
        timing["ae_step_cuda_total_ms"] = sum(request.ae_step_cuda_ms)
    if request.ae_batch_sizes:
        timing["ae_effective_batch"] = sum(request.ae_batch_sizes) / len(request.ae_batch_sizes)
    if "prefix_lane_ingest_ms" in timing or request.lane_compact_ms:
        ingest_ms = float(timing.get("prefix_lane_ingest_ms", 0.0))
        compact_ms = float(request.lane_compact_ms)
        timing["prefix_lane_compact_ms"] = compact_ms
        timing["prefix_lane_overhead_ms"] = ingest_ms + compact_ms
    return timing
