from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from dataclasses import replace
import queue
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import DeviceSlab
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabHandle
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxDenoiseBatchSlots
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxSlotMoved
from openpi.serving.va_split_jax.types import JaxWorkerError


@dataclass
class JaxAERequestState:
    request_id: str
    active_lane_id: int
    prefix_slot_id: int
    x_t: jax.Array
    step_idx: int
    num_steps: int
    dt: jax.Array
    started_ns: int
    timing: dict[str, float]
    ae_step_ms: list[float]
    ae_batch_sizes: list[int]
    lane_compact_ms: float = 0.0


class JaxAEWorker:
    """Runs AE denoise steps over active requests while reading VLM-owned prefix slabs."""

    def __init__(
        self,
        *,
        model: Any,
        max_batch_size: int,
        max_prefix_slots: int | None = None,
        backend: DeviceSlabBackend | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_prefix_slots is None:
            max_prefix_slots = max_batch_size
        if max_prefix_slots <= 0:
            raise ValueError("max_prefix_slots must be positive")
        self._model = model
        self._max_batch_size = max_batch_size
        self._max_prefix_slots = max_prefix_slots
        self._backend = backend or make_default_device_slab_backend()
        self._mapped_prefix_slabs: dict[str, Any] | None = None
        self._prefix_slab_map_ms = 0.0
        self._lanes: list[JaxAERequestState | None] = [None for _ in range(max_prefix_slots)]
        self._active_count = 0
        self.active: dict[str, JaxAERequestState] = {}

    @property
    def can_accept_prefix(self) -> bool:
        return self._active_count < self._max_prefix_slots

    @property
    def has_mapped_prefix_slabs(self) -> bool:
        return self._mapped_prefix_slabs is not None

    def open_prefix_slab(self, ready: JaxPrefixSlabReady) -> None:
        if self._mapped_prefix_slabs is not None:
            return
        start_ns = time.monotonic_ns()
        self._mapped_prefix_slabs = _open_slab_tree(self._backend, ready.slab.slab_handle_tree)
        self._prefix_slab_map_ms = (time.monotonic_ns() - start_ns) / 1_000_000

    def set_mapped_prefix_slabs(self, slab_tree: dict[str, Any], *, map_ms: float = 0.0) -> None:
        self.close()
        start_ns = time.monotonic_ns()
        self._mapped_prefix_slabs = _open_slab_tree(self._backend, slab_tree) if _contains_slab_handles(slab_tree) else slab_tree
        map_elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        self._prefix_slab_map_ms = map_ms
        if map_ms == 0.0 and map_elapsed_ms > 0.0 and _contains_device_slabs(self._mapped_prefix_slabs):
            self._prefix_slab_map_ms = map_elapsed_ms

    def close(self) -> None:
        if self._mapped_prefix_slabs is not None:
            _close_slab_tree(self._mapped_prefix_slabs)
            self._mapped_prefix_slabs = None

    def add_prefix(self, ready: JaxPrefixReady) -> None:
        if self._active_count >= self._max_prefix_slots:
            raise RuntimeError(f"AE prefix slot table is full ({self._max_prefix_slots} active requests)")
        sample_kwargs = dict(ready.sample_kwargs)
        noise = sample_kwargs.get("noise")
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
        timing["prefix_slab_map_ms"] = self._prefix_slab_map_ms
        timing["prefix_pool_write_ms"] = 0.0
        if prefix_get_end_ns is not None:
            timing["prefix_admit_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_get_end_ns)) / 1_000_000)
        elif prefix_enqueue_ns is not None:
            timing["prefix_admit_wait_ms"] = 0.0
            timing["prefix_queue_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_enqueue_ns)) / 1_000_000)
            timing["prefix_transfer_ms"] = 0.0
        else:
            timing["prefix_admit_wait_ms"] = 0.0

        denoise_state = self._model.init_denoise_state(None, ready.slot_handle.batch_rows, noise, ready.num_steps)
        lane_id = self._active_count
        state = JaxAERequestState(
            request_id=ready.request_id,
            active_lane_id=lane_id,
            prefix_slot_id=ready.slot_handle.slot_id,
            x_t=denoise_state.x_t,
            step_idx=int(np.asarray(denoise_state.step_idx)),
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

    def apply_slot_moved(self, message: JaxSlotMoved) -> None:
        request = self.active.get(message.request_id)
        if request is not None and request.prefix_slot_id == message.old_slot_id:
            request.prefix_slot_id = message.new_slot_id

    def select_ready_lanes(self) -> list[JaxAERequestState]:
        if self._active_count == 0:
            return []
        by_slot = {request.prefix_slot_id: request for request in self.active.values()}
        selected: list[JaxAERequestState] = []
        for slot_id in range(min(self._max_batch_size, self._active_count)):
            request = by_slot.get(slot_id)
            if request is None:
                break
            selected.append(request)
        return selected

    def step_once(self) -> tuple[list[JaxActionResult], list[JaxReleaseFeature]]:
        batch = self.select_ready_lanes()
        if not batch:
            return [], []
        if self._mapped_prefix_slabs is None:
            raise RuntimeError("AE prefix slabs have not been mapped")

        batch_slots = JaxDenoiseBatchSlots(
            request_ids=tuple(request.request_id for request in batch),
            slot_ids=tuple(request.prefix_slot_id for request in batch),
        )
        step_start_ns = time.monotonic_ns()
        prefix_batch = _slice_prefix_batch(self._backend, self._mapped_prefix_slabs, batch_slots.slot_ids)
        x_t = jnp.concatenate([request.x_t for request in batch], axis=0)
        step_idx = jnp.asarray([request.step_idx for request in batch], dtype=jnp.int32)
        dt = jnp.stack([jnp.asarray(request.dt) for request in batch])
        denoise_batch = JaxDenoiseState(x_t=x_t, step_idx=step_idx, num_steps=batch[0].num_steps, dt=dt)
        v_t = self._model.denoise_one_batch(prefix_batch, denoise_batch)
        v_t.block_until_ready()

        results: list[JaxActionResult] = []
        releases: list[JaxReleaseFeature] = []
        for row, request in enumerate(batch):
            request.x_t = request.x_t + request.dt * v_t[row : row + 1]
            request.x_t.block_until_ready()
            request.step_idx += 1
        step_ms = (time.monotonic_ns() - step_start_ns) / 1_000_000

        for request in batch:
            request.ae_step_ms.append(step_ms)
            request.ae_batch_sizes.append(len(batch))
            if request.step_idx == request.num_steps:
                results.append(
                    JaxActionResult(
                        request_id=request.request_id,
                        actions=request.x_t,
                        timing=_finish_timing(request),
                    )
                )
                releases.append(JaxReleaseFeature(request_id=request.request_id, slot_id=request.prefix_slot_id))
        for lane_id in sorted(
            (request.active_lane_id for request in batch if request.step_idx == request.num_steps), reverse=True
        ):
            self._remove_active_lane(lane_id)
        return results, releases

    def clear_active(self) -> list[JaxReleaseFeature]:
        releases = [
            JaxReleaseFeature(request_id=request_id, slot_id=request.prefix_slot_id)
            for request_id, request in self.active.items()
        ]
        self.active.clear()
        self._lanes = [None for _ in range(self._max_prefix_slots)]
        self._active_count = 0
        return releases

    def _remove_active_lane(self, lane_id: int) -> None:
        request = self._lanes[lane_id]
        if request is None:
            raise RuntimeError(f"Cannot remove empty AE lane {lane_id}")
        del self.active[request.request_id]
        last_lane = self._active_count - 1
        if lane_id != last_lane:
            moved = self._lanes[last_lane]
            if moved is None:
                raise RuntimeError(f"Cannot compact empty AE lane {last_lane}")
            compact_start_ns = time.monotonic_ns()
            moved.active_lane_id = lane_id
            self._lanes[lane_id] = moved
            moved.lane_compact_ms += (time.monotonic_ns() - compact_start_ns) / 1_000_000
        self._lanes[last_lane] = None
        self._active_count -= 1


class JaxAEProcess:
    def __init__(
        self,
        *,
        model: Any,
        prefix_queue,
        result_queue,
        release_queue,
        max_batch_size: int,
        max_prefix_slots: int | None = None,
        backend: DeviceSlabBackend | None = None,
    ):
        self.worker = JaxAEWorker(
            model=model,
            max_batch_size=max_batch_size,
            max_prefix_slots=max_prefix_slots,
            backend=backend,
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
        except Exception as exc:  # pragma: no cover
            self._fail_active_requests(error=str(exc), traceback_text=traceback.format_exc())
            return

        for result in results:
            actions = np.asarray(result.actions)
            timing = dict(result.timing or {})
            timing["_ae_result_enqueue_ns"] = float(time.monotonic_ns())
            self._result_queue.put(JaxActionResult(request_id=result.request_id, actions=actions, timing=timing))
        for release in releases:
            self._release_queue.put(release)

    def drain_prefix_ready(self, *, block: bool) -> None:
        while True:
            try:
                message = self._next_prefix_message(block=block)
            except queue.Empty:
                return
            if isinstance(message, JaxShutdown):
                self._result_queue.put(message)
                self.worker.close()
                raise SystemExit
            if isinstance(message, JaxWorkerError):
                self._result_queue.put(message)
                continue
            if isinstance(message, JaxPrefixSlabReady):
                self.worker.open_prefix_slab(message)
                if block:
                    block = False
                continue
            if isinstance(message, JaxSlotMoved):
                self.worker.apply_slot_moved(message)
                if block:
                    block = False
                continue
            if not isinstance(message, JaxPrefixReady):
                self._result_queue.put(JaxWorkerError(request_id=None, error=f"Unexpected AE message: {type(message)}"))
                continue
            if not self.worker.has_mapped_prefix_slabs:
                self._prefix_backlog.appendleft(message)
                return
            if not self.worker.can_accept_prefix:
                self._prefix_backlog.appendleft(message)
                return
            try:
                self.worker.add_prefix(message)
            except Exception as exc:  # pragma: no cover
                self._result_queue.put(
                    JaxWorkerError(request_id=message.request_id, error=str(exc), traceback=traceback.format_exc())
                )
                self._release_queue.put(
                    JaxReleaseFeature(request_id=message.request_id, slot_id=message.slot_handle.slot_id)
                )
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
            self._result_queue.put(JaxWorkerError(request_id=request_id, error=error, traceback=traceback_text))
        for release in releases:
            self._release_queue.put(release)


def _stamp_prefix_get_timing(message: object, *, get_start_ns: int, get_end_ns: int) -> object:
    if not isinstance(message, JaxPrefixReady):
        return message
    timing = dict(message.timing or {})
    timing["_prefix_get_start_ns"] = float(get_start_ns)
    timing["_prefix_get_end_ns"] = float(get_end_ns)
    return replace(message, timing=timing)


def _slice_prefix_batch(backend: DeviceSlabBackend, slab_tree: dict[str, Any], slot_ids: tuple[int, ...]) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=_slice_slab_tree(backend, slab_tree["past_key_values"], slot_ids),
        prefix_pad_masks=backend.slice_lanes(slab_tree["prefix_pad_masks"], slot_ids),
        state=backend.slice_lanes(slab_tree["state"], slot_ids) if slab_tree["state"] is not None else None,
    )


def _slice_slab_tree(backend: DeviceSlabBackend, value: Any, slot_ids: tuple[int, ...]) -> Any:
    if isinstance(value, DeviceSlab):
        return backend.slice_lanes(value, slot_ids)
    if isinstance(value, tuple):
        return tuple(_slice_slab_tree(backend, item, slot_ids) for item in value)
    if isinstance(value, list):
        return [_slice_slab_tree(backend, item, slot_ids) for item in value]
    if isinstance(value, dict):
        return {key: _slice_slab_tree(backend, item, slot_ids) for key, item in value.items()}
    raise TypeError(f"Unsupported prefix slab tree node: {type(value)}")


def _open_slab_tree(backend: DeviceSlabBackend, value: Any) -> Any:
    if isinstance(value, DeviceSlabHandle):
        return backend.open_slab(value)
    if isinstance(value, tuple):
        return tuple(_open_slab_tree(backend, item) for item in value)
    if isinstance(value, list):
        return [_open_slab_tree(backend, item) for item in value]
    if isinstance(value, dict):
        return {key: _open_slab_tree(backend, item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(f"Unsupported prefix slab handle tree node: {type(value)}")


def _contains_slab_handles(value: Any) -> bool:
    if isinstance(value, DeviceSlabHandle):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_slab_handles(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_slab_handles(item) for item in value.values())
    return False


def _contains_device_slabs(value: Any) -> bool:
    if isinstance(value, DeviceSlab):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_device_slabs(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_device_slabs(item) for item in value.values())
    return False


def _close_slab_tree(value: Any) -> None:
    if isinstance(value, DeviceSlab):
        value.close()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _close_slab_tree(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _close_slab_tree(item)


def _finish_timing(request: JaxAERequestState) -> dict[str, float]:
    timing = dict(request.timing)
    if request.ae_step_ms:
        timing["ae_step_ms"] = sum(request.ae_step_ms) / len(request.ae_step_ms)
        timing["ae_step_total_ms"] = sum(request.ae_step_ms)
    if request.ae_batch_sizes:
        timing["ae_effective_batch"] = sum(request.ae_batch_sizes) / len(request.ae_batch_sizes)
    compact_ms = float(request.lane_compact_ms)
    timing["prefix_pool_compact_ms"] = compact_ms
    timing["prefix_pool_overhead_ms"] = float(timing.get("prefix_pool_write_ms", 0.0)) + compact_ms
    return timing
