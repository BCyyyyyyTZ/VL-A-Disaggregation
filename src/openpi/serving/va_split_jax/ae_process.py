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
from openpi.models.jax_split_types import JaxPrefixSlabHandleTree
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import warmup_ae_denoise_on_mapped_slabs
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax import timeline_log
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxLaneCredits
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxWorkerError


@dataclass
class JaxAERequestState:
    request_id: str
    active_lane_id: int
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
    """Owns the sole prefix lane pool; VLM writes via IPC, AE admits without ingest."""

    def __init__(
        self,
        *,
        model: Any,
        max_batch_size: int,
        max_prefix_slots: int | None = None,
        backend: DeviceSlabBackend | None = None,
        compile_config: JaxCompileConfig | None = None,
        noise_factory=None,
        owned_pool: JaxVlmPrefixCacheLanePool | None = None,
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
        self._owned_pool = owned_pool or JaxVlmPrefixCacheLanePool(max_lanes=max_prefix_slots, backend=self._backend)
        if self._owned_pool.max_lanes != max_prefix_slots:
            raise ValueError("owned_pool.max_lanes must equal max_prefix_slots")
        self._compile_config = compile_config
        self._noise_factory = noise_factory
        self._pool_ready = False
        self._pool_init_ms = 0.0
        self._ipc_warmup_batches = 0.0
        self._ipc_warmup_done = False
        self._warmup_template: JaxPrefixFeature | None = None
        self._lanes: list[JaxAERequestState | None] = [None for _ in range(max_prefix_slots)]
        self._active_count = 0
        self.active: dict[str, JaxAERequestState] = {}
        self._rng = jax.random.key(0)
        self._pending_lane_credits: list[int] = []

    @property
    def owned_pool(self) -> JaxVlmPrefixCacheLanePool:
        return self._owned_pool

    @property
    def ipc_warmup_batches(self) -> float:
        return self._ipc_warmup_batches

    @property
    def can_accept_prefix(self) -> bool:
        return self._active_count < self._max_prefix_slots

    @property
    def has_owned_pool(self) -> bool:
        return self._pool_ready

    # Back-compat alias used by older call sites / tests during migration.
    @property
    def has_mapped_prefix_slabs(self) -> bool:
        return self.has_owned_pool

    def initialize_owned_pool(self, template: JaxPrefixFeature) -> tuple[JaxPrefixSlabReady, JaxLaneCredits]:
        """Allocate AE-owned slabs, export handles, and grant all lane credits to VLM."""
        if self._pool_ready:
            raise RuntimeError("AE owned prefix pool is already initialized")
        start_ns = time.monotonic_ns()
        self._owned_pool.initialize_from_feature(template)
        self._warmup_template = template
        self._pool_init_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        self._pool_ready = True
        self._maybe_warmup_owned_pool()
        return self.export_slab_ready(), self.initial_lane_credits()

    def export_slab_ready(self) -> JaxPrefixSlabReady:
        if not self._pool_ready:
            raise RuntimeError("Cannot export AE prefix slabs before initialization")
        slab_tree = self._owned_pool.export_slab_handle_tree()
        return JaxPrefixSlabReady(
            slab=JaxPrefixSlabHandleTree(
                max_lanes=self._owned_pool.max_lanes,
                prefix_shape_tree=_shape_tree_from_slab_handles(slab_tree),
                prefix_dtype_tree=_dtype_tree_from_slab_handles(slab_tree),
                slab_handle_tree=slab_tree,
            ),
            timing={"prefix_slab_map_ms": self._pool_init_ms},
        )

    def initial_lane_credits(self) -> JaxLaneCredits:
        return JaxLaneCredits(lane_ids=tuple(range(self._max_prefix_slots)))

    def take_pending_lane_credits(self) -> JaxLaneCredits | None:
        if not self._pending_lane_credits:
            return None
        credits = JaxLaneCredits(lane_ids=tuple(self._pending_lane_credits))
        self._pending_lane_credits.clear()
        return credits

    def attach_initialized_pool(self, pool: JaxVlmPrefixCacheLanePool) -> None:
        """In-process: adopt an already-initialized shared pool (LocalVASplitRuntime)."""
        if pool.max_lanes != self._max_prefix_slots:
            raise ValueError("owned_pool.max_lanes must equal max_prefix_slots")
        if pool._past_slabs is None:
            raise RuntimeError("attach_initialized_pool requires an initialized pool")
        self._owned_pool = pool
        self._pool_ready = True
        self._ipc_warmup_done = True

    def _maybe_warmup_owned_pool(self) -> None:
        if self._ipc_warmup_done or not self._pool_ready:
            return
        if self._compile_config is None or self._noise_factory is None:
            self._ipc_warmup_done = True
            return
        if not self._compile_config.warmup_enabled:
            self._ipc_warmup_done = True
            return
        print(
            f"[jax-ae] starting owned-pool denoise warmup "
            f"(max_ae={self._max_batch_size} max_slots={self._max_prefix_slots} "
            f"warmup_max={self._compile_config.warmup_max_batch_size})",
            flush=True,
        )
        warm_start_ns = time.monotonic_ns()

        def make_prefix_batch(slot_ids: tuple[int, ...]) -> JaxPrefixFeature:
            batch_size = len(slot_ids)
            self._clear_owned_pool_active()
            for idx, _slot_id in enumerate(slot_ids):
                row = _warmup_prefix_row(self._warmup_template, fill=float(idx + 1))
                self._owned_pool.put_lane(f"__warm-{idx}", row)
            prefix_batch = self._owned_pool.view_prefix_batch(batch_size)
            jax.block_until_ready(prefix_batch.prefix_pad_masks)
            return prefix_batch

        stats = warmup_ae_denoise_on_mapped_slabs(
            model=self._model,
            noise_factory=self._noise_factory,
            max_ae_batch_size=self._max_batch_size,
            max_prefix_slots=self._max_prefix_slots,
            config=self._compile_config,
            make_prefix_batch=make_prefix_batch,
        )
        self._clear_owned_pool_active()
        self._ipc_warmup_batches = float(stats["jax_warmup_batches"])
        self._ipc_warmup_done = True
        warm_ms = (time.monotonic_ns() - warm_start_ns) / 1_000_000
        print(
            f"[jax-ae] owned-pool denoise warmup done: batches={self._ipc_warmup_batches} "
            f"wall_ms={warm_ms:.1f}",
            flush=True,
        )

    def _clear_owned_pool_active(self) -> None:
        while self._owned_pool.active_count > 0:
            request_id = self._owned_pool._lane_to_request[0]
            if request_id is None:
                break
            self._owned_pool.release_lane(request_id)

    def close(self) -> None:
        self._clear_owned_pool_active()
        self._ipc_warmup_done = False
        self._ipc_warmup_batches = 0.0

    def add_prefix(self, ready: JaxPrefixReady) -> None:
        if not self._pool_ready:
            raise RuntimeError("AE owned prefix pool has not been initialized")
        if self._active_count >= self._max_prefix_slots:
            raise RuntimeError(f"AE prefix slot table is full ({self._max_prefix_slots} active requests)")
        sample_kwargs = dict(ready.sample_kwargs)
        noise = sample_kwargs.get("noise")
        if noise is not None:
            noise = jnp.asarray(noise)
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
        timing["prefix_slab_map_ms"] = self._pool_init_ms
        timing["prefix_pool_write_ms"] = 0.0
        timing["prefix_lane_ingest_ms"] = 0.0
        if prefix_get_end_ns is not None:
            timing["prefix_admit_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_get_end_ns)) / 1_000_000)
        elif prefix_enqueue_ns is not None:
            timing["prefix_admit_wait_ms"] = 0.0
            timing["prefix_queue_wait_ms"] = max(0.0, (time.monotonic_ns() - float(prefix_enqueue_ns)) / 1_000_000)
            timing["prefix_transfer_ms"] = 0.0
        else:
            timing["prefix_admit_wait_ms"] = 0.0

        init_denoise_start_ns = time.monotonic_ns()
        self._rng, request_rng = jax.random.split(self._rng)
        denoise_state = self._model.init_denoise_state(
            request_rng, ready.slot_handle.batch_rows, noise, ready.num_steps
        )
        timing["ae_init_denoise_ms"] = (time.monotonic_ns() - init_denoise_start_ns) / 1_000_000
        dt = jnp.asarray(-1.0 / float(ready.num_steps), dtype=jnp.float32)
        lane_id = self._active_count
        admit_start_ns = time.monotonic_ns()
        dense_lane, vacated = self._owned_pool.claim_written_lane(ready.request_id, ready.slot_handle.slot_id)
        if dense_lane != lane_id:
            raise RuntimeError(f"AE owned prefix lane mismatch: pool={dense_lane} ae={lane_id}")
        if vacated is not None:
            self._pending_lane_credits.append(int(vacated))
        timing["prefix_lane_ingest_ms"] = 0.0
        timing["prefix_pool_write_ms"] = (time.monotonic_ns() - admit_start_ns) / 1_000_000
        x_t = jnp.array(denoise_state.x_t)
        state = JaxAERequestState(
            request_id=ready.request_id,
            active_lane_id=lane_id,
            x_t=x_t,
            step_idx=0,
            num_steps=ready.num_steps,
            dt=dt,
            started_ns=time.monotonic_ns(),
            timing=timing,
            ae_step_ms=[],
            ae_batch_sizes=[],
        )
        self.active[ready.request_id] = state
        self._lanes[lane_id] = state
        self._active_count += 1

    def select_ready_lanes(self) -> list[JaxAERequestState]:
        if self._active_count == 0:
            return []
        batch_size = min(self._max_batch_size, self._active_count)
        lanes = self._lanes[:batch_size]
        if any(lane is None for lane in lanes):
            raise RuntimeError("AE lane table is not dense")
        return [lane for lane in lanes if lane is not None]

    def step_once(self) -> tuple[list[JaxActionResult], list[JaxReleaseFeature]]:
        batch = self.select_ready_lanes()
        if not batch:
            return [], []

        step_start_ns = time.monotonic_ns()
        timeline_log.emit(
            "ae_denoise_begin",
            batch=len(batch),
            active=self._active_count,
            reqs=[request.request_id for request in batch],
            step_idx=batch[0].step_idx,
        )
        prefix_batch = self._owned_pool.view_prefix_batch(len(batch))
        x_t = jnp.concatenate([request.x_t for request in batch], axis=0)
        step_idx = jnp.asarray([request.step_idx for request in batch], dtype=jnp.int32)
        dt = jnp.stack([jnp.asarray(request.dt) for request in batch])
        denoise_batch = JaxDenoiseState(x_t=x_t, step_idx=step_idx, num_steps=batch[0].num_steps, dt=dt)
        v_t = self._model.denoise_one_batch(prefix_batch, denoise_batch)
        dt_b = dt.reshape((-1,) + (1,) * (v_t.ndim - 1))
        x_t = x_t + dt_b * v_t
        results: list[JaxActionResult] = []
        releases: list[JaxReleaseFeature] = []
        completed: list[JaxAERequestState] = []
        for row, request in enumerate(batch):
            request.x_t = x_t[row : row + 1]
            request.step_idx += 1
            if request.step_idx == request.num_steps:
                request.x_t.block_until_ready()
        step_ms = (time.monotonic_ns() - step_start_ns) / 1_000_000
        timeline_log.emit(
            "ae_denoise_end",
            batch=len(batch),
            active=self._active_count,
            reqs=[request.request_id for request in batch],
            ms=round(step_ms, 3),
        )

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
                completed.append(request)
        freed_by_request: dict[str, int] = {}
        for request in sorted(completed, key=lambda item: item.active_lane_id, reverse=True):
            freed_by_request[request.request_id] = self._remove_active_lane(request.active_lane_id)
        for request in completed:
            releases.append(JaxReleaseFeature(request_id=request.request_id, slot_id=freed_by_request[request.request_id]))
        return results, releases

    def clear_active(self) -> list[JaxReleaseFeature]:
        releases: list[JaxReleaseFeature] = []
        for request_id in list(self.active):
            freed = self._owned_pool.release_lane(request_id)
            if freed is not None:
                releases.append(JaxReleaseFeature(request_id=request_id, slot_id=freed))
        self.active.clear()
        self._lanes = [None for _ in range(self._max_prefix_slots)]
        self._active_count = 0
        return releases

    def _remove_active_lane(self, lane_id: int) -> int:
        request = self._lanes[lane_id]
        if request is None:
            raise RuntimeError(f"Cannot remove empty AE lane {lane_id}")
        del self.active[request.request_id]
        last_lane = self._active_count - 1
        compact_start_ns = time.monotonic_ns()
        freed = self._owned_pool.release_lane(request.request_id)
        if freed is None:
            raise RuntimeError(f"Owned pool missing lane for {request.request_id}")
        if lane_id != last_lane:
            moved = self._lanes[last_lane]
            if moved is None:
                raise RuntimeError(f"Cannot compact empty AE lane {last_lane}")
            moved.active_lane_id = lane_id
            self._lanes[lane_id] = moved
            moved.lane_compact_ms += (time.monotonic_ns() - compact_start_ns) / 1_000_000
        self._lanes[last_lane] = None
        self._active_count -= 1
        return freed


def _warmup_prefix_row(template: JaxPrefixFeature | None, *, fill: float) -> JaxPrefixFeature:
    """Build a single-row warmup feature with logical dtypes matching the AE template."""
    if template is None:
        raise RuntimeError("AE warmup template is missing")
    state = None if template.state is None else jnp.full_like(template.state, fill)
    return JaxPrefixFeature(
        past_key_values=template.past_key_values,
        prefix_pad_masks=template.prefix_pad_masks,
        state=state,
    )


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
    timing["prefix_lane_ingest_ms"] = float(timing.get("prefix_lane_ingest_ms", 0.0))
    timing["prefix_lane_compact_ms"] = compact_ms
    timing["prefix_lane_overhead_ms"] = timing["prefix_pool_overhead_ms"]
    return timing


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
        compile_config: JaxCompileConfig | None = None,
        noise_factory=None,
        owned_pool: JaxVlmPrefixCacheLanePool | None = None,
    ):
        self.worker = JaxAEWorker(
            model=model,
            max_batch_size=max_batch_size,
            max_prefix_slots=max_prefix_slots,
            backend=backend,
            compile_config=compile_config,
            noise_factory=noise_factory,
            owned_pool=owned_pool,
        )
        self._prefix_queue = prefix_queue
        self._result_queue = result_queue
        self._release_queue = release_queue
        self._prefix_backlog: deque[object] = deque()

    def bootstrap_owned_pool(self, template: JaxPrefixFeature) -> float:
        """Create AE-owned slabs, export to VLM via release_queue, warm denoise."""
        slab_ready, credits = self.worker.initialize_owned_pool(template)
        self._release_queue.put(slab_ready)
        self._release_queue.put(credits)
        return float(self.worker.ipc_warmup_batches)

    def run(self) -> None:
        timeline_log.configure("ae")
        timeline_log.emit("ae_run_begin")
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

        for release in releases:
            self._release_queue.put(release)
        pending = self.worker.take_pending_lane_credits()
        if pending is not None:
            self._release_queue.put(pending)
        # Publish results after releases so the parent cannot submit the next batch
        # while VLM still holds a stale outstanding credit for an AE-active lane.
        for result in results:
            actions = np.asarray(result.actions)
            timing = dict(result.timing or {})
            timing["_ae_result_enqueue_ns"] = float(time.monotonic_ns())
            self._result_queue.put(JaxActionResult(request_id=result.request_id, actions=actions, timing=timing))

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
            if not isinstance(message, JaxPrefixReady):
                self._result_queue.put(JaxWorkerError(request_id=None, error=f"Unexpected AE message: {type(message)}"))
                continue
            if not self.worker.has_owned_pool:
                self._prefix_backlog.appendleft(message)
                return
            if not self.worker.can_accept_prefix:
                self._prefix_backlog.appendleft(message)
                return
            try:
                self.worker.add_prefix(message)
                pending = self.worker.take_pending_lane_credits()
                if pending is not None:
                    self._release_queue.put(pending)
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
