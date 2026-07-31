from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import replace
import multiprocessing as mp
import os
import threading
import time
import uuid

import jax.numpy as jnp
import numpy as np

from openpi.serving.va_split_jax.ae_process import JaxAEProcess
from openpi.serving.va_split_jax.ae_process import JaxAEWorker
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxBatchRequestEnvelope
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxWorkerError
from openpi.serving.va_split_jax.vlm_process import JaxVLMProcess
from openpi.serving.va_split_jax.vlm_process import JaxVLMWorker


class JaxLocalVASplitRuntime:
    """In-process runtime for numeric and scheduler tests."""

    def __init__(self, *, model, max_ae_batch_size: int = 8, max_prefix_slots: int | None = None):
        if max_prefix_slots is None:
            max_prefix_slots = max_ae_batch_size
        backend = make_default_device_slab_backend()
        self._prefix_pool = JaxVlmPrefixCacheLanePool(max_lanes=max_prefix_slots, backend=backend)
        self.vlm_worker = JaxVLMWorker(model=model, max_live_features=max_prefix_slots, prefix_pool=self._prefix_pool)
        self.ae_worker = JaxAEWorker(
            model=model,
            max_batch_size=max_ae_batch_size,
            max_prefix_slots=max_prefix_slots,
            backend=backend,
        )

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        request_id = str(uuid.uuid4())
        ready = self.vlm_worker.handle_request(
            JaxRequestEnvelope(
                request_id=request_id,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        if not self.ae_worker.has_mapped_prefix_slabs:
            self.ae_worker.set_mapped_prefix_slabs(self._prefix_pool.local_slab_tree())
        self.ae_worker.add_prefix(ready)
        while request_id in self.ae_worker.active:
            results, releases = self.ae_worker.step_once()
            for release in releases:
                moved = self.vlm_worker.release(release)
                if moved is not None:
                    self.ae_worker.apply_slot_moved(moved)
            for result in results:
                if result.request_id == request_id:
                    return result
        raise RuntimeError(f"Request {request_id} finished without an action result")

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        ready_messages = self.vlm_worker.handle_batch_request(
            JaxBatchRequestEnvelope(
                batch_id=batch_id,
                request_ids=request_ids,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        if not self.ae_worker.has_mapped_prefix_slabs:
            self.ae_worker.set_mapped_prefix_slabs(self._prefix_pool.local_slab_tree())
        for ready in ready_messages:
            self.ae_worker.add_prefix(ready)

        results_by_id: dict[str, JaxActionResult] = {}
        while len(results_by_id) < batch_size:
            results, releases = self.ae_worker.step_once()
            for release in releases:
                moved = self.vlm_worker.release(release)
                if moved is not None:
                    self.ae_worker.apply_slot_moved(moved)
            for result in results:
                results_by_id[result.request_id] = result
        return _combine_ordered_results(batch_id, request_ids, results_by_id)


def _apply_env_updates(env_updates: dict[str, str | None] | None) -> None:
    for key, value in (env_updates or {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_jax_vlm_process(
    model_factory,
    request_queue,
    prefix_queue,
    release_queue,
    max_vlm_batch_size,
    max_vlm_wait_ms,
    max_prefix_slots,
    env_updates=None,
) -> None:
    _apply_env_updates(env_updates)
    model = model_factory()
    backend = make_default_device_slab_backend()
    prefix_pool = JaxVlmPrefixCacheLanePool(max_lanes=max_prefix_slots, backend=backend)
    JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=release_queue,
        prefix_pool=prefix_pool,
        max_batch_size=max_vlm_batch_size,
        max_wait_ms=max_vlm_wait_ms,
        max_live_features=max_prefix_slots,
    ).run()


def _run_jax_ae_process(
    model_factory,
    prefix_queue,
    result_queue,
    release_queue,
    max_ae_batch_size,
    max_prefix_slots,
    env_updates=None,
) -> None:
    _apply_env_updates(env_updates)
    model = model_factory()
    JaxAEProcess(
        model=model,
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
    ).run()


class JaxProcessVASplitRuntime:
    def __init__(
        self,
        *,
        model_factory: Callable[[], object],
        max_ae_batch_size: int = 8,
        max_vlm_batch_size: int = 8,
        max_vlm_wait_ms: float = 2.0,
        max_prefix_slots: int | None = None,
        start_method: str = "spawn",
        result_timeout_s: float = 120.0,
        vlm_env_updates: dict[str, str | None] | None = None,
        ae_env_updates: dict[str, str | None] | None = None,
    ):
        if max_prefix_slots is None:
            max_prefix_slots = max_vlm_batch_size * 3
        self._result_timeout_s = result_timeout_s
        self._pending_results: dict[str, JaxActionResult] = {}
        self._pending_errors: dict[str | None, JaxWorkerError] = {}
        self._shutdown_seen = False
        self._closed = False
        self._condition = threading.Condition()

        ctx = mp.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._release_queue = ctx.Queue()
        self._vlm_process = ctx.Process(
            target=_run_jax_vlm_process,
            args=(
                model_factory,
                self._request_queue,
                self._prefix_queue,
                self._release_queue,
                max_vlm_batch_size,
                max_vlm_wait_ms,
                max_prefix_slots,
                vlm_env_updates,
            ),
            daemon=True,
        )
        self._ae_process = ctx.Process(
            target=_run_jax_ae_process,
            args=(
                model_factory,
                self._prefix_queue,
                self._result_queue,
                self._release_queue,
                max_ae_batch_size,
                max_prefix_slots,
                ae_env_updates,
            ),
            daemon=True,
        )
        self._vlm_process.start()
        self._ae_process.start()
        self._result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self._result_thread.start()

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        if self._closed:
            raise RuntimeError("JAX VA split runtime is shut down")
        request_id = str(uuid.uuid4())
        self._request_queue.put(
            JaxRequestEnvelope(
                request_id=request_id,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        return self._wait_for_result(request_id)

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        if self._closed:
            raise RuntimeError("JAX VA split runtime is shut down")
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        self._request_queue.put(
            JaxBatchRequestEnvelope(
                batch_id=batch_id,
                request_ids=request_ids,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        results_by_id = {request_id: self._wait_for_result(request_id) for request_id in request_ids}
        return _combine_ordered_results(batch_id, request_ids, results_by_id)

    def _wait_for_result(self, request_id: str) -> JaxActionResult:
        deadline = time.monotonic() + self._result_timeout_s
        with self._condition:
            while True:
                if request_id in self._pending_results:
                    return self._pending_results.pop(request_id)
                if request_id in self._pending_errors:
                    raise _worker_error_to_runtime_error(self._pending_errors.pop(request_id))
                if None in self._pending_errors:
                    raise _worker_error_to_runtime_error(self._pending_errors[None])
                if self._shutdown_seen:
                    raise RuntimeError("JAX VA split worker shut down before producing a result")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for JAX VA split result {request_id}")
                self._condition.wait(timeout=remaining)

    def _collect_results(self) -> None:
        while True:
            try:
                message, get_start_ns, get_end_ns = timed_queue_get(self._result_queue)
            except (EOFError, OSError):
                return
            with self._condition:
                if isinstance(message, JaxActionResult):
                    self._pending_results[message.request_id] = _mark_collected_result(
                        message,
                        get_start_ns=get_start_ns,
                        get_end_ns=get_end_ns,
                    )
                elif isinstance(message, JaxWorkerError):
                    self._pending_errors[message.request_id] = message
                elif isinstance(message, JaxShutdown):
                    self._shutdown_seen = True
                self._condition.notify_all()
            if isinstance(message, JaxShutdown):
                return

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._request_queue.put(JaxShutdown())
        for process in (self._vlm_process, self._ae_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._result_thread.join(timeout=5)

    def reset(self) -> None:
        pass


def _worker_error_to_runtime_error(error: JaxWorkerError) -> RuntimeError:
    detail = f"\n{error.traceback}" if error.traceback else ""
    return RuntimeError(f"JAX VA split worker failed for {error.request_id}: {error.error}{detail}")


def _combine_ordered_results(
    batch_id: str,
    request_ids: tuple[str, ...],
    results_by_id: dict[str, JaxActionResult],
) -> JaxActionResult:
    ordered = [results_by_id[request_id] for request_id in request_ids]
    actions = jnp.concatenate([jnp.asarray(result.actions) for result in ordered], axis=0)
    timing = _aggregate_batch_timing([dict(result.timing or {}) for result in ordered], batch_size=len(request_ids))
    return JaxActionResult(request_id=batch_id, actions=actions, timing=timing)


def _aggregate_batch_timing(row_timings: list[dict[str, float]], *, batch_size: int) -> dict[str, float]:
    timing: dict[str, float] = {
        "effective_batch": float(batch_size),
        "policy_effective_batch": float(batch_size),
    }
    for key in (
        "vlm_prefix_forward_ms",
        "vlm_queue_wait_ms",
        "vlm_effective_batch",
        "ae_step_ms",
        "ae_step_total_ms",
        "prefix_slab_map_ms",
    ):
        values = [float(row[key]) for row in row_timings if key in row]
        if values:
            timing[key] = sum(values) / len(values)
    for key in (
        "vlm_request_queue_wait_ms",
        "vlm_request_transfer_ms",
        "prefix_queue_wait_ms",
        "prefix_transfer_ms",
        "prefix_admit_wait_ms",
        "ae_result_queue_wait_ms",
        "ae_result_transfer_ms",
        "va_split_transfer_ms",
        "va_split_queue_wait_ms",
        "prefix_pool_write_ms",
        "prefix_pool_compact_ms",
        "prefix_pool_overhead_ms",
        "infer_queue_wait_ms",
    ):
        values = [float(row[key]) for row in row_timings if key in row]
        if values:
            timing[key] = sum(values) / len(values)
    ae_batch_values = [float(row["ae_effective_batch"]) for row in row_timings if "ae_effective_batch" in row]
    if ae_batch_values:
        timing["ae_effective_batch"] = sum(ae_batch_values) / len(ae_batch_values)
        timing["ae_effective_batch_mean"] = timing["ae_effective_batch"]
    return timing


def _mark_collected_result(
    result: JaxActionResult,
    *,
    get_start_ns: int | None = None,
    get_end_ns: int | None = None,
) -> JaxActionResult:
    timing = dict(result.timing or {})
    result_enqueue_ns = timing.pop("_ae_result_enqueue_ns", None)
    queue_wait_ms, transfer_ms = queue_wait_and_transfer_ms(
        enqueue_ns=result_enqueue_ns,
        get_start_ns=get_start_ns,
        get_end_ns=get_end_ns,
    )
    if result_enqueue_ns is not None and get_start_ns is not None and get_end_ns is not None:
        timing["ae_result_queue_wait_ms"] = queue_wait_ms
        timing["ae_result_transfer_ms"] = transfer_ms
    elif result_enqueue_ns is not None:
        timing["ae_result_queue_wait_ms"] = max(0.0, (time.monotonic_ns() - float(result_enqueue_ns)) / 1_000_000)
        timing["ae_result_transfer_ms"] = 0.0
    transfer_keys = ("vlm_request_transfer_ms", "prefix_transfer_ms", "ae_result_transfer_ms")
    transfer_values = [float(timing[key]) for key in transfer_keys if key in timing]
    if transfer_values:
        timing["va_split_transfer_ms"] = sum(transfer_values)
    queue_wait_keys = (
        "vlm_request_queue_wait_ms",
        "vlm_queue_wait_ms",
        "prefix_queue_wait_ms",
        "prefix_admit_wait_ms",
        "ae_result_queue_wait_ms",
    )
    queue_wait_values = [float(timing[key]) for key in queue_wait_keys if key in timing]
    if queue_wait_values:
        timing["va_split_queue_wait_ms"] = sum(queue_wait_values)
    return replace(result, timing=timing)
