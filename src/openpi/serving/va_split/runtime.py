from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import replace
import os
import threading
import time
import uuid

import torch

from openpi.serving.va_split.ae_process import AEProcess
from openpi.serving.va_split.ae_process import AEWorker
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import BatchRequestEnvelope
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError
from openpi.serving.va_split.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split.timing import timed_queue_get
from openpi.serving.va_split.vlm_process import VLMProcess
from openpi.serving.va_split.vlm_process import VLMWorker


class LocalVASplitRuntime:
    """In-process runtime that exercises the same VLM/AE worker split."""

    def __init__(self, *, model, device: str, max_ae_batch_size: int = 8, max_prefix_slots: int | None = None):
        self.vlm_worker = VLMWorker(model=model, device=device, max_live_features=max_prefix_slots)
        self.ae_worker = AEWorker(
            model=model,
            device=device,
            max_batch_size=max_ae_batch_size,
            max_prefix_slots=max_prefix_slots,
        )

    def infer(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        request_id = str(uuid.uuid4())
        request = RequestEnvelope(
            request_id=request_id,
            observation=observation,
            sample_kwargs=dict(sample_kwargs),
            enqueue_ns=time.monotonic_ns(),
        )
        ready = self.vlm_worker.handle_request(request)
        self.ae_worker.add_prefix(ready)

        while request_id in self.ae_worker.active:
            results, releases = self.ae_worker.step_once()
            for release in releases:
                self.vlm_worker.release(release)
            for result in results:
                if result.request_id == request_id:
                    return result
        raise RuntimeError(f"Request {request_id} finished without an ActionResult")

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        ready_messages = self.vlm_worker.handle_batch_request(
            BatchRequestEnvelope(
                batch_id=batch_id,
                request_ids=request_ids,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        for ready in ready_messages:
            self.ae_worker.add_prefix(ready)

        results_by_id: dict[str, ActionResult] = {}
        while len(results_by_id) < batch_size:
            results, releases = self.ae_worker.step_once()
            for release in releases:
                self.vlm_worker.release(release)
            for result in results:
                results_by_id[result.request_id] = result
        return _combine_ordered_results(batch_id, request_ids, results_by_id)


def _prepare_model(model_factory: Callable[[], object], device: str):
    model = model_factory()
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _apply_env_updates(env_updates: dict[str, str | None] | None) -> None:
    for key, value in (env_updates or {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run_vlm_process(
    model_factory,
    device,
    request_queue,
    prefix_queue,
    release_queue,
    max_vlm_batch_size,
    max_vlm_wait_ms,
    max_live_features,
    env_updates=None,
) -> None:
    _apply_env_updates(env_updates)
    model = _prepare_model(model_factory, device)
    VLMProcess(
        model=model,
        device=device,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=release_queue,
        max_batch_size=max_vlm_batch_size,
        max_wait_ms=max_vlm_wait_ms,
        max_live_features=max_live_features,
    ).run()


def _run_ae_process(
    model_factory,
    device,
    prefix_queue,
    result_queue,
    release_queue,
    max_ae_batch_size,
    max_prefix_slots,
    env_updates=None,
) -> None:
    _apply_env_updates(env_updates)
    model = _prepare_model(model_factory, device)
    AEProcess(
        model=model,
        device=device,
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
    ).run()


class ProcessVASplitRuntime:
    """Queue-based VLM/AE process runtime for the basic tensor-sharing implementation."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], object],
        device: str,
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
        self._model_factory = model_factory
        self._device = device
        self._max_ae_batch_size = max_ae_batch_size
        self._max_vlm_batch_size = max_vlm_batch_size
        self._max_vlm_wait_ms = max_vlm_wait_ms
        self._max_prefix_slots = max_prefix_slots
        self._result_timeout_s = result_timeout_s
        self._pending_results: dict[str, ActionResult] = {}
        self._pending_errors: dict[str | None, WorkerError] = {}
        self._shutdown_seen = False
        self._closed = False
        self._condition = threading.Condition()

        ctx = torch.multiprocessing.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._release_queue = ctx.Queue()
        self._vlm_process = ctx.Process(
            target=_run_vlm_process,
            args=(
                model_factory,
                device,
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
            target=_run_ae_process,
            args=(
                model_factory,
                device,
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

    def infer(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        if self._closed:
            raise RuntimeError("VA split runtime is shut down")
        request_id = str(uuid.uuid4())
        self._request_queue.put(
            RequestEnvelope(
                request_id=request_id,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        return self._wait_for_result(request_id)

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        if self._closed:
            raise RuntimeError("VA split runtime is shut down")
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        self._request_queue.put(
            BatchRequestEnvelope(
                batch_id=batch_id,
                request_ids=request_ids,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        results_by_id = {request_id: self._wait_for_result(request_id) for request_id in request_ids}
        return _combine_ordered_results(batch_id, request_ids, results_by_id)

    def _wait_for_result(self, request_id: str) -> ActionResult:
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
                    raise RuntimeError("VA split worker shut down before producing a result")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for VA split result {request_id}")
                self._condition.wait(timeout=remaining)

    def _collect_results(self) -> None:
        while True:
            try:
                message, get_start_ns, get_end_ns = timed_queue_get(self._result_queue)
            except (EOFError, OSError):
                return
            with self._condition:
                if isinstance(message, ActionResult):
                    self._pending_results[message.request_id] = _mark_collected_result(
                        message,
                        get_start_ns=get_start_ns,
                        get_end_ns=get_end_ns,
                    )
                elif isinstance(message, WorkerError):
                    self._pending_errors[message.request_id] = message
                elif isinstance(message, Shutdown):
                    self._shutdown_seen = True
                self._condition.notify_all()
            if isinstance(message, Shutdown):
                return

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._request_queue.put(Shutdown())
        for process in (self._vlm_process, self._ae_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._result_thread.join(timeout=5)

    def reset(self) -> None:
        pass


def _worker_error_to_runtime_error(error: WorkerError) -> RuntimeError:
    detail = f"\n{error.traceback}" if error.traceback else ""
    return RuntimeError(f"VA split worker failed for {error.request_id}: {error.error}{detail}")


def _combine_ordered_results(
    batch_id: str,
    request_ids: tuple[str, ...],
    results_by_id: dict[str, ActionResult],
) -> ActionResult:
    ordered = [results_by_id[request_id] for request_id in request_ids]
    actions = torch.cat([result.actions for result in ordered], dim=0)
    timing = _aggregate_batch_timing([dict(result.timing or {}) for result in ordered], batch_size=len(request_ids))
    return ActionResult(request_id=batch_id, actions=actions, timing=timing)


def _aggregate_batch_timing(row_timings: list[dict[str, float]], *, batch_size: int) -> dict[str, float]:
    timing: dict[str, float] = {
        "effective_batch": float(batch_size),
        "policy_effective_batch": float(batch_size),
    }
    for key in ("vlm_prefix_forward_ms", "vlm_queue_wait_ms", "vlm_effective_batch", "ae_step_ms", "ae_step_total_ms"):
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
        "prefix_lane_ingest_ms",
        "prefix_lane_compact_ms",
        "prefix_lane_overhead_ms",
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
    result: ActionResult,
    *,
    get_start_ns: int | None = None,
    get_end_ns: int | None = None,
) -> ActionResult:
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
        # Legacy fallback when get split is unavailable.
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
