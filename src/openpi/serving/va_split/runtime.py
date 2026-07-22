from __future__ import annotations

from collections.abc import Callable
import os
import threading
import time
import uuid

import torch

from openpi.serving.va_split.ae_process import AEProcess
from openpi.serving.va_split.ae_process import AEWorker
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError
from openpi.serving.va_split.vlm_process import VLMProcess
from openpi.serving.va_split.vlm_process import VLMWorker


class LocalVASplitRuntime:
    """In-process runtime that exercises the same VLM/AE worker split."""

    def __init__(self, *, model, device: str, max_ae_batch_size: int = 8):
        self.vlm_worker = VLMWorker(model=model, device=device)
        self.ae_worker = AEWorker(model=model, device=device, max_batch_size=max_ae_batch_size)

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


def _run_vlm_process(model_factory, device, request_queue, prefix_queue, release_queue, env_updates=None) -> None:
    _apply_env_updates(env_updates)
    model = _prepare_model(model_factory, device)
    VLMProcess(
        model=model,
        device=device,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=release_queue,
    ).run()


def _run_ae_process(
    model_factory, device, prefix_queue, result_queue, release_queue, max_ae_batch_size, env_updates=None
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
    ).run()


class ProcessVASplitRuntime:
    """Queue-based VLM/AE process runtime for the basic tensor-sharing implementation."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], object],
        device: str,
        max_ae_batch_size: int = 8,
        start_method: str = "spawn",
        result_timeout_s: float = 120.0,
        vlm_env_updates: dict[str, str | None] | None = None,
        ae_env_updates: dict[str, str | None] | None = None,
    ):
        self._model_factory = model_factory
        self._device = device
        self._max_ae_batch_size = max_ae_batch_size
        self._result_timeout_s = result_timeout_s
        self._pending_results: dict[str, ActionResult] = {}
        self._pending_errors: dict[str | None, WorkerError] = {}
        self._shutdown_seen = False
        self._condition = threading.Condition()

        ctx = torch.multiprocessing.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._release_queue = ctx.Queue()
        self._vlm_process = ctx.Process(
            target=_run_vlm_process,
            args=(model_factory, device, self._request_queue, self._prefix_queue, self._release_queue, vlm_env_updates),
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
                ae_env_updates,
            ),
            daemon=True,
        )
        self._vlm_process.start()
        self._ae_process.start()
        self._result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self._result_thread.start()

    def infer(self, observation: dict, sample_kwargs: dict) -> ActionResult:
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
                message = self._result_queue.get()
            except (EOFError, OSError):
                return
            with self._condition:
                if isinstance(message, ActionResult):
                    self._pending_results[message.request_id] = message
                elif isinstance(message, WorkerError):
                    self._pending_errors[message.request_id] = message
                elif isinstance(message, Shutdown):
                    self._shutdown_seen = True
                self._condition.notify_all()
            if isinstance(message, Shutdown):
                return

    def shutdown(self) -> None:
        self._request_queue.put(Shutdown())
        for process in (self._vlm_process, self._ae_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def reset(self) -> None:
        pass


def _worker_error_to_runtime_error(error: WorkerError) -> RuntimeError:
    detail = f"\n{error.traceback}" if error.traceback else ""
    return RuntimeError(f"VA split worker failed for {error.request_id}: {error.error}{detail}")
