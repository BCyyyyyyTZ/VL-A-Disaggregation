from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import replace
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid

import jax.numpy as jnp
import numpy as np

from openpi.serving.va_split_jax.ae_process import JaxAEProcess
from openpi.serving.va_split_jax.ae_process import JaxAEWorker
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import make_model_noise_factory
from openpi.serving.va_split_jax.compile import make_model_observation_factory
from openpi.serving.va_split_jax.compile import maybe_jit_split_model
from openpi.serving.va_split_jax.compile import warmup_vlm_ae_slab_writes
from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxBatchRequestEnvelope
from openpi.serving.va_split_jax.types import JaxCompileWarmupDone
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
        self._max_prefix_slots = max_prefix_slots
        self.ae_worker = JaxAEWorker(
            model=model,
            max_batch_size=max_ae_batch_size,
            max_prefix_slots=max_prefix_slots,
            backend=backend,
            owned_pool=self._prefix_pool,
        )
        self.vlm_worker = JaxVLMWorker(
            model=model,
            max_live_features=max_prefix_slots,
            backend=backend,
            shared_pool=self._prefix_pool,
        )
        self._model = model
        self._bootstrapped = False

    def _ensure_bootstrapped(self, observation: dict) -> None:
        if self._bootstrapped:
            return
        # Build a 1-row template from the request observation shapes via the model.
        from openpi.models import model as _model
        from openpi.serving.va_split_jax.vlm_process import _to_jax_tree

        obs = _model.Observation.from_dict(_to_jax_tree(_single_row_observation(observation)))
        template = self._model.build_prefix_feature(None, obs)
        slab_ready, credits = self.ae_worker.initialize_owned_pool(template)
        self.vlm_worker.attach_shared_pool(self._prefix_pool, credits)
        del slab_ready
        self._bootstrapped = True

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        self._ensure_bootstrapped(observation)
        request_id = str(uuid.uuid4())
        ready = self.vlm_worker.handle_request(
            JaxRequestEnvelope(
                request_id=request_id,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        self.ae_worker.add_prefix(ready)
        pending = self.ae_worker.take_pending_lane_credits()
        if pending is not None:
            self.vlm_worker.grant_lane_credits(pending)
        while request_id in self.ae_worker.active:
            results, releases = self.ae_worker.step_once()
            pending = self.ae_worker.take_pending_lane_credits()
            if pending is not None:
                self.vlm_worker.grant_lane_credits(pending)
            for release in releases:
                self.vlm_worker.release(release)
            for result in results:
                if result.request_id == request_id:
                    return result
        raise RuntimeError(f"Request {request_id} finished without an action result")

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        self._ensure_bootstrapped(observation)
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
        for ready in ready_messages:
            self.ae_worker.add_prefix(ready)
            pending = self.ae_worker.take_pending_lane_credits()
            if pending is not None:
                self.vlm_worker.grant_lane_credits(pending)

        results_by_id: dict[str, JaxActionResult] = {}
        while len(results_by_id) < batch_size:
            results, releases = self.ae_worker.step_once()
            pending = self.ae_worker.take_pending_lane_credits()
            if pending is not None:
                self.vlm_worker.grant_lane_credits(pending)
            for release in releases:
                self.vlm_worker.release(release)
            for result in results:
                results_by_id[result.request_id] = result
        return _combine_ordered_results(batch_id, request_ids, results_by_id)

    @property
    def compile_timing(self) -> dict[str, float]:
        return {"jax_warmup_batches": 0.0}


def _single_row_observation(observation: dict) -> dict:
    """Take the first row of a possibly batched observation dict."""

    def _slice(value):
        if isinstance(value, dict):
            return {key: _slice(item) for key, item in value.items()}
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            return value[:1]
        return value

    return _slice(observation)


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
    compile_config=None,
    env_updates=None,
    warmup_queue=None,
) -> None:
    _apply_env_updates(env_updates)
    model = model_factory()
    observation_factory = make_model_observation_factory(model)
    vlm_warmup_batches = 0.0
    if compile_config is not None:
        model = maybe_jit_split_model(model, compile_config)
        if warmup_queue is not None:
            stats = warmup_vlm_prefix_model(
                model=model,
                observation_factory=observation_factory,
                max_vlm_batch_size=max_vlm_batch_size,
                config=compile_config,
            )
            vlm_warmup_batches = float(stats["jax_warmup_batches"])
    backend = make_default_device_slab_backend()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=release_queue,
        max_batch_size=max_vlm_batch_size,
        max_wait_ms=max_vlm_wait_ms,
        max_live_features=max_prefix_slots,
        backend=backend,
    )
    # AE-first handshake: wait for slab export + credits on release_queue.
    process.wait_for_ae_export()
    if compile_config is not None and warmup_queue is not None and process.worker._writable_slab_tree is not None:
        pool_stats = warmup_vlm_ae_slab_writes(
            model=model,
            observation_factory=observation_factory,
            backend=backend,
            slab_tree=process.worker._writable_slab_tree,
            max_lanes=max_prefix_slots,
            max_vlm_batch_size=max_vlm_batch_size,
            config=compile_config,
        )
        vlm_warmup_batches += float(pool_stats["jax_warmup_batches"])
    if warmup_queue is not None:
        warmup_queue.put(JaxCompileWarmupDone(role="vlm", jax_warmup_batches=vlm_warmup_batches))
    process.run()


def _run_jax_ae_process(
    model_factory,
    prefix_queue,
    result_queue,
    release_queue,
    max_ae_batch_size,
    max_prefix_slots,
    compile_config=None,
    env_updates=None,
    warmup_queue=None,
) -> None:
    _apply_env_updates(env_updates)
    model = model_factory()
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model) if compile_config is not None else None
    if compile_config is not None:
        model = maybe_jit_split_model(model, compile_config)
    process = JaxAEProcess(
        model=model,
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
        compile_config=compile_config,
        noise_factory=noise_factory,
    )
    template = model.build_prefix_feature(None, observation_factory(1))
    ipc_batches = process.bootstrap_owned_pool(template)
    if warmup_queue is not None:
        warmup_queue.put(JaxCompileWarmupDone(role="ae", jax_warmup_batches=float(ipc_batches)))
    process.run()


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
        compile_config: JaxCompileConfig | None = None,
        warmup_timeout_s: float | None = None,
    ):
        if max_prefix_slots is None:
            max_prefix_slots = max_vlm_batch_size * 3
        self._result_timeout_s = result_timeout_s
        self._pending_results: dict[str, JaxActionResult] = {}
        self._pending_errors: dict[str | None, JaxWorkerError] = {}
        self._shutdown_seen = False
        self._closed = False
        self._condition = threading.Condition()
        self._compile_timing = {"jax_warmup_batches": 0.0}

        ctx = mp.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._release_queue = ctx.Queue()
        self._warmup_queue = ctx.Queue()
        # Start AE first so slab export is available when VLM waits.
        self._ae_process = ctx.Process(
            target=_run_jax_ae_process,
            args=(
                model_factory,
                self._prefix_queue,
                self._result_queue,
                self._release_queue,
                max_ae_batch_size,
                max_prefix_slots,
                compile_config,
                ae_env_updates,
                self._warmup_queue,
            ),
            daemon=True,
        )
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
                compile_config,
                vlm_env_updates,
                self._warmup_queue,
            ),
            daemon=True,
        )
        self._ae_process.start()
        self._vlm_process.start()
        self._compile_timing = _collect_compile_warmup(
            self._warmup_queue,
            expected_roles=("vlm", "ae"),
            timeout_s=warmup_timeout_s if warmup_timeout_s is not None else result_timeout_s,
            vlm_process=self._vlm_process,
            ae_process=self._ae_process,
        )
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

    @property
    def compile_timing(self) -> dict[str, float]:
        return dict(self._compile_timing)


def _worker_error_to_runtime_error(error: JaxWorkerError) -> RuntimeError:
    detail = f"\n{error.traceback}" if error.traceback else ""
    return RuntimeError(f"JAX VA split worker failed for {error.request_id}: {error.error}{detail}")


def _collect_compile_warmup(
    warmup_queue,
    *,
    expected_roles: tuple[str, ...],
    timeout_s: float,
    vlm_process,
    ae_process,
) -> dict[str, float]:
    deadline = time.monotonic() + max(timeout_s, 1.0)
    seen: dict[str, float] = {}
    while len(seen) < len(expected_roles):
        if not vlm_process.is_alive() and "vlm" not in seen:
            raise RuntimeError("VLM process exited before reporting compile warmup")
        if not ae_process.is_alive() and "ae" not in seen:
            raise RuntimeError("AE process exited before reporting compile warmup")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(set(expected_roles) - set(seen))
            raise TimeoutError(f"Timed out waiting for JAX compile warmup from: {missing}")
        try:
            message = warmup_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if not isinstance(message, JaxCompileWarmupDone):
            raise RuntimeError(f"Unexpected warmup queue message: {type(message)}")
        seen[message.role] = float(message.jax_warmup_batches)
    return {"jax_warmup_batches": float(sum(seen[role] for role in expected_roles))}


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
        "vlm_batch_wait_ms",
        "vlm_input_stage_ms",
        "vlm_effective_batch",
        "vlm_slab_write_ms",
        "ae_step_ms",
        "ae_step_total_ms",
        "prefix_slab_map_ms",
        "prefix_lane_ingest_ms",
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
    # Only true IPC/queue residency waits. Exclude post-dequeue VLM scheduling
    # (vlm_queue_wait_ms) and post-get admit delay (prefix_admit_wait_ms).
    queue_wait_keys = (
        "vlm_request_queue_wait_ms",
        "prefix_queue_wait_ms",
        "ae_result_queue_wait_ms",
    )
    queue_wait_values = [float(timing[key]) for key in queue_wait_keys if key in timing]
    if queue_wait_values:
        timing["va_split_queue_wait_ms"] = sum(queue_wait_values)
    return replace(result, timing=timing)
