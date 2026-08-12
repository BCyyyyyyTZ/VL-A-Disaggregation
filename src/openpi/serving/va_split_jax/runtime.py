from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import replace
import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any
import uuid

import jax
import jax.numpy as jnp

from openpi.serving.va_split_jax.ae_process import JaxAEProcess
from openpi.serving.va_split_jax.ae_process import JaxAEWorker
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import ModelWithPrefixTemplate
from openpi.serving.va_split_jax.compile import make_model_noise_factory
from openpi.serving.va_split_jax.compile import make_model_observation_factory
from openpi.serving.va_split_jax.compile import make_prefix_feature_template
from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
from openpi.serving.va_split_jax.compile import prune_split_model_for_role
from openpi.serving.va_split_jax.compile import warmup_vlm_ae_slab_writes
from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
from openpi.serving.va_split_jax.device_slab import DeviceSlabHandle
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.multigpu_config import JaxMultiGpuVASplitConfig
from openpi.serving.va_split_jax.multigpu_router import JaxVlmRouteState
from openpi.serving.va_split_jax.multigpu_router import LeastBacklogVlmRouter
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.process_entry import run_ae_worker_entry
from openpi.serving.va_split_jax.process_entry import run_vlm_worker_entry
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionBatchRow
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxBatchRequestEnvelope
from openpi.serving.va_split_jax.types import JaxCompileWarmupDone
from openpi.serving.va_split_jax.types import JaxLaneCredits
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
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
                    return replace(result, actions=_result_actions_array(result.actions))
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
    source_worker_id=None,
    cross_card_transfer_strategy="device-direct",
) -> None:
    _apply_env_updates(env_updates)
    model = model_factory()
    observation_factory = make_model_observation_factory(model)
    model = prune_split_model_for_role(model, role="vlm")
    vlm_warmup_batches = 0.0
    if compile_config is not None:
        model = maybe_jit_vlm_model(model, compile_config)
        if warmup_queue is not None:
            stats = warmup_vlm_prefix_model(
                model=model,
                observation_factory=observation_factory,
                max_vlm_batch_size=max_vlm_batch_size,
                config=compile_config,
            )
            vlm_warmup_batches = float(stats["jax_warmup_batches"])
    backend = make_default_device_slab_backend(
        device_ordinal=1 if len(jax.devices()) > 1 else 0,
        cross_card_transfer_strategy=cross_card_transfer_strategy,
    )
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=release_queue,
        max_batch_size=max_vlm_batch_size,
        max_wait_ms=max_vlm_wait_ms,
        max_live_features=max_prefix_slots,
        source_worker_id=source_worker_id,
        backend=backend,
    )
    # AE-first handshake: wait for slab export + credits on release_queue.
    process.wait_for_ae_export()
    if compile_config is not None and warmup_queue is not None and process.worker._writable_slab_tree is not None:  # noqa: SLF001
        pool_stats = warmup_vlm_ae_slab_writes(
            model=model,
            observation_factory=observation_factory,
            backend=backend,
            slab_tree=process.worker._writable_slab_tree,  # noqa: SLF001
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
    max_prefix_admits_per_drain=1,
) -> None:
    _apply_env_updates(env_updates)
    model, template = _unwrap_model_with_prefix_template(model_factory())
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model) if compile_config is not None else None
    template = template or getattr(model, "_va_split_prefix_feature_template", None)
    if template is None:
        template = make_prefix_feature_template(model, observation_factory)
    model = prune_split_model_for_role(model, role="ae")
    if compile_config is not None:
        model = maybe_jit_ae_model(model, compile_config)
    process = JaxAEProcess(
        model=model,
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
        compile_config=compile_config,
        noise_factory=noise_factory,
        max_prefix_admits_per_drain=max_prefix_admits_per_drain,
    )
    ipc_batches = process.bootstrap_owned_pool(template)
    if warmup_queue is not None:
        warmup_queue.put(JaxCompileWarmupDone(role="ae", jax_warmup_batches=float(ipc_batches)))
    process.run()


def _unwrap_model_with_prefix_template(value: Any) -> tuple[Any, Any | None]:
    if isinstance(value, ModelWithPrefixTemplate):
        return value.model, value.prefix_template
    return value, None


class JaxMultiGpuReleaseFanout:
    """Route AE control messages back to the VLM worker that owns each lane."""

    def __init__(self, release_queues: dict[str, Any], *, slab_device_ordinals: dict[str, int] | None = None):
        if not release_queues:
            raise ValueError("release_queues must be non-empty")
        self._release_queues = dict(release_queues)
        self._worker_ids = tuple(release_queues)
        self._next_credit_worker = 0
        self._slab_device_ordinals = dict(slab_device_ordinals or {})

    def put(self, message: object) -> None:
        if isinstance(message, JaxPrefixSlabReady):
            for worker_id in self._worker_ids:
                self._release_queues[worker_id].put(
                    _prefix_slab_ready_for_worker(message, self._slab_device_ordinals.get(worker_id))
                )
            return
        if isinstance(message, JaxLaneCredits):
            for worker_id, credits in self._partition_lane_credits(message).items():
                if credits.lane_ids:
                    self._release_queues[worker_id].put(credits)
            return
        if isinstance(message, JaxReleaseFeature):
            worker_id = _release_queue_key(message)
            if worker_id in self._release_queues:
                self._release_queues[worker_id].put(message)
                return
            self._release_queues[self._worker_ids[0]].put(message)
            return
        for worker_id in self._worker_ids:
            self._release_queues[worker_id].put(message)

    def _partition_lane_credits(self, credits: JaxLaneCredits) -> dict[str, JaxLaneCredits]:
        lanes_by_worker = {worker_id: [] for worker_id in self._worker_ids}
        for lane_id in credits.lane_ids:
            worker_id = self._worker_ids[self._next_credit_worker % len(self._worker_ids)]
            lanes_by_worker[worker_id].append(int(lane_id))
            self._next_credit_worker += 1
        return {worker_id: JaxLaneCredits(lane_ids=tuple(lanes)) for worker_id, lanes in lanes_by_worker.items()}


def _prefix_slab_ready_for_worker(message: JaxPrefixSlabReady, device_ordinal: int | None) -> JaxPrefixSlabReady:
    if device_ordinal is None:
        return message
    rewritten_tree = _replace_slab_handle_device_ordinal(
        message.slab.slab_handle_tree,
        device_ordinal=device_ordinal,
    )
    if rewritten_tree is message.slab.slab_handle_tree:
        return message
    return replace(message, slab=replace(message.slab, slab_handle_tree=rewritten_tree))


def _replace_slab_handle_device_ordinal(value: Any, *, device_ordinal: int) -> Any:
    if isinstance(value, DeviceSlabHandle):
        return replace(value, device_ordinal=int(device_ordinal))
    if isinstance(value, tuple):
        return tuple(_replace_slab_handle_device_ordinal(item, device_ordinal=device_ordinal) for item in value)
    if isinstance(value, list):
        return [_replace_slab_handle_device_ordinal(item, device_ordinal=device_ordinal) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_slab_handle_device_ordinal(item, device_ordinal=device_ordinal)
            for key, item in value.items()
        }
    return value


class JaxMultiGpuProcessVASplitRuntime:
    """One AE process with N VLM processes pinned to separate CUDA-visible devices."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], object] | None = None,
        vlm_model_factory: Callable[[], object] | None = None,
        ae_model_factory: Callable[[], object] | None = None,
        config: JaxMultiGpuVASplitConfig,
        result_timeout_s: float = 120.0,
        vlm_env_updates: dict[str, str | None] | None = None,
        ae_env_updates: dict[str, str | None] | None = None,
        compile_config: JaxCompileConfig | None = None,
        warmup_timeout_s: float | None = None,
    ):
        self._config = config
        self._result_timeout_s = result_timeout_s
        self._pending_results: dict[str, JaxActionResult] = {}
        self._pending_errors: dict[str | None, JaxWorkerError] = {}
        self._request_to_worker: dict[str, str] = {}
        self._shutdown_seen = False
        self._closed = False
        self._condition = threading.Condition()
        self._compile_timing = {"jax_warmup_batches": 0.0}
        self._router = LeastBacklogVlmRouter(config.vlm_worker_ids)
        self._model_factory = model_factory or vlm_model_factory or ae_model_factory
        self._vlm_model_factory = vlm_model_factory or self._model_factory
        self._ae_model_factory = ae_model_factory or self._model_factory
        initial_per_worker = max(1, int(config.max_prefix_slots or 1) // max(1, config.num_vlm_workers))
        for worker_id in config.vlm_worker_ids:
            self._router.update(worker_id, JaxVlmRouteState(available_credits=initial_per_worker))

        ctx = mp.get_context(config.start_method)
        self._request_queues = {worker_id: ctx.Queue() for worker_id in config.vlm_worker_ids}
        self._release_queues = {worker_id: ctx.Queue() for worker_id in config.vlm_worker_ids}
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._warmup_queue = ctx.Queue()
        release_fanout = JaxMultiGpuReleaseFanout(
            self._release_queues,
            slab_device_ordinals=dict.fromkeys(config.vlm_worker_ids, 0),
        )

        self._ae_process = ctx.Process(
            target=run_ae_worker_entry,
            args=(
                self._ae_model_factory,
                self._prefix_queue,
                self._result_queue,
                release_fanout,
                config.max_ae_batch_size,
                config.max_prefix_slots,
                compile_config,
                None,
                self._warmup_queue,
                config.max_prefix_admits_per_drain,
            ),
            kwargs={"device": config.ae_device, "env_updates": ae_env_updates},
            daemon=True,
        )
        self._vlm_processes = {}
        for worker_id, device in zip(config.vlm_worker_ids, config.vlm_devices, strict=True):
            self._vlm_processes[worker_id] = ctx.Process(
                target=run_vlm_worker_entry,
                args=(
                    self._vlm_model_factory,
                    self._request_queues[worker_id],
                    self._prefix_queue,
                    self._release_queues[worker_id],
                    config.max_vlm_batch_size,
                    config.max_vlm_wait_ms,
                    config.max_prefix_slots,
                    compile_config,
                    None,
                    self._warmup_queue,
                    worker_id,
                    config.cross_card_transfer_strategy,
                ),
                # VLM sees the AE slab GPU first and its compute GPU second.
                kwargs={"device": f"{config.ae_device},{device}", "env_updates": vlm_env_updates},
                daemon=True,
            )

        timeout = warmup_timeout_s if warmup_timeout_s is not None else result_timeout_s
        self._ae_process.start()
        ae_compile = _collect_compile_warmup(
            self._warmup_queue,
            expected_roles=("ae",),
            timeout_s=timeout,
            vlm_process=None,
            ae_process=self._ae_process,
        )
        for process in self._vlm_processes.values():
            process.start()
        vlm_compile = _collect_compile_warmup_count(
            self._warmup_queue,
            expected_role="vlm",
            expected_count=len(self._vlm_processes),
            timeout_s=timeout,
            processes=(*self._vlm_processes.values(), self._ae_process),
        )
        self._compile_timing = {
            "jax_warmup_batches": float(ae_compile["jax_warmup_batches"] + vlm_compile["jax_warmup_batches"])
        }
        self._result_thread = threading.Thread(target=self._collect_results, daemon=True)
        self._result_thread.start()

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        if self._closed:
            raise RuntimeError("JAX multi-GPU VA split runtime is shut down")
        request_id = str(uuid.uuid4())
        worker_id = self._choose_worker(count=1)
        self._request_to_worker[request_id] = worker_id
        self._request_queues[worker_id].put(
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
            raise RuntimeError("JAX multi-GPU VA split runtime is shut down")
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        worker_id = self._choose_worker(count=batch_size)
        for request_id in request_ids:
            self._request_to_worker[request_id] = worker_id
        self._request_queues[worker_id].put(
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

    def _choose_worker(self, *, count: int) -> str:
        decision = self._router.choose_worker()
        self._router.mark_enqueued(decision.worker_id, count=count)
        self._router.mark_dispatched(decision.worker_id, count=count)
        return decision.worker_id

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
                    raise RuntimeError("JAX multi-GPU VA split worker shut down before producing a result")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for JAX multi-GPU VA split result {request_id}")
                self._condition.wait(timeout=remaining)

    def _collect_results(self) -> None:
        while True:
            try:
                message, get_start_ns, get_end_ns = timed_queue_get(self._result_queue)
            except (EOFError, OSError):
                return
            with self._condition:
                if isinstance(message, JaxActionResult):
                    worker_id = self._request_to_worker.pop(message.request_id, None)
                    if worker_id is not None:
                        self._router.mark_released(worker_id)
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
        for request_queue in self._request_queues.values():
            with contextlib.suppress(Exception):
                request_queue.put(JaxShutdown())
        for process in (*self._vlm_processes.values(), self._ae_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._result_thread.join(timeout=5)

    def reset(self) -> None:
        pass

    @property
    def compile_timing(self) -> dict[str, float]:
        return dict(self._compile_timing)


class JaxProcessVASplitRuntime:
    def __init__(
        self,
        *,
        model_factory: Callable[[], object] | None = None,
        vlm_model_factory: Callable[[], object] | None = None,
        ae_model_factory: Callable[[], object] | None = None,
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
        self._model_factory = model_factory or vlm_model_factory or ae_model_factory
        self._vlm_model_factory = vlm_model_factory or self._model_factory
        self._ae_model_factory = ae_model_factory or self._model_factory

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
                self._ae_model_factory,
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
                self._vlm_model_factory,
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
        ae_compile_timing = _collect_compile_warmup(
            self._warmup_queue,
            expected_roles=("ae",),
            timeout_s=warmup_timeout_s if warmup_timeout_s is not None else result_timeout_s,
            vlm_process=None,
            ae_process=self._ae_process,
        )
        self._vlm_process.start()
        vlm_compile_timing = _collect_compile_warmup(
            self._warmup_queue,
            expected_roles=("vlm",),
            timeout_s=warmup_timeout_s if warmup_timeout_s is not None else result_timeout_s,
            vlm_process=self._vlm_process,
            ae_process=self._ae_process,
        )
        self._compile_timing = {
            "jax_warmup_batches": float(
                ae_compile_timing["jax_warmup_batches"] + vlm_compile_timing["jax_warmup_batches"]
            )
        }
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


def _release_queue_key(release: JaxReleaseFeature) -> str | None:
    return release.source_worker_id


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
        if vlm_process is not None and not vlm_process.is_alive() and "vlm" not in seen:
            raise RuntimeError("VLM process exited before reporting compile warmup")
        if ae_process is not None and not ae_process.is_alive() and "ae" not in seen:
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


def _collect_compile_warmup_count(
    warmup_queue,
    *,
    expected_role: str,
    expected_count: int,
    timeout_s: float,
    processes: tuple[Any, ...],
) -> dict[str, float]:
    if expected_count <= 0:
        return {"jax_warmup_batches": 0.0}
    deadline = time.monotonic() + max(timeout_s, 1.0)
    seen = 0
    total = 0.0
    while seen < expected_count:
        if any(process is not None and not process.is_alive() for process in processes) and seen == 0:
            raise RuntimeError(f"Process exited before reporting {expected_role} compile warmup")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for {expected_count} {expected_role} compile warmup reports; got {seen}"
            )
        try:
            message = warmup_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if not isinstance(message, JaxCompileWarmupDone):
            raise RuntimeError(f"Unexpected warmup queue message: {type(message)}")
        if message.role != expected_role:
            raise RuntimeError(f"Unexpected warmup role {message.role!r}; expected {expected_role!r}")
        total += float(message.jax_warmup_batches)
        seen += 1
    return {"jax_warmup_batches": total}


def _combine_ordered_results(
    batch_id: str,
    request_ids: tuple[str, ...],
    results_by_id: dict[str, JaxActionResult],
) -> JaxActionResult:
    ordered = [results_by_id[request_id] for request_id in request_ids]
    actions = jnp.concatenate([_result_actions_array(result.actions) for result in ordered], axis=0)
    timing = _aggregate_batch_timing([dict(result.timing or {}) for result in ordered], batch_size=len(request_ids))
    return JaxActionResult(request_id=batch_id, actions=actions, timing=timing)


def _result_actions_array(actions: Any) -> jax.Array:
    if isinstance(actions, JaxActionBatchRow):
        return jnp.asarray(actions.batch[actions.row : actions.row + 1])
    return jnp.asarray(actions)


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
        "vlm_sample_kwargs_stage_ms",
        "vlm_observation_stack_ms",
        "vlm_to_jax_tree_ms",
        "vlm_observation_from_dict_ms",
        "vlm_observation_uint8_normalize_ms",
        "vlm_observation_construct_ms",
        "vlm_observation_uint8_images",
        "vlm_observation_float32_images",
        "vlm_observation_other_images",
        "vlm_effective_batch",
        "vlm_slab_write_ms",
        "vlm_slab_write_total_ms",
        "vlm_slab_write_contiguous_batch",
        "vlm_slab_write_rows",
        "ae_init_denoise_ms",
        "ae_init_denoise_noise_fast_path",
        "ae_step_ms",
        "ae_step_total_ms",
        "ae_prefix_view_ms",
        "ae_prefix_view_cache_hit",
        "ae_prefix_view_total_ms",
        "ae_state_batch_stage_ms",
        "ae_state_batch_stage_total_ms",
        "ae_denoise_enqueue_ms",
        "ae_denoise_enqueue_total_ms",
        "ae_update_stage_ms",
        "ae_update_stage_total_ms",
        "ae_complete_block_ms",
        "ae_complete_block_total_ms",
        "ae_result_slice_ms",
        "ae_result_slice_total_ms",
        "ae_result_device_get_ms",
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
        "prefix_shadow_copy_ms",
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
