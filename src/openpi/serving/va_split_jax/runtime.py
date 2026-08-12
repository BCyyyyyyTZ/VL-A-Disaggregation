from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import traceback
from typing import Any
import uuid

import jax
import jax.numpy as jnp

from openpi.serving.va_split_jax.ae_process import JaxAEProcess
from openpi.serving.va_split_jax.ae_process import JaxAEWorker
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import make_model_noise_factory
from openpi.serving.va_split_jax.compile import make_model_observation_factory
from openpi.serving.va_split_jax.compile import make_prefix_feature_template
from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
from openpi.serving.va_split_jax.compile import prune_split_model_for_role
from openpi.serving.va_split_jax.compile import warmup_vlm_ae_slab_writes
from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxActionBatchRow
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
        # Defer lane credits until the whole admit burst settles.
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


def _apply_jax_compilation_cache_config() -> None:
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)

    enable_cache = os.environ.get("JAX_ENABLE_COMPILATION_CACHE")
    if enable_cache is not None:
        jax.config.update("jax_enable_compilation_cache", _parse_env_bool(enable_cache))

    min_compile_time = os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS")
    if min_compile_time is not None:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time))

    min_entry_size = os.environ.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES")
    if min_entry_size is not None:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry_size))


def _parse_env_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off"}


_MPS_ENV_KEYS = (
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
)


def _without_mps_env(env_updates: dict[str, str | None] | None) -> dict[str, str | None]:
    updates = dict(env_updates or {})
    for key in _MPS_ENV_KEYS:
        updates[key] = None
    return updates


def _cleanup_mps_pipe_dir(pipe_dir: str) -> None:
    for name in (
        "control",
        "control_privileged",
        "control_lock",
        "log",
        "nvidia-cuda-mps-control.pid",
    ):
        with contextlib.suppress(OSError):
            Path(pipe_dir, name).unlink()


def _stop_mps_daemon() -> bool:
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe_dir:
        return False
    print(f"[jax-split] stopping MPS before final weight load ({pipe_dir})", flush=True)
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
    if log_dir:
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        check=False,
        capture_output=True,
    )
    time.sleep(0.5)
    _cleanup_mps_pipe_dir(pipe_dir)
    return True


def _start_mps_daemon(*, timeout_s: float = 10.0) -> bool:
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    if not pipe_dir:
        return False
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY", pipe_dir)
    Path(pipe_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _cleanup_mps_pipe_dir(pipe_dir)
    print(f"[jax-split] starting MPS before profile ({pipe_dir})", flush=True)
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=False)
    deadline = time.monotonic() + max(timeout_s, 1.0)
    while time.monotonic() < deadline:
        if (
            Path(pipe_dir, "control").exists()
            or Path(pipe_dir, "control_lock").exists()
            or Path(pipe_dir, "log").exists()
        ):
            return True
        time.sleep(0.2)
    raise RuntimeError(f"Failed to restart MPS control daemon at {pipe_dir}")


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
    ready_queue=None,
    warmup_start_queue=None,
    *,
    run_after_warmup: bool = True,
    wait_for_ae_export: bool = True,
    warmup_ae_slab_writes: bool = True,
) -> None:
    def _phase(msg: str) -> None:
        print(f"[jax-vlm] {msg}", flush=True)

    _apply_env_updates(env_updates)
    _apply_jax_compilation_cache_config()
    _phase("load model begin")
    model = model_factory()
    _phase("load model done")
    observation_factory = make_model_observation_factory(model)
    model = prune_split_model_for_role(model, role="vlm")
    _phase("prune done")
    vlm_warmup_batches = 0.0
    if compile_config is not None:
        model = maybe_jit_vlm_model(model, compile_config)
        _phase("maybe_jit done")
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
    _phase("process constructed")
    if ready_queue is not None:
        ready_queue.put("vlm")
        _phase("ready sent")
    if warmup_start_queue is not None:
        _phase("wait warmup_start")
        warmup_start_queue.get()
        _phase("warmup_start received")
    if compile_config is not None and warmup_queue is not None:
        _phase("prefix warmup begin")
        stats = warmup_vlm_prefix_model(
            model=model,
            observation_factory=observation_factory,
            max_vlm_batch_size=max_vlm_batch_size,
            config=compile_config,
        )
        vlm_warmup_batches = float(stats["jax_warmup_batches"])
        _phase(f"prefix warmup done batches={vlm_warmup_batches}")
    if wait_for_ae_export:
        # AE-first handshake: wait for slab export + credits on release_queue.
        _phase("wait_for_ae_export begin")
        process.wait_for_ae_export()
        _phase("wait_for_ae_export done")
    writable_slab_tree = process.worker._writable_slab_tree if wait_for_ae_export else None  # noqa: SLF001
    if (
        wait_for_ae_export
        and warmup_ae_slab_writes
        and compile_config is not None
        and warmup_queue is not None
        and writable_slab_tree is not None
    ):
        _phase("slab-write warmup begin")
        pool_stats = warmup_vlm_ae_slab_writes(
            model=model,
            observation_factory=observation_factory,
            backend=backend,
            slab_tree=writable_slab_tree,
            max_lanes=max_prefix_slots,
            max_vlm_batch_size=max_vlm_batch_size,
            config=compile_config,
        )
        vlm_warmup_batches += float(pool_stats["jax_warmup_batches"])
        _phase(f"slab-write warmup done batches={pool_stats['jax_warmup_batches']}")
    if warmup_queue is not None:
        warmup_queue.put(JaxCompileWarmupDone(role="vlm", jax_warmup_batches=vlm_warmup_batches))
        _phase("WarmupDone sent")
    if not run_after_warmup:
        _phase("exit after warmup")
        return
    _phase("process.run begin")
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
    ready_queue=None,
    warmup_start_queue=None,
    prefix_template_factory=None,
    *,
    run_after_warmup: bool = True,
) -> None:
    _apply_env_updates(env_updates)
    _apply_jax_compilation_cache_config()
    model = model_factory()
    observation_factory = make_model_observation_factory(model)
    noise_factory = make_model_noise_factory(model) if compile_config is not None else None
    template = (
        prefix_template_factory()
        if prefix_template_factory is not None
        else make_prefix_feature_template(model, observation_factory)
    )
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
    )
    if ready_queue is not None:
        ready_queue.put("ae")
    if warmup_start_queue is not None:
        warmup_start_queue.get()
    ipc_batches = process.bootstrap_owned_pool(template)
    if warmup_queue is not None:
        warmup_queue.put(JaxCompileWarmupDone(role="ae", jax_warmup_batches=float(ipc_batches)))
    if not run_after_warmup:
        return
    process.run()


def _run_jax_process_entrypoint(role: str, target, warmup_queue, target_args: tuple, target_kwargs: dict) -> None:
    try:
        target(*target_args, **target_kwargs)
    except Exception as exc:
        if warmup_queue is not None:
            with contextlib.suppress(Exception):
                warmup_queue.put(
                    JaxWorkerError(
                        request_id=None,
                        error=f"{role} process failed before/while reporting compile warmup: {exc}",
                        traceback=traceback.format_exc(),
                    )
                )
        raise


def _start_jax_ae_process(
    ctx,
    *,
    model_factory,
    prefix_queue,
    result_queue,
    release_queue,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    compile_config: JaxCompileConfig | None,
    env_updates: dict[str, str | None] | None,
    warmup_queue,
    ready_queue,
    warmup_start_queue,
    prefix_template_factory,
    run_after_warmup: bool,
):
    target_args = (
        model_factory,
        prefix_queue,
        result_queue,
        release_queue,
        max_ae_batch_size,
        max_prefix_slots,
        compile_config,
        env_updates,
        warmup_queue,
        ready_queue,
        warmup_start_queue,
        prefix_template_factory,
    )
    process = ctx.Process(
        target=_run_jax_process_entrypoint,
        args=(
            "ae",
            _run_jax_ae_process,
            warmup_queue,
            target_args,
            {"run_after_warmup": run_after_warmup},
        ),
        daemon=True,
    )
    process.start()
    return process


def _start_jax_vlm_process(
    ctx,
    *,
    model_factory,
    request_queue,
    prefix_queue,
    release_queue,
    max_vlm_batch_size: int,
    max_vlm_wait_ms: float,
    max_prefix_slots: int,
    compile_config: JaxCompileConfig | None,
    env_updates: dict[str, str | None] | None,
    warmup_queue,
    ready_queue,
    warmup_start_queue,
    run_after_warmup: bool,
    wait_for_ae_export: bool,
    warmup_ae_slab_writes: bool,
):
    target_args = (
        model_factory,
        request_queue,
        prefix_queue,
        release_queue,
        max_vlm_batch_size,
        max_vlm_wait_ms,
        max_prefix_slots,
        compile_config,
        env_updates,
        warmup_queue,
        ready_queue,
        warmup_start_queue,
    )
    process = ctx.Process(
        target=_run_jax_process_entrypoint,
        args=(
            "vlm",
            _run_jax_vlm_process,
            warmup_queue,
            target_args,
            {
                "run_after_warmup": run_after_warmup,
                "wait_for_ae_export": wait_for_ae_export,
                "warmup_ae_slab_writes": warmup_ae_slab_writes,
            },
        ),
        daemon=True,
    )
    process.start()
    return process


def _should_stage_compile_warmup(compile_config: JaxCompileConfig | None) -> bool:
    return (
        compile_config is not None
        and compile_config.enabled
        and compile_config.warmup_enabled
        and (compile_config.compile_ae or compile_config.compile_vlm)
    )


def _run_staged_compile_warmup(
    ctx,
    *,
    model_factory=None,
    vlm_model_factory=None,
    ae_model_factory=None,
    prefix_template_factory=None,
    max_ae_batch_size: int,
    max_vlm_batch_size: int,
    max_vlm_wait_ms: float,
    max_prefix_slots: int,
    ae_env_updates: dict[str, str | None] | None,
    vlm_env_updates: dict[str, str | None] | None,
    compile_config: JaxCompileConfig,
    timeout_s: float,
) -> dict[str, float]:
    vlm_model_factory = vlm_model_factory or model_factory
    ae_model_factory = ae_model_factory or model_factory
    if vlm_model_factory is None or ae_model_factory is None:
        raise ValueError("JAX staged compile warmup requires model factories")

    warmed = 0.0
    if compile_config.compile_ae:
        ae_timing = _run_staged_ae_compile_warmup(
            ctx,
            model_factory=ae_model_factory,
            prefix_template_factory=prefix_template_factory,
            max_ae_batch_size=max_ae_batch_size,
            max_prefix_slots=max_prefix_slots,
            ae_env_updates=ae_env_updates,
            compile_config=compile_config,
            timeout_s=timeout_s,
        )
        warmed += float(ae_timing["jax_warmup_batches"])
    if compile_config.compile_vlm:
        vlm_timing = _run_staged_vlm_compile_warmup(
            ctx,
            model_factory=vlm_model_factory,
            max_vlm_batch_size=max_vlm_batch_size,
            max_vlm_wait_ms=max_vlm_wait_ms,
            max_prefix_slots=max_prefix_slots,
            vlm_env_updates=vlm_env_updates,
            compile_config=compile_config,
            timeout_s=timeout_s,
        )
        warmed += float(vlm_timing["jax_warmup_batches"])
    return {"jax_warmup_batches": warmed}


def _run_staged_ae_compile_warmup(
    ctx,
    *,
    model_factory,
    prefix_template_factory,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    ae_env_updates: dict[str, str | None] | None,
    compile_config: JaxCompileConfig,
    timeout_s: float,
) -> dict[str, float]:
    prefix_queue = ctx.Queue()
    result_queue = ctx.Queue()
    release_queue = ctx.Queue()
    warmup_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    warmup_start_queue = ctx.Queue()
    process = None
    try:
        process = _start_jax_ae_process(
            ctx,
            model_factory=model_factory,
            prefix_queue=prefix_queue,
            result_queue=result_queue,
            release_queue=release_queue,
            max_ae_batch_size=max_ae_batch_size,
            max_prefix_slots=max_prefix_slots,
            compile_config=compile_config,
            env_updates=ae_env_updates,
            warmup_queue=warmup_queue,
            ready_queue=ready_queue,
            warmup_start_queue=warmup_start_queue,
            prefix_template_factory=prefix_template_factory,
            run_after_warmup=False,
        )
        _collect_worker_ready(
            ready_queue,
            expected_roles=("ae",),
            timeout_s=timeout_s,
            vlm_process=None,
            ae_process=process,
        )
        warmup_start_queue.put("go")
        timing = _collect_compile_warmup(
            warmup_queue,
            expected_roles=("ae",),
            timeout_s=timeout_s,
            vlm_process=None,
            ae_process=process,
        )
        _join_compile_warmup_process(process, role="AE", timeout_s=timeout_s)
        process = None
        return timing
    except BaseException:
        _terminate_processes((process,))
        raise


def _run_staged_vlm_compile_warmup(
    ctx,
    *,
    model_factory,
    max_vlm_batch_size: int,
    max_vlm_wait_ms: float,
    max_prefix_slots: int,
    vlm_env_updates: dict[str, str | None] | None,
    compile_config: JaxCompileConfig,
    timeout_s: float,
) -> dict[str, float]:
    request_queue = ctx.Queue()
    prefix_queue = ctx.Queue()
    release_queue = ctx.Queue()
    warmup_queue = ctx.Queue()
    ready_queue = ctx.Queue()
    warmup_start_queue = ctx.Queue()
    process = None
    try:
        process = _start_jax_vlm_process(
            ctx,
            model_factory=model_factory,
            request_queue=request_queue,
            prefix_queue=prefix_queue,
            release_queue=release_queue,
            max_vlm_batch_size=max_vlm_batch_size,
            max_vlm_wait_ms=max_vlm_wait_ms,
            max_prefix_slots=max_prefix_slots,
            compile_config=compile_config,
            env_updates=vlm_env_updates,
            warmup_queue=warmup_queue,
            ready_queue=ready_queue,
            warmup_start_queue=warmup_start_queue,
            run_after_warmup=False,
            wait_for_ae_export=False,
            warmup_ae_slab_writes=False,
        )
        _collect_worker_ready(
            ready_queue,
            expected_roles=("vlm",),
            timeout_s=timeout_s,
            vlm_process=process,
            ae_process=None,
        )
        warmup_start_queue.put("go")
        timing = _collect_compile_warmup(
            warmup_queue,
            expected_roles=("vlm",),
            timeout_s=timeout_s,
            vlm_process=process,
            ae_process=None,
        )
        _join_compile_warmup_process(process, role="VLM", timeout_s=timeout_s)
        process = None
        return timing
    except BaseException:
        _terminate_processes((process,))
        raise


def _join_compile_warmup_process(process, *, role: str, timeout_s: float) -> None:
    process.join(timeout=max(timeout_s, 1.0))
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    raise RuntimeError(f"{role} compile warmup process did not exit after reporting warmup complete")


def _terminate_processes(processes) -> None:
    for process in processes:
        if process is None:
            continue
        with contextlib.suppress(Exception):
            if process.is_alive():
                process.terminate()
        with contextlib.suppress(Exception):
            process.join(timeout=5)


def _start_serial_serving_pair(
    ctx,
    *,
    ae_model_factory,
    vlm_model_factory,
    prefix_template_factory,
    request_queue,
    prefix_queue,
    result_queue,
    release_queue,
    warmup_queue,
    ready_queue,
    ae_warmup_start_queue,
    vlm_warmup_start_queue,
    max_ae_batch_size: int,
    max_vlm_batch_size: int,
    max_vlm_wait_ms: float,
    max_prefix_slots: int,
    compile_config: JaxCompileConfig | None,
    ae_env_updates: dict[str, str | None] | None,
    vlm_env_updates: dict[str, str | None] | None,
    final_warmup_queue,
    final_ae_warmup_start_queue,
    final_vlm_warmup_start_queue,
    startup_timeout_s: float,
    startup_phase: str,
):
    """Start AE then VLM serially; wait for each role's ready/handshake barrier."""
    ae_process = _start_jax_ae_process(
        ctx,
        model_factory=ae_model_factory,
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_ae_batch_size=max_ae_batch_size,
        max_prefix_slots=max_prefix_slots,
        compile_config=compile_config,
        env_updates=ae_env_updates,
        warmup_queue=final_warmup_queue,
        ready_queue=ready_queue,
        warmup_start_queue=final_ae_warmup_start_queue,
        prefix_template_factory=prefix_template_factory,
        run_after_warmup=True,
    )
    try:
        _collect_worker_ready(
            ready_queue,
            expected_roles=("ae",),
            timeout_s=startup_timeout_s,
            vlm_process=None,
            ae_process=ae_process,
        )
        ae_compile_timing = {"jax_warmup_batches": 0.0}
        if final_warmup_queue is not None:
            if final_ae_warmup_start_queue is not None:
                ae_warmup_start_queue.put("go")
            ae_compile_timing = _collect_compile_warmup(
                warmup_queue,
                expected_roles=("ae",),
                timeout_s=startup_timeout_s,
                vlm_process=None,
                ae_process=ae_process,
                phase=startup_phase,
            )

        vlm_process = _start_jax_vlm_process(
            ctx,
            model_factory=vlm_model_factory,
            request_queue=request_queue,
            prefix_queue=prefix_queue,
            release_queue=release_queue,
            max_vlm_batch_size=max_vlm_batch_size,
            max_vlm_wait_ms=max_vlm_wait_ms,
            max_prefix_slots=max_prefix_slots,
            compile_config=compile_config,
            env_updates=vlm_env_updates,
            warmup_queue=final_warmup_queue,
            ready_queue=ready_queue,
            warmup_start_queue=final_vlm_warmup_start_queue,
            run_after_warmup=True,
            wait_for_ae_export=True,
            warmup_ae_slab_writes=True,
        )
        _collect_worker_ready(
            ready_queue,
            expected_roles=("vlm",),
            timeout_s=startup_timeout_s,
            vlm_process=vlm_process,
            ae_process=ae_process,
        )
        vlm_compile_timing = {"jax_warmup_batches": 0.0}
        if final_warmup_queue is not None:
            if final_vlm_warmup_start_queue is not None:
                vlm_warmup_start_queue.put("go")
            vlm_compile_timing = _collect_compile_warmup(
                warmup_queue,
                expected_roles=("vlm",),
                timeout_s=startup_timeout_s,
                vlm_process=vlm_process,
                ae_process=ae_process,
                phase=startup_phase,
            )
        return ae_process, vlm_process, ae_compile_timing, vlm_compile_timing
    except BaseException:
        _terminate_processes((ae_process, locals().get("vlm_process")))
        raise


class JaxProcessVASplitRuntime:
    def __init__(
        self,
        *,
        model_factory: Callable[[], object],
        vlm_model_factory: Callable[[], object] | None = None,
        ae_model_factory: Callable[[], object] | None = None,
        prefix_template_factory: Callable[[], object] | None = None,
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
        self._admission_limit = max(1, int(max_prefix_slots))
        self._admission_semaphore = threading.BoundedSemaphore(self._admission_limit)
        self._admitted_request_ids: set[str] = set()
        self._compile_timing = {"jax_warmup_batches": 0.0}
        self._model_factory = model_factory
        self._vlm_model_factory = vlm_model_factory or model_factory
        self._ae_model_factory = ae_model_factory or model_factory
        self._prefix_template_factory = prefix_template_factory

        ctx = mp.get_context(start_method)
        self._request_queue = ctx.Queue()
        self._prefix_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._release_queue = ctx.Queue()
        self._warmup_queue = ctx.Queue()
        self._ready_queue = ctx.Queue()
        self._ae_warmup_start_queue = ctx.Queue()
        self._vlm_warmup_start_queue = ctx.Queue()
        startup_timeout_s = warmup_timeout_s if warmup_timeout_s is not None else result_timeout_s
        self._ae_process = None
        self._vlm_process = None
        self._result_thread = None
        final_compile_config = compile_config
        staged_compile_warmup = _should_stage_compile_warmup(compile_config)
        try:
            if staged_compile_warmup:
                self._compile_timing = _run_staged_compile_warmup(
                    ctx,
                    vlm_model_factory=self._vlm_model_factory,
                    ae_model_factory=self._ae_model_factory,
                    prefix_template_factory=self._prefix_template_factory,
                    max_ae_batch_size=max_ae_batch_size,
                    max_vlm_batch_size=max_vlm_batch_size,
                    max_vlm_wait_ms=max_vlm_wait_ms,
                    max_prefix_slots=max_prefix_slots,
                    ae_env_updates=ae_env_updates,
                    vlm_env_updates=vlm_env_updates,
                    compile_config=compile_config,
                    timeout_s=startup_timeout_s,
                )
                final_compile_config = replace(compile_config, warmup_enabled=False)
                final_warmup_queue = self._warmup_queue
                final_ae_warmup_start_queue = None
                final_vlm_warmup_start_queue = None
            else:
                final_warmup_queue = self._warmup_queue
                final_ae_warmup_start_queue = self._ae_warmup_start_queue
                final_vlm_warmup_start_queue = self._vlm_warmup_start_queue

            # After staged compile, load the serving pair without MPS first (avoids
            # MPS+IPC dual-client crashes during weight load / slab attach). Then
            # restart MPS and respawn the serving pair under MPS for profiling.
            mps_pipe_configured = bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
            reload_under_mps = staged_compile_warmup and mps_pipe_configured
            startup_phase = (
                "reporting startup handshake" if staged_compile_warmup else "reporting compile warmup"
            )
            if reload_under_mps:
                _stop_mps_daemon()
                ae_env_for_load = _without_mps_env(ae_env_updates)
                vlm_env_for_load = _without_mps_env(vlm_env_updates)
                load_phase = "reporting startup handshake (no-MPS load)"
            else:
                ae_env_for_load = ae_env_updates
                vlm_env_for_load = vlm_env_updates
                load_phase = startup_phase

            self._ae_process, self._vlm_process, ae_compile_timing, vlm_compile_timing = _start_serial_serving_pair(
                ctx,
                ae_model_factory=self._ae_model_factory,
                vlm_model_factory=self._vlm_model_factory,
                prefix_template_factory=self._prefix_template_factory,
                request_queue=self._request_queue,
                prefix_queue=self._prefix_queue,
                result_queue=self._result_queue,
                release_queue=self._release_queue,
                warmup_queue=self._warmup_queue,
                ready_queue=self._ready_queue,
                ae_warmup_start_queue=self._ae_warmup_start_queue,
                vlm_warmup_start_queue=self._vlm_warmup_start_queue,
                max_ae_batch_size=max_ae_batch_size,
                max_vlm_batch_size=max_vlm_batch_size,
                max_vlm_wait_ms=max_vlm_wait_ms,
                max_prefix_slots=max_prefix_slots,
                compile_config=final_compile_config,
                ae_env_updates=ae_env_for_load,
                vlm_env_updates=vlm_env_for_load,
                final_warmup_queue=final_warmup_queue,
                final_ae_warmup_start_queue=final_ae_warmup_start_queue,
                final_vlm_warmup_start_queue=final_vlm_warmup_start_queue,
                startup_timeout_s=startup_timeout_s,
                startup_phase=load_phase,
            )

            if reload_under_mps:
                print(
                    "[jax-split] no-MPS final load+handshake ok; "
                    "respawning AE/VLM under MPS for profile",
                    flush=True,
                )
                _terminate_processes((self._vlm_process, self._ae_process))
                self._vlm_process = None
                self._ae_process = None
                self._request_queue = ctx.Queue()
                self._prefix_queue = ctx.Queue()
                self._result_queue = ctx.Queue()
                self._release_queue = ctx.Queue()
                self._warmup_queue = ctx.Queue()
                self._ready_queue = ctx.Queue()
                self._ae_warmup_start_queue = ctx.Queue()
                self._vlm_warmup_start_queue = ctx.Queue()
                _start_mps_daemon()
                self._ae_process, self._vlm_process, _, _ = _start_serial_serving_pair(
                    ctx,
                    ae_model_factory=self._ae_model_factory,
                    vlm_model_factory=self._vlm_model_factory,
                    prefix_template_factory=self._prefix_template_factory,
                    request_queue=self._request_queue,
                    prefix_queue=self._prefix_queue,
                    result_queue=self._result_queue,
                    release_queue=self._release_queue,
                    warmup_queue=self._warmup_queue,
                    ready_queue=self._ready_queue,
                    ae_warmup_start_queue=self._ae_warmup_start_queue,
                    vlm_warmup_start_queue=self._vlm_warmup_start_queue,
                    max_ae_batch_size=max_ae_batch_size,
                    max_vlm_batch_size=max_vlm_batch_size,
                    max_vlm_wait_ms=max_vlm_wait_ms,
                    max_prefix_slots=max_prefix_slots,
                    compile_config=final_compile_config,
                    ae_env_updates=ae_env_updates,
                    vlm_env_updates=vlm_env_updates,
                    final_warmup_queue=self._warmup_queue,
                    final_ae_warmup_start_queue=None,
                    final_vlm_warmup_start_queue=None,
                    startup_timeout_s=startup_timeout_s,
                    startup_phase="reporting startup handshake (MPS profile)",
                )

            if not staged_compile_warmup:
                self._compile_timing = {
                    "jax_warmup_batches": float(
                        ae_compile_timing["jax_warmup_batches"] + vlm_compile_timing["jax_warmup_batches"]
                    )
                }
            self._result_thread = threading.Thread(target=self._collect_results, daemon=True)
            self._result_thread.start()
        except BaseException:
            _terminate_processes((self._vlm_process, self._ae_process))
            raise

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        if self._closed:
            raise RuntimeError("JAX VA split runtime is shut down")
        request_id = str(uuid.uuid4())
        self._acquire_request_admission((request_id,), timeout_s=self._result_timeout_s)
        try:
            self._request_queue.put(
                JaxRequestEnvelope(
                    request_id=request_id,
                    observation=observation,
                    sample_kwargs=dict(sample_kwargs),
                    enqueue_ns=time.monotonic_ns(),
                )
            )
        except Exception:
            with self._condition:
                self._release_request_admission_locked(request_id)
            raise
        return self._wait_for_result(request_id)

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        if self._closed:
            raise RuntimeError("JAX VA split runtime is shut down")
        batch_size = int(observation["state"].shape[0])
        batch_id = str(uuid.uuid4())
        request_ids = tuple(f"{batch_id}:{row}" for row in range(batch_size))
        self._acquire_request_admission(request_ids, timeout_s=self._result_timeout_s)
        try:
            self._request_queue.put(
                JaxBatchRequestEnvelope(
                    batch_id=batch_id,
                    request_ids=request_ids,
                    observation=observation,
                    sample_kwargs=dict(sample_kwargs),
                    enqueue_ns=time.monotonic_ns(),
                )
            )
        except Exception:
            with self._condition:
                for request_id in request_ids:
                    self._release_request_admission_locked(request_id)
            raise
        results_by_id = {request_id: self._wait_for_result(request_id) for request_id in request_ids}
        return _combine_ordered_results(batch_id, request_ids, results_by_id)

    def _acquire_request_admission(self, request_ids: tuple[str, ...], *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        acquired: list[str] = []
        try:
            for request_id in request_ids:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._admission_semaphore.acquire(timeout=remaining):
                    raise TimeoutError(
                        "Timed out waiting for JAX VA split request admission "
                        f"({len(self._admitted_request_ids)}/{self._admission_limit} admitted)"
                    )
                acquired.append(request_id)
                with self._condition:
                    self._admitted_request_ids.add(request_id)
        except Exception:
            with self._condition:
                for request_id in acquired:
                    self._release_request_admission_locked(request_id)
            raise

    def _release_request_admission_locked(self, request_id: str) -> None:
        if request_id not in self._admitted_request_ids:
            return
        self._admitted_request_ids.remove(request_id)
        self._admission_semaphore.release()

    def _release_all_request_admissions_locked(self) -> None:
        for request_id in tuple(self._admitted_request_ids):
            self._release_request_admission_locked(request_id)

    def _wait_for_result(self, request_id: str) -> JaxActionResult:
        deadline = time.monotonic() + self._result_timeout_s
        with self._condition:
            while True:
                if request_id in self._pending_results:
                    return self._pending_results.pop(request_id)
                if request_id in self._pending_errors:
                    raise _worker_error_to_runtime_error(self._pending_errors.pop(request_id))
                if None in self._pending_errors:
                    self._release_all_request_admissions_locked()
                    raise _worker_error_to_runtime_error(self._pending_errors[None])
                if self._shutdown_seen:
                    self._release_all_request_admissions_locked()
                    raise RuntimeError("JAX VA split worker shut down before producing a result")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._release_request_admission_locked(request_id)
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
                    self._release_request_admission_locked(message.request_id)
                    self._pending_results[message.request_id] = _mark_collected_result(
                        message,
                        get_start_ns=get_start_ns,
                        get_end_ns=get_end_ns,
                    )
                elif isinstance(message, JaxWorkerError):
                    if message.request_id is None:
                        self._release_all_request_admissions_locked()
                    else:
                        self._release_request_admission_locked(message.request_id)
                    self._pending_errors[message.request_id] = message
                elif isinstance(message, JaxShutdown):
                    self._release_all_request_admissions_locked()
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


def _collect_worker_ready(
    ready_queue,
    *,
    expected_roles: tuple[str, ...],
    timeout_s: float,
    vlm_process,
    ae_process,
) -> None:
    deadline = time.monotonic() + max(timeout_s, 1.0)
    seen: set[str] = set()
    while len(seen) < len(expected_roles):
        if vlm_process is not None and not vlm_process.is_alive() and "vlm" not in seen:
            raise RuntimeError(_process_exited_message("VLM", vlm_process, "reporting ready"))
        if ae_process is not None and not ae_process.is_alive() and "ae" not in seen:
            raise RuntimeError(_process_exited_message("AE", ae_process, "reporting ready"))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(set(expected_roles) - seen)
            raise TimeoutError(f"Timed out waiting for JAX worker ready from: {missing}")
        try:
            role = ready_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if role not in expected_roles:
            raise RuntimeError(f"Unexpected JAX worker ready message: {role!r}")
        seen.add(role)


def _worker_error_to_runtime_error(error: JaxWorkerError) -> RuntimeError:
    detail = f"\n{error.traceback}" if error.traceback else ""
    return RuntimeError(f"JAX VA split worker failed for {error.request_id}: {error.error}{detail}")


def _process_exited_message(role: str, process, phase: str) -> str:
    exitcode = getattr(process, "exitcode", None)
    if exitcode is None:
        return f"{role} process exited before {phase}"
    signal_name = ""
    if exitcode < 0:
        with contextlib.suppress(ValueError):
            signal_name = f" ({signal.Signals(-exitcode).name})"
    return f"{role} process exited before {phase} (exitcode={exitcode}{signal_name})"


def _collect_compile_warmup(
    warmup_queue,
    *,
    expected_roles: tuple[str, ...],
    timeout_s: float,
    vlm_process,
    ae_process,
    phase: str = "reporting compile warmup",
) -> dict[str, float]:
    deadline = time.monotonic() + max(timeout_s, 1.0)
    seen: dict[str, float] = {}
    while len(seen) < len(expected_roles):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(set(expected_roles) - set(seen))
            raise TimeoutError(f"Timed out waiting for JAX compile warmup from: {missing}")
        try:
            message = warmup_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            if vlm_process is not None and not vlm_process.is_alive() and "vlm" not in seen:
                raise RuntimeError(_process_exited_message("VLM", vlm_process, phase)) from None
            if ae_process is not None and not ae_process.is_alive() and "ae" not in seen:
                raise RuntimeError(_process_exited_message("AE", ae_process, phase)) from None
            continue
        if isinstance(message, JaxWorkerError):
            raise _worker_error_to_runtime_error(message)
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
