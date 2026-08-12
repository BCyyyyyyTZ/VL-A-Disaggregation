# ruff: noqa: N802, SLF001

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax import runtime as runtime_module
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.launcher import build_jax_mps_process_envs
from openpi.serving.va_split_jax.runtime import JaxLocalVASplitRuntime
from openpi.serving.va_split_jax.runtime import JaxProcessVASplitRuntime
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxCompileWarmupDone
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxWorkerError


class FakeJaxSplitModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.prefix_batch_sizes: list[int] = []

    def build_prefix_feature(self, rng, observation) -> JaxPrefixFeature:
        del rng
        batch = int(observation.state.shape[0])
        self.prefix_batch_sizes.append(batch)
        return JaxPrefixFeature(
            past_key_values=(
                jnp.ones((3, batch, 2, 4), dtype=jnp.float32),
                jnp.full((3, batch, 2, 4), 2.0, dtype=jnp.float32),
            ),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=observation.state,
        )

    def init_denoise_state(self, rng, batch_size: int, noise: jax.Array | None, num_steps: int) -> JaxDenoiseState:
        del rng
        if noise is None:
            noise = jnp.zeros((batch_size, self.config.action_horizon, self.config.action_dim), dtype=jnp.float32)
        return JaxDenoiseState(
            x_t=noise,
            step_idx=jnp.asarray(0, dtype=jnp.int32),
            num_steps=num_steps,
            dt=jnp.asarray(-1.0 / num_steps, dtype=jnp.float32),
        )

    def denoise_one_batch(self, prefix_batch: JaxPrefixFeature, denoise_batch: JaxDenoiseState) -> jax.Array:
        assert prefix_batch.prefix_pad_masks.shape[0] == denoise_batch.x_t.shape[0]
        return jnp.ones_like(denoise_batch.x_t)


def _observation(batch_size: int = 1) -> dict:
    image = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_),
        },
        "state": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "tokenized_prompt": jnp.ones((batch_size, 8), dtype=jnp.int32),
        "tokenized_prompt_mask": jnp.ones((batch_size, 8), dtype=jnp.bool_),
    }


def test_jax_local_va_split_runtime_runs_vlm_then_ae_and_releases_prefix():
    runtime = JaxLocalVASplitRuntime(model=FakeJaxSplitModel(), max_ae_batch_size=2)

    result = runtime.infer(_observation(), {"num_steps": 1, "noise": jnp.zeros((1, 2, 1), dtype=jnp.float32)})

    assert result.request_id
    np.testing.assert_allclose(result.actions, -jnp.ones((1, 2, 1), dtype=jnp.float32))
    assert runtime.vlm_worker.active_count == 0


def test_jax_local_va_split_runtime_infer_batch_builds_prefix_once_and_returns_row_order():
    model = FakeJaxSplitModel()
    runtime = JaxLocalVASplitRuntime(model=model, max_ae_batch_size=2, max_prefix_slots=4)

    result = runtime.infer_batch(
        _observation(batch_size=2),
        {"num_steps": 1, "noise": jnp.zeros((2, 2, 1), dtype=jnp.float32)},
    )

    assert result.request_id
    assert model.prefix_batch_sizes == [1, 2]  # bootstrap template + batched batch
    np.testing.assert_allclose(result.actions, -jnp.ones((2, 2, 1), dtype=jnp.float32))
    assert result.timing is not None
    assert result.timing["effective_batch"] == 2
    assert result.timing["vlm_effective_batch"] == 2.0
    assert result.timing["ae_effective_batch_mean"] == 2.0
    assert runtime.vlm_worker.active_count == 0


def test_build_jax_mps_process_envs_sets_preallocate_and_sm_percent(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "99")

    ae_env, vlm_env = build_jax_mps_process_envs(
        cuda_visible_devices="0",
        mps_pipe_dir="/tmp/mps-pipe",
        mps_log_dir="/tmp/mps-log",
        ae_sm_percent=20,
        vlm_sm_percent=0,
    )

    assert ae_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert vlm_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert ae_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "20"
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in vlm_env


class _ResultQueue:
    def __init__(self, messages, *, get_delay_s: float = 0.0):
        self._messages = list(messages)
        self._get_delay_s = get_delay_s

    def get(self):
        if self._get_delay_s:
            time.sleep(self._get_delay_s)
        return self._messages.pop(0)


def test_jax_process_runtime_collect_results_records_ae_result_transfer_latency():
    enqueue_ns = time.monotonic_ns() - 5_000_000
    runtime = object.__new__(JaxProcessVASplitRuntime)
    runtime._result_queue = _ResultQueue(
        [
            JaxActionResult(
                request_id="req-1",
                actions=jnp.zeros((1, 2, 1), dtype=jnp.float32),
                timing={"_ae_result_enqueue_ns": float(enqueue_ns)},
            ),
            JaxShutdown(),
        ],
        get_delay_s=0.01,
    )
    runtime._condition = threading.Condition()
    runtime._pending_results = {}
    runtime._pending_errors = {}
    runtime._shutdown_seen = False
    runtime._admission_semaphore = threading.BoundedSemaphore(1)
    assert runtime._admission_semaphore.acquire(timeout=0.01)
    runtime._admission_limit = 1
    runtime._admitted_request_ids = {"req-1"}

    runtime._collect_results()

    timing = runtime._pending_results["req-1"].timing
    assert timing is not None
    assert timing["ae_result_queue_wait_ms"] >= 4.0
    assert timing["ae_result_transfer_ms"] >= 8.0
    assert timing["va_split_transfer_ms"] == timing["ae_result_transfer_ms"]
    assert timing["va_split_queue_wait_ms"] == timing["ae_result_queue_wait_ms"]
    assert "_ae_result_enqueue_ns" not in timing
    assert runtime._admitted_request_ids == set()


def test_jax_process_runtime_request_admission_is_released_by_request_id():
    runtime = object.__new__(JaxProcessVASplitRuntime)
    runtime._condition = threading.Condition()
    runtime._admission_semaphore = threading.BoundedSemaphore(1)
    runtime._admission_limit = 1
    runtime._admitted_request_ids = set()

    runtime._acquire_request_admission(("req-1",), timeout_s=0.01)
    with pytest.raises(TimeoutError, match="admission"):
        runtime._acquire_request_admission(("req-2",), timeout_s=0.01)

    with runtime._condition:
        runtime._release_request_admission_locked("req-1")

    runtime._acquire_request_admission(("req-2",), timeout_s=0.01)
    assert runtime._admitted_request_ids == {"req-2"}


def test_jax_process_runtime_infer_after_shutdown_raises():
    runtime = object.__new__(JaxProcessVASplitRuntime)
    runtime._closed = True
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.infer(_observation(), {})


def test_jax_process_runtime_stages_compile_warmup_before_final_workers(monkeypatch):
    events: list[str] = []
    processes: dict[str, list[object]] = {"ae": [], "vlm": []}

    class FakeQueue:
        def put(self, value):
            events.append(f"queue-put:{type(value).__name__}")

    class FakeProcess:
        def __init__(self, *, target, args, kwargs, daemon):
            self.role = args[0] if target is runtime_module._run_jax_process_entrypoint else "unknown"
            self.args = args
            self.target_args = args[3]
            self.target_kwargs = args[4]
            self.kwargs = kwargs
            self.daemon = daemon
            self.started = False
            self.exited = False
            processes[self.role].append(self)

        def start(self):
            self.started = True
            events.append(f"start:{self.role}:{len(processes[self.role])}")

        def join(self, timeout=None):
            del timeout
            self.exited = True
            events.append(f"join:{self.role}:{processes[self.role].index(self) + 1}")

        def terminate(self):
            self.exited = True
            events.append(f"terminate:{self.role}:{processes[self.role].index(self) + 1}")

        def is_alive(self):
            return self.started and not self.exited

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, *, target, args, kwargs=None, daemon):
            return FakeProcess(target=target, args=args, kwargs=kwargs or {}, daemon=daemon)

    class FakeThread:
        def __init__(self, *, target, daemon):
            del target
            self.daemon = daemon

        def start(self):
            events.append("thread:start")

        def join(self, timeout=None):
            del timeout

    def collect_startup(_queue, *, expected_roles, timeout_s, vlm_process, ae_process):
        del timeout_s
        events.append("startup:" + ",".join(expected_roles))
        if expected_roles == ("ae",):
            assert vlm_process is None
            assert ae_process is processes["ae"][-1]
            assert ae_process.started
            return
        if expected_roles == ("vlm",):
            assert vlm_process is processes["vlm"][-1]
            assert vlm_process.started
            if vlm_process is processes["vlm"][0]:
                assert ae_process is None
                assert processes["ae"][0].exited
                return
            assert ae_process is processes["ae"][1]
            assert processes["ae"][1].started
            return
        raise AssertionError(f"Unexpected ready roles: {expected_roles!r}")

    def collect_warmup(_queue, *, expected_roles, timeout_s, vlm_process, ae_process, phase="reporting compile warmup"):
        del timeout_s
        events.append("warmup:" + ",".join(expected_roles))
        assert expected_roles in (("ae",), ("vlm",))
        if expected_roles == ("ae",):
            assert ae_process is processes["ae"][-1]
            assert ae_process.started
            if ae_process is processes["ae"][0]:
                assert vlm_process is None
                assert phase == "reporting compile warmup"
                return {"jax_warmup_batches": 1.0}
            assert vlm_process is None
            assert "startup handshake" in phase
            return {"jax_warmup_batches": 0.0}
        assert vlm_process is processes["vlm"][-1]
        if vlm_process is processes["vlm"][0]:
            assert processes["ae"][0].exited
            assert ae_process is None
            assert phase == "reporting compile warmup"
            return {"jax_warmup_batches": 1.0}
        assert ae_process is processes["ae"][1]
        assert "startup handshake" in phase
        return {"jax_warmup_batches": 0.0}

    monkeypatch.setattr(runtime_module.mp, "get_context", lambda _start_method: FakeContext())
    monkeypatch.setattr(runtime_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(runtime_module, "_collect_worker_ready", collect_startup, raising=False)
    monkeypatch.setattr(runtime_module, "_collect_compile_warmup", collect_warmup)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_MPS_LOG_DIRECTORY", raising=False)

    base_factory = object()
    vlm_factory = object()
    ae_factory = object()
    runtime = JaxProcessVASplitRuntime(
        model_factory=base_factory,
        vlm_model_factory=vlm_factory,
        ae_model_factory=ae_factory,
        start_method="spawn",
        result_timeout_s=1.0,
        warmup_timeout_s=1.0,
        compile_config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=4),
    )

    assert runtime._model_factory is base_factory
    assert runtime._vlm_model_factory is vlm_factory
    assert runtime._ae_model_factory is ae_factory
    assert runtime.compile_timing == {"jax_warmup_batches": 2.0}
    assert processes["ae"][0].target_args[0] is ae_factory
    assert processes["ae"][0].target_args[6].warmup_enabled is True
    assert processes["ae"][0].target_kwargs["run_after_warmup"] is False
    assert processes["vlm"][0].target_args[0] is vlm_factory
    assert processes["vlm"][0].target_args[7].warmup_enabled is True
    assert processes["vlm"][0].target_kwargs["run_after_warmup"] is False
    assert processes["vlm"][0].target_kwargs["wait_for_ae_export"] is False
    assert processes["vlm"][0].target_kwargs["warmup_ae_slab_writes"] is False
    assert processes["ae"][1].target_args[0] is ae_factory
    assert processes["ae"][1].target_args[6].warmup_enabled is False
    assert processes["ae"][1].target_kwargs["run_after_warmup"] is True
    assert processes["vlm"][1].target_args[0] is vlm_factory
    assert processes["vlm"][1].target_args[7].warmup_enabled is False
    assert processes["vlm"][1].target_kwargs["run_after_warmup"] is True
    assert processes["vlm"][1].target_kwargs["wait_for_ae_export"] is True
    assert events.index("start:ae:1") < events.index("warmup:ae")
    assert events.index("join:ae:1") < events.index("start:vlm:1")
    assert events.index("join:vlm:1") < events.index("start:ae:2")
    final_ae_ready = max(i for i, event in enumerate(events) if event == "startup:ae")
    final_ae_warmup = max(i for i, event in enumerate(events) if event == "warmup:ae")
    final_vlm_ready = max(i for i, event in enumerate(events) if event == "startup:vlm")
    final_vlm_warmup = max(i for i, event in enumerate(events) if event == "warmup:vlm")
    assert events.index("start:ae:2") < final_ae_ready < final_ae_warmup
    # Final AE handshake completes before VLM role weights are loaded.
    assert final_ae_warmup < events.index("start:vlm:2")
    assert events.index("start:vlm:2") < final_vlm_ready < final_vlm_warmup


def test_staged_compile_warmup_skips_disabled_vlm(monkeypatch):
    calls: list[str] = []

    def ae_warmup(*args, **kwargs):
        del args, kwargs
        calls.append("ae")
        return {"jax_warmup_batches": 2.0}

    def vlm_warmup(*args, **kwargs):
        del args, kwargs
        raise AssertionError("VLM warmup should be skipped")

    monkeypatch.setattr(runtime_module, "_run_staged_ae_compile_warmup", ae_warmup)
    monkeypatch.setattr(runtime_module, "_run_staged_vlm_compile_warmup", vlm_warmup)

    timing = runtime_module._run_staged_compile_warmup(
        object(),
        model_factory=lambda: object(),
        max_ae_batch_size=8,
        max_vlm_batch_size=8,
        max_vlm_wait_ms=1.0,
        max_prefix_slots=24,
        ae_env_updates=None,
        vlm_env_updates=None,
        compile_config=JaxCompileConfig(enabled=True, warmup_enabled=True, compile_ae=True, compile_vlm=False),
        timeout_s=1.0,
    )

    assert calls == ["ae"]
    assert timing == {"jax_warmup_batches": 2.0}


def test_collect_compile_warmup_accepts_done_from_exited_temp_process():
    class FakeQueue:
        def get(self, timeout):
            del timeout
            return JaxCompileWarmupDone(role="ae", jax_warmup_batches=3.0)

    class ExitedProcess:
        def is_alive(self):
            return False

    timing = runtime_module._collect_compile_warmup(
        FakeQueue(),
        expected_roles=("ae",),
        timeout_s=1.0,
        vlm_process=None,
        ae_process=ExitedProcess(),
    )

    assert timing == {"jax_warmup_batches": 3.0}


def test_collect_compile_warmup_reports_process_exitcode_signal():
    class EmptyQueue:
        def get(self, timeout):
            del timeout
            raise queue.Empty

    class KilledProcess:
        exitcode = -9

        def is_alive(self):
            return False

    with pytest.raises(RuntimeError, match=r"VLM process exited.*exitcode=-9.*SIGKILL"):
        runtime_module._collect_compile_warmup(
            EmptyQueue(),
            expected_roles=("vlm",),
            timeout_s=1.0,
            vlm_process=KilledProcess(),
            ae_process=None,
        )


def test_collect_compile_warmup_raises_worker_error_traceback():
    class ErrorQueue:
        def get(self, timeout):
            del timeout
            return JaxWorkerError(request_id=None, error="vlm failed", traceback="Traceback text")

    with pytest.raises(RuntimeError, match=r"(?s)vlm failed.*Traceback text"):
        runtime_module._collect_compile_warmup(
            ErrorQueue(),
            expected_roles=("vlm",),
            timeout_s=1.0,
            vlm_process=None,
            ae_process=None,
        )


def test_apply_jax_compilation_cache_config_uses_env(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeJaxConfig:
        def update(self, name, value):
            calls.append((name, value))

    monkeypatch.setattr(runtime_module.jax, "config", FakeJaxConfig())
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "true")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "1024")

    runtime_module._apply_jax_compilation_cache_config()

    assert calls == [
        ("jax_compilation_cache_dir", "/tmp/jax-cache"),
        ("jax_enable_compilation_cache", True),
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
        ("jax_persistent_cache_min_entry_size_bytes", 1024),
    ]
