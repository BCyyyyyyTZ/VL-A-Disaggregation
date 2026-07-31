# ruff: noqa: SLF001

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.launcher import build_jax_mps_process_envs
from openpi.serving.va_split_jax.runtime import JaxLocalVASplitRuntime
from openpi.serving.va_split_jax.runtime import JaxProcessVASplitRuntime
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.serving.va_split_jax.types import JaxShutdown


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
    assert model.prefix_batch_sizes == [2]
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

    runtime._collect_results()

    timing = runtime._pending_results["req-1"].timing
    assert timing is not None
    assert timing["ae_result_queue_wait_ms"] >= 4.0
    assert timing["ae_result_transfer_ms"] >= 8.0
    assert timing["va_split_transfer_ms"] == timing["ae_result_transfer_ms"]
    assert timing["va_split_queue_wait_ms"] == timing["ae_result_queue_wait_ms"]
    assert "_ae_result_enqueue_ns" not in timing


def test_jax_process_runtime_infer_after_shutdown_raises():
    runtime = object.__new__(JaxProcessVASplitRuntime)
    runtime._closed = True
    with pytest.raises(RuntimeError, match="shut down"):
        runtime.infer(_observation(), {})
