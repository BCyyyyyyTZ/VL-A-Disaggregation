from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
import pytest

from scripts import profile_va_split


def test_make_poisson_arrival_times_is_reproducible_and_monotonic():
    arrivals = profile_va_split.make_poisson_arrival_times(num_requests=8, request_rate_hz=16.0, seed=7)
    repeated = profile_va_split.make_poisson_arrival_times(num_requests=8, request_rate_hz=16.0, seed=7)

    np.testing.assert_array_equal(arrivals, repeated)
    assert arrivals.dtype == np.float64
    assert np.all(np.diff(arrivals) > 0)


@pytest.mark.parametrize("request_rate_hz", [0.0, -1.0])
def test_make_poisson_arrival_times_rejects_invalid_rate(request_rate_hz):
    with pytest.raises(ValueError, match="request_rate_hz must be positive"):
        profile_va_split.make_poisson_arrival_times(num_requests=1, request_rate_hz=request_rate_hz, seed=0)


def test_make_poisson_arrival_times_rejects_invalid_request_count():
    with pytest.raises(ValueError, match="num_requests must be positive"):
        profile_va_split.make_poisson_arrival_times(num_requests=0, request_rate_hz=1.0, seed=0)


def test_make_synthetic_libero_requests_matches_raw_policy_contract():
    requests = profile_va_split.make_synthetic_libero_requests(
        num_requests=3,
        request_rate_hz=128.0,
        seed=11,
        action_horizon=10,
        action_dim=32,
        state_dim=8,
        image_size=224,
        fixed_noise=True,
    )
    repeated = profile_va_split.make_synthetic_libero_requests(
        num_requests=3,
        request_rate_hz=128.0,
        seed=11,
        action_horizon=10,
        action_dim=32,
        state_dim=8,
        image_size=224,
        fixed_noise=True,
    )

    assert [request.request_id for request in requests] == ["req-000000", "req-000001", "req-000002"]
    np.testing.assert_array_equal(
        [request.scheduled_at_s for request in requests], [r.scheduled_at_s for r in repeated]
    )

    observation = requests[0].observation
    assert observation["observation/state"].shape == (8,)
    assert observation["observation/state"].dtype == np.float32
    assert observation["observation/image"].shape == (224, 224, 3)
    assert observation["observation/image"].dtype == np.uint8
    assert observation["observation/wrist_image"].shape == (224, 224, 3)
    assert observation["observation/wrist_image"].dtype == np.uint8
    assert observation["prompt"] == "do something"
    assert requests[0].noise is not None
    assert requests[0].noise.shape == (10, 32)
    assert requests[0].noise.dtype == np.float32
    np.testing.assert_array_equal(requests[0].noise, repeated[0].noise)
    np.testing.assert_array_equal(observation["observation/image"], repeated[0].observation["observation/image"])


def test_make_synthetic_libero_requests_can_omit_noise():
    requests = profile_va_split.make_synthetic_libero_requests(
        num_requests=1,
        request_rate_hz=128.0,
        seed=11,
        fixed_noise=False,
    )

    assert requests[0].noise is None


class _ConcurrentFakePolicy:
    supports_concurrent_infer = True

    def __init__(self, *, sleep_s: float = 0.02):
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0
        self.completed_calls = 0

    def infer(self, obs, *, noise=None):
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(self._sleep_s)
        finally:
            with self._lock:
                self._active_calls -= 1
                self.completed_calls += 1
        return {
            "actions": np.asarray([obs["value"]], dtype=np.float32),
            "policy_timing": {
                "vlm_prefix_forward_ms": 1.0,
                "ae_step_ms": 2.0,
                "ae_effective_batch": 3.0,
            },
        }


class _SerialFakePolicy(_ConcurrentFakePolicy):
    supports_concurrent_infer = False


def _fake_requests(count: int) -> list[profile_va_split.SyntheticRequest]:
    return [
        profile_va_split.SyntheticRequest(
            request_id=f"req-{idx:06d}",
            scheduled_at_s=idx * 0.001,
            observation={"value": idx},
            noise=np.full((2, 3), idx, dtype=np.float32),
        )
        for idx in range(count)
    ]


def test_run_benchmark_allows_concurrent_policy_overlap():
    policy = _ConcurrentFakePolicy()
    traces = asyncio.run(
        profile_va_split.run_benchmark_requests(
            policy,
            _fake_requests(4),
            max_inflight=4,
            timeout_s=1.0,
        )
    )
    summary = profile_va_split.summarize_traces(traces, target_request_rate_hz=1000.0, inflight_peak=4)

    assert [trace.status for trace in traces] == ["ok", "ok", "ok", "ok"]
    assert all(trace.actions is not None for trace in traces)
    assert summary["completed_requests"] == 4
    assert summary["failed_requests"] == 0
    assert summary["timeout_requests"] == 0
    assert summary["target_request_rate_hz"] == 1000.0
    assert summary["inflight_peak"] == 4
    assert summary["vlm_prefix_forward_p50_ms"] == 1.0
    assert summary["ae_step_p50_ms"] == 2.0
    assert summary["ae_effective_batch_mean"] == 3.0
    assert policy.max_active_calls > 1


def test_run_benchmark_serializes_non_concurrent_policy():
    policy = _SerialFakePolicy()

    traces = asyncio.run(
        profile_va_split.run_benchmark_requests(
            policy,
            _fake_requests(4),
            max_inflight=4,
            timeout_s=1.0,
        )
    )

    assert [trace.status for trace in traces] == ["ok", "ok", "ok", "ok"]
    assert policy.max_active_calls == 1


def test_run_benchmark_keeps_open_loop_submission_for_serial_policy():
    policy = _SerialFakePolicy(sleep_s=0.03)

    traces = asyncio.run(
        profile_va_split.run_benchmark_requests(
            policy,
            _fake_requests(4),
            max_inflight=4,
            timeout_s=1.0,
        )
    )

    assert [trace.status for trace in traces] == ["ok", "ok", "ok", "ok"]
    assert max(trace.submitted_at_s for trace in traces) < 0.02
    assert policy.max_active_calls == 1


def test_run_benchmark_timeout_keeps_inflight_slot_until_infer_finishes():
    policy = _ConcurrentFakePolicy(sleep_s=0.03)

    traces = asyncio.run(
        profile_va_split.run_benchmark_requests(
            policy,
            _fake_requests(2),
            max_inflight=1,
            timeout_s=0.001,
        )
    )

    assert [trace.status for trace in traces] == ["timeout", "timeout"]
    assert traces[1].submitted_at_s >= 0.025
    assert policy.completed_calls == 2


def test_compare_action_traces_reports_max_abs_diff():
    expected = [
        profile_va_split.RequestTrace(
            request_id="req-1",
            scheduled_at_s=0.0,
            submitted_at_s=0.0,
            completed_at_s=0.1,
            status="ok",
            policy_timing={},
            actions=np.asarray([1.0, 2.0], dtype=np.float32),
        )
    ]
    actual = [
        profile_va_split.RequestTrace(
            request_id="req-1",
            scheduled_at_s=0.0,
            submitted_at_s=0.0,
            completed_at_s=0.1,
            status="ok",
            policy_timing={},
            actions=np.asarray([1.0, 2.25], dtype=np.float32),
        )
    ]

    result = profile_va_split.compare_action_traces(expected, actual, rtol=1e-4, atol=1e-4)

    assert result["all_close"] is False
    assert result["max_abs_diff"] == 0.25
    assert result["per_request_max_abs_diff"] == {"req-1": 0.25}


def test_main_writes_json_with_numpy_actions(tmp_path, monkeypatch):
    output_path = tmp_path / "profile.json"
    trace = profile_va_split.RequestTrace(
        request_id="req-1",
        scheduled_at_s=0.0,
        submitted_at_s=0.0,
        completed_at_s=0.1,
        status="ok",
        policy_timing={"infer_ms": 1.0},
        actions=np.asarray([[1.0, 2.0]], dtype=np.float32),
    )
    result = profile_va_split.BenchmarkResult(traces=[trace], summary={"num_requests": 1})
    monkeypatch.setattr(profile_va_split, "run_profile", lambda args: result)

    profile_va_split.main(
        profile_va_split.Args(
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            json_output=output_path,
        )
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"] == {"num_requests": 1}
    assert payload["traces"][0]["actions"] == [[1.0, 2.0]]


def test_validate_mps_environment_requires_pipe_dir(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)

    with pytest.raises(RuntimeError, match="split-mps requires an active MPS environment"):
        profile_va_split.validate_mps_environment("split-mps", require_mps_env=True)


def test_validate_mps_environment_allows_split_no_mps_without_pipe_dir(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)

    profile_va_split.validate_mps_environment("split-no-mps", require_mps_env=True)


def test_resolve_gpu_device_index_uses_override_and_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    assert profile_va_split.resolve_gpu_device_index(pytorch_device="cuda:1", override=7) == 7
    assert profile_va_split.resolve_gpu_device_index(pytorch_device="cuda:1", override=None) == 5
    assert profile_va_split.resolve_gpu_device_index(pytorch_device="cuda", override=None) == 3
    assert profile_va_split.resolve_gpu_device_index(pytorch_device="cpu", override=None) is None


def test_resolve_gpu_device_index_ignores_uuid_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd,GPU-efgh")

    assert profile_va_split.resolve_gpu_device_index(pytorch_device="cuda:1", override=None) == 1
