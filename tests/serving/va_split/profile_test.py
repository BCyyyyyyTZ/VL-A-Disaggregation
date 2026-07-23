from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time

import numpy as np
import pytest

from openpi.policies import va_split_policy
from openpi.training import config as training_config
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


def test_args_defaults_to_profile_checkpoint():
    args = profile_va_split.Args()

    assert args.policy.config == "pi05_libero"
    assert args.policy.dir == "/data2/gaobowen/model/RLinf-Pi05-LIBERO-SFT"


def test_args_defaults_to_two_warmup_requests():
    assert profile_va_split.Args().warmup_requests == 2


def test_args_disables_pytorch_compile_by_default():
    assert profile_va_split.Args().pytorch_compile_mode is None


class _ConcurrentFakePolicy:
    supports_concurrent_infer = True

    def __init__(self, *, sleep_s: float = 0.02):
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0
        self.completed_calls = 0
        self.infer_calls = 0
        self.infer_batch_calls = 0
        self.observed_batch_sizes: list[int] = []

    def infer(self, obs, *, noise=None):
        self.infer_calls += 1
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
                "vlm_request_transfer_ms": 0.25,
                "prefix_transfer_ms": 0.5,
                "ae_result_transfer_ms": 0.75,
                "va_split_transfer_ms": 1.5,
                "ae_step_ms": 2.0,
                "ae_effective_batch": 3.0,
            },
        }

    def infer_batch(self, obs, *, noise=None):
        self.infer_batch_calls += 1
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(self._sleep_s)
            batch_size = len(obs["value"])
            self.observed_batch_sizes.append(batch_size)
            return {
                "actions": np.asarray(obs["value"], dtype=np.float32).reshape(batch_size, 1),
                "policy_timing": {
                    "effective_batch": float(batch_size),
                    "policy_effective_batch": float(batch_size),
                    "baseline_vlm_ms": 4.0,
                    "baseline_ae_ms": 6.0,
                    "baseline_ae_step_ms": 3.0,
                    "baseline_ae_steps": 2.0,
                    "baseline_effective_batch": float(batch_size),
                },
            }
        finally:
            with self._lock:
                self._active_calls -= 1
                self.completed_calls += 1


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
    assert summary["target_requests_per_second"] == 1000.0
    assert summary["throughput_requests_per_second"] is not None
    assert summary["inflight_peak"] == 4
    assert summary["submit_lateness_mean_ms"] is not None
    assert summary["action_latency_mean_ms"] is not None
    assert summary["end_to_end_latency_mean_ms"] is not None
    assert summary["vlm_prefix_forward_mean_ms"] == 1.0
    assert summary["vlm_prefix_forward_p50_ms"] == 1.0
    assert summary["vlm_request_transfer_mean_ms"] == 0.25
    assert summary["prefix_transfer_mean_ms"] == 0.5
    assert summary["ae_result_transfer_mean_ms"] == 0.75
    assert summary["va_split_transfer_mean_ms"] == 1.5
    assert summary["ae_step_mean_ms"] == 2.0
    assert summary["ae_step_p50_ms"] == 2.0
    assert summary["ae_effective_batch_mean"] == 3.0
    assert summary["effective_batch_mean"] == 3.0
    assert summary["effective_batch_p50"] == 3.0
    assert summary["effective_batch_p95"] == 3.0
    assert summary["effective_batch_p99"] == 3.0
    assert summary["effective_batch_max"] == 3.0
    assert policy.max_active_calls > 1


def test_make_fcfs_batched_synthetic_requests_uses_max_batch_size_and_max_wait():
    requests = [
        dataclasses.replace(request, scheduled_at_s=scheduled_at_s)
        for request, scheduled_at_s in zip(
            _fake_requests(5),
            [0.0, 0.001, 0.010, 0.011, 0.012],
            strict=True,
        )
    ]

    batches = profile_va_split.make_fcfs_batched_synthetic_requests(
        requests,
        max_batch_size=4,
        max_wait_ms=2.0,
    )

    assert [batch.request_ids for batch in batches] == [
        ("req-000000", "req-000001"),
        ("req-000002", "req-000003", "req-000004"),
    ]
    assert batches[0].scheduled_at_s == pytest.approx(0.002)
    assert batches[1].scheduled_at_s == pytest.approx(0.012)
    assert batches[0].observation["value"].shape == (2,)
    assert batches[0].noise.shape == (2, 2, 3)
    assert batches[-1].observation["value"].shape == (3,)


def test_make_fcfs_batched_synthetic_requests_dispatches_immediately_when_full():
    batches = profile_va_split.make_fcfs_batched_synthetic_requests(
        _fake_requests(4),
        max_batch_size=3,
        max_wait_ms=100.0,
    )

    assert batches[0].request_ids == ("req-000000", "req-000001", "req-000002")
    assert batches[0].scheduled_at_s == pytest.approx(0.002)
    assert batches[1].request_ids == ("req-000003",)
    assert batches[1].scheduled_at_s == pytest.approx(0.103)


def test_make_batched_synthetic_requests_stacks_libero_prompt_batch():
    requests = profile_va_split.make_synthetic_libero_requests(
        num_requests=2,
        request_rate_hz=128.0,
        seed=11,
        fixed_noise=True,
    )

    batches = profile_va_split.make_fcfs_batched_synthetic_requests(
        requests,
        max_batch_size=2,
        max_wait_ms=1000.0,
    )

    assert batches[0].observation["observation/state"].shape == (2, 8)
    assert batches[0].observation["observation/image"].shape == (2, 224, 224, 3)
    assert batches[0].observation["prompt"] == ["do something", "do something"]
    assert batches[0].noise.shape == (2, 10, 32)


def test_run_benchmark_batches_call_policy_infer_batch_once_per_wave():
    policy = _ConcurrentFakePolicy(sleep_s=0.0)
    batches = profile_va_split.make_fcfs_batched_synthetic_requests(
        _fake_requests(4),
        max_batch_size=2,
        max_wait_ms=2.0,
    )

    traces = asyncio.run(
        profile_va_split.run_benchmark_batch_requests(
            policy,
            batches,
            max_inflight=2,
            timeout_s=1.0,
        )
    )
    summary = profile_va_split.summarize_traces(traces, target_request_rate_hz=1000.0, inflight_peak=2)

    assert [trace.request_id for trace in traces] == ["req-000000", "req-000001", "req-000002", "req-000003"]
    assert [trace.status for trace in traces] == ["ok", "ok", "ok", "ok"]
    assert policy.completed_calls == 2
    assert policy.infer_batch_calls == 2
    assert all(trace.policy_timing["effective_batch"] == 2.0 for trace in traces)
    assert summary["policy_effective_batch_mean"] == 2.0
    assert summary["baseline_vlm_latency_mean_ms"] == 4.0
    assert summary["baseline_ae_latency_mean_ms"] == 6.0
    assert summary["baseline_ae_step_latency_mean_ms"] == 3.0
    assert summary["baseline_ae_steps_mean"] == 2.0
    assert summary["baseline_effective_batch_mean"] == 2.0
    assert summary["batch_wait_p95_ms"] is not None


def test_summarize_traces_reports_realized_offered_rate_and_slo_goodput():
    traces = [
        profile_va_split.RequestTrace(
            request_id="req-0",
            scheduled_at_s=0.0,
            submitted_at_s=0.0,
            completed_at_s=0.1,
            status="ok",
            policy_timing={},
        ),
        profile_va_split.RequestTrace(
            request_id="req-1",
            scheduled_at_s=1.0,
            submitted_at_s=1.0,
            completed_at_s=1.3,
            status="ok",
            policy_timing={},
        ),
        profile_va_split.RequestTrace(
            request_id="req-2",
            scheduled_at_s=2.0,
            submitted_at_s=2.0,
            completed_at_s=2.1,
            status="ok",
            policy_timing={},
        ),
    ]

    summary = profile_va_split.summarize_traces(traces, target_request_rate_hz=2.0, inflight_peak=1)

    assert summary["realized_offered_requests_per_second"] == 1.5
    assert summary["slo_ms"] == 200.0
    assert summary["slo_goodput_requests_per_second"] == 1.0
    assert summary["slo_good_requests"] == 2


def test_run_profile_warms_up_policy_before_timed_workload(monkeypatch):
    policy = _SerialFakePolicy(sleep_s=0.0)
    calls = []
    worker_thread_ids = []

    def infer(obs, *, noise=None):
        calls.append(obs["value"])
        worker_thread_ids.append(threading.get_ident())
        return {
            "actions": np.asarray([obs["value"]], dtype=np.float32),
            "policy_timing": {"infer_ms": 0.0},
        }

    policy.infer = infer
    monkeypatch.setattr(profile_va_split, "create_policy_for_mode", lambda args, mode: policy)
    monkeypatch.setattr(profile_va_split, "make_synthetic_libero_requests", lambda **kwargs: _fake_requests(2))

    result = profile_va_split.run_profile(
        profile_va_split.Args(
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            num_requests=2,
            request_rate_hz=1000.0,
            warmup_requests=2,
            pytorch_device="cpu",
        )
    )

    assert calls[:2] == [0, 0]
    assert [trace.request_id for trace in result.traces] == ["req-000000", "req-000001"]
    assert policy.completed_calls == 0
    assert len(set(worker_thread_ids)) == 1
    assert worker_thread_ids[0] != threading.get_ident()


def test_run_profile_uses_baseline_fcfs_batching_only_for_monolithic(monkeypatch):
    policy = _SerialFakePolicy(sleep_s=0.0)
    monkeypatch.setattr(profile_va_split, "create_policy_for_mode", lambda args, mode: policy)
    monkeypatch.setattr(profile_va_split, "make_synthetic_libero_requests", lambda **kwargs: _fake_requests(5))

    result = profile_va_split.run_profile(
        profile_va_split.Args(
            mode="monolithic",
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            num_requests=5,
            request_rate_hz=1000.0,
            batch_size=4,
            max_vlm_wait_ms=2.0,
            warmup_requests=0,
            pytorch_device="cpu",
        )
    )

    assert [trace.request_id for trace in result.traces] == [f"req-{idx:06d}" for idx in range(5)]
    assert policy.infer_calls == 0
    assert policy.infer_batch_calls == 2
    assert policy.observed_batch_sizes == [3, 2]
    assert result.summary["policy_effective_batch_mean"] == 2.6


def test_run_profile_monolithic_batching_uses_one_model_lane(monkeypatch):
    policy = _ConcurrentFakePolicy(sleep_s=0.03)
    monkeypatch.setattr(profile_va_split, "create_policy_for_mode", lambda args, mode: policy)
    monkeypatch.setattr(profile_va_split, "make_synthetic_libero_requests", lambda **kwargs: _fake_requests(6))

    result = profile_va_split.run_profile(
        profile_va_split.Args(
            mode="monolithic",
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            num_requests=6,
            request_rate_hz=1000.0,
            max_inflight=8,
            batch_size=3,
            max_vlm_wait_ms=2.0,
            warmup_requests=0,
            pytorch_device="cpu",
        )
    )

    assert [trace.request_id for trace in result.traces] == [f"req-{idx:06d}" for idx in range(6)]
    assert policy.infer_calls == 0
    assert policy.infer_batch_calls == 2
    assert policy.observed_batch_sizes == [3, 3]
    assert policy.max_active_calls == 1


def test_run_profile_does_not_add_outer_batching_for_split_modes(monkeypatch):
    policy = _SerialFakePolicy(sleep_s=0.0)

    def fail_infer_batch(obs, *, noise=None):
        raise AssertionError("split profile should let the VLM FCFS collector batch single requests")

    policy.infer_batch = fail_infer_batch
    monkeypatch.setattr(profile_va_split, "create_policy_for_mode", lambda args, mode: policy)
    monkeypatch.setattr(profile_va_split, "make_synthetic_libero_requests", lambda **kwargs: _fake_requests(3))

    result = profile_va_split.run_profile(
        profile_va_split.Args(
            mode="split-mps",
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            num_requests=3,
            request_rate_hz=1000.0,
            batch_size=4,
            max_vlm_batch_size=4,
            max_vlm_wait_ms=2.0,
            warmup_requests=0,
            pytorch_device="cpu",
            require_mps_env=False,
        )
    )

    assert [trace.request_id for trace in result.traces] == ["req-000000", "req-000001", "req-000002"]
    assert policy.infer_calls == 3
    assert policy.infer_batch_calls == 0


def test_create_policy_for_mode_uses_profile_timeout_for_split_runtime(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    train_config = FakeTrainConfig()
    captured_kwargs = {}

    def create_split_policy(received_train_config, *args, **kwargs):
        captured_kwargs["train_config"] = received_train_config
        captured_kwargs.update(kwargs)
        return "split-policy"

    monkeypatch.setattr(training_config, "get_config", lambda config_name: train_config)
    monkeypatch.setattr(va_split_policy, "create_trained_va_split_policy", create_split_policy)

    policy = profile_va_split.create_policy_for_mode(
        profile_va_split.Args(
            mode="split-no-mps",
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            timeout_s=321.0,
            max_vlm_batch_size=6,
            max_vlm_wait_ms=1.25,
        ),
        "split-no-mps",
    )

    assert policy == "split-policy"
    assert captured_kwargs["train_config"].model.pytorch_compile_mode is None
    assert captured_kwargs["result_timeout_s"] == 321.0
    assert captured_kwargs["max_vlm_batch_size"] == 6
    assert captured_kwargs["max_vlm_wait_ms"] == 1.25


def test_create_policy_for_mode_allows_explicit_pytorch_compile_opt_in(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    captured_kwargs = {}

    def create_split_policy(received_train_config, *args, **kwargs):
        captured_kwargs["train_config"] = received_train_config
        return "split-policy"

    monkeypatch.setattr(training_config, "get_config", lambda config_name: FakeTrainConfig())
    monkeypatch.setattr(va_split_policy, "create_trained_va_split_policy", create_split_policy)

    policy = profile_va_split.create_policy_for_mode(
        profile_va_split.Args(
            mode="split-no-mps",
            policy=profile_va_split.Checkpoint(config="dummy", dir="/tmp/checkpoint"),
            pytorch_compile_mode="default",
        ),
        "split-no-mps",
    )

    assert policy == "split-policy"
    assert captured_kwargs["train_config"].model.pytorch_compile_mode == "default"


def test_print_summary_uses_requests_per_second_units(capsys):
    profile_va_split.print_summary(
        {
            "target_requests_per_second": 4.0,
            "throughput_requests_per_second": 2.5,
            "completed_requests": 128,
        }
    )

    output = capsys.readouterr().out
    assert "target_requests_per_second: 4.000" in output
    assert "throughput_requests_per_second: 2.500" in output
    assert "hz" not in output.lower()


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
    assert "traces" not in payload
    assert payload["per_request_e2e_ms"] == {"req-1": 100.0}


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
