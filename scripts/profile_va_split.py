from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import contextlib
import dataclasses
import json
import math
import os
import pathlib
import threading
import time
from typing import Any, Literal

import numpy as np
import tyro

Mode = Literal["monolithic", "split-no-mps", "split-mps", "jax-monolithic", "jax-split-ipc"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
TraceStatus = Literal["ok", "error", "timeout"]
DEFAULT_PROFILE_CHECKPOINT_CONFIG = "pi05_libero"
DEFAULT_PROFILE_CHECKPOINT_DIR = "/data1/miliang/models/RLinf-Pi05-LIBERO-SFT"
# When torch.compile is enabled, warm these batch shapes before timed measurement.
# Actual warmup sizes are clamped to profile_warmup_max_batch_size(), which for
# VA-split follows runtime prefix capacity (max_vlm_batch_size * 3, typically 24).
COMPILE_WARMUP_MAX_BATCH_SIZE = 32
COMPILE_WARMUP_BATCH_PLAN: tuple[tuple[int, int], ...] = (
    (1, 2),
    (4, 1),
    (8, 2),
    (16, 1),
    (20, 2),
    (24, 1),
    (32, 1),
)


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticRequest:
    request_id: str
    scheduled_at_s: float
    observation: dict[str, Any]
    noise: np.ndarray | None


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticBatchRequest:
    request_ids: tuple[str, ...]
    scheduled_at_s: float
    original_scheduled_at_s: tuple[float, ...]
    observation: dict[str, Any]
    noise: np.ndarray | None


@dataclasses.dataclass(frozen=True, slots=True)
class RequestTrace:
    request_id: str
    scheduled_at_s: float
    submitted_at_s: float
    completed_at_s: float
    status: TraceStatus
    policy_timing: dict[str, float]
    error: str | None = None
    actions: Any | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkResult:
    traces: list[RequestTrace]
    summary: dict[str, float | int | None]
    consistency: dict[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    config: str
    dir: str


def _default_profile_checkpoint() -> Checkpoint:
    return Checkpoint(config=DEFAULT_PROFILE_CHECKPOINT_CONFIG, dir=DEFAULT_PROFILE_CHECKPOINT_DIR)


@dataclasses.dataclass
class Args:
    policy: Checkpoint = dataclasses.field(default_factory=_default_profile_checkpoint)
    mode: Mode = "split-mps"
    num_requests: int = 128
    request_rate_hz: float = 16.0
    max_inflight: int = 64
    batch_size: int = 8
    seed: int = 0
    num_steps: int = 10
    action_horizon: int = 10
    action_dim: int = 32
    state_dim: int = 8
    image_size: int = 224
    prompt: str = "do something"
    fixed_noise: bool = True
    timeout_s: float = 60.0
    warmup_requests: int = 2
    # After compile/shape warmup, keep issuing real e2e requests until latency is steady
    # (jax-split / jax-monolithic only). Avoids counting first-request cold spikes in profile.
    warmup_until_steady: bool = True
    warmup_steady_window: int = 4
    warmup_steady_max_requests: int = 48
    warmup_steady_rel_tol: float = 0.20
    warmup_steady_spike_factor: float = 3.0
    warmup_concurrent_inflight: int = 4
    jax_compile: bool = True
    jax_compile_warmup: bool = True
    # None => VA-split resolves to max_vlm_batch_size * 3 (prefix capacity).
    jax_compile_warmup_max_batch_size: int | None = None
    slo_ms: float = 200.0
    pytorch_device: str | None = None
    pytorch_compile_mode: CompileMode | None = None
    max_ae_batch_size: int = 8
    max_vlm_batch_size: int = 8
    max_vlm_wait_ms: float = 2.0
    ae_sm_percent: int = 20
    vlm_sm_percent: int = 0
    require_mps_env: bool = True
    gpu_device_index: int | None = None
    check_consistency: bool = False
    consistency_atol: float = 1e-3
    consistency_rtol: float = 1e-3
    json_output: pathlib.Path | None = None


def make_poisson_arrival_times(*, num_requests: int, request_rate_hz: float, seed: int) -> np.ndarray:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if request_rate_hz <= 0:
        raise ValueError("request_rate_hz must be positive")
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(scale=1.0 / request_rate_hz, size=num_requests)
    return np.cumsum(inter_arrivals, dtype=np.float64)


def make_synthetic_libero_requests(
    *,
    num_requests: int,
    request_rate_hz: float,
    seed: int,
    action_horizon: int = 10,
    action_dim: int = 32,
    state_dim: int = 8,
    image_size: int = 224,
    prompt: str = "do something",
    fixed_noise: bool = True,
) -> list[SyntheticRequest]:
    scheduled_at_s = make_poisson_arrival_times(
        num_requests=num_requests,
        request_rate_hz=request_rate_hz,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    requests = []
    for idx, scheduled_at in enumerate(scheduled_at_s):
        observation = {
            "observation/state": rng.normal(size=(state_dim,)).astype(np.float32),
            "observation/image": rng.integers(
                0,
                256,
                size=(image_size, image_size, 3),
                dtype=np.uint8,
            ),
            "observation/wrist_image": rng.integers(
                0,
                256,
                size=(image_size, image_size, 3),
                dtype=np.uint8,
            ),
            "prompt": prompt,
        }
        noise = None
        if fixed_noise:
            noise = rng.normal(size=(action_horizon, action_dim)).astype(np.float32)
        requests.append(
            SyntheticRequest(
                request_id=f"req-{idx:06d}",
                scheduled_at_s=float(scheduled_at),
                observation=observation,
                noise=noise,
            )
        )
    return requests


def make_fcfs_batched_synthetic_requests(
    requests: Sequence[SyntheticRequest],
    *,
    max_batch_size: int,
    max_wait_ms: float,
) -> list[SyntheticBatchRequest]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if max_wait_ms < 0:
        raise ValueError("max_wait_ms must be non-negative")

    batches = []
    max_wait_s = max_wait_ms / 1000.0
    start = 0
    while start < len(requests):
        group = [requests[start]]
        deadline_s = requests[start].scheduled_at_s + max_wait_s
        next_index = start + 1
        while next_index < len(requests) and len(group) < max_batch_size:
            candidate = requests[next_index]
            if candidate.scheduled_at_s > deadline_s:
                break
            group.append(candidate)
            next_index += 1
        dispatch_at_s = group[-1].scheduled_at_s if len(group) >= max_batch_size else deadline_s
        batches.append(
            SyntheticBatchRequest(
                request_ids=tuple(request.request_id for request in group),
                scheduled_at_s=float(dispatch_at_s),
                original_scheduled_at_s=tuple(request.scheduled_at_s for request in group),
                observation=_stack_observation_batch([request.observation for request in group]),
                noise=_stack_noise_batch([request.noise for request in group]),
            )
        )
        start = next_index
    return batches


def make_batched_synthetic_requests(
    requests: Sequence[SyntheticRequest],
    *,
    batch_size: int,
) -> list[SyntheticBatchRequest]:
    return make_fcfs_batched_synthetic_requests(requests, max_batch_size=batch_size, max_wait_ms=0.0)


async def run_benchmark_requests(
    policy: Any,
    requests: list[SyntheticRequest],
    *,
    max_inflight: int,
    timeout_s: float,
    executor: ThreadPoolExecutor | None = None,
) -> list[RequestTrace]:
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    start_s = time.monotonic()
    pending: set[asyncio.Task[RequestTrace]] = set()
    traces: list[RequestTrace] = []
    owns_executor = executor is None
    if executor is None:
        supports_concurrent_infer = bool(getattr(policy, "supports_concurrent_infer", False))
        max_workers = max_inflight if supports_concurrent_infer else 1
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="va-profile")

    try:
        for request in requests:
            await _sleep_until(start_s + request.scheduled_at_s)
            while len(pending) >= max_inflight:
                done, pending = await _wait_for_completed_requests(pending)
                traces.extend(task.result() for task in done)
            submitted_abs_s = time.monotonic()
            submitted_ns = time.monotonic_ns()
            task = asyncio.create_task(
                _run_one_request(
                    policy,
                    request,
                    submitted_at_s=submitted_abs_s - start_s,
                    submitted_ns=submitted_ns,
                    timeout_s=timeout_s,
                    start_s=start_s,
                    executor=executor,
                )
            )
            pending.add(task)

        while pending:
            done, pending = await _wait_for_completed_requests(pending)
            traces.extend(task.result() for task in done)
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    return sorted(traces, key=lambda trace: trace.request_id)


async def run_benchmark_batch_requests(
    policy: Any,
    requests: list[SyntheticBatchRequest],
    *,
    max_inflight: int,
    timeout_s: float,
    executor: ThreadPoolExecutor | None = None,
) -> list[RequestTrace]:
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    start_s = time.monotonic()
    pending: set[asyncio.Task[list[RequestTrace]]] = set()
    traces: list[RequestTrace] = []
    owns_executor = executor is None
    if executor is None:
        supports_concurrent_infer = bool(getattr(policy, "supports_concurrent_infer", False))
        max_workers = max_inflight if supports_concurrent_infer else 1
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="va-profile")

    try:
        for request in requests:
            await _sleep_until(start_s + request.scheduled_at_s)
            while len(pending) >= max_inflight:
                done, pending = await _wait_for_completed_batch_requests(pending)
                for task in done:
                    traces.extend(task.result())
            submitted_abs_s = time.monotonic()
            submitted_ns = time.monotonic_ns()
            task = asyncio.create_task(
                _run_one_batch_request(
                    policy,
                    request,
                    submitted_at_s=submitted_abs_s - start_s,
                    submitted_ns=submitted_ns,
                    timeout_s=timeout_s,
                    start_s=start_s,
                    executor=executor,
                )
            )
            pending.add(task)

        while pending:
            done, pending = await _wait_for_completed_batch_requests(pending)
            for task in done:
                traces.extend(task.result())
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    return sorted(traces, key=lambda trace: trace.request_id)


async def run_benchmark_runtime_batch_requests(
    policy: Any,
    requests: list[SyntheticRequest],
    *,
    max_inflight: int,
    timeout_s: float,
    max_batch_size: int,
    max_wait_ms: float,
    executor: ThreadPoolExecutor | None = None,
) -> list[RequestTrace]:
    if max_inflight <= 0:
        raise ValueError("max_inflight must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if max_wait_ms < 0:
        raise ValueError("max_wait_ms must be non-negative")

    start_s = time.monotonic()
    traces: list[RequestTrace] = []
    in_flight = 0
    owns_executor = executor is None
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="va-profile-baseline-model")

    try:
        queued_requests: list[SyntheticRequest] = []
        next_request_index = 0
        running_task: asyncio.Task[list[RequestTrace]] | None = None
        running_batch_size = 0

        def admit_due_requests() -> None:
            nonlocal in_flight, next_request_index
            now_s = time.monotonic()
            while (
                next_request_index < len(requests)
                and in_flight < max_inflight
                and start_s + requests[next_request_index].scheduled_at_s <= now_s
            ):
                queued_requests.append(requests[next_request_index])
                in_flight += 1
                next_request_index += 1

        def next_admittable_arrival_abs_s() -> float | None:
            if next_request_index >= len(requests) or in_flight >= max_inflight:
                return None
            return start_s + requests[next_request_index].scheduled_at_s

        async def collect_runtime_batch() -> list[SyntheticRequest]:
            batch = [queued_requests.pop(0)]
            deadline_s = time.monotonic() + max_wait_ms / 1000.0
            while len(batch) < max_batch_size:
                admit_due_requests()
                if queued_requests:
                    batch.append(queued_requests.pop(0))
                    continue
                if max_wait_ms == 0:
                    break
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    break
                next_arrival_s = next_admittable_arrival_abs_s()
                sleep_until_s = deadline_s if next_arrival_s is None else min(deadline_s, next_arrival_s)
                await _sleep_until(sleep_until_s)
            return batch

        def start_batch(batch: list[SyntheticRequest]) -> tuple[asyncio.Task[list[RequestTrace]], int]:
            submitted_ns = time.monotonic_ns()
            submitted_at_s = time.monotonic() - start_s
            batch_request = SyntheticBatchRequest(
                request_ids=tuple(request.request_id for request in batch),
                scheduled_at_s=submitted_at_s,
                original_scheduled_at_s=tuple(request.scheduled_at_s for request in batch),
                observation=_stack_observation_batch([request.observation for request in batch]),
                noise=_stack_noise_batch([request.noise for request in batch]),
            )
            return (
                asyncio.create_task(
                    _run_one_batch_request(
                        policy,
                        batch_request,
                        submitted_at_s=submitted_at_s,
                        submitted_ns=submitted_ns,
                        timeout_s=timeout_s,
                        start_s=start_s,
                        executor=executor,
                    )
                ),
                len(batch),
            )

        while next_request_index < len(requests) or queued_requests or running_task is not None:
            admit_due_requests()

            if running_task is not None and running_task.done():
                traces.extend(running_task.result())
                in_flight -= running_batch_size
                running_task = None
                running_batch_size = 0
                continue

            if running_task is None and queued_requests:
                running_task, running_batch_size = start_batch(await collect_runtime_batch())
                continue

            next_arrival_s = next_admittable_arrival_abs_s()
            if running_task is not None:
                if next_arrival_s is None:
                    await asyncio.sleep(0.001)
                else:
                    await asyncio.sleep(min(max(0.0, next_arrival_s - time.monotonic()), 0.001))
                continue

            if next_arrival_s is not None:
                await _sleep_until(next_arrival_s)
                continue

            await asyncio.sleep(0.001)
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    return sorted(traces, key=lambda trace: trace.request_id)


def summarize_traces(
    traces: list[RequestTrace],
    *,
    target_request_rate_hz: float,
    inflight_peak: int,
    slo_ms: float = 200.0,
    gpu_sm_util_mean: float | None = None,
    gpu_mem_bw_util_mean: float | None = None,
) -> dict[str, float | int | None]:
    completed = [trace for trace in traces if trace.status == "ok"]
    failed = [trace for trace in traces if trace.status == "error"]
    timed_out = [trace for trace in traces if trace.status == "timeout"]
    action_latency_ms = [(trace.completed_at_s - trace.submitted_at_s) * 1000.0 for trace in completed]
    end_to_end_latency_ms = [(trace.completed_at_s - trace.scheduled_at_s) * 1000.0 for trace in completed]
    submit_lateness_ms = [(trace.submitted_at_s - trace.scheduled_at_s) * 1000.0 for trace in traces]
    slo_good = _slo_good_traces(completed, slo_ms=slo_ms)

    summary: dict[str, float | int | None] = {
        "target_request_rate_hz": float(target_request_rate_hz),
        "target_requests_per_second": float(target_request_rate_hz),
        "realized_offered_requests_per_second": _rate_over_scheduled_span(len(traces), traces),
        "throughput_requests_per_second": _throughput_requests_per_second(completed),
        "slo_ms": float(slo_ms),
        "slo_good_requests": len(slo_good),
        "slo_goodput_requests_per_second": _rate_over_scheduled_span(len(slo_good), traces),
        "num_requests": len(traces),
        "completed_requests": len(completed),
        "failed_requests": len(failed),
        "timeout_requests": len(timed_out),
        "submit_lateness_mean_ms": _mean(submit_lateness_ms),
        "submit_lateness_p50_ms": _percentile(submit_lateness_ms, 50),
        "submit_lateness_p95_ms": _percentile(submit_lateness_ms, 95),
        "action_latency_mean_ms": _mean(action_latency_ms),
        "action_latency_p50_ms": _percentile(action_latency_ms, 50),
        "action_latency_p95_ms": _percentile(action_latency_ms, 95),
        "action_latency_p99_ms": _percentile(action_latency_ms, 99),
        "end_to_end_latency_mean_ms": _mean(end_to_end_latency_ms),
        "end_to_end_latency_p50_ms": _percentile(end_to_end_latency_ms, 50),
        "end_to_end_latency_p95_ms": _percentile(end_to_end_latency_ms, 95),
        "vlm_prefix_forward_mean_ms": _timing_mean(completed, "vlm_prefix_forward_ms"),
        "vlm_prefix_forward_p50_ms": _timing_percentile(completed, "vlm_prefix_forward_ms", 50),
        "baseline_vlm_latency_mean_ms": _timing_mean(completed, "baseline_vlm_ms"),
        "baseline_vlm_latency_p50_ms": _timing_percentile(completed, "baseline_vlm_ms", 50),
        "baseline_vlm_latency_p95_ms": _timing_percentile(completed, "baseline_vlm_ms", 95),
        "baseline_ae_latency_mean_ms": _timing_mean(completed, "baseline_ae_ms"),
        "baseline_ae_latency_p50_ms": _timing_percentile(completed, "baseline_ae_ms", 50),
        "baseline_ae_latency_p95_ms": _timing_percentile(completed, "baseline_ae_ms", 95),
        "baseline_ae_step_latency_mean_ms": _timing_mean(completed, "baseline_ae_step_ms"),
        "baseline_ae_step_latency_p50_ms": _timing_percentile(completed, "baseline_ae_step_ms", 50),
        "baseline_ae_step_latency_p95_ms": _timing_percentile(completed, "baseline_ae_step_ms", 95),
        "baseline_ae_steps_mean": _timing_mean(completed, "baseline_ae_steps"),
        "baseline_effective_batch_mean": _timing_mean(completed, "baseline_effective_batch"),
        "infer_queue_wait_mean_ms": _timing_mean(completed, "infer_queue_wait_ms"),
        "infer_queue_wait_p50_ms": _timing_percentile(completed, "infer_queue_wait_ms", 50),
        "infer_queue_wait_p95_ms": _timing_percentile(completed, "infer_queue_wait_ms", 95),
        "vlm_request_queue_wait_mean_ms": _timing_mean(completed, "vlm_request_queue_wait_ms"),
        "vlm_request_queue_wait_p50_ms": _timing_percentile(completed, "vlm_request_queue_wait_ms", 50),
        "vlm_request_queue_wait_p95_ms": _timing_percentile(completed, "vlm_request_queue_wait_ms", 95),
        "vlm_request_transfer_mean_ms": _timing_mean(completed, "vlm_request_transfer_ms"),
        "vlm_request_transfer_p50_ms": _timing_percentile(completed, "vlm_request_transfer_ms", 50),
        "vlm_request_transfer_p95_ms": _timing_percentile(completed, "vlm_request_transfer_ms", 95),
        "vlm_queue_wait_mean_ms": _timing_mean(completed, "vlm_queue_wait_ms"),
        "vlm_queue_wait_p50_ms": _timing_percentile(completed, "vlm_queue_wait_ms", 50),
        "vlm_queue_wait_p95_ms": _timing_percentile(completed, "vlm_queue_wait_ms", 95),
        "vlm_batch_wait_mean_ms": _timing_mean(completed, "vlm_batch_wait_ms"),
        "vlm_batch_wait_p50_ms": _timing_percentile(completed, "vlm_batch_wait_ms", 50),
        "vlm_batch_wait_p95_ms": _timing_percentile(completed, "vlm_batch_wait_ms", 95),
        "vlm_input_stage_mean_ms": _timing_mean(completed, "vlm_input_stage_ms"),
        "vlm_input_stage_p50_ms": _timing_percentile(completed, "vlm_input_stage_ms", 50),
        "vlm_input_stage_p95_ms": _timing_percentile(completed, "vlm_input_stage_ms", 95),
        "vlm_sample_kwargs_stage_mean_ms": _timing_mean(completed, "vlm_sample_kwargs_stage_ms"),
        "vlm_sample_kwargs_stage_p50_ms": _timing_percentile(completed, "vlm_sample_kwargs_stage_ms", 50),
        "vlm_sample_kwargs_stage_p95_ms": _timing_percentile(completed, "vlm_sample_kwargs_stage_ms", 95),
        "vlm_observation_stack_mean_ms": _timing_mean(completed, "vlm_observation_stack_ms"),
        "vlm_observation_stack_p50_ms": _timing_percentile(completed, "vlm_observation_stack_ms", 50),
        "vlm_observation_stack_p95_ms": _timing_percentile(completed, "vlm_observation_stack_ms", 95),
        "vlm_to_jax_tree_mean_ms": _timing_mean(completed, "vlm_to_jax_tree_ms"),
        "vlm_to_jax_tree_p50_ms": _timing_percentile(completed, "vlm_to_jax_tree_ms", 50),
        "vlm_to_jax_tree_p95_ms": _timing_percentile(completed, "vlm_to_jax_tree_ms", 95),
        "vlm_observation_from_dict_mean_ms": _timing_mean(completed, "vlm_observation_from_dict_ms"),
        "vlm_observation_from_dict_p50_ms": _timing_percentile(completed, "vlm_observation_from_dict_ms", 50),
        "vlm_observation_from_dict_p95_ms": _timing_percentile(completed, "vlm_observation_from_dict_ms", 95),
        "prefix_queue_wait_mean_ms": _timing_mean(completed, "prefix_queue_wait_ms"),
        "prefix_queue_wait_p50_ms": _timing_percentile(completed, "prefix_queue_wait_ms", 50),
        "prefix_queue_wait_p95_ms": _timing_percentile(completed, "prefix_queue_wait_ms", 95),
        "prefix_transfer_mean_ms": _timing_mean(completed, "prefix_transfer_ms"),
        "prefix_transfer_p50_ms": _timing_percentile(completed, "prefix_transfer_ms", 50),
        "prefix_transfer_p95_ms": _timing_percentile(completed, "prefix_transfer_ms", 95),
        "prefix_admit_wait_mean_ms": _timing_mean(completed, "prefix_admit_wait_ms"),
        "prefix_admit_wait_p50_ms": _timing_percentile(completed, "prefix_admit_wait_ms", 50),
        "prefix_admit_wait_p95_ms": _timing_percentile(completed, "prefix_admit_wait_ms", 95),
        "ae_result_queue_wait_mean_ms": _timing_mean(completed, "ae_result_queue_wait_ms"),
        "ae_result_queue_wait_p50_ms": _timing_percentile(completed, "ae_result_queue_wait_ms", 50),
        "ae_result_queue_wait_p95_ms": _timing_percentile(completed, "ae_result_queue_wait_ms", 95),
        "ae_result_transfer_mean_ms": _timing_mean(completed, "ae_result_transfer_ms"),
        "ae_result_transfer_p50_ms": _timing_percentile(completed, "ae_result_transfer_ms", 50),
        "ae_result_transfer_p95_ms": _timing_percentile(completed, "ae_result_transfer_ms", 95),
        "va_split_transfer_mean_ms": _timing_mean(completed, "va_split_transfer_ms"),
        "va_split_transfer_p50_ms": _timing_percentile(completed, "va_split_transfer_ms", 50),
        "va_split_transfer_p95_ms": _timing_percentile(completed, "va_split_transfer_ms", 95),
        "va_split_queue_wait_mean_ms": _timing_mean(completed, "va_split_queue_wait_ms"),
        "va_split_queue_wait_p50_ms": _timing_percentile(completed, "va_split_queue_wait_ms", 50),
        "va_split_queue_wait_p95_ms": _timing_percentile(completed, "va_split_queue_wait_ms", 95),
        "prefix_lane_ingest_mean_ms": _timing_mean(completed, "prefix_lane_ingest_ms"),
        "prefix_lane_ingest_p50_ms": _timing_percentile(completed, "prefix_lane_ingest_ms", 50),
        "prefix_lane_ingest_p95_ms": _timing_percentile(completed, "prefix_lane_ingest_ms", 95),
        "prefix_lane_compact_mean_ms": _timing_mean(completed, "prefix_lane_compact_ms"),
        "prefix_lane_compact_p50_ms": _timing_percentile(completed, "prefix_lane_compact_ms", 50),
        "prefix_lane_compact_p95_ms": _timing_percentile(completed, "prefix_lane_compact_ms", 95),
        "prefix_lane_overhead_mean_ms": _timing_mean(completed, "prefix_lane_overhead_ms"),
        "prefix_lane_overhead_p50_ms": _timing_percentile(completed, "prefix_lane_overhead_ms", 50),
        "prefix_lane_overhead_p95_ms": _timing_percentile(completed, "prefix_lane_overhead_ms", 95),
        "prefix_pool_write_mean_ms": _timing_mean(completed, "prefix_pool_write_ms"),
        "prefix_pool_write_p50_ms": _timing_percentile(completed, "prefix_pool_write_ms", 50),
        "prefix_pool_write_p95_ms": _timing_percentile(completed, "prefix_pool_write_ms", 95),
        "prefix_pool_compact_mean_ms": _timing_mean(completed, "prefix_pool_compact_ms"),
        "prefix_pool_compact_p50_ms": _timing_percentile(completed, "prefix_pool_compact_ms", 50),
        "prefix_pool_compact_p95_ms": _timing_percentile(completed, "prefix_pool_compact_ms", 95),
        "prefix_pool_overhead_mean_ms": _timing_mean(completed, "prefix_pool_overhead_ms"),
        "prefix_pool_overhead_p50_ms": _timing_percentile(completed, "prefix_pool_overhead_ms", 50),
        "prefix_pool_overhead_p95_ms": _timing_percentile(completed, "prefix_pool_overhead_ms", 95),
        "prefix_slab_map_mean_ms": _timing_mean(completed, "prefix_slab_map_ms"),
        "prefix_slab_map_p50_ms": _timing_percentile(completed, "prefix_slab_map_ms", 50),
        "prefix_slab_map_p95_ms": _timing_percentile(completed, "prefix_slab_map_ms", 95),
        "ae_init_denoise_mean_ms": _timing_mean(completed, "ae_init_denoise_ms"),
        "ae_init_denoise_p50_ms": _timing_percentile(completed, "ae_init_denoise_ms", 50),
        "ae_init_denoise_p95_ms": _timing_percentile(completed, "ae_init_denoise_ms", 95),
        "ae_step_mean_ms": _timing_mean(completed, "ae_step_ms"),
        "ae_step_p50_ms": _timing_percentile(completed, "ae_step_ms", 50),
        "ae_effective_batch_mean": _timing_mean(completed, "ae_effective_batch"),
        "vlm_effective_batch_mean": _timing_mean(completed, "vlm_effective_batch"),
        "policy_effective_batch_mean": _timing_mean(completed, "policy_effective_batch"),
        "batch_wait_mean_ms": _timing_mean(completed, "batch_wait_ms"),
        "batch_wait_p50_ms": _timing_percentile(completed, "batch_wait_ms", 50),
        "batch_wait_p95_ms": _timing_percentile(completed, "batch_wait_ms", 95),
        "effective_batch_mean": _mean(_effective_batch_values(completed)),
        "effective_batch_p50": _percentile(_effective_batch_values(completed), 50),
        "effective_batch_p95": _percentile(_effective_batch_values(completed), 95),
        "effective_batch_p99": _percentile(_effective_batch_values(completed), 99),
        "effective_batch_max": _max(_effective_batch_values(completed)),
        "inflight_peak": int(inflight_peak),
        "gpu_sm_util_mean": gpu_sm_util_mean,
        "gpu_mem_bw_util_mean": gpu_mem_bw_util_mean,
    }
    return summary


def compare_action_traces(
    expected: list[RequestTrace],
    actual: list[RequestTrace],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_by_id = {trace.request_id: trace for trace in actual}
    per_request: dict[str, float] = {}
    missing: list[str] = []
    all_close = True
    max_abs_diff = 0.0

    for expected_trace in expected:
        actual_trace = actual_by_id.get(expected_trace.request_id)
        if actual_trace is None or expected_trace.actions is None or actual_trace.actions is None:
            missing.append(expected_trace.request_id)
            all_close = False
            continue
        expected_actions = np.asarray(expected_trace.actions)
        actual_actions = np.asarray(actual_trace.actions)
        diff = float(np.max(np.abs(expected_actions - actual_actions))) if expected_actions.size else 0.0
        per_request[expected_trace.request_id] = diff
        max_abs_diff = max(max_abs_diff, diff)
        if not np.allclose(expected_actions, actual_actions, rtol=rtol, atol=atol):
            all_close = False

    return {
        "all_close": all_close,
        "max_abs_diff": max_abs_diff,
        "per_request_max_abs_diff": per_request,
        "missing_request_ids": missing,
    }


def print_summary(summary: dict[str, float | int | None]) -> None:
    for key, value in summary.items():
        if value is None:
            print(f"{key}: null")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def compute_inflight_peak(traces: list[RequestTrace]) -> int:
    events = []
    for trace in traces:
        events.append((trace.submitted_at_s, 1))
        events.append((trace.completed_at_s, -1))

    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], -event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def run_profile(args: Args) -> BenchmarkResult:
    requests = make_synthetic_libero_requests(
        num_requests=args.num_requests,
        request_rate_hz=args.request_rate_hz,
        seed=args.seed,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        image_size=args.image_size,
        prompt=args.prompt,
        fixed_noise=args.fixed_noise,
    )
    policy = create_policy_for_mode(args, args.mode)
    policy_device = getattr(policy, "_pytorch_device", args.pytorch_device)
    sampler = GpuUtilizationSampler(device_index=resolve_gpu_device_index(policy_device, args.gpu_device_index))
    try:
        if args.mode in ("monolithic", "jax-monolithic") and args.batch_size > 1:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="va-profile-baseline-model") as executor:
                _warmup_before_timed_workload(policy, requests, args=args, executor=executor)
                sampler.start()
                traces = asyncio.run(
                    run_benchmark_runtime_batch_requests(
                        policy,
                        requests,
                        max_inflight=args.max_inflight,
                        timeout_s=args.timeout_s,
                        max_batch_size=args.batch_size,
                        max_wait_ms=args.max_vlm_wait_ms,
                        executor=executor,
                    )
                )
        else:
            supports_concurrent_infer = bool(getattr(policy, "supports_concurrent_infer", False))
            max_workers = args.max_inflight if supports_concurrent_infer else 1
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="va-profile") as executor:
                _warmup_before_timed_workload(policy, requests, args=args, executor=executor)
                sampler.start()
                traces = asyncio.run(
                    run_benchmark_requests(
                        policy,
                        requests,
                        max_inflight=args.max_inflight,
                        timeout_s=args.timeout_s,
                        executor=executor,
                    )
                )
    finally:
        sampler.stop()
        close_policy(policy)

    summary = summarize_traces(
        traces,
        target_request_rate_hz=args.request_rate_hz,
        inflight_peak=compute_inflight_peak(traces),
        slo_ms=args.slo_ms,
        gpu_sm_util_mean=sampler.gpu_sm_util_mean,
        gpu_mem_bw_util_mean=sampler.gpu_mem_bw_util_mean,
    )
    summary["jax_compile_enabled"] = bool(args.jax_compile)
    summary["jax_compile_warmup_enabled"] = bool(args.jax_compile_warmup)
    summary["jax_warmup_batches"] = _timing_max([trace for trace in traces if trace.status == "ok"], "jax_warmup_batches")
    if summary["jax_warmup_batches"] is None:
        summary["jax_warmup_batches"] = 0.0
    e2e_warmup = getattr(policy, "_profile_e2e_warmup_stats", None)
    if isinstance(e2e_warmup, dict):
        summary["e2e_warmup_requests"] = float(e2e_warmup.get("e2e_warmup_requests", 0.0))
        summary["e2e_warmup_wall_ms"] = float(e2e_warmup.get("e2e_warmup_wall_ms", 0.0))
        if e2e_warmup.get("e2e_warmup_steady_p50_ms") is not None:
            summary["e2e_warmup_steady_p50_ms"] = float(e2e_warmup["e2e_warmup_steady_p50_ms"])
    consistency = _run_consistency_check(args, requests) if args.check_consistency else None
    return BenchmarkResult(traces=traces, summary=summary, consistency=consistency)


def warmup_policy(
    policy: Any,
    request: SyntheticRequest,
    *,
    num_requests: int,
    executor: ThreadPoolExecutor,
) -> None:
    if num_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    for _ in range(num_requests):
        executor.submit(_call_policy_infer, policy, request).result()


def warmup_policy_batch(
    policy: Any,
    request: SyntheticBatchRequest,
    *,
    num_requests: int,
    executor: ThreadPoolExecutor,
) -> None:
    if num_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    for _ in range(num_requests):
        executor.submit(_call_policy_infer_batch, policy, request).result()


def pytorch_compile_warmup_batch_plan(max_batch_size: int) -> tuple[tuple[int, int], ...]:
    """Return compile warmup (batch_size, repeats), clamped to max_batch_size."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    return tuple((min(batch_size, max_batch_size), repeats) for batch_size, repeats in COMPILE_WARMUP_BATCH_PLAN)


def make_fixed_size_batch_request(
    requests: Sequence[SyntheticRequest],
    *,
    batch_size: int,
) -> SyntheticBatchRequest:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not requests:
        raise ValueError("requests must be non-empty")
    selected = [requests[index % len(requests)] for index in range(batch_size)]
    return SyntheticBatchRequest(
        request_ids=tuple(request.request_id for request in selected),
        scheduled_at_s=float(selected[-1].scheduled_at_s),
        original_scheduled_at_s=tuple(request.scheduled_at_s for request in selected),
        observation=_stack_observation_batch([request.observation for request in selected]),
        noise=_stack_noise_batch([request.noise for request in selected]),
    )


def warmup_pytorch_compile_shapes(
    policy: Any,
    requests: Sequence[SyntheticRequest],
    *,
    max_batch_size: int,
    executor: ThreadPoolExecutor,
) -> None:
    """Warm torch.compile paths with explicit batch shapes before timed runs."""
    for batch_size, repeats in pytorch_compile_warmup_batch_plan(max_batch_size):
        batch_request = make_fixed_size_batch_request(requests, batch_size=batch_size)
        warmup_policy_batch(policy, batch_request, num_requests=repeats, executor=executor)


def profile_warmup_max_batch_size(args: Args) -> int:
    """Compile-shape warmup ceiling.

    For VA-split this follows runtime prefix/live-feature capacity
    (`max_vlm_batch_size * 3` in VASplitRuntime), not the VLM FCFS cap alone.
    """
    absolute_cap = COMPILE_WARMUP_MAX_BATCH_SIZE
    if args.mode in ("monolithic", "jax-monolithic"):
        return min(absolute_cap, max(1, int(args.batch_size)))
    # Keep in sync with openpi.serving.va_split.runtime.VASplitRuntime default slots.
    prefix_capacity = max(1, int(args.max_vlm_batch_size) * 3)
    return min(absolute_cap, prefix_capacity, max(1, int(args.max_ae_batch_size)))


def _median_ms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def is_e2e_latency_steady(
    latencies_ms: Sequence[float],
    *,
    window: int,
    rel_tol: float,
    spike_factor: float,
) -> bool:
    """Return True when the last two windows have similar p50 and no huge spikes remain."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(latencies_ms) < 2 * window:
        return False
    recent = list(latencies_ms[-window:])
    prev = list(latencies_ms[-2 * window : -window])
    recent_p50 = _median_ms(recent)
    prev_p50 = _median_ms(prev)
    if recent_p50 <= 0.0 or prev_p50 <= 0.0:
        return False
    if abs(recent_p50 - prev_p50) / prev_p50 > rel_tol:
        return False
    if max(recent) > recent_p50 * spike_factor:
        return False
    return True


def warmup_jax_e2e_pipeline_until_steady(
    policy: Any,
    requests: Sequence[SyntheticRequest],
    *,
    args: Args,
    executor: ThreadPoolExecutor,
) -> dict[str, float]:
    """Warm the full VLM→AE serving path until e2e latency looks steady.

    Component XLA warmup does not cover first real queue/IPC/credit handoff; those cold
    spikes must finish before timed profile measurement.
    """
    if not requests:
        raise ValueError("requests must be non-empty")
    request = requests[0]
    wall_start = time.perf_counter()
    latencies_ms: list[float] = []
    e2e_count = 0

    print(
        f"[profile] e2e pipeline warmup starting (mode={args.mode} "
        f"min={args.warmup_requests} max={args.warmup_steady_max_requests} "
        f"window={args.warmup_steady_window} rel_tol={args.warmup_steady_rel_tol})",
        flush=True,
    )

    def _one_e2e() -> float:
        nonlocal e2e_count
        t0 = time.perf_counter()
        executor.submit(_call_policy_infer, policy, request).result()
        ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(ms)
        e2e_count += 1
        return ms

    for _ in range(max(0, args.warmup_requests)):
        ms = _one_e2e()
        print(f"[profile]   e2e warmup #{e2e_count}: {ms:.1f}ms", flush=True)

    if args.mode in ("jax-split-ipc", "jax-monolithic") and callable(getattr(policy, "infer_batch", None)):
        batch_sizes: list[int] = []
        if args.mode == "jax-monolithic":
            # Baseline FCFS can emit any size in [1, batch_size]; warm all of them so timed
            # measurement does not pay first-hit XLA compile behind the single-worker queue.
            cap = max(1, int(args.batch_size))
            batch_sizes = list(range(1, cap + 1))
        else:
            cap = max(1, min(int(args.max_vlm_batch_size), 8))
            for batch_size in (1, 2, 4, 8):
                if batch_size <= cap and batch_size not in batch_sizes:
                    batch_sizes.append(batch_size)
            if cap not in batch_sizes:
                batch_sizes.append(cap)
        print(f"[profile]   batch-shape e2e warmup: B={batch_sizes}", flush=True)
        for batch_size in batch_sizes:
            batch_request = make_fixed_size_batch_request(requests, batch_size=batch_size)
            t0 = time.perf_counter()
            executor.submit(_call_policy_infer_batch, policy, batch_request).result()
            ms = (time.perf_counter() - t0) * 1000.0
            e2e_count += 1
            print(f"[profile]   e2e batch warmup B={batch_size}: {ms:.1f}ms", flush=True)

    burst = max(0, min(int(args.warmup_concurrent_inflight), int(args.max_inflight)))
    if burst > 1 and bool(getattr(policy, "supports_concurrent_infer", False)):
        print(f"[profile]   concurrent burst warmup: inflight={burst}", flush=True)
        t0 = time.perf_counter()
        futures = [executor.submit(_call_policy_infer, policy, request) for _ in range(burst)]
        for future in futures:
            future.result()
            e2e_count += 1
        burst_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[profile]   concurrent burst done: wall={burst_ms:.1f}ms", flush=True)
        for _ in range(min(burst, args.warmup_steady_window)):
            ms = _one_e2e()
            print(f"[profile]   e2e warmup #{e2e_count}: {ms:.1f}ms (post-burst)", flush=True)

    steady_p50 = 0.0
    if args.warmup_until_steady:
        max_requests = max(args.warmup_steady_max_requests, args.warmup_requests)
        while len(latencies_ms) < max_requests:
            if is_e2e_latency_steady(
                latencies_ms,
                window=args.warmup_steady_window,
                rel_tol=args.warmup_steady_rel_tol,
                spike_factor=args.warmup_steady_spike_factor,
            ):
                break
            ms = _one_e2e()
            recent_p50 = _median_ms(latencies_ms[-args.warmup_steady_window :])
            print(
                f"[profile]   e2e warmup #{e2e_count}: {ms:.1f}ms "
                f"(window_p50={recent_p50:.1f}ms, n_singles={len(latencies_ms)})",
                flush=True,
            )
        steady = is_e2e_latency_steady(
            latencies_ms,
            window=args.warmup_steady_window,
            rel_tol=args.warmup_steady_rel_tol,
            spike_factor=args.warmup_steady_spike_factor,
        )
        steady_p50 = _median_ms(latencies_ms[-args.warmup_steady_window :]) if latencies_ms else 0.0
        if steady:
            print(
                f"[profile] e2e pipeline warmup steady after {len(latencies_ms)} timed singles "
                f"(p50={steady_p50:.1f}ms); total e2e ops≈{e2e_count}",
                flush=True,
            )
        else:
            print(
                f"[profile] e2e pipeline warmup hit max without full steady "
                f"(last_p50={steady_p50:.1f}ms); proceeding to timed profile",
                flush=True,
            )
    elif latencies_ms:
        steady_p50 = _median_ms(latencies_ms[-args.warmup_steady_window :])

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    print(f"[profile] timed measurement starting (e2e_warmup_wall={wall_ms:.1f}ms)", flush=True)
    return {
        "e2e_warmup_requests": float(e2e_count),
        "e2e_warmup_wall_ms": float(wall_ms),
        "e2e_warmup_steady_p50_ms": float(steady_p50),
    }


def _warmup_before_timed_workload(
    policy: Any,
    requests: Sequence[SyntheticRequest],
    *,
    args: Args,
    executor: ThreadPoolExecutor,
) -> None:
    if args.pytorch_compile_mode is not None:
        warmup_pytorch_compile_shapes(
            policy,
            requests,
            max_batch_size=profile_warmup_max_batch_size(args),
            executor=executor,
        )
        return
    if args.mode in ("jax-split-ipc", "jax-monolithic"):
        stats = warmup_jax_e2e_pipeline_until_steady(policy, requests, args=args, executor=executor)
        policy._profile_e2e_warmup_stats = stats  # noqa: SLF001 — profile-only annotation
        return
    if args.mode == "monolithic" and args.batch_size > 1:
        warmup_batch_requests = make_fcfs_batched_synthetic_requests(
            requests,
            max_batch_size=args.batch_size,
            max_wait_ms=args.max_vlm_wait_ms,
        )
        warmup_policy_batch(
            policy,
            warmup_batch_requests[0],
            num_requests=args.warmup_requests,
            executor=executor,
        )
        return
    warmup_policy(
        policy,
        requests[0],
        num_requests=args.warmup_requests,
        executor=executor,
    )


def _with_pytorch_compile_mode(train_config, compile_mode: CompileMode | None):
    if not hasattr(train_config.model, "pytorch_compile_mode"):
        return train_config
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=compile_mode),
    )


def create_policy_for_mode(args: Args, mode: Mode):
    validate_mps_environment(mode, require_mps_env=args.require_mps_env)

    from openpi.policies import policy_config as _policy_config  # noqa: PLC0415
    from openpi.policies import jax_va_split_policy as _jax_va_split_policy  # noqa: PLC0415
    from openpi.policies import va_split_policy as _va_split_policy  # noqa: PLC0415
    from openpi.training import config as _config  # noqa: PLC0415

    train_config = _with_pytorch_compile_mode(_config.get_config(args.policy.config), args.pytorch_compile_mode)
    sample_kwargs = {"num_steps": args.num_steps}
    if mode == "monolithic":
        return _policy_config.create_trained_policy(
            train_config,
            args.policy.dir,
            sample_kwargs=sample_kwargs,
            pytorch_device=args.pytorch_device,
        )
    if mode == "jax-monolithic":
        return _policy_config.create_trained_policy(
            train_config,
            args.policy.dir,
            sample_kwargs=sample_kwargs,
        )
    if mode == "jax-split-ipc":
        return _jax_va_split_policy.create_trained_jax_va_split_policy(
            train_config,
            args.policy.dir,
            sample_kwargs=sample_kwargs,
            max_ae_batch_size=args.max_ae_batch_size,
            max_vlm_batch_size=args.max_vlm_batch_size,
            max_vlm_wait_ms=args.max_vlm_wait_ms,
            ae_sm_percent=args.ae_sm_percent,
            vlm_sm_percent=args.vlm_sm_percent,
            result_timeout_s=args.timeout_s,
            jax_compile=args.jax_compile,
            jax_compile_warmup=args.jax_compile_warmup,
            jax_compile_warmup_max_batch_size=args.jax_compile_warmup_max_batch_size,
        )
    ae_sm_percent = args.ae_sm_percent if mode == "split-mps" else 0
    vlm_sm_percent = args.vlm_sm_percent if mode == "split-mps" else 0
    return _va_split_policy.create_trained_va_split_policy(
        train_config,
        args.policy.dir,
        sample_kwargs=sample_kwargs,
        pytorch_device=args.pytorch_device,
        max_ae_batch_size=args.max_ae_batch_size,
        max_vlm_batch_size=args.max_vlm_batch_size,
        max_vlm_wait_ms=args.max_vlm_wait_ms,
        ae_sm_percent=ae_sm_percent,
        vlm_sm_percent=vlm_sm_percent,
        result_timeout_s=args.timeout_s,
    )


class GpuUtilizationSampler:
    def __init__(self, *, interval_s: float = 0.2, device_index: int | None = 0):
        self._interval_s = interval_s
        self._device_index = device_index
        self._running = False
        self._thread: threading.Thread | None = None
        self._sm_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        if self._device_index is None:
            return
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception:
            self._nvml = None
            self._handle = None
            return

        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()

    @property
    def gpu_sm_util_mean(self) -> float | None:
        return _mean(self._sm_samples)

    @property
    def gpu_mem_bw_util_mean(self) -> float | None:
        return _mean(self._mem_samples)

    def _sample_loop(self) -> None:
        assert self._nvml is not None
        assert self._handle is not None
        while self._running:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._sm_samples.append(float(util.gpu))
                self._mem_samples.append(float(util.memory))
            except Exception:
                return
            time.sleep(self._interval_s)


async def _sleep_until(deadline_s: float) -> None:
    delay_s = deadline_s - time.monotonic()
    if delay_s > 0:
        await asyncio.sleep(delay_s)


async def _wait_for_completed_requests(
    pending: set[asyncio.Task[RequestTrace]],
) -> tuple[set[asyncio.Task[RequestTrace]], set[asyncio.Task[RequestTrace]]]:
    done, pending = await asyncio.wait(pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
    return done, pending


async def _wait_for_completed_batch_requests(
    pending: set[asyncio.Task[list[RequestTrace]]],
) -> tuple[set[asyncio.Task[list[RequestTrace]]], set[asyncio.Task[list[RequestTrace]]]]:
    done, pending = await asyncio.wait(pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
    return done, pending


async def _run_one_request(
    policy: Any,
    request: SyntheticRequest,
    *,
    submitted_at_s: float,
    submitted_ns: int,
    timeout_s: float,
    start_s: float,
    executor: ThreadPoolExecutor,
) -> RequestTrace:
    try:
        result = await _infer_policy(policy, request, submitted_ns=submitted_ns, executor=executor)
        completed_at_s = time.monotonic() - start_s
        if completed_at_s - submitted_at_s > timeout_s:
            return _timeout_trace(request, submitted_at_s, completed_at_s, timeout_s)
        actions, policy_timing = _extract_policy_result(result)
        return RequestTrace(
            request_id=request.request_id,
            scheduled_at_s=request.scheduled_at_s,
            submitted_at_s=submitted_at_s,
            completed_at_s=completed_at_s,
            status="ok",
            policy_timing=policy_timing,
            actions=actions,
        )
    except TimeoutError as exc:
        return _failed_trace(request, submitted_at_s, start_s, "timeout", str(exc))
    except Exception as exc:  # pragma: no cover - exact model failures are environment dependent.
        return _failed_trace(request, submitted_at_s, start_s, "error", repr(exc))


async def _run_one_batch_request(
    policy: Any,
    request: SyntheticBatchRequest,
    *,
    submitted_at_s: float,
    submitted_ns: int,
    timeout_s: float,
    start_s: float,
    executor: ThreadPoolExecutor,
) -> list[RequestTrace]:
    try:
        result = await _infer_policy_batch(policy, request, submitted_ns=submitted_ns, executor=executor)
        completed_at_s = time.monotonic() - start_s
        if completed_at_s - submitted_at_s > timeout_s:
            return _timeout_batch_trace(request, submitted_at_s, completed_at_s, timeout_s)
        actions, policy_timing = _extract_policy_result(result)
        return _batch_result_traces(
            request,
            submitted_at_s=submitted_at_s,
            completed_at_s=completed_at_s,
            actions=actions,
            policy_timing=policy_timing,
        )
    except TimeoutError as exc:
        return _failed_batch_traces(request, submitted_at_s, start_s, "timeout", str(exc))
    except Exception as exc:  # pragma: no cover - exact model failures are environment dependent.
        return _failed_batch_traces(request, submitted_at_s, start_s, "error", repr(exc))


async def _infer_policy(
    policy: Any,
    request: SyntheticRequest,
    *,
    submitted_ns: int,
    executor: ThreadPoolExecutor,
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _call_policy_infer, policy, request, submitted_ns)


async def _infer_policy_batch(
    policy: Any,
    request: SyntheticBatchRequest,
    *,
    submitted_ns: int,
    executor: ThreadPoolExecutor,
) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _call_policy_infer_batch, policy, request, submitted_ns)


def _call_policy_infer(policy: Any, request: SyntheticRequest, submitted_ns: int | None = None) -> Any:
    started_ns = time.monotonic_ns()
    if request.noise is None:
        result = policy.infer(request.observation)
    else:
        result = policy.infer(request.observation, noise=request.noise)
    return _attach_infer_queue_wait(result, submitted_ns=submitted_ns, started_ns=started_ns)


def _call_policy_infer_batch(policy: Any, request: SyntheticBatchRequest, submitted_ns: int | None = None) -> Any:
    started_ns = time.monotonic_ns()
    infer_batch = getattr(policy, "infer_batch", None)
    if not callable(infer_batch):
        raise AttributeError("policy does not support infer_batch")
    if request.noise is None:
        result = infer_batch(request.observation)
    else:
        result = infer_batch(request.observation, noise=request.noise)
    return _attach_infer_queue_wait(result, submitted_ns=submitted_ns, started_ns=started_ns)


def _attach_infer_queue_wait(result: Any, *, submitted_ns: int | None, started_ns: int) -> Any:
    if submitted_ns is None or not isinstance(result, dict):
        return result
    timing = dict(result.get("policy_timing") or {})
    timing["infer_queue_wait_ms"] = max(0.0, (started_ns - submitted_ns) / 1_000_000)
    return {**result, "policy_timing": timing}


def _extract_policy_result(result: Any) -> tuple[Any | None, dict[str, float]]:
    if isinstance(result, dict):
        return result.get("actions"), dict(result.get("policy_timing") or {})
    actions = getattr(result, "actions", None)
    timing = getattr(result, "timing", None)
    return actions, dict(timing or {})


def _batch_result_traces(
    request: SyntheticBatchRequest,
    *,
    submitted_at_s: float,
    completed_at_s: float,
    actions: Any | None,
    policy_timing: dict[str, float],
) -> list[RequestTrace]:
    actions_array = None if actions is None else np.asarray(actions)
    traces = []
    for row, request_id in enumerate(request.request_ids):
        row_timing = dict(policy_timing)
        row_timing.setdefault("effective_batch", float(len(request.request_ids)))
        row_timing.setdefault("policy_effective_batch", float(len(request.request_ids)))
        row_timing["batch_wait_ms"] = max(
            0.0,
            (request.scheduled_at_s - request.original_scheduled_at_s[row]) * 1000.0,
        )
        traces.append(
            RequestTrace(
                request_id=request_id,
                scheduled_at_s=request.original_scheduled_at_s[row],
                submitted_at_s=submitted_at_s,
                completed_at_s=completed_at_s,
                status="ok",
                policy_timing=row_timing,
                actions=None if actions_array is None else actions_array[row],
            )
        )
    return traces


def _failed_trace(
    request: SyntheticRequest,
    submitted_at_s: float,
    start_s: float,
    status: TraceStatus,
    error: str,
) -> RequestTrace:
    return RequestTrace(
        request_id=request.request_id,
        scheduled_at_s=request.scheduled_at_s,
        submitted_at_s=submitted_at_s,
        completed_at_s=time.monotonic() - start_s,
        status=status,
        policy_timing={},
        error=error,
    )


def _failed_batch_traces(
    request: SyntheticBatchRequest,
    submitted_at_s: float,
    start_s: float,
    status: TraceStatus,
    error: str,
) -> list[RequestTrace]:
    completed_at_s = time.monotonic() - start_s
    return [
        RequestTrace(
            request_id=request_id,
            scheduled_at_s=scheduled_at_s,
            submitted_at_s=submitted_at_s,
            completed_at_s=completed_at_s,
            status=status,
            policy_timing={},
            error=error,
        )
        for request_id, scheduled_at_s in zip(request.request_ids, request.original_scheduled_at_s, strict=True)
    ]


def _timeout_trace(
    request: SyntheticRequest,
    submitted_at_s: float,
    completed_at_s: float,
    timeout_s: float,
) -> RequestTrace:
    return RequestTrace(
        request_id=request.request_id,
        scheduled_at_s=request.scheduled_at_s,
        submitted_at_s=submitted_at_s,
        completed_at_s=completed_at_s,
        status="timeout",
        policy_timing={},
        error=f"exceeded timeout_s={timeout_s}",
    )


def _timeout_batch_trace(
    request: SyntheticBatchRequest,
    submitted_at_s: float,
    completed_at_s: float,
    timeout_s: float,
) -> list[RequestTrace]:
    return [
        RequestTrace(
            request_id=request_id,
            scheduled_at_s=scheduled_at_s,
            submitted_at_s=submitted_at_s,
            completed_at_s=completed_at_s,
            status="timeout",
            policy_timing={},
            error=f"exceeded timeout_s={timeout_s}",
        )
        for request_id, scheduled_at_s in zip(request.request_ids, request.original_scheduled_at_s, strict=True)
    ]


def _run_consistency_check(args: Args, requests: list[SyntheticRequest]) -> dict[str, Any]:
    serial_requests = [dataclasses.replace(request, scheduled_at_s=0.0) for request in requests]
    monolithic = create_policy_for_mode(args, "monolithic")
    split = create_policy_for_mode(args, "split-no-mps")
    try:
        monolithic_traces = asyncio.run(
            run_benchmark_requests(
                monolithic,
                serial_requests,
                max_inflight=1,
                timeout_s=args.timeout_s,
            )
        )
        split_traces = asyncio.run(
            run_benchmark_requests(
                split,
                serial_requests,
                max_inflight=1,
                timeout_s=args.timeout_s,
            )
        )
    finally:
        close_policy(monolithic)
        close_policy(split)
    return compare_action_traces(
        monolithic_traces,
        split_traces,
        rtol=args.consistency_rtol,
        atol=args.consistency_atol,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _timing_values(traces: list[RequestTrace], key: str) -> list[float]:
    values = []
    for trace in traces:
        value = trace.policy_timing.get(key)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def _timing_percentile(traces: list[RequestTrace], key: str, percentile: float) -> float | None:
    return _percentile(_timing_values(traces, key), percentile)


def _timing_mean(traces: list[RequestTrace], key: str) -> float | None:
    return _mean(_timing_values(traces, key))


def _timing_max(traces: list[RequestTrace], key: str) -> float | None:
    return _max(_timing_values(traces, key))


def _effective_batch_values(traces: list[RequestTrace]) -> list[float]:
    values = []
    for trace in traces:
        value = trace.policy_timing.get(
            "ae_effective_batch",
            trace.policy_timing.get("effective_batch", trace.policy_timing.get("policy_effective_batch", 1.0)),
        )
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return values


def _stack_observation_batch(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _stack_tree(list(observations))


def _stack_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack_tree([value[key] for value in values]) for key in first}
    if isinstance(first, str):
        return list(values)
    return np.asarray(values).copy()


def _stack_noise_batch(noises: Sequence[np.ndarray | None]) -> np.ndarray | None:
    if all(noise is None for noise in noises):
        return None
    if any(noise is None for noise in noises):
        raise ValueError("Cannot batch a mix of fixed-noise and noise-free requests")
    return np.asarray(noises).copy()


def _slo_good_traces(completed: list[RequestTrace], *, slo_ms: float) -> list[RequestTrace]:
    return [trace for trace in completed if (trace.completed_at_s - trace.scheduled_at_s) * 1000.0 <= slo_ms]


def _rate_over_scheduled_span(count: int, traces: list[RequestTrace]) -> float | None:
    if not traces:
        return None
    first_scheduled_at_s = min(trace.scheduled_at_s for trace in traces)
    last_scheduled_at_s = max(trace.scheduled_at_s for trace in traces)
    duration_s = last_scheduled_at_s - first_scheduled_at_s
    if duration_s <= 0:
        return None
    return count / duration_s


def _throughput_requests_per_second(completed: list[RequestTrace]) -> float | None:
    if not completed:
        return None
    first_submitted_at_s = min(trace.submitted_at_s for trace in completed)
    last_completed_at_s = max(trace.completed_at_s for trace in completed)
    duration_s = last_completed_at_s - first_submitted_at_s
    if duration_s <= 0:
        return None
    return len(completed) / duration_s


def validate_mps_environment(mode: Mode, *, require_mps_env: bool) -> None:
    if mode not in ("split-mps", "jax-split-ipc") or not require_mps_env:
        return
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return
    raise RuntimeError(
        f"{mode} requires an active MPS environment. Start MPS with scripts/run_va_split_mps.sh "
        "or set CUDA_MPS_PIPE_DIRECTORY, or pass --no-require-mps-env to run without this guard."
    )


def resolve_gpu_device_index(pytorch_device: str | None, override: int | None) -> int | None:
    if override is not None:
        return override
    if pytorch_device is not None and not pytorch_device.startswith("cuda"):
        return None

    logical_index = 0
    if pytorch_device and ":" in pytorch_device:
        try:
            logical_index = int(pytorch_device.rsplit(":", maxsplit=1)[1])
        except ValueError:
            logical_index = 0

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        entries = [entry.strip() for entry in visible_devices.split(",") if entry.strip()]
        if logical_index < len(entries):
            try:
                return int(entries[logical_index])
            except ValueError:
                return logical_index
    return logical_index


def close_policy(policy: Any) -> None:
    shutdown = getattr(policy, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    close = getattr(policy, "close", None)
    if callable(close):
        close()
        return
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()


def main(args: Args) -> None:
    result = run_profile(args)
    print_summary(result.summary)
    if result.consistency is not None:
        print("consistency:")
        print(json.dumps(result.consistency, default=_json_default, indent=2, sort_keys=True))
    if args.json_output is not None:
        payload = {
            "summary": result.summary,
            "per_request_e2e_ms": per_request_e2e_ms(result.traces),
            "consistency": result.consistency,
        }
        args.json_output.write_text(
            json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def per_request_e2e_ms(traces: list[RequestTrace]) -> dict[str, float]:
    return {
        trace.request_id: (trace.completed_at_s - trace.scheduled_at_s) * 1000.0
        for trace in sorted(traces, key=lambda trace: trace.request_id)
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


if __name__ == "__main__":
    main(tyro.cli(Args))
