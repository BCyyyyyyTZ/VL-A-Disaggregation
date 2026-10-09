from __future__ import annotations

import queue

import numpy as np

from openpi.serving.va_split_multinode.credit import take_credit
from openpi.serving.va_split_multinode.host_lane_pool import HostLanePool
from openpi.serving.va_split_multinode.messages import RunConfig
from openpi.serving.va_split_multinode.runtime import run_benchmark


def _cfg(**overrides: object) -> RunConfig:
    cfg = RunConfig(
        backend="fake",
        mode="ours",
        vlm_devices=(-1, -1),
        ae_device=-1,
        transport="local",
        launch="thread",
        num_requests=6,
        rate=1000.0,
        max_batch_size=2,
        ae_max_batch_size=4,
        max_prefix_slots=8,
        max_wait_ms=5.0,
        num_steps=3,
        timeout_s=20.0,
        startup_timeout_s=20.0,
        state_dim=4,
        action_dim=2,
        action_horizon=4,
        token_len=8,
        overlap_d2h=True,
        packed_noise=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_fair_credit_returns_leftover_to_the_queue() -> None:
    credit_q: queue.Queue[list[int] | int] = queue.Queue()
    credit_q.put(list(range(6)))
    local: list[int] = []
    stats = [0.0, 0.0]
    first = take_credit(local, credit_q, max_batch_size=2, stats=stats)
    assert first == 0
    assert local == [1]
    leftover = credit_q.get_nowait()
    assert leftover == [2, 3, 4, 5]


def test_host_lane_pool_roundtrip_via_shared_memory() -> None:
    example = {
        "pad": np.ones((3,), dtype=np.int32),
        "noise": np.zeros((2, 2), dtype=np.float32),
    }
    pool = HostLanePool.create(example, max_lanes=4, shared=True, try_host_register=False)
    try:
        ready = pool.export_ready()
        attached = HostLanePool.attach(ready)
        row = {"pad": np.array([1, 2, 3], dtype=np.int32), "noise": np.ones((2, 2), dtype=np.float32)}
        pool.write_rows([1], [row])
        attached.wait_ready(1, timeout_s=1.0)
        got = attached.read_row(1)
        np.testing.assert_array_equal(got["pad"], row["pad"])
        np.testing.assert_array_equal(got["noise"], row["noise"])
        attached.clear_ready(1)
        attached.close(unlink=False)
    finally:
        pool.close(unlink=True)


def test_local_threads_complete_split_requests() -> None:
    payload = run_benchmark(_cfg())
    assert payload["failed"] == 0
    assert payload["completed"] == 6
    noises = {f"req-{index:06d}": None for index in range(6)}
    assert {item["request_id"] for item in payload["results"]} == set(noises)
    assert payload["segments"]["ae_forward_ms"]["avg_ms"] >= 0.0


def test_spawned_host_shm_split_matches_noise() -> None:
    payload = run_benchmark(_cfg(transport="host_shm", launch="process", vlm_devices=(-1,), num_requests=4))
    assert payload["completed"] == 4
    for item in payload["results"]:
        assert item["status"] == "pass"
        assert item["actions"].shape == (4, 2)


def test_tcp_multinode_loopback() -> None:
    payload = run_benchmark(_cfg(transport="tcp", launch="thread", num_requests=4, vlm_devices=(-1,)))
    assert payload["transport"] == "tcp"
    assert payload["completed"] == 4
    assert payload["failed"] == 0


def test_baseline_threads() -> None:
    payload = run_benchmark(
        _cfg(mode="baseline", baseline_devices=(-1, -1), num_requests=4, transport="local", launch="thread")
    )
    assert payload["completed"] == 4
    assert payload["mode"] == "baseline"
