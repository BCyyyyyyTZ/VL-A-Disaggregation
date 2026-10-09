"""Local and multi-node VA-split benchmark orchestration."""

from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
import queue
import statistics
import threading
import time
from typing import Any

import numpy as np

from openpi.serving.va_split_multinode.backends import synthetic_observation
from openpi.serving.va_split_multinode.backends import unbatch_observation
from openpi.serving.va_split_multinode.bootstrap import entry as process_entry
from openpi.serving.va_split_multinode.host_lane_pool import unlink_pool
from openpi.serving.va_split_multinode.messages import RunConfig
from openpi.serving.va_split_multinode.messages import WorkItem
from openpi.serving.va_split_multinode.tcp import Duplex
from openpi.serving.va_split_multinode.tcp import connect
from openpi.serving.va_split_multinode.tcp import listen
from openpi.serving.va_split_multinode.tcp import serve_ae
from openpi.serving.va_split_multinode.tcp import serve_vlm
from openpi.serving.va_split_multinode.workers import run_worker


def run_benchmark(cfg: RunConfig) -> dict[str, Any]:
    cfg.validate()
    _validate_gpus(cfg)
    if cfg.role == "ae":
        if not cfg.result_addr:
            raise ValueError("AE role requires result_addr")
        serve_ae(cfg, result_addr=cfg.result_addr, bind_host=cfg.bind_host)
        return {"role": "ae", "status": "done"}
    if cfg.role == "vlm":
        if not cfg.ae_addr:
            raise ValueError("VLM role requires ae_addr")
        serve_vlm(cfg, bind_host=cfg.bind_host)
        return {"role": "vlm", "worker": cfg.worker_index, "status": "done"}
    if cfg.resolved_transport() == "tcp":
        return _run_tcp(cfg)
    return _run_local(cfg)


def _run_local(cfg: RunConfig) -> dict[str, Any]:
    if cfg.launch == "process":
        ctx = mp.get_context("spawn")
        result_q: Any = ctx.Queue()
        prefix_q: Any = ctx.Queue()
        pool_q: Any = ctx.Queue()
        credit_q: Any = ctx.Queue()
        workers, holders = _spawn_local_processes(ctx, cfg, result_q, prefix_q, pool_q, credit_q)
        closer = _join_processes
    else:
        result_q = queue.Queue()
        prefix_q = queue.Queue()
        pool_q = queue.Queue()
        credit_q = queue.Queue()
        workers, holders = _spawn_local_threads(cfg, result_q, prefix_q, pool_q, credit_q)
        closer = _join_threads
    pool_ready = None
    try:
        pool_ready = _wait_ready(result_q, expected=_ready_count(cfg), timeout_s=cfg.startup_timeout_s)
        started, schedule = _dispatch(cfg, workers)
        for work_q, _device in workers:
            work_q.put(None)
        results, done, errors = _collect(
            result_q,
            expected_results=cfg.num_requests,
            expected_done=_ready_count(cfg),
            timeout_s=cfg.timeout_s,
        )
        if errors:
            raise RuntimeError(errors[0]["error"])
    finally:
        closer(holders)
        if pool_ready is not None:
            unlink_pool(pool_ready)
    return _payload(cfg, started, schedule, results, done)


def _run_tcp(cfg: RunConfig) -> dict[str, Any]:
    if cfg.role == "coordinator":
        return _coordinate_remote(cfg)
    result_listener, result_addr = listen("127.0.0.1", 0)
    address_q: queue.Queue[str] = queue.Queue()
    prefix_box: dict[str, str] = {}
    holders: list[threading.Thread] = []
    try:
        ae_thread = threading.Thread(
            target=lambda: serve_ae(cfg, result_addr=result_addr, bind_host="127.0.0.1"),
            name="va-ae",
            daemon=True,
        )
        ae_thread.start()
        holders.append(ae_thread)
        result_conn = _accept_one(result_listener, timeout_s=cfg.startup_timeout_s)
        result = Duplex(result_conn)
        listen_msg = result.recv(timeout=cfg.startup_timeout_s)
        if not isinstance(listen_msg, dict) or listen_msg.get("kind") != "listen":
            raise RuntimeError(f"AE did not publish its prefix address: {listen_msg!r}")
        prefix_box["addr"] = str(listen_msg["addr"])
        for index in range(len(cfg.vlm_devices)):
            vlm_cfg = replace(
                cfg,
                worker_index=index,
                ae_addr=prefix_box["addr"],
                extra={**cfg.extra, "address_queue": address_q},
            )
            holders.append(_thread(serve_vlm, vlm_cfg, bind_host="127.0.0.1"))
        for holder in holders[1:]:
            holder.start()
        vlm_addrs = _collect_vlm_addrs(address_q, expected=len(cfg.vlm_devices), timeout_s=cfg.startup_timeout_s)
        return _coordinate_connected(cfg, result, tuple(vlm_addrs))
    finally:
        result_listener.close()
        for holder in holders:
            holder.join(timeout=5)


def _coordinate_remote(cfg: RunConfig) -> dict[str, Any]:
    if not cfg.result_addr or not cfg.vlm_addrs:
        raise ValueError("coordinator role requires result_addr and vlm_addrs")
    host, port = _split_addr(cfg.result_addr)
    result_listener, _published = listen(host, port)
    try:
        result = Duplex(_accept_one(result_listener, timeout_s=cfg.startup_timeout_s))
        listen_msg = result.recv(timeout=cfg.startup_timeout_s)
        if not isinstance(listen_msg, dict) or listen_msg.get("kind") != "listen":
            raise RuntimeError(f"AE did not publish its prefix address: {listen_msg!r}")
        return _coordinate_connected(cfg, result, cfg.vlm_addrs)
    finally:
        result_listener.close()


def _coordinate_connected(cfg: RunConfig, result: Duplex, vlm_addrs: tuple[str, ...]) -> dict[str, Any]:
    peers: list[Duplex] = []
    try:
        _wait_kind(result, "ready", timeout_s=cfg.startup_timeout_s)
        peers = [Duplex(connect(addr, timeout_s=cfg.startup_timeout_s)) for addr in vlm_addrs]
        started, schedule = _dispatch_tcp(cfg, peers)
        for peer in peers:
            peer.send({"op": "close"})
        results, done, errors = _collect_tcp(result, expected_results=cfg.num_requests, timeout_s=cfg.timeout_s)
        if errors:
            raise RuntimeError(errors[0]["error"])
        return _payload(cfg, started, schedule, results, done)
    finally:
        for peer in peers:
            peer.close()
        result.close()


def _spawn_local_threads(
    cfg: RunConfig,
    result_q: Any,
    prefix_q: Any,
    pool_q: Any,
    credit_q: Any,
) -> tuple[list[tuple[Any, int]], list[threading.Thread]]:
    workers: list[tuple[Any, int]] = []
    threads: list[threading.Thread] = []
    if cfg.mode == "baseline":
        devices = tuple(cfg.baseline_devices or cfg.vlm_devices)
        for index, device in enumerate(devices):
            work_q: Any = queue.Queue()
            workers.append((work_q, int(device)))
            threads.append(
                _thread(
                    run_worker,
                    "baseline",
                    cfg,
                    device=int(device),
                    worker_index=index,
                    num_vlm=len(devices),
                    work_q=work_q,
                    prefix_q=None,
                    result_q=result_q,
                    pool_q=None,
                    credit_q=None,
                )
            )
    else:
        for index, device in enumerate(cfg.vlm_devices):
            work_q = queue.Queue()
            workers.append((work_q, int(device)))
            threads.append(
                _thread(
                    run_worker,
                    "vlm",
                    cfg,
                    device=int(device),
                    worker_index=index,
                    num_vlm=len(cfg.vlm_devices),
                    work_q=work_q,
                    prefix_q=prefix_q,
                    result_q=result_q,
                    pool_q=pool_q,
                    credit_q=credit_q,
                )
            )
        threads.append(
            _thread(
                run_worker,
                "ae",
                cfg,
                device=int(cfg.ae_device),
                worker_index=0,
                num_vlm=len(cfg.vlm_devices),
                work_q=None,
                prefix_q=prefix_q,
                result_q=result_q,
                pool_q=pool_q,
                credit_q=credit_q,
            )
        )
    for thread in threads:
        thread.start()
    return workers, threads


def _spawn_local_processes(
    ctx: Any,
    cfg: RunConfig,
    result_q: Any,
    prefix_q: Any,
    pool_q: Any,
    credit_q: Any,
) -> tuple[list[tuple[Any, int]], list[Any]]:
    workers: list[tuple[Any, int]] = []
    procs: list[Any] = []
    if cfg.mode == "baseline":
        devices = tuple(cfg.baseline_devices or cfg.vlm_devices)
        for index, device in enumerate(devices):
            work_q = ctx.Queue()
            workers.append((work_q, int(device)))
            procs.append(
                ctx.Process(
                    target=process_entry,
                    args=("baseline", int(device), cfg, work_q, None, result_q, None, None, index, len(devices)),
                    daemon=True,
                )
            )
    else:
        for index, device in enumerate(cfg.vlm_devices):
            work_q = ctx.Queue()
            workers.append((work_q, int(device)))
            procs.append(
                ctx.Process(
                    target=process_entry,
                    args=(
                        "vlm",
                        int(device),
                        cfg,
                        work_q,
                        prefix_q,
                        result_q,
                        pool_q,
                        credit_q,
                        index,
                        len(cfg.vlm_devices),
                    ),
                    daemon=True,
                )
            )
        procs.append(
            ctx.Process(
                target=process_entry,
                args=(
                    "ae",
                    int(cfg.ae_device),
                    cfg,
                    None,
                    prefix_q,
                    result_q,
                    pool_q,
                    credit_q,
                    0,
                    len(cfg.vlm_devices),
                ),
                daemon=True,
            )
        )
    for proc in procs:
        proc.start()
    return workers, procs


def _ready_count(cfg: RunConfig) -> int:
    if cfg.mode == "baseline":
        return len(cfg.baseline_devices or cfg.vlm_devices)
    return len(cfg.vlm_devices) + 1


def _wait_ready(result_q: Any, *, expected: int, timeout_s: float) -> Any:
    ready = 0
    pool_ready = None
    deadline = time.perf_counter() + timeout_s
    while ready < expected:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"workers did not become ready ({ready}/{expected})")
        message = result_q.get(timeout=remaining)
        kind = message.get("kind")
        if kind == "ready":
            ready += 1
        elif kind == "pool":
            pool_ready = message["ready"]
        elif kind == "worker_error":
            raise RuntimeError(message["error"])
    return pool_ready


def _dispatch(cfg: RunConfig, workers: list[tuple[Any, int]]) -> tuple[float, list[float]]:
    plan = _arrival_plan(cfg, len(workers))
    schedule = [item[0] for item in plan]
    started = time.perf_counter()
    for scheduled_s, worker_index, seq in plan:
        delay = started + scheduled_s - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        workers[worker_index][0].put(_work_item(cfg, seq=seq, scheduled_s=scheduled_s))
    return started, schedule


def _dispatch_tcp(cfg: RunConfig, peers: list[Duplex]) -> tuple[float, list[float]]:
    plan = _arrival_plan(cfg, len(peers))
    schedule = [item[0] for item in plan]
    started = time.perf_counter()
    for scheduled_s, worker_index, seq in plan:
        delay = started + scheduled_s - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        peers[worker_index].send({"op": "work", "item": _work_item(cfg, seq=seq, scheduled_s=scheduled_s)})
    return started, schedule


def _work_item(cfg: RunConfig, *, seq: int, scheduled_s: float) -> WorkItem:
    observation = unbatch_observation(synthetic_observation(cfg, batch_size=1, seed=cfg.seed + seq + 1))[0]
    rng = np.random.default_rng(cfg.seed + 17 + seq)
    noise = rng.standard_normal((cfg.action_horizon, cfg.action_dim)).astype(np.float32)
    return WorkItem(
        request_id=f"req-{seq:06d}",
        observation=observation,
        noise=noise,
        num_steps=cfg.num_steps,
        scheduled_at_s=scheduled_s,
        enqueue_ns=time.monotonic_ns(),
    )


def _arrival_plan(cfg: RunConfig, workers: int) -> list[tuple[float, int, int]]:
    if cfg.arrival_scope == "global":
        return [(float(stamp), index % workers, index) for index, stamp in enumerate(_arrivals(cfg.num_requests, cfg.rate, cfg.seed))]
    plan: list[tuple[float, int, int]] = []
    counts = _split_counts(cfg.num_requests, workers)
    seq = 0
    for worker_index, count in enumerate(counts):
        for stamp in _arrivals(count, cfg.rate, cfg.seed + worker_index * 9973):
            plan.append((float(stamp), worker_index, seq))
            seq += 1
    plan.sort(key=lambda item: (item[0], item[1]))
    return plan


def _arrivals(count: int, rate: float, seed: int) -> list[float]:
    if count <= 0:
        return []
    if rate <= 0:
        return [0.0] * count
    rng = np.random.default_rng(seed)
    return list(np.cumsum(rng.exponential(scale=1.0 / rate, size=count)))


def _split_counts(total: int, workers: int) -> list[int]:
    base, rem = divmod(total, workers)
    return [base + (1 if index < rem else 0) for index in range(workers)]


def _collect(
    result_q: Any,
    *,
    expected_results: int,
    expected_done: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    deadline = time.perf_counter() + timeout_s
    while len(results) < expected_results or len(done) < expected_done:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            message = result_q.get(timeout=remaining)
        except queue.Empty:
            break
        kind = message.get("kind")
        if kind == "result":
            results.append(message)
        elif kind == "done":
            done.append(message)
        elif kind == "worker_error":
            errors.append(message)
        elif kind == "pool":
            continue
    return results, done, errors


def _collect_tcp(
    result: Duplex,
    *,
    expected_results: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    deadline = time.perf_counter() + timeout_s
    while len(results) < expected_results or not done:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            message = result.recv(timeout=remaining)
        except (queue.Empty, EOFError, TimeoutError):
            break
        if not isinstance(message, dict):
            continue
        kind = message.get("kind")
        if kind == "result":
            results.append(message)
        elif kind == "done":
            done.append(message)
        elif kind == "worker_error":
            errors.append(message)
    return results, done, errors


def _wait_kind(result: Duplex, kind: str, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {kind}")
        message = result.recv(timeout=remaining)
        if isinstance(message, dict) and message.get("kind") == "worker_error":
            raise RuntimeError(message["error"])
        if isinstance(message, dict) and message.get("kind") == kind:
            return message


def _collect_vlm_addrs(address_q: Any, *, expected: int, timeout_s: float) -> list[str]:
    found: list[str] = []
    deadline = time.perf_counter() + timeout_s
    while len(found) < expected:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"VLM workers did not publish work addresses ({len(found)}/{expected})")
        found.append(str(address_q.get(timeout=remaining)))
    return found


def _payload(
    cfg: RunConfig,
    started: float,
    schedule: list[float],
    results: list[dict[str, Any]],
    done: list[dict[str, Any]],
) -> dict[str, Any]:
    del started
    ok = [item for item in results if item.get("status") == "pass"]
    e2e = [float(item["timing"]["e2e_ms"]) for item in ok if "e2e_ms" in item.get("timing", {})]
    good = [value for value in e2e if value <= cfg.slo_ms]
    span_s = (max(schedule) - min(schedule)) if len(schedule) > 1 else 0.0
    segments = (
        "vlm_compute_ms",
        "vlm_export_ms",
        "ae_admit_wait_ms",
        "ae_queue_ms",
        "ae_prefix_ms",
        "ae_pack_ms",
        "ae_forward_ms",
        "ae_unpack_ms",
        "ae_tick_ms",
        "vlm_effective_batch",
        "ae_effective_batch",
    )
    return {
        "backend": cfg.backend,
        "mode": cfg.mode,
        "transport": cfg.resolved_transport(),
        "launch": cfg.launch,
        "vlm_devices": list(cfg.vlm_devices),
        "ae_device": cfg.ae_device,
        "baseline_devices": list(cfg.baseline_devices or cfg.vlm_devices),
        "num_requests": cfg.num_requests,
        "rate": cfg.rate,
        "num_steps": cfg.num_steps,
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "done_workers": len(done),
        "e2e": _summary(e2e),
        "throughput_rps": (len(ok) / span_s) if span_s > 0 else None,
        "goodput_rps": (len(good) / span_s) if span_s > 0 else None,
        "slo_ms": cfg.slo_ms,
        "segments": {key: _summary(_timing_values(ok, key)) for key in segments},
        "results": results,
    }


def _timing_values(results: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for item in results:
        value = item.get("timing", {}).get(key)
        if value is not None:
            values.append(float(value))
    return values


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)

    def _pct(prob: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * prob
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (1.0 - (rank - lo)) + ordered[hi] * (rank - lo)

    return {
        "avg_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(_pct(0.50)),
        "p95_ms": float(_pct(0.95)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _validate_gpus(cfg: RunConfig) -> None:
    if cfg.backend == "fake":
        return
    devices: list[int] = []
    if cfg.mode == "baseline" or cfg.role == "vlm":
        devices.extend(cfg.baseline_devices or cfg.vlm_devices)
    if cfg.mode == "ours" or cfg.role == "ae":
        devices.extend([*cfg.vlm_devices, cfg.ae_device])
    physical = sorted({device for device in devices if device >= 0})
    if not physical:
        return
    count = _gpu_count()
    missing = [device for device in physical if device >= count]
    if missing:
        raise RuntimeError(f"GPU ids {missing} are outside visible device count {count}")


def _gpu_count() -> int:
    import subprocess

    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi is required to validate physical GPU ids") from exc
    return len([line for line in output.splitlines() if line.startswith("GPU ")])


def _accept_one(listener: Any, *, timeout_s: float) -> Any:
    box: dict[str, Any] = {}

    def _run() -> None:
        box["conn"] = listener.accept()

    thread = threading.Thread(target=_run, name="va-accept", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if "conn" not in box:
        raise TimeoutError("timed out accepting a worker connection")
    return box["conn"]


def _thread(target: Any, *args: Any, **kwargs: Any) -> threading.Thread:
    return threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)


def _join_threads(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=5)


def _join_processes(procs: list[Any]) -> None:
    for proc in procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


def _split_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    if not host or not port:
        raise ValueError(f"expected host:port, got {addr!r}")
    return host, int(port)
