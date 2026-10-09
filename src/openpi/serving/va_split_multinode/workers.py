"""VLM / AE / baseline worker loops shared by local processes and TCP roles."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
import traceback
from typing import Any

import numpy as np

from openpi.serving.va_split_multinode.backends import build_backend
from openpi.serving.va_split_multinode.credit import take_credit
from openpi.serving.va_split_multinode.fcfs import BatchScheduler
from openpi.serving.va_split_multinode.host_lane_pool import HostLanePool
from openpi.serving.va_split_multinode.host_lane_pool import stack_rows
from openpi.serving.va_split_multinode.messages import PrefixMsg
from openpi.serving.va_split_multinode.messages import RunConfig
from openpi.serving.va_split_multinode.messages import WorkItem


@dataclass(slots=True)
class _Job:
    item: WorkItem
    future: Any


@dataclass(slots=True)
class ActiveRequest:
    request_id: str
    lane_id: int
    num_steps: int
    step_idx: int
    scheduled_at_s: float
    enqueue_ns: int
    vlm_done_ns: int
    vlm_batch_size: int
    vlm_batch_start_ns: int
    vlm_compute_done_ns: int
    vlm_export_done_ns: int
    ae_first_seen_ns: int
    payload: dict[str, np.ndarray] | None = None


class _Publisher:
    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="va-split-d2h", daemon=True)
        self._thread.start()

    def submit(self, fn: Any) -> None:
        if self._error is not None:
            raise self._error
        self._queue.put(fn)

    def close(self, *, timeout_s: float) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                return
            try:
                fn()
            except BaseException as exc:
                self._error = exc
                return


def _uses_host_pool(cfg: RunConfig) -> bool:
    return cfg.resolved_transport() in {"host_shm", "local"}


def _bind_torch_device(device: int) -> None:
    if device < 0:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(device)


def _fit_noise(noise: np.ndarray, *, horizon: int, action_dim: int) -> np.ndarray:
    array = np.asarray(noise, dtype=np.float32)
    if array.shape != (horizon, action_dim):
        raise ValueError(f"noise shape {array.shape} != {(horizon, action_dim)}")
    return array


def run_vlm_worker(
    cfg: RunConfig,
    *,
    device: int,
    worker_index: int,
    num_vlm: int,
    work_q: Any,
    prefix_q: Any,
    result_q: Any,
    pool_q: Any,
    credit_q: Any,
) -> None:
    backend = build_backend(cfg, "vlm", device)
    backend.warmup_vlm()
    pool: HostLanePool | None = None
    if _uses_host_pool(cfg):
        pool = _publish_or_attach_pool(
            cfg,
            backend=backend,
            worker_index=worker_index,
            num_vlm=num_vlm,
            pool_q=pool_q,
            result_q=result_q,
            credit_q=credit_q,
        )
    elif worker_index == 0 and cfg.resolved_transport() != "tcp":
        credit_q.put(list(range(cfg.max_prefix_slots)))
    local_credits: list[int] = []
    credit_stats = [0.0, 0.0]
    _emit_prefix_example(backend, worker_index)
    publisher = _Publisher() if cfg.overlap_d2h else None
    pending: list[Any] = []

    def _on_batch(jobs: list[_Job]) -> None:
        try:
            _bind_torch_device(device)
            _run_vlm_batch(jobs)
        except BaseException as exc:
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
            raise

    def _run_vlm_batch(jobs: list[_Job]) -> None:
        batch_start = time.monotonic_ns()
        export = backend.forward_vlm([job.item.observation for job in jobs])
        compute_done = time.monotonic_ns()
        lane_ids = [
            take_credit(local_credits, credit_q, max_batch_size=cfg.max_batch_size, stats=credit_stats)
            for _ in jobs
        ]

        def _publish() -> None:
            try:
                rows = export.materialize()
                export_done = time.monotonic_ns()
                prepared: list[dict[str, np.ndarray]] = []
                for row, job in zip(rows, jobs, strict=True):
                    payload = dict(row)
                    payload["noise"] = _fit_noise(
                        job.item.noise, horizon=backend.horizon, action_dim=backend.action_dim
                    )
                    prepared.append(payload)
                if pool is not None:
                    pool.write_rows(lane_ids, prepared)
                for index, job in enumerate(jobs):
                    prefix_q.put(
                        PrefixMsg(
                            request_id=job.item.request_id,
                            lane_id=lane_ids[index],
                            num_steps=job.item.num_steps,
                            scheduled_at_s=job.item.scheduled_at_s,
                            enqueue_ns=job.item.enqueue_ns,
                            vlm_done_ns=compute_done,
                            vlm_batch_size=len(jobs),
                            vlm_batch_start_ns=batch_start,
                            vlm_compute_done_ns=compute_done,
                            vlm_export_done_ns=export_done,
                            payload=None if pool is not None else prepared[index],
                        )
                    )
                    if not job.future.done():
                        job.future.set_result(None)
            except BaseException as exc:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)
                raise

        try:
            if publisher is not None:
                publisher.submit(_publish)
            else:
                _publish()
        except BaseException as exc:
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
            raise

    scheduler = BatchScheduler(max_batch_size=cfg.max_batch_size, max_wait_ms=cfg.max_wait_ms, on_batch=_on_batch)
    scheduler.start()
    result_q.put({"kind": "ready", "role": "vlm", "worker": worker_index, "device": device})
    try:
        while True:
            item = work_q.get()
            if item is None:
                break
            job = _Job(item=item, future=_Future())
            scheduler.submit(job)
            pending.append(job.future)
        for future in pending:
            future.result(timeout=cfg.timeout_s)
    except Exception as exc:
        result_q.put(
            {
                "kind": "worker_error",
                "role": "vlm",
                "worker": worker_index,
                "error": f"{exc}\n{traceback.format_exc()}",
            }
        )
        raise
    finally:
        scheduler.stop(timeout_s=cfg.timeout_s)
        if publisher is not None:
            publisher.close(timeout_s=cfg.timeout_s)
        prefix_q.put(None)
        result_q.put(
            {
                "kind": "done",
                "role": "vlm",
                "worker": worker_index,
                "credit_wait_ms_sum": credit_stats[0],
                "credit_block_count": credit_stats[1],
            }
        )


def run_ae_worker(
    cfg: RunConfig,
    *,
    device: int,
    num_vlm: int,
    prefix_q: Any,
    result_q: Any,
    pool_q: Any,
    credit_q: Any,
    example_wait: Any = None,
) -> None:
    backend = build_backend(cfg, "ae", device)
    pool: HostLanePool | None = None
    if _uses_host_pool(cfg):
        token = pool_q.get(timeout=cfg.startup_timeout_s)
        pool = token if isinstance(token, HostLanePool) else HostLanePool.attach(token)
        backend._example = pool.row_template()
    elif example_wait is not None:
        backend._example = example_wait()
    backend.warmup_ae()
    result_q.put({"kind": "ready", "role": "ae", "device": device})
    active: list[ActiveRequest] = []
    closed = 0
    cached_key: tuple[Any, ...] | None = None
    try:
        while closed < num_vlm or active:
            admit_started = time.monotonic_ns()
            while len(active) < cfg.max_prefix_slots and closed < num_vlm:
                try:
                    message = prefix_q.get_nowait() if active else prefix_q.get(timeout=0.05)
                except queue.Empty:
                    break
                if message is None:
                    closed += 1
                    continue
                seen = time.monotonic_ns()
                row = None
                if pool is not None:
                    pool.wait_ready(int(message.lane_id), timeout_s=cfg.timeout_s)
                    row = pool.read_row(int(message.lane_id))
                else:
                    row = message.payload
                if row is None or "noise" not in row:
                    raise RuntimeError(f"prefix for {message.request_id} is missing noise")
                backend.admit_noise(int(message.lane_id), row["noise"])
                active.append(
                    ActiveRequest(
                        request_id=message.request_id,
                        lane_id=int(message.lane_id),
                        num_steps=int(message.num_steps),
                        step_idx=0,
                        scheduled_at_s=float(message.scheduled_at_s),
                        enqueue_ns=int(message.enqueue_ns),
                        vlm_done_ns=int(message.vlm_done_ns),
                        vlm_batch_size=int(message.vlm_batch_size),
                        vlm_batch_start_ns=int(message.vlm_batch_start_ns),
                        vlm_compute_done_ns=int(message.vlm_compute_done_ns),
                        vlm_export_done_ns=int(message.vlm_export_done_ns),
                        ae_first_seen_ns=seen,
                        payload=None if pool is not None else row,
                    )
                )
            if not active:
                continue
            batch = active[: cfg.ae_max_batch_size]
            rest = active[cfg.ae_max_batch_size :]
            prefix_started = time.monotonic_ns()
            cache_key: tuple[Any, ...]
            if pool is not None:
                cache_key = tuple(item.lane_id for item in batch)
                if cache_key != cached_key:
                    rows = [pool.read_row(item.lane_id) for item in batch]
                    backend.import_prefix(stack_rows(rows))
                    cached_key = cache_key
            else:
                cache_key = tuple(item.request_id for item in batch)
                if cache_key != cached_key:
                    rows = []
                    for item in batch:
                        assert item.payload is not None
                        rows.append(item.payload)
                    backend.import_prefix(stack_rows(rows))
                    cached_key = cache_key
            prefix_ms = (time.monotonic_ns() - prefix_started) / 1e6
            step_timing = backend.step_batch(batch)
            finished: list[ActiveRequest] = []
            still: list[ActiveRequest] = []
            freed: list[int] = []
            for item in batch:
                if item.step_idx >= item.num_steps:
                    finished.append(item)
                    freed.append(item.lane_id)
                    if pool is not None:
                        pool.clear_ready(item.lane_id)
                else:
                    still.append(item)
            done_ns = time.monotonic_ns()
            tick_ms = (done_ns - prefix_started) / 1e6
            for item in finished:
                actions = backend.copy_noise(item.lane_id)
                export_ns = item.vlm_export_done_ns or item.vlm_compute_done_ns or item.vlm_done_ns
                compute_ns = item.vlm_compute_done_ns or item.vlm_done_ns
                result_q.put(
                    {
                        "kind": "result",
                        "request_id": item.request_id,
                        "status": "pass",
                        "actions": actions,
                        "timing": _ae_timing(
                            item,
                            done_ns=done_ns,
                            admit_started=admit_started,
                            prefix_ms=prefix_ms,
                            step_timing=step_timing,
                            tick_ms=tick_ms,
                            batch_size=len(batch),
                            export_ns=export_ns,
                            compute_ns=compute_ns,
                        ),
                    }
                )
            if freed:
                credit_q.put([int(lane) for lane in freed])
            active = still + rest
    except Exception as exc:
        result_q.put({"kind": "worker_error", "role": "ae", "error": f"{exc}\n{traceback.format_exc()}"})
        raise
    finally:
        if pool is not None and not isinstance(pool, HostLanePool):
            pass
        if pool is not None:
            pool.close(unlink=False)
        result_q.put({"kind": "done", "role": "ae"})


def run_baseline_worker(
    cfg: RunConfig,
    *,
    device: int,
    worker_index: int,
    work_q: Any,
    result_q: Any,
) -> None:
    backend = build_backend(cfg, "full", device)
    if cfg.backend != "fake":
        backend.warmup_vlm()
    pending: list[Any] = []

    def _on_batch(jobs: list[_Job]) -> None:
        _bind_torch_device(device)
        started = time.monotonic_ns()
        actions = backend.sample_batch(
            [job.item.observation for job in jobs],
            [np.asarray(job.item.noise, dtype=np.float32) for job in jobs],
            jobs[0].item.num_steps,
        )
        done = time.monotonic_ns()
        for job, action in zip(jobs, actions, strict=True):
            result_q.put(
                {
                    "kind": "result",
                    "request_id": job.item.request_id,
                    "status": "pass",
                    "actions": action,
                    "timing": {
                        "e2e_ms": (done - job.item.enqueue_ns) / 1e6,
                        "worker_compute_ms": (done - started) / 1e6,
                        "effective_batch": float(len(jobs)),
                        "vlm_effective_batch": float(len(jobs)),
                    },
                }
            )
            if not job.future.done():
                job.future.set_result(None)

    scheduler = BatchScheduler(max_batch_size=cfg.max_batch_size, max_wait_ms=cfg.max_wait_ms, on_batch=_on_batch)
    scheduler.start()
    result_q.put({"kind": "ready", "role": "baseline", "worker": worker_index, "device": device})
    try:
        while True:
            item = work_q.get()
            if item is None:
                break
            job = _Job(item=item, future=_Future())
            scheduler.submit(job)
            pending.append(job.future)
        for future in pending:
            future.result(timeout=cfg.timeout_s)
    except Exception as exc:
        result_q.put(
            {
                "kind": "worker_error",
                "role": "baseline",
                "worker": worker_index,
                "error": f"{exc}\n{traceback.format_exc()}",
            }
        )
        raise
    finally:
        scheduler.stop(timeout_s=cfg.timeout_s)
        result_q.put({"kind": "done", "role": "baseline", "worker": worker_index})


def run_worker(
    role: str,
    cfg: RunConfig,
    *,
    device: int,
    worker_index: int,
    num_vlm: int,
    work_q: Any,
    prefix_q: Any,
    result_q: Any,
    pool_q: Any,
    credit_q: Any,
    example_wait: Any = None,
) -> None:
    if role == "vlm":
        run_vlm_worker(
            cfg,
            device=device,
            worker_index=worker_index,
            num_vlm=num_vlm,
            work_q=work_q,
            prefix_q=prefix_q,
            result_q=result_q,
            pool_q=pool_q,
            credit_q=credit_q,
        )
        return
    if role == "ae":
        run_ae_worker(
            cfg,
            device=device,
            num_vlm=num_vlm,
            prefix_q=prefix_q,
            result_q=result_q,
            pool_q=pool_q,
            credit_q=credit_q,
            example_wait=example_wait,
        )
        return
    if role == "baseline":
        run_baseline_worker(cfg, device=device, worker_index=worker_index, work_q=work_q, result_q=result_q)
        return
    raise ValueError(f"unsupported worker role {role}")


def _publish_or_attach_pool(
    cfg: RunConfig,
    *,
    backend: Any,
    worker_index: int,
    num_vlm: int,
    pool_q: Any,
    result_q: Any,
    credit_q: Any,
) -> HostLanePool:
    shared = cfg.resolved_transport() == "host_shm"
    if worker_index == 0:
        pool = HostLanePool.create(
            backend.example_row(),
            max_lanes=cfg.max_prefix_slots,
            shared=shared,
            try_host_register=shared,
        )
        token: Any = pool.export_ready() if shared else pool
        if shared:
            result_q.put({"kind": "pool", "ready": token})
        for _ in range(num_vlm):
            pool_q.put(token)
        credit_q.put(list(range(cfg.max_prefix_slots)))
        return pool
    token = pool_q.get(timeout=cfg.startup_timeout_s)
    if isinstance(token, HostLanePool):
        return token
    return HostLanePool.attach(token)


def _ae_timing(
    item: ActiveRequest,
    *,
    done_ns: int,
    admit_started: int,
    prefix_ms: float,
    step_timing: dict[str, float],
    tick_ms: float,
    batch_size: int,
    export_ns: int,
    compute_ns: int,
) -> dict[str, float]:
    start_ns = item.vlm_batch_start_ns
    return {
        "e2e_ms": (done_ns - item.enqueue_ns) / 1e6,
        "vlm_compute_ms": ((compute_ns - start_ns) / 1e6) if start_ns else 0.0,
        "vlm_export_ms": ((export_ns - compute_ns) / 1e6) if export_ns and compute_ns else 0.0,
        "ae_admit_wait_ms": ((item.ae_first_seen_ns - export_ns) / 1e6) if export_ns else 0.0,
        "ae_queue_ms": max(0.0, (admit_started - item.ae_first_seen_ns) / 1e6),
        "ae_prefix_ms": prefix_ms,
        "ae_pack_ms": float(step_timing.get("ae_pack_ms", 0.0)),
        "ae_forward_ms": float(step_timing.get("ae_forward_ms", 0.0)),
        "ae_unpack_ms": float(step_timing.get("ae_unpack_ms", 0.0)),
        "ae_tick_ms": tick_ms,
        "vlm_effective_batch": float(item.vlm_batch_size),
        "ae_effective_batch": float(batch_size),
        "effective_batch": float(batch_size),
    }


def _emit_prefix_example(backend: Any, worker_index: int) -> None:
    from openpi.serving.va_split_multinode.tcp import consume_ready_hook

    hook = consume_ready_hook()
    if hook is not None and worker_index == 0:
        hook(backend)


class _Future:
    """Tiny future that does not depend on concurrent.futures for the scheduler thread."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._result: Any = None
        self._error: BaseException | None = None

    def set_result(self, value: Any) -> None:
        self._result = value
        self._done.set()

    def set_exception(self, exc: BaseException) -> None:
        self._error = exc
        self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: float | None = None) -> Any:
        if not self._done.wait(timeout):
            raise TimeoutError("worker future timed out")
        if self._error is not None:
            raise self._error
        return self._result
