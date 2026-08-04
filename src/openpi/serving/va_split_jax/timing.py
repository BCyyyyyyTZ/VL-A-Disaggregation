from __future__ import annotations

import queue
import time
from typing import Any

import jax


def synchronize_jax_if_needed() -> None:
    # JAX has no global synchronize API; callers should block on concrete arrays at ownership boundaries.
    for device in jax.devices():
        del device


def timed_queue_get(q, *, block: bool = True, timeout: float | None = None) -> tuple[Any, int, int]:
    start_ns = time.monotonic_ns()
    try:
        if block:
            message = q.get() if timeout is None else q.get(timeout=timeout)
        else:
            message = q.get_nowait()
    except queue.Empty:
        raise
    end_ns = time.monotonic_ns()
    return message, start_ns, end_ns


def queue_wait_and_transfer_ms(
    *,
    enqueue_ns: float | int | None,
    get_start_ns: int | None,
    get_end_ns: int | None,
) -> tuple[float, float]:
    """Split IPC queue residency into wait vs transfer.

    Definitions:
    - ``queue_wait``: time the message spent in the queue before the consumer began
      retrieving it (``max(0, get_start - enqueue)``).
    - ``transfer``: time spent in ``Queue.get()`` *after* the message existed
      (``get_end - max(get_start, enqueue)``).

    This matters for blocking gets that start *before* the producer enqueues: the
    idle blocked time must not be billed as transfer.
    """
    if enqueue_ns is None or get_start_ns is None or get_end_ns is None:
        return 0.0, 0.0
    enqueue = float(enqueue_ns)
    get_start = float(get_start_ns)
    get_end = float(get_end_ns)
    effective_get_start = max(get_start, enqueue)
    queue_wait_ms = max(0.0, (effective_get_start - enqueue) / 1_000_000)
    transfer_ms = max(0.0, (get_end - effective_get_start) / 1_000_000)
    return queue_wait_ms, transfer_ms
