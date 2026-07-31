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
    if enqueue_ns is None or get_start_ns is None or get_end_ns is None:
        return 0.0, 0.0
    return (
        max(0.0, (float(get_start_ns) - float(enqueue_ns)) / 1_000_000),
        max(0.0, (float(get_end_ns) - float(get_start_ns)) / 1_000_000),
    )
