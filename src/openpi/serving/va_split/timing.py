from __future__ import annotations

import time
from typing import Any

import torch


def synchronize_cuda_if_needed(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def timed_queue_get(queue_obj, *, block: bool = True, timeout: float | None = None) -> tuple[Any, int, int]:
    """Return ``(message, get_start_ns, get_end_ns)`` for queue-wait vs transfer splits."""
    get_start_ns = time.monotonic_ns()
    if block:
        message = queue_obj.get() if timeout is None else queue_obj.get(timeout=timeout)
    else:
        message = queue_obj.get_nowait()
    get_end_ns = time.monotonic_ns()
    return message, get_start_ns, get_end_ns


def queue_wait_and_transfer_ms(
    *,
    enqueue_ns: float | int | None,
    get_start_ns: float | int | None,
    get_end_ns: float | int | None,
) -> tuple[float, float]:
    """Split queue residency into wait-before-get and get()/IPC transfer."""
    if enqueue_ns is None or get_start_ns is None or get_end_ns is None:
        return 0.0, 0.0
    queue_wait_ms = max(0.0, (float(get_start_ns) - float(enqueue_ns)) / 1_000_000)
    transfer_ms = max(0.0, (float(get_end_ns) - float(get_start_ns)) / 1_000_000)
    return queue_wait_ms, transfer_ms
