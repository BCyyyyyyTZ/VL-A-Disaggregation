from __future__ import annotations

import time
from typing import Any

import torch


def synchronize_cuda_if_needed(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


class CudaEventTimer:
    """Per-op CUDA timing without draining the whole device between stages.

    Records start/end events on the current stream. ``elapsed_ms()`` only waits
    for the end event (happens-before for this op), matching the realtime-vla
    approach of avoiding mid-pipeline ``cuda.synchronize()`` host drains.
    """

    def __init__(self, device: str | torch.device):
        self._device = torch.device(device)
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._wall_start_ns: int | None = None
        self._enabled = self._device.type == "cuda" and torch.cuda.is_available()

    def start(self) -> None:
        if self._enabled:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        self._wall_start_ns = time.monotonic_ns()

    def stop(self) -> None:
        if self._enabled and self._end is not None:
            self._end.record()
        elif self._wall_start_ns is None:
            self._wall_start_ns = time.monotonic_ns()

    def elapsed_ms(self) -> float:
        if self._enabled and self._start is not None and self._end is not None:
            self._end.synchronize()
            return float(self._start.elapsed_time(self._end))
        if self._wall_start_ns is None:
            return 0.0
        return (time.monotonic_ns() - self._wall_start_ns) / 1_000_000


def timed_queue_get(queue_obj, *, block: bool = True, timeout: float | None = None) -> tuple[Any, int, int]:
    """Return ``(message, get_start_ns, get_end_ns)`` for queue-wait vs transfer splits.

    Important: when ``block=True``, a blocking ``get()`` would attribute idle wait
    to transfer. Prefer non-blocking polls with a short sleep so wait stays in
    ``queue_wait`` (enqueue → get_start) and transfer is only deserialize/IPC open.
    """
    if block and timeout is None:
        return _timed_blocking_queue_get_without_idle_transfer(queue_obj)
    get_start_ns = time.monotonic_ns()
    if block:
        message = queue_obj.get(timeout=timeout)
    else:
        message = queue_obj.get_nowait()
    get_end_ns = time.monotonic_ns()
    return message, get_start_ns, get_end_ns


def _timed_blocking_queue_get_without_idle_transfer(queue_obj) -> tuple[Any, int, int]:
    poll_s = 0.0002
    while True:
        get_start_ns = time.monotonic_ns()
        try:
            message = queue_obj.get_nowait()
        except Exception as exc:  # queue.Empty across stdlib / mp queue types
            if type(exc).__name__ != "Empty":
                raise
            time.sleep(poll_s)
            continue
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
