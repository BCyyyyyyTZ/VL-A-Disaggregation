"""Fair lane-credit prefetch.

A bulk credit message must not be swallowed by one VLM. Keep at most
`max_batch_size` ids locally and push the rest back onto the queue.
"""

from __future__ import annotations

import time
from typing import Any


def take_credit(local: list[int], credit_q: Any, *, max_batch_size: int, stats: list[float]) -> int:
    if local:
        return int(local.pop(0))
    started = time.monotonic_ns()
    got = credit_q.get()
    waited_ms = (time.monotonic_ns() - started) / 1e6
    if waited_ms >= 0.05:
        stats[0] += waited_ms
        stats[1] += 1.0
    if isinstance(got, int):
        return int(got)
    ids = [int(item) for item in got]
    if not ids:
        raise RuntimeError("empty credit message")
    lane = ids[0]
    prefetch = max(0, int(max_batch_size) - 1)
    local.extend(ids[1 : 1 + prefetch])
    leftover = ids[1 + prefetch :]
    if leftover:
        credit_q.put(leftover)
    return lane
