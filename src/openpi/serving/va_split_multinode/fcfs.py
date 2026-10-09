"""FCFS micro-batcher.

Already-queued work is drained before the wait window starts. `on_batch` should
block for the GPU forward and only then kick asynchronous export, so the next
collect overlaps D2H with the following forward.
"""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Any
from typing import Callable


_SHUTDOWN = object()


class BatchScheduler:
    def __init__(self, *, max_batch_size: int, max_wait_ms: float, on_batch: Callable[[list[Any]], None]) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be non-negative")
        self.max_batch_size = int(max_batch_size)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        self._on_batch = on_batch
        self._queue: queue.Queue[Any] = queue.Queue()
        self._backlog: deque[Any] = deque()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, name="va-split-fcfs", daemon=True)
        self._thread.start()

    def submit(self, item: Any) -> None:
        if self._stopped.is_set():
            raise RuntimeError("BatchScheduler is stopped")
        self._queue.put(item)

    def stop(self, *, timeout_s: float = 30.0) -> None:
        self._stopped.set()
        self._queue.put(_SHUTDOWN)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                message = self._next(timeout=0.05)
            except queue.Empty:
                continue
            if message is _SHUTDOWN:
                return
            batch = self._collect(message)
            if batch:
                self._on_batch(batch)

    def _collect(self, first: Any) -> list[Any]:
        batch = [first]
        self._drain_ready(batch)
        if len(batch) >= self.max_batch_size:
            return batch
        deadline = time.perf_counter() + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self._drain_ready(batch)
                break
            try:
                message = self._next(timeout=remaining)
            except queue.Empty:
                break
            if message is _SHUTDOWN:
                self._backlog.appendleft(_SHUTDOWN)
                self._stopped.set()
                break
            batch.append(message)
            self._drain_ready(batch)
        return batch

    def _drain_ready(self, batch: list[Any]) -> None:
        while len(batch) < self.max_batch_size and self._backlog:
            message = self._backlog.popleft()
            if message is _SHUTDOWN:
                self._backlog.appendleft(_SHUTDOWN)
                self._stopped.set()
                return
            batch.append(message)
        while len(batch) < self.max_batch_size:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                return
            if message is _SHUTDOWN:
                self._backlog.appendleft(_SHUTDOWN)
                self._stopped.set()
                return
            batch.append(message)

    def _next(self, *, timeout: float) -> Any:
        if self._backlog:
            return self._backlog.popleft()
        return self._queue.get(timeout=timeout)
