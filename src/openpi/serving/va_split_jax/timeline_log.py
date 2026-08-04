"""Optional cross-process timeline events for VLM/AE overlap analysis.

Enable with:
  VA_SPLIT_TIMELINE_PATH=/path/to/timeline.ndjson
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_lock = threading.Lock()
_path: str | None = None
_proc: str | None = None
_enabled: bool | None = None


def configure(process: str, path: str | None = None) -> None:
    global _path, _proc, _enabled
    _proc = process
    _path = path if path is not None else os.environ.get("VA_SPLIT_TIMELINE_PATH")
    _enabled = bool(_path)


def enabled() -> bool:
    global _enabled, _path
    if _enabled is None:
        _path = os.environ.get("VA_SPLIT_TIMELINE_PATH")
        _enabled = bool(_path)
    return bool(_enabled)


def emit(event: str, **fields: Any) -> None:
    if not enabled():
        return
    assert _path is not None
    record = {
        "t_ns": time.monotonic_ns(),
        "proc": _proc or os.environ.get("VA_SPLIT_TIMELINE_PROC", "?"),
        "event": event,
        **{k: v for k, v in fields.items() if v is not None},
    }
    line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
    with _lock:
        with open(_path, "a", encoding="utf-8") as fh:
            fh.write(line)
