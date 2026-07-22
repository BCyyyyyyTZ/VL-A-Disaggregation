from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpi.models_pytorch.pi0_split_types import PrefixFeature


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    request_id: str
    observation: dict[str, Any]
    sample_kwargs: dict[str, Any]
    enqueue_ns: int


@dataclass(frozen=True, slots=True)
class PrefixReady:
    request_id: str
    feature: PrefixFeature
    num_steps: int
    sample_kwargs: dict[str, Any]
    slot_id: int = -1


@dataclass(frozen=True, slots=True)
class ActionResult:
    request_id: str
    actions: Any
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class ReleaseFeature:
    request_id: str
    slot_id: int


@dataclass(frozen=True, slots=True)
class WorkerError:
    request_id: str | None
    error: str
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass
