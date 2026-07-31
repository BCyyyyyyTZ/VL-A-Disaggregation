from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpi.models.jax_split_types import JaxPrefixSlabHandleTree
from openpi.models.jax_split_types import JaxPrefixSlotHandle


@dataclass(frozen=True, slots=True)
class JaxRequestEnvelope:
    request_id: str
    observation: dict[str, Any]
    sample_kwargs: dict[str, Any]
    enqueue_ns: int
    dequeue_ns: int | None = None
    dequeue_start_ns: int | None = None


@dataclass(frozen=True, slots=True)
class JaxBatchRequestEnvelope:
    batch_id: str
    request_ids: tuple[str, ...]
    observation: dict[str, Any]
    sample_kwargs: dict[str, Any]
    enqueue_ns: int
    dequeue_ns: int | None = None
    dequeue_start_ns: int | None = None


@dataclass(frozen=True, slots=True)
class JaxPrefixReady:
    request_id: str
    slot_handle: JaxPrefixSlotHandle
    num_steps: int
    sample_kwargs: dict[str, Any]
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxActionResult:
    request_id: str
    actions: Any
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxReleaseFeature:
    request_id: str
    slot_id: int


@dataclass(frozen=True, slots=True)
class JaxSlotMoved:
    request_id: str
    old_slot_id: int
    new_slot_id: int


@dataclass(frozen=True, slots=True)
class JaxPrefixSlabReady:
    slab: JaxPrefixSlabHandleTree
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxDenoiseBatchSlots:
    request_ids: tuple[str, ...]
    slot_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JaxWorkerError:
    request_id: str | None
    error: str
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class JaxShutdown:
    pass
