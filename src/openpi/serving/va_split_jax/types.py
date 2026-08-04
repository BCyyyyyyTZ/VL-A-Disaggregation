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
    """VLM finished writing prefix KV into AE-owned slab lane ``slot_handle.slot_id``."""

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
    """AE finished a request; ``slot_id`` is the recycled physical lane credit for VLM."""

    request_id: str
    slot_id: int


@dataclass(frozen=True, slots=True)
class JaxLaneCredits:
    """Physical lane ids VLM may write before handing ownership to AE via PrefixReady."""

    lane_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JaxPrefixSlabReady:
    """AE-owned prefix slab export (sent AE → VLM on the release/control path)."""

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


@dataclass(frozen=True, slots=True)
class JaxCompileWarmupDone:
    role: str
    jax_warmup_batches: float
