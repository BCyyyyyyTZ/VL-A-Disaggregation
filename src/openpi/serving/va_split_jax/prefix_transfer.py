from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class PrefixTransferTicket:
    kind: str
    payload: Any = None


def make_composite_prefix_ticket(tickets: Iterable[PrefixTransferTicket | Any]) -> PrefixTransferTicket:
    return PrefixTransferTicket(kind="composite", payload=tuple(ticket for ticket in tickets if ticket is not None))


def wait_for_prefix_ticket(ticket: PrefixTransferTicket | None) -> float:
    if ticket is None or ticket.kind == "synchronous":
        return 0.0
    start_ns = time.monotonic_ns()
    _wait_payload(ticket.payload)
    return (time.monotonic_ns() - start_ns) / 1_000_000


def _wait_payload(payload: Any) -> None:
    if payload is None:
        return
    if isinstance(payload, PrefixTransferTicket):
        wait_for_prefix_ticket(payload)
        return
    if isinstance(payload, dict):
        for item in payload.values():
            _wait_payload(item)
        return
    if isinstance(payload, (tuple, list)):
        for item in payload:
            _wait_payload(item)
        return
    wait = getattr(payload, "wait", None)
    if callable(wait):
        wait()
        return
    synchronize = getattr(payload, "synchronize", None)
    if callable(synchronize):
        synchronize()
