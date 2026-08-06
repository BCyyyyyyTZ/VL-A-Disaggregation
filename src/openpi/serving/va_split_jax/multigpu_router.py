from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JaxVlmRouteState:
    inflight: int = 0
    queued: int = 0
    available_credits: int = 0

    @property
    def backlog(self) -> int:
        return int(self.inflight) + int(self.queued)


@dataclass(frozen=True, slots=True)
class JaxVlmRouteDecision:
    worker_id: str
    state: JaxVlmRouteState


class LeastBacklogVlmRouter:
    def __init__(self, worker_ids: tuple[str, ...]):
        if not worker_ids:
            raise ValueError("worker_ids must be non-empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker_ids must be unique")
        self._worker_ids = tuple(worker_ids)
        self._states = {worker_id: JaxVlmRouteState() for worker_id in self._worker_ids}

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return self._worker_ids

    def update(self, worker_id: str, state: JaxVlmRouteState) -> None:
        if worker_id not in self._states:
            raise KeyError(f"unknown VLM worker {worker_id!r}")
        self._states[worker_id] = state

    def mark_enqueued(self, worker_id: str, *, count: int = 1) -> None:
        state = self._states[worker_id]
        self._states[worker_id] = JaxVlmRouteState(
            inflight=state.inflight,
            queued=state.queued + count,
            available_credits=max(0, state.available_credits - count),
        )

    def mark_dispatched(self, worker_id: str, *, count: int = 1) -> None:
        state = self._states[worker_id]
        self._states[worker_id] = JaxVlmRouteState(
            inflight=state.inflight + count,
            queued=max(0, state.queued - count),
            available_credits=state.available_credits,
        )

    def mark_released(self, worker_id: str, *, count: int = 1) -> None:
        state = self._states[worker_id]
        self._states[worker_id] = JaxVlmRouteState(
            inflight=max(0, state.inflight - count),
            queued=state.queued,
            available_credits=state.available_credits + count,
        )

    def choose_worker(self) -> JaxVlmRouteDecision:
        candidates = [
            (state.queued, state.inflight, -state.available_credits, idx, worker_id, state)
            for idx, worker_id in enumerate(self._worker_ids)
            for state in (self._states[worker_id],)
            if state.available_credits > 0
        ]
        if not candidates:
            candidates = [
                (state.queued, state.inflight, -state.available_credits, idx, worker_id, state)
                for idx, worker_id in enumerate(self._worker_ids)
                for state in (self._states[worker_id],)
            ]
        _, _, _, _, worker_id, state = min(candidates)
        return JaxVlmRouteDecision(worker_id=worker_id, state=state)
