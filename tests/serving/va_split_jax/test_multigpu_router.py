from __future__ import annotations

from openpi.serving.va_split_jax.multigpu_router import JaxVlmRouteState
from openpi.serving.va_split_jax.multigpu_router import LeastBacklogVlmRouter


def test_router_chooses_worker_with_lowest_backlog_then_inflight():
    router = LeastBacklogVlmRouter(("vlm-0", "vlm-1", "vlm-2"))
    router.update("vlm-0", JaxVlmRouteState(inflight=8, queued=2, available_credits=16))
    router.update("vlm-1", JaxVlmRouteState(inflight=4, queued=0, available_credits=16))
    router.update("vlm-2", JaxVlmRouteState(inflight=4, queued=1, available_credits=16))

    assert router.choose_worker().worker_id == "vlm-1"


def test_router_skips_workers_without_credit():
    router = LeastBacklogVlmRouter(("vlm-0", "vlm-1"))
    router.update("vlm-0", JaxVlmRouteState(inflight=0, queued=0, available_credits=0))
    router.update("vlm-1", JaxVlmRouteState(inflight=2, queued=3, available_credits=1))

    assert router.choose_worker().worker_id == "vlm-1"
