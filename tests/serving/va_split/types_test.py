from __future__ import annotations

import multiprocessing as mp

import torch

from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import BatchRequestEnvelope
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError


def _round_trip(message: object) -> object:
    queue: mp.Queue[object] = mp.Queue()
    queue.put(message)
    return queue.get(timeout=1)


def test_request_envelope_round_trips_through_queue():
    message = RequestEnvelope(
        request_id="req-1",
        observation={"state": [1.0, 2.0]},
        sample_kwargs={"num_steps": 4},
        enqueue_ns=123,
    )

    received = _round_trip(message)

    assert received == message


def test_batch_request_envelope_round_trips_through_queue():
    message = BatchRequestEnvelope(
        batch_id="batch-1",
        request_ids=("req-1", "req-2"),
        observation={"state": [1.0, 2.0]},
        sample_kwargs={"num_steps": 4},
        enqueue_ns=123,
    )

    received = _round_trip(message)

    assert received == message


def test_prefix_ready_round_trips_feature_payload_through_queue():
    feature = PrefixFeature(
        past_key_values=("kv-handle",),
        prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
        state=torch.zeros(1, 2),
    )
    message = PrefixReady(
        request_id="req-2",
        feature=feature,
        num_steps=4,
        sample_kwargs={"noise": None},
        slot_id=-1,
    )

    received = _round_trip(message)

    assert isinstance(received, PrefixReady)
    assert received.request_id == "req-2"
    assert received.num_steps == 4
    assert received.sample_kwargs == {"noise": None}
    assert received.slot_id == -1
    assert received.feature.past_key_values == ("kv-handle",)
    torch.testing.assert_close(received.feature.prefix_pad_masks, feature.prefix_pad_masks)
    torch.testing.assert_close(received.feature.state, feature.state)


def test_action_result_round_trips_through_queue():
    actions = torch.ones(1, 4, 8)
    message = ActionResult(request_id="req-3", actions=actions, timing={"ae_ms": 1.5})

    received = _round_trip(message)

    assert isinstance(received, ActionResult)
    assert received.request_id == "req-3"
    assert received.timing == {"ae_ms": 1.5}
    torch.testing.assert_close(received.actions, actions)


def test_release_error_and_shutdown_round_trip_through_queue():
    assert _round_trip(ReleaseFeature(request_id="req-4", slot_id=-1)) == ReleaseFeature(request_id="req-4", slot_id=-1)
    assert _round_trip(WorkerError(request_id="req-5", error="boom", traceback="trace")) == WorkerError(
        request_id="req-5", error="boom", traceback="trace"
    )
    assert _round_trip(Shutdown()) == Shutdown()
