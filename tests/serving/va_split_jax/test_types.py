from __future__ import annotations

import multiprocessing as mp
import queue
import time

import pytest

from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.timing import queue_wait_and_transfer_ms
from openpi.serving.va_split_jax.timing import timed_queue_get
from openpi.serving.va_split_jax.types import JaxDenoiseBatchSlots
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxSlotMoved


def test_jax_prefix_ready_round_trips_metadata_only():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    message = JaxPrefixReady(
        request_id="req-1",
        slot_handle=JaxPrefixSlotHandle(
            slot_id=3,
            batch_rows=1,
            prefix_shape_tree=("shape",),
            prefix_dtype_tree=("dtype",),
        ),
        num_steps=10,
        sample_kwargs={"num_steps": 10},
    )
    q.put(message)
    received = q.get(timeout=5)
    assert received.request_id == "req-1"
    assert received.slot_handle.slot_id == 3


def test_jax_slot_moved_round_trips_after_vlm_compaction():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put(JaxSlotMoved(request_id="req-2", old_slot_id=1, new_slot_id=0))
    received = q.get(timeout=5)
    assert received.request_id == "req-2"
    assert received.old_slot_id == 1
    assert received.new_slot_id == 0


def test_jax_denoise_batch_slots_round_trips_metadata_only():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put(JaxDenoiseBatchSlots(request_ids=("req-1", "req-2"), slot_ids=(0, 1)))
    received = q.get(timeout=5)
    assert received.request_ids == ("req-1", "req-2")
    assert received.slot_ids == (0, 1)


def test_jax_request_envelope_has_enqueue_timestamp():
    request = JaxRequestEnvelope(request_id="req-1", observation={}, sample_kwargs={}, enqueue_ns=time.monotonic_ns())
    assert request.enqueue_ns > 0


def test_jax_timing_helpers_split_queue_wait_and_transfer():
    q: queue.Queue[str] = queue.Queue()
    enqueue_ns = time.monotonic_ns()
    q.put("msg")
    message, get_start_ns, get_end_ns = timed_queue_get(q, timeout=1)
    queue_wait_ms, transfer_ms = queue_wait_and_transfer_ms(
        enqueue_ns=enqueue_ns,
        get_start_ns=get_start_ns,
        get_end_ns=get_end_ns,
    )
    assert message == "msg"
    assert queue_wait_ms >= 0.0
    assert transfer_ms >= 0.0


def test_jax_timed_queue_get_reraises_empty():
    q: queue.Queue[str] = queue.Queue()
    with pytest.raises(queue.Empty):
        timed_queue_get(q, block=False)
