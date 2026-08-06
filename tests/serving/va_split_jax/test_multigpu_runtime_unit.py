from __future__ import annotations

from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.prefix_transfer import PrefixTransferTicket
from openpi.serving.va_split_jax.process_entry import _child_env_updates
from openpi.serving.va_split_jax.runtime import _release_queue_key
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature


def test_prefix_ready_carries_source_worker_and_transfer_ticket():
    ticket = PrefixTransferTicket(kind="synchronous")
    ready = JaxPrefixReady(
        request_id="req-1",
        slot_handle=JaxPrefixSlotHandle(slot_id=3, batch_rows=1, prefix_shape_tree={}, prefix_dtype_tree={}),
        num_steps=5,
        sample_kwargs={},
        source_worker_id="vlm-1",
        prefix_ready_ticket=ticket,
    )

    assert ready.source_worker_id == "vlm-1"
    assert ready.prefix_ready_ticket is ticket


def test_child_env_updates_pin_single_visible_device():
    env = _child_env_updates("GPU-abc")

    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_release_queue_key_routes_to_source_worker():
    release = JaxReleaseFeature(request_id="req-1", slot_id=7, source_worker_id="vlm-2")

    assert _release_queue_key(release) == "vlm-2"
