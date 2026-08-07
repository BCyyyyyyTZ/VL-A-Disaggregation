from __future__ import annotations

import sys
from types import SimpleNamespace

from openpi.models.jax_split_types import JaxPrefixSlabHandleTree
from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax import device_slab
from openpi.serving.va_split_jax.device_slab import DeviceSlabHandle
from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec
from openpi.serving.va_split_jax.prefix_transfer import PrefixTransferTicket
from openpi.serving.va_split_jax.process_entry import _child_env_updates
from openpi.serving.va_split_jax.process_entry import run_ae_worker_entry
from openpi.serving.va_split_jax.runtime import JaxMultiGpuReleaseFanout
from openpi.serving.va_split_jax.runtime import _release_queue_key
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
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


def test_process_entry_applies_env_without_forwarding_duplicate_env_updates(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "openpi.serving.va_split_jax.runtime",
        SimpleNamespace(_run_jax_ae_process=fake_run),
    )

    run_ae_worker_entry(
        "model_factory",
        "prefix_queue",
        "result_queue",
        "release_queue",
        8,
        24,
        "compile_config",
        None,
        "warmup_queue",
        device="3",
        env_updates={"CUSTOM_ENV": "1"},
    )

    assert captured["args"][-2:] == (None, "warmup_queue")
    assert "env_updates" not in captured["kwargs"]


class _Queue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


def test_multigpu_release_fanout_rewrites_slab_device_ordinal_per_worker():
    spec = DeviceSlabSpec(name="kv", shape=(1, 2), dtype="float32", max_lanes=4)
    handle = DeviceSlabHandle(
        spec=spec,
        transport="cuda-ipc-numba",
        device_ordinal=0,
        handle_bytes=b"handle",
    )
    slab_ready = JaxPrefixSlabReady(
        slab=JaxPrefixSlabHandleTree(
            max_lanes=4,
            prefix_shape_tree={},
            prefix_dtype_tree={},
            slab_handle_tree={"past_key_values": (handle,), "prefix_pad_masks": handle, "state": None},
        )
    )
    queues = {"vlm-0": _Queue(), "vlm-1": _Queue()}
    fanout = JaxMultiGpuReleaseFanout(queues, slab_device_ordinals={"vlm-0": 0, "vlm-1": 0})

    fanout.put(slab_ready)

    first = queues["vlm-0"].messages[0].slab.slab_handle_tree["past_key_values"][0]
    second = queues["vlm-1"].messages[0].slab.slab_handle_tree["prefix_pad_masks"]
    assert first.device_ordinal == 0
    assert second.device_ordinal == 0
    assert handle.device_ordinal == 0


def test_numba_device_ordinal_prefers_cuda_visible_devices_mapping(monkeypatch):
    fake_device = SimpleNamespace(id=3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(device_slab.cuda, "gpus", [object() for _ in range(8)])

    ordinal = device_slab._numba_device_ordinal_for_jax_device(fake_device)  # noqa: SLF001

    assert ordinal == 0
