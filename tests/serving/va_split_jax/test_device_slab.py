from __future__ import annotations

import multiprocessing as mp
import queue

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.serving.va_split_jax.device_slab import CudaIpcDeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec
from openpi.serving.va_split_jax.device_slab import has_cuda_device
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend


def test_device_slab_put_lane_and_view_batch():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.float32))
    slab = backend.copy_lane_from_array(slab, 1, jnp.full((1, 2, 3), 2.0, dtype=jnp.float32))
    batch = backend.view_batch(slab, 2)
    np.testing.assert_allclose(np.asarray(batch[0]), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch[1]), np.full((2, 3), 2.0, dtype=np.float32))
    slab.close()


def _producer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    backend = CudaIpcDeviceSlabBackend()
    slab = backend.create_slab(DeviceSlabSpec(name="ipc", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    try:
        slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.float32))
        slab = backend.copy_lane_from_array(slab, 1, jnp.full((1, 2, 3), 2.0, dtype=jnp.float32))
        control_queue.put(slab.handle)
        ack = result_queue.get(timeout=30)
        if ack != "ok":
            raise RuntimeError(f"consumer returned {ack!r}")
    except Exception as exc:
        control_queue.put({"error": repr(exc)})
        raise
    finally:
        slab.close()


def _consumer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    message = control_queue.get(timeout=30)
    if isinstance(message, dict) and "error" in message:
        result_queue.put(f"producer-error: {message['error']}")
        return
    try:
        slab = CudaIpcDeviceSlabBackend().open_slab(message)
        try:
            batch = CudaIpcDeviceSlabBackend().slice_lanes(slab, (0, 1))
            np.testing.assert_allclose(np.asarray(batch[0]), np.ones((2, 3), dtype=np.float32))
            np.testing.assert_allclose(np.asarray(batch[1]), np.full((2, 3), 2.0, dtype=np.float32))
        finally:
            slab.close()
        result_queue.put("ok")
    except Exception as exc:
        result_queue.put(f"consumer-error: {exc!r}")


def test_device_slab_handle_opens_in_consumer_process():
    if not has_cuda_device():
        pytest.skip("CUDA JAX device is required for cross-process device slab IPC")

    ctx = mp.get_context("spawn")
    control_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    producer = ctx.Process(target=_producer, args=(control_queue, result_queue))
    consumer = ctx.Process(target=_consumer, args=(control_queue, result_queue))
    producer.start()
    consumer.start()
    producer.join(timeout=45)
    consumer.join(timeout=45)

    assert producer.exitcode == 0
    assert consumer.exitcode == 0


def test_device_slab_rejects_sparse_slot_view():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    with pytest.raises(ValueError, match="dense-prefix"):
        backend.slice_lanes(slab, (0, 2))
    slab.close()
