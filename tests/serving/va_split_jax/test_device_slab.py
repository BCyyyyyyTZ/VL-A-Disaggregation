from __future__ import annotations

import multiprocessing as mp
import threading
import time
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import CudaIpcDeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec
from openpi.serving.va_split_jax.device_slab import has_cuda_device
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import write_feature_to_slab_tree


def test_device_slab_put_lane_and_view_batch():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.float32))
    slab = backend.copy_lane_from_array(slab, 1, jnp.full((1, 2, 3), 2.0, dtype=jnp.float32))
    batch = backend.view_batch(slab, 2)
    np.testing.assert_allclose(np.asarray(batch[0]), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch[1]), np.full((2, 3), 2.0, dtype=np.float32))
    slab.close()


def test_device_slab_preserves_bfloat16_logical_dtype():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="bfloat16", max_lanes=4))
    try:
        slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.bfloat16))
        batch = backend.view_batch(slab, 1)
        assert batch.dtype == jnp.bfloat16
        np.testing.assert_allclose(np.asarray(batch[0]), np.ones((2, 3), dtype=np.float32))
    finally:
        slab.close()


def test_device_slab_copy_batch_from_array_writes_contiguous_lanes():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    try:
        slab = backend.copy_lane_from_array(slab, 0, jnp.full((1, 2, 3), 7.0, dtype=jnp.float32))
        slab = backend.copy_batch_from_array(
            slab,
            1,
            jnp.stack(
                [
                    jnp.full((2, 3), 2.0, dtype=jnp.float32),
                    jnp.full((2, 3), 3.0, dtype=jnp.float32),
                ],
                axis=0,
            ),
        )
        batch = backend.view_batch(slab, 3)
        np.testing.assert_allclose(np.asarray(batch[0]), np.full((2, 3), 7.0, dtype=np.float32))
        np.testing.assert_allclose(np.asarray(batch[1]), np.full((2, 3), 2.0, dtype=np.float32))
        np.testing.assert_allclose(np.asarray(batch[2]), np.full((2, 3), 3.0, dtype=np.float32))
    finally:
        slab.close()


def test_device_slab_copy_batch_from_array_supports_nonzero_lane_axis_and_bfloat16():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(
        DeviceSlabSpec(name="past", shape=(3, 1, 2), dtype="bfloat16", max_lanes=4, lane_axis=1)
    )
    try:
        value = jnp.concatenate(
            [
                jnp.full((3, 1, 2), 4.0, dtype=jnp.bfloat16),
                jnp.full((3, 1, 2), 5.0, dtype=jnp.bfloat16),
            ],
            axis=1,
        )
        slab = backend.copy_batch_from_array(slab, 1, value)
        batch = backend.view_batch(slab, 3)
        assert batch.dtype == jnp.bfloat16
        np.testing.assert_allclose(np.asarray(batch[:, 1]), np.full((3, 2), 4.0, dtype=np.float32))
        np.testing.assert_allclose(np.asarray(batch[:, 2]), np.full((3, 2), 5.0, dtype=np.float32))
    finally:
        slab.close()


def _producer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    backend = CudaIpcDeviceSlabBackend()
    slab = backend.create_slab(DeviceSlabSpec(name="ipc", shape=(1, 2, 3), dtype="bfloat16", max_lanes=4))
    try:
        slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.bfloat16))
        slab = backend.copy_lane_from_array(slab, 1, jnp.full((1, 2, 3), 2.0, dtype=jnp.bfloat16))
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
            assert batch.dtype == jnp.bfloat16
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


def test_cuda_copy_lane_uses_stream_sync_not_device_or_slab_sync():
    if not has_cuda_device():
        pytest.skip("CUDA JAX device is required")

    backend = CudaIpcDeviceSlabBackend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    value = jnp.ones((1, 2, 3), dtype=jnp.float32)
    value.block_until_ready()
    try:
        with mock.patch("openpi.serving.va_split_jax.device_slab.cuda.synchronize") as device_sync:
            stream_sync = mock.Mock()
            fake_stream = mock.Mock()
            fake_stream.synchronize = stream_sync
            with mock.patch.object(backend, "_write_stream", return_value=fake_stream), mock.patch(
                "openpi.serving.va_split_jax.device_slab._copy_lane_with_kernel"
            ) as copy_kernel:
                backend.copy_lane_from_array(slab, 2, value)

            copy_kernel.assert_called_once()
            assert copy_kernel.call_args.kwargs.get("stream") is fake_stream
            stream_sync.assert_called_once()
            device_sync.assert_not_called()
    finally:
        slab.close()


def test_cuda_write_stream_uses_selected_device_context():
    backend = CudaIpcDeviceSlabBackend()
    fake_stream = mock.Mock()
    fake_context = mock.Mock()
    fake_context.create_stream.return_value = fake_stream

    with mock.patch(
        "openpi.serving.va_split_jax.device_slab._select_numba_device",
        return_value=fake_context,
    ) as select_device, mock.patch(
        "openpi.serving.va_split_jax.device_slab.cuda.stream",
        side_effect=AssertionError("cuda.stream() should not be used"),
    ):
        stream = backend._write_stream(2)  # noqa: SLF001

    select_device.assert_called_once_with(2)
    fake_context.create_stream.assert_called_once_with()
    assert stream is fake_stream


def test_cuda_deferred_lane_writes_sync_once_and_preserve_content():
    if not has_cuda_device():
        pytest.skip("CUDA JAX device is required")

    backend = CudaIpcDeviceSlabBackend()
    past = backend.create_slab(DeviceSlabSpec(name="past", shape=(1, 1, 2), dtype="float32", max_lanes=4))
    masks = backend.create_slab(DeviceSlabSpec(name="mask", shape=(1, 2), dtype="float32", max_lanes=4))
    state = backend.create_slab(DeviceSlabSpec(name="state", shape=(1, 3), dtype="float32", max_lanes=4))
    try:
        # Active dense prefix lane 0 stays readable while we write free lane 2.
        past = backend.copy_lane_from_array(past, 0, jnp.full((1, 1, 2), 7.0, dtype=jnp.float32))
        masks = backend.copy_lane_from_array(masks, 0, jnp.full((1, 2), 7.0, dtype=jnp.float32))
        state = backend.copy_lane_from_array(state, 0, jnp.full((1, 3), 7.0, dtype=jnp.float32))

        sync_calls: list[int] = []
        real_sync = backend.sync_write_stream

        def counting_sync(device_ordinal=None):
            sync_calls.append(1)
            return real_sync(device_ordinal)

        backend.sync_write_stream = counting_sync  # type: ignore[method-assign]
        feature = JaxPrefixFeature(
            past_key_values=jnp.full((1, 1, 2), 3.0, dtype=jnp.float32),
            prefix_pad_masks=jnp.full((1, 2), 3.0, dtype=jnp.float32),
            state=jnp.full((1, 3), 3.0, dtype=jnp.float32),
        )
        tree = write_feature_to_slab_tree(
            backend,
            {"past_key_values": past, "prefix_pad_masks": masks, "state": state},
            2,
            feature,
        )
        assert len(sync_calls) == 1

        active = backend.view_batch(tree["past_key_values"], 1)
        np.testing.assert_allclose(np.asarray(active[0]), np.full((1, 2), 7.0, dtype=np.float32))
        written_past = backend.view_batch(tree["past_key_values"], 3)
        np.testing.assert_allclose(np.asarray(written_past[2]), np.full((1, 2), 3.0, dtype=np.float32))
        written_mask = backend.view_batch(tree["prefix_pad_masks"], 3)
        np.testing.assert_allclose(np.asarray(written_mask[2]), np.full((2,), 3.0, dtype=np.float32))
        written_state = backend.view_batch(tree["state"], 3)
        np.testing.assert_allclose(np.asarray(written_state[2]), np.full((3,), 3.0, dtype=np.float32))
    finally:
        past.close()
        masks.close()
        state.close()


def test_free_lane_write_concurrent_with_active_prefix_read():
    """Free-lane IPC write must not require syncing / mutating active dense lanes."""
    if not has_cuda_device():
        pytest.skip("CUDA JAX device is required")

    backend = CudaIpcDeviceSlabBackend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 4, 8), dtype="float32", max_lanes=8))
    try:
        active_value = jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8)
        free_value = jnp.full((1, 4, 8), -1.0, dtype=jnp.float32)
        slab = backend.copy_lane_from_array(slab, 0, active_value)
        slab = backend.copy_lane_from_array(slab, 1, active_value * 2)

        errors: list[BaseException] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                expected0 = np.asarray(active_value[0])
                expected1 = np.asarray((active_value * 2)[0])
                while not stop.is_set():
                    batch = backend.view_batch(slab, 2)
                    batch.block_until_ready()
                    np.testing.assert_allclose(np.asarray(batch[0]), expected0)
                    np.testing.assert_allclose(np.asarray(batch[1]), expected1)
                    time.sleep(0.001)
            except BaseException as exc:  # surface in main thread
                errors.append(exc)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        time.sleep(0.01)
        with mock.patch("openpi.serving.va_split_jax.device_slab.cuda.synchronize") as device_sync:
            backend.copy_lane_from_array(slab, 5, free_value)
            device_sync.assert_not_called()
        stop.set()
        reader_thread.join(timeout=5)
        assert not reader_thread.is_alive()
        assert errors == []

        batch = backend.view_batch(slab, 6)
        np.testing.assert_allclose(np.asarray(batch[0]), np.asarray(active_value[0]))
        np.testing.assert_allclose(np.asarray(batch[1]), np.asarray((active_value * 2)[0]))
        np.testing.assert_allclose(np.asarray(batch[5]), np.full((4, 8), -1.0, dtype=np.float32))
    finally:
        slab.close()
