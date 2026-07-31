from __future__ import annotations

import dataclasses
import queue
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.ae_process import JaxAEProcess
from openpi.serving.va_split_jax.ae_process import JaxAEWorker
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxSlotMoved
from openpi.serving.va_split_jax.types import JaxWorkerError


class FakeJaxAEModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.batch_sizes: list[int] = []
        self.prefix_slot_batch_shapes: list[tuple[int, ...]] = []

    def init_denoise_state(self, rng, batch_size: int, noise: jax.Array | None, num_steps: int) -> JaxDenoiseState:
        del rng
        if noise is None:
            noise = jnp.zeros((batch_size, self.config.action_horizon, self.config.action_dim), dtype=jnp.float32)
        return JaxDenoiseState(
            x_t=noise,
            step_idx=jnp.asarray(0, dtype=jnp.int32),
            num_steps=num_steps,
            dt=jnp.asarray(-1.0 / num_steps, dtype=jnp.float32),
        )

    def denoise_one_batch(self, prefix_batch: JaxPrefixFeature, denoise_batch: JaxDenoiseState) -> jax.Array:
        self.batch_sizes.append(int(denoise_batch.x_t.shape[0]))
        self.prefix_slot_batch_shapes.append(tuple(prefix_batch.prefix_pad_masks.shape))
        return jnp.ones_like(denoise_batch.x_t)


class FailingStepModel(FakeJaxAEModel):
    def denoise_one_batch(self, prefix_batch: JaxPrefixFeature, denoise_batch: JaxDenoiseState) -> jax.Array:
        raise RuntimeError("bad step")


class SimpleQueue:
    def __init__(self, messages=()):
        self._messages = list(messages)
        self.items = []

    def get(self):
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)

    def get_nowait(self):
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)

    def put(self, item):
        self.items.append(item)

    def remaining(self):
        return list(self._messages)


def _feature(fill: float) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=(
            jnp.full((1, 3, 2, 4), fill, dtype=jnp.float32),
            jnp.full((1, 3, 2, 4), fill + 1, dtype=jnp.float32),
        ),
        prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.full((1, 8), fill, dtype=jnp.float32),
    )


def _mapped_prefix_slabs(*request_ids: str):
    backend = make_default_device_slab_backend()
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=backend)
    for idx, request_id in enumerate(request_ids):
        pool.put_lane(request_id, _feature(float(idx + 1)))
    return backend, pool.local_slab_tree()


def _ready(request_id: str, slot_id: int, *, num_steps: int = 1) -> JaxPrefixReady:
    return JaxPrefixReady(
        request_id=request_id,
        slot_handle=JaxPrefixSlotHandle(
            slot_id=slot_id,
            batch_rows=1,
            prefix_shape_tree=("shape",),
            prefix_dtype_tree=("dtype",),
        ),
        num_steps=num_steps,
        sample_kwargs={"noise": jnp.zeros((1, 2, 1), dtype=jnp.float32)},
        timing={"vlm_prefix_forward_ms": 1.5},
    )


def test_jax_ae_worker_batches_two_ready_requests_for_two_denoise_steps():
    model = FakeJaxAEModel()
    backend, slab_tree = _mapped_prefix_slabs("req-1", "req-2")
    worker = JaxAEWorker(model=model, max_batch_size=2, max_prefix_slots=4, backend=backend)
    worker.set_mapped_prefix_slabs(slab_tree, map_ms=0.25)
    worker.add_prefix(_ready("req-1", 0, num_steps=2))
    worker.add_prefix(_ready("req-2", 1, num_steps=2))

    first_results, first_releases = worker.step_once()
    second_results, second_releases = worker.step_once()

    assert first_results == []
    assert first_releases == []
    assert model.batch_sizes == [2, 2]
    assert model.prefix_slot_batch_shapes == [(2, 3), (2, 3)]
    assert [result.request_id for result in second_results] == ["req-1", "req-2"]
    assert [release.request_id for release in second_releases] == ["req-1", "req-2"]
    for result in second_results:
        np.testing.assert_allclose(result.actions, -jnp.ones((1, 2, 1), dtype=jnp.float32))
        assert result.timing is not None
        assert result.timing["ae_effective_batch"] == 2.0
        assert result.timing["prefix_pool_write_ms"] == 0.0
        assert result.timing["prefix_pool_compact_ms"] >= 0.0
        assert result.timing["prefix_slab_map_ms"] == 0.25


def test_jax_ae_worker_updates_prefix_slot_after_vlm_compaction():
    backend, slab_tree = _mapped_prefix_slabs("req-1", "req-2")
    worker = JaxAEWorker(model=FakeJaxAEModel(), max_batch_size=2, max_prefix_slots=4, backend=backend)
    worker.set_mapped_prefix_slabs(slab_tree)
    worker.add_prefix(_ready("req-1", 0, num_steps=1))
    worker.add_prefix(_ready("req-2", 1, num_steps=2))

    results, releases = worker.step_once()
    assert [result.request_id for result in results] == ["req-1"]
    assert releases == [JaxReleaseFeature(request_id="req-1", slot_id=0)]
    assert worker.active["req-2"].prefix_slot_id == 1

    worker.apply_slot_moved(JaxSlotMoved(request_id="req-2", old_slot_id=1, new_slot_id=0))

    assert worker.active["req-2"].prefix_slot_id == 0
    assert [request.request_id for request in worker.select_ready_lanes()] == ["req-2"]


def test_jax_ae_process_waits_for_free_prefix_slot_before_draining_more_ready_messages():
    backend, slab_tree = _mapped_prefix_slabs("req-1", "req-2")
    prefix_queue = SimpleQueue([_ready("req-1", 0, num_steps=2), _ready("req-2", 1, num_steps=1)])
    process = JaxAEProcess(
        model=FakeJaxAEModel(),
        prefix_queue=prefix_queue,
        result_queue=SimpleQueue(),
        release_queue=SimpleQueue(),
        max_batch_size=1,
        max_prefix_slots=1,
        backend=backend,
    )
    process.worker.set_mapped_prefix_slabs(slab_tree)

    process.drain_prefix_ready(block=True)

    assert list(process.worker.active) == ["req-1"]
    assert [message.request_id for message in prefix_queue.remaining()] == ["req-2"]

    process.step_active_once()
    process.drain_prefix_ready(block=False)

    assert list(process.worker.active) == ["req-1"]
    assert [message.request_id for message in prefix_queue.remaining()] == ["req-2"]

    process.step_active_once()
    process.worker.apply_slot_moved(JaxSlotMoved(request_id="req-2", old_slot_id=1, new_slot_id=0))
    process.drain_prefix_ready(block=True)

    assert list(process.worker.active) == ["req-2"]


def test_jax_ae_process_releases_active_features_when_step_fails():
    backend, slab_tree = _mapped_prefix_slabs("req-1", "req-2")
    result_queue = SimpleQueue()
    release_queue = SimpleQueue()
    process = JaxAEProcess(
        model=FailingStepModel(),
        prefix_queue=SimpleQueue(),
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=2,
        backend=backend,
    )
    process.worker.set_mapped_prefix_slabs(slab_tree)
    process.worker.add_prefix(_ready("req-1", 0))
    process.worker.add_prefix(_ready("req-2", 1))

    process.step_active_once()

    assert [item.request_id for item in result_queue.items] == ["req-1", "req-2"]
    assert all(isinstance(item, JaxWorkerError) for item in result_queue.items)
    assert [item.request_id for item in release_queue.items] == ["req-1", "req-2"]
    assert process.worker.active == {}


def test_jax_ae_process_shutdown_closes_worker():
    process = JaxAEProcess(
        model=FakeJaxAEModel(),
        prefix_queue=SimpleQueue([JaxShutdown()]),
        result_queue=SimpleQueue(),
        release_queue=SimpleQueue(),
        max_batch_size=1,
    )
    with pytest.raises(SystemExit):
        process.drain_prefix_ready(block=True)
    assert isinstance(process._result_queue.items[0], JaxShutdown)


def test_jax_ae_worker_splits_prefix_queue_wait_transfer_and_admit():
    backend, slab_tree = _mapped_prefix_slabs("req-1")
    worker = JaxAEWorker(model=FakeJaxAEModel(), max_batch_size=1, backend=backend)
    worker.set_mapped_prefix_slabs(slab_tree)
    get_end_ns = time.monotonic_ns() - 2_000_000
    prefix_ready = dataclasses.replace(
        _ready("req-1", 0),
        timing={
            "vlm_prefix_forward_ms": 1.5,
            "_prefix_enqueue_ns": float(get_end_ns - 5_000_000),
            "_prefix_get_start_ns": float(get_end_ns - 1_000_000),
            "_prefix_get_end_ns": float(get_end_ns),
        },
    )
    time.sleep(0.005)
    worker.add_prefix(prefix_ready)

    timing = worker.active["req-1"].timing
    assert timing["prefix_queue_wait_ms"] == pytest.approx(4.0, abs=0.1)
    assert timing["prefix_transfer_ms"] == pytest.approx(1.0, abs=0.1)
    assert timing["prefix_admit_wait_ms"] >= 4.0
    assert "_prefix_enqueue_ns" not in timing
