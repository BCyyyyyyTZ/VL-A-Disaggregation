from __future__ import annotations

import queue

import jax
import jax.numpy as jnp

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxPrefixSlabReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.types import JaxSlotMoved
from openpi.serving.va_split_jax.vlm_process import JaxVLMProcess
from openpi.serving.va_split_jax.vlm_process import JaxVLMWorker


class FakeJaxSplitModel:
    def __init__(self):
        self.prefix_batch_sizes: list[int] = []

    def build_prefix_feature(self, rng, observation):
        del rng
        batch = int(observation.state.shape[0])
        self.prefix_batch_sizes.append(batch)
        return JaxPrefixFeature(
            past_key_values=(
                jnp.ones((3, batch, 2, 4), dtype=jnp.float32),
                jnp.full((3, batch, 2, 4), 2.0, dtype=jnp.float32),
            ),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=observation.state,
        )


class SimpleQueue:
    def __init__(self, messages=()):
        self._messages = list(messages)
        self.items = []

    def get(self, timeout=None):
        del timeout
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)

    def get_nowait(self):
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)

    def put(self, item):
        self.items.append(item)


def _request_observation(*, prompt_len: int = 8) -> dict:
    image = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((1,), dtype=jnp.bool_),
        },
        "state": jnp.zeros((1, 8), dtype=jnp.float32),
        "tokenized_prompt": jnp.ones((1, prompt_len), dtype=jnp.int32),
        "tokenized_prompt_mask": jnp.ones((1, prompt_len), dtype=jnp.bool_),
    }


def _request(request_id: str, *, prompt_len: int = 8) -> JaxRequestEnvelope:
    return JaxRequestEnvelope(
        request_id=request_id,
        observation=_request_observation(prompt_len=prompt_len),
        sample_kwargs={"num_steps": 4},
        enqueue_ns=123,
    )


def _pool(max_lanes: int = 4) -> JaxVlmPrefixCacheLanePool:
    return JaxVlmPrefixCacheLanePool(max_lanes=max_lanes, backend=make_default_device_slab_backend())


def test_jax_vlm_worker_publishes_slot_handle_and_releases_slot():
    worker = JaxVLMWorker(model=FakeJaxSplitModel(), max_live_features=2, prefix_pool=_pool(max_lanes=2))

    ready = worker.handle_request(_request("req-1"))

    assert ready.request_id == "req-1"
    assert ready.num_steps == 4
    assert ready.slot_handle.slot_id == 0
    assert ready.slot_handle.batch_rows == 1
    assert ready.slot_handle.prefix_shape_tree["prefix_pad_masks"] == (1, 3)
    assert worker.available_live_feature_slots == 1

    moved = worker.release(JaxReleaseFeature(request_id="req-1", slot_id=0))

    assert moved is None
    assert worker.available_live_feature_slots == 2


def test_jax_vlm_process_fcfs_batches_compatible_requests_and_sends_slab_once():
    model = FakeJaxSplitModel()
    request_queue = SimpleQueue([_request("req-1"), _request("req-2"), JaxShutdown()])
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        prefix_pool=_pool(max_lanes=4),
        max_batch_size=4,
        max_wait_ms=0.0,
    )

    process.run()

    assert model.prefix_batch_sizes == [2]
    assert isinstance(prefix_queue.items[0], JaxPrefixSlabReady)
    ready = [item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert [item.request_id for item in ready] == ["req-1", "req-2"]
    assert [item.slot_handle.slot_id for item in ready] == [0, 1]
    assert [item.timing["vlm_effective_batch"] for item in ready] == [2.0, 2.0]
    assert sum(isinstance(item, JaxPrefixSlabReady) for item in prefix_queue.items) == 1
    assert isinstance(prefix_queue.items[-1], JaxShutdown)


def test_jax_vlm_process_release_compaction_sends_slot_moved():
    model = FakeJaxSplitModel()
    release_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=SimpleQueue([_request("req-1"), _request("req-2"), JaxShutdown()]),
        prefix_queue=SimpleQueue(),
        release_queue=release_queue,
        prefix_pool=_pool(max_lanes=4),
        max_batch_size=4,
        max_wait_ms=0.0,
    )
    process.run()

    release_queue._messages.append(JaxReleaseFeature(request_id="req-1", slot_id=0))
    process._drain_releases()

    moved = process._prefix_queue.items[-1]
    assert moved == JaxSlotMoved(request_id="req-2", old_slot_id=1, new_slot_id=0)


def test_jax_vlm_process_fcfs_splits_incompatible_prompt_lengths():
    model = FakeJaxSplitModel()
    process = JaxVLMProcess(
        model=model,
        request_queue=SimpleQueue([_request("req-1", prompt_len=8), _request("req-2", prompt_len=9), JaxShutdown()]),
        prefix_queue=SimpleQueue(),
        release_queue=SimpleQueue(),
        prefix_pool=_pool(max_lanes=4),
        max_batch_size=4,
        max_wait_ms=0.0,
    )

    process.run()

    assert model.prefix_batch_sizes == [1, 1]
    ready = [item for item in process._prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert [item.request_id for item in ready] == ["req-1", "req-2"]


def test_jax_prefix_feature_is_valid_jit_output():
    @jax.jit
    def make_feature(x):
        return JaxPrefixFeature(
            past_key_values=(x[None, :, :], x[None, :, :] + 1),
            prefix_pad_masks=jnp.ones((x.shape[0], x.shape[1]), dtype=jnp.bool_),
            state=x,
        )

    feature = make_feature(jnp.ones((2, 3), dtype=jnp.float32))

    assert isinstance(feature, JaxPrefixFeature)
    assert feature.state.shape == (2, 3)
