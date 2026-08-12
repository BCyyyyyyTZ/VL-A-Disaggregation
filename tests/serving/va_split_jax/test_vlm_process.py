from __future__ import annotations

import queue
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.serving.va_split_jax.types import JaxLaneCredits
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxReleaseFeature
from openpi.serving.va_split_jax.types import JaxRequestEnvelope
from openpi.serving.va_split_jax.types import JaxShutdown
from openpi.serving.va_split_jax.vlm_process import JaxVLMProcess
from openpi.serving.va_split_jax.vlm_process import JaxVLMWorker
from openpi.serving.va_split_jax.vlm_process import _asarray_model_input
from openpi.serving.va_split_jax.vlm_process import _vlm_request_queue_timings


class FakeJaxSplitModel:
    def __init__(self):
        self.prefix_batch_sizes: list[int] = []
        self.image_dtypes: list[dict[str, str]] = []

    def build_prefix_feature(self, rng, observation):
        del rng
        batch = int(observation.state.shape[0])
        self.prefix_batch_sizes.append(batch)
        self.image_dtypes.append({key: str(value.dtype) for key, value in observation.images.items()})
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


class SlowFeederQueue:
    def __init__(self, messages=(), *, message_delay_s: float):
        self._messages = list(messages)
        self._message_delay_s = message_delay_s
        self._next_ready_s = 0.0

    def get(self, timeout=None):
        if not self._messages:
            raise queue.Empty
        now_s = time.monotonic()
        wait_s = self._next_ready_s - now_s
        if wait_s > 0:
            if timeout is not None and timeout < wait_s:
                time.sleep(timeout)
                raise queue.Empty
            time.sleep(wait_s)
        return self._pop_ready()

    def get_nowait(self):
        if not self._messages or time.monotonic() < self._next_ready_s:
            raise queue.Empty
        return self._pop_ready()

    def _pop_ready(self):
        message = self._messages.pop(0)
        self._next_ready_s = time.monotonic() + self._message_delay_s
        return message


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


def _request(request_id: str, *, prompt_len: int = 8, enqueue_ns: int = 123) -> JaxRequestEnvelope:
    return JaxRequestEnvelope(
        request_id=request_id,
        observation=_request_observation(prompt_len=prompt_len),
        sample_kwargs={"num_steps": 4},
        enqueue_ns=enqueue_ns,
    )


def _shared_pool(max_lanes: int = 4) -> JaxVlmPrefixCacheLanePool:
    pool = JaxVlmPrefixCacheLanePool(max_lanes=max_lanes, backend=make_default_device_slab_backend())
    template = JaxPrefixFeature(
        past_key_values=(
            jnp.zeros((3, 1, 2, 4), dtype=jnp.float32),
            jnp.zeros((3, 1, 2, 4), dtype=jnp.float32),
        ),
        prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.zeros((1, 8), dtype=jnp.float32),
    )
    pool.initialize_from_feature(template)
    return pool


def _worker_with_credits(max_lanes: int = 2) -> JaxVLMWorker:
    pool = _shared_pool(max_lanes=max_lanes)
    worker = JaxVLMWorker(
        model=FakeJaxSplitModel(),
        max_live_features=max_lanes,
        backend=make_default_device_slab_backend(),
        shared_pool=pool,
    )
    worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=tuple(range(max_lanes))))
    return worker


def test_jax_vlm_worker_writes_ae_lane_and_recycles_credit_on_release():
    worker = _worker_with_credits(max_lanes=2)

    ready = worker.handle_request(_request("req-1"))

    assert ready.request_id == "req-1"
    assert ready.num_steps == 4
    assert ready.slot_handle.slot_id == 0
    assert ready.slot_handle.batch_rows == 1
    assert ready.slot_handle.prefix_shape_tree["prefix_pad_masks"] == (1, 3)
    assert ready.timing is not None
    assert ready.timing["vlm_input_stage_ms"] >= 0.0
    assert ready.timing["vlm_sample_kwargs_stage_ms"] >= 0.0
    assert ready.timing["vlm_observation_stack_ms"] >= 0.0
    assert ready.timing["vlm_to_jax_tree_ms"] >= 0.0
    assert ready.timing["vlm_observation_from_dict_ms"] >= 0.0
    assert ready.timing["vlm_observation_uint8_normalize_ms"] >= 0.0
    assert ready.timing["vlm_observation_construct_ms"] >= 0.0
    assert ready.timing["vlm_observation_uint8_images"] == 0.0
    assert ready.timing["vlm_observation_float32_images"] == 3.0
    assert ready.timing["vlm_observation_other_images"] == 0.0
    assert ready.timing["vlm_batch_wait_ms"] >= 0.0
    assert ready.timing["vlm_queue_wait_ms"] == ready.timing["vlm_batch_wait_ms"]
    assert worker.available_live_feature_slots == 1

    worker.release(JaxReleaseFeature(request_id="req-1", slot_id=0))

    assert worker.available_live_feature_slots == 2


def test_jax_vlm_worker_returns_credits_when_slab_write_fails():
    worker = _worker_with_credits(max_lanes=2)
    assert worker.available_live_feature_slots == 2

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated slab write failure")

    worker._write_feature_to_lanes = _boom  # type: ignore[method-assign]

    try:
        worker.handle_batch([_request("req-1"), _request("req-2")])
        raise AssertionError("expected slab write failure")
    except RuntimeError as exc:
        assert "simulated slab write failure" in str(exc)

    assert worker.available_live_feature_slots == 2
    assert worker.active_count == 0


def test_jax_vlm_worker_rejects_credit_when_capacity_is_already_held():
    worker = _worker_with_credits(max_lanes=2)

    first = worker.handle_request(_request("req-1"))
    assert first.slot_handle.slot_id == 0
    assert worker.available_live_feature_slots == 1
    assert worker.active_count == 1

    worker.grant_lane_credits(JaxLaneCredits(lane_ids=(0,)))

    assert worker.available_live_feature_slots == 1
    second = worker.handle_request(_request("req-2"))
    assert second.slot_handle.slot_id == 1


def test_jax_vlm_worker_recycles_original_physical_release_slot():
    worker = _worker_with_credits(max_lanes=3)

    first = worker.handle_request(_request("req-1"))
    second = worker.handle_request(_request("req-2"))

    assert first.slot_handle.slot_id == 0
    assert second.slot_handle.slot_id == 1
    assert worker.available_live_feature_slots == 1
    assert worker.active_count == 2

    worker.release(JaxReleaseFeature(request_id="req-1", slot_id=0))

    assert worker.available_live_feature_slots == 2
    assert worker.active_count == 1
    third = worker.handle_request(_request("req-3"))
    assert third.slot_handle.slot_id == 0


def test_jax_vlm_worker_ignores_duplicate_release_for_completed_request():
    worker = _worker_with_credits(max_lanes=2)

    first = worker.handle_request(_request("req-1"))
    assert first.slot_handle.slot_id == 0
    worker.release(JaxReleaseFeature(request_id="req-1", slot_id=0))
    second = worker.handle_request(_request("req-2"))

    assert second.slot_handle.slot_id == 0
    assert worker.available_live_feature_slots == 1
    assert worker.active_count == 1

    worker.release(JaxReleaseFeature(request_id="req-1", slot_id=0))

    assert worker.available_live_feature_slots == 1
    assert worker.active_count == 1
    third = worker.handle_request(_request("req-3"))
    assert third.slot_handle.slot_id == 1


def test_jax_vlm_process_returns_unsent_credits_when_prefix_queue_put_fails():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=2)
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=SimpleQueue(),
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=2,
        max_wait_ms=0.0,
        max_live_features=2,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0, 1)))
    process._ae_export_ready = True

    ready = process.worker.handle_batch([_request("req-1"), _request("req-2")])
    assert process.worker.available_live_feature_slots == 0
    assert process.worker.active_count == 2

    original_put = prefix_queue.put
    calls = {"n": 0}

    def _flaky_put(item):
        calls["n"] += 1
        if calls["n"] == 1:
            original_put(item)
            return
        raise RuntimeError("simulated prefix queue failure")

    prefix_queue.put = _flaky_put  # type: ignore[method-assign]

    try:
        process._put_prefix_batch(ready)
        raise AssertionError("expected prefix queue failure")
    except RuntimeError as exc:
        assert "simulated prefix queue failure" in str(exc)

    # First PrefixReady was handed off; only the unsent credit must come back.
    assert process.worker.available_live_feature_slots == 1
    assert process.worker.active_count == 1
    assert len([item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]) == 1


def test_jax_vlm_request_transfer_excludes_pre_enqueue_blocking_get_wait():
    queue_wait_ms, transfer_ms = _vlm_request_queue_timings(
        enqueue_ns=1_000,
        dequeue_start_ns=0,
        dequeue_ns=1_500,
    )

    assert queue_wait_ms == 0.0
    assert transfer_ms == 0.0005


def test_jax_vlm_input_stage_keeps_uint8_until_observation_from_dict():
    staged = _asarray_model_input(np.zeros((1, 224, 224, 3), dtype=np.uint8))

    assert staged.dtype == jnp.uint8


def test_jax_vlm_input_stage_keeps_existing_jax_arrays_on_device():
    array = jnp.zeros((2, 4), dtype=jnp.float32)

    staged = _asarray_model_input(array)

    assert staged is array


def test_jax_vlm_worker_normalizes_unexpected_uint8_images_before_model():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=2)
    worker = JaxVLMWorker(
        model=model,
        max_live_features=2,
        backend=make_default_device_slab_backend(),
        shared_pool=pool,
    )
    worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0, 1)))
    obs = _request_observation()
    image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    obs["image"] = {
        "base_0_rgb": image,
        "left_wrist_0_rgb": image,
        "right_wrist_0_rgb": image,
    }

    ready = worker.handle_request(
        JaxRequestEnvelope(
            request_id="req-uint8",
            observation=obs,
            sample_kwargs={"num_steps": 4},
            enqueue_ns=123,
        )
    )

    assert ready.timing is not None
    assert ready.timing["vlm_observation_uint8_images"] == 3.0
    assert ready.timing["vlm_observation_float32_images"] == 0.0
    assert all(dtype == "float32" for dtype in model.image_dtypes[-1].values())


def test_jax_vlm_process_fcfs_batches_compatible_requests_without_slab_export():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=4)
    request_queue = SimpleQueue([_request("req-1"), _request("req-2"), JaxShutdown()])
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=4,
        max_wait_ms=0.0,
        max_live_features=4,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0, 1, 2, 3)))
    process._ae_export_ready = True

    process.run()

    assert model.prefix_batch_sizes == [2]
    ready = [item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert [item.request_id for item in ready] == ["req-1", "req-2"]
    assert [item.slot_handle.slot_id for item in ready] == [0, 1]
    assert [item.timing["vlm_effective_batch"] for item in ready] == [2.0, 2.0]
    assert [item.timing["vlm_slab_write_contiguous_batch"] for item in ready] == [1.0, 1.0]
    assert all(item.timing["vlm_slab_write_total_ms"] >= item.timing["vlm_slab_write_ms"] for item in ready)
    assert isinstance(prefix_queue.items[-1], JaxShutdown)


def test_jax_vlm_process_fcfs_drains_pending_credits_before_fixing_batch_limit():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=4)
    request_queue = SimpleQueue([_request(f"req-{idx}") for idx in range(4)] + [JaxShutdown()])
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue([JaxLaneCredits(lane_ids=(1, 2, 3))]),
        max_batch_size=4,
        max_wait_ms=0.0,
        max_live_features=4,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0,)))
    process._ae_export_ready = True

    process.run()

    ready = [item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert model.prefix_batch_sizes == [4]
    assert [item.request_id for item in ready] == [f"req-{idx}" for idx in range(4)]
    assert [item.slot_handle.slot_id for item in ready] == [0, 1, 2, 3]
    assert [item.timing["vlm_slab_write_contiguous_batch"] for item in ready] == [1.0] * 4


def test_jax_vlm_process_fcfs_drains_slow_feeder_when_head_request_is_already_late():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=8)
    old_enqueue_ns = time.monotonic_ns() - 50_000_000
    request_queue = SlowFeederQueue(
        [_request(f"req-{idx}", enqueue_ns=old_enqueue_ns) for idx in range(8)] + [JaxShutdown()],
        message_delay_s=0.0015,
    )
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=8,
        max_wait_ms=1.0,
        max_live_features=8,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=tuple(range(8))))
    process._ae_export_ready = True

    process.run()

    ready = [item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert model.prefix_batch_sizes == [6, 2]
    assert [item.request_id for item in ready] == [f"req-{idx}" for idx in range(8)]
    assert [item.timing["vlm_effective_batch"] for item in ready] == [6.0] * 6 + [2.0] * 2
    assert isinstance(prefix_queue.items[-1], JaxShutdown)


def test_jax_vlm_process_keeps_full_batch_when_late_head_has_deep_backlog():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=8)
    old_enqueue_ns = time.monotonic_ns() - 50_000_000
    request_queue = SimpleQueue(
        [_request(f"req-{idx}", enqueue_ns=old_enqueue_ns) for idx in range(8)] + [JaxShutdown()]
    )
    prefix_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=8,
        max_wait_ms=1.0,
        max_live_features=8,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=tuple(range(8))))
    process._ae_export_ready = True

    process.run()

    ready = [item for item in prefix_queue.items if isinstance(item, JaxPrefixReady)]
    assert model.prefix_batch_sizes == [8]
    assert [item.request_id for item in ready] == [f"req-{idx}" for idx in range(8)]
    assert [item.timing["vlm_effective_batch"] for item in ready] == [8.0] * 8


def test_jax_vlm_worker_keeps_row_noise_host_side_for_prefix_ready():
    worker = _worker_with_credits(max_lanes=2)
    req1 = JaxRequestEnvelope(
        request_id="req-1",
        observation=_request_observation(),
        sample_kwargs={"num_steps": 4, "noise": np.ones((1, 2, 1), dtype=np.float32)},
        enqueue_ns=123,
    )
    req2 = JaxRequestEnvelope(
        request_id="req-2",
        observation=_request_observation(),
        sample_kwargs={"num_steps": 4, "noise": np.full((1, 2, 1), 2.0, dtype=np.float32)},
        enqueue_ns=123,
    )

    ready = worker.handle_batch([req1, req2])

    assert isinstance(ready[0].sample_kwargs["noise"], np.ndarray)
    assert isinstance(ready[1].sample_kwargs["noise"], np.ndarray)
    np.testing.assert_allclose(ready[0].sample_kwargs["noise"], np.ones((1, 2, 1), dtype=np.float32))
    np.testing.assert_allclose(ready[1].sample_kwargs["noise"], np.full((1, 2, 1), 2.0, dtype=np.float32))


def test_jax_vlm_process_release_returns_lane_credit():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=4)
    release_queue = SimpleQueue()
    process = JaxVLMProcess(
        model=model,
        request_queue=SimpleQueue([_request("req-1"), _request("req-2"), JaxShutdown()]),
        prefix_queue=SimpleQueue(),
        release_queue=release_queue,
        max_batch_size=4,
        max_wait_ms=0.0,
        max_live_features=4,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0, 1, 2, 3)))
    process._ae_export_ready = True
    process.run()

    assert process.worker.available_live_feature_slots == 2
    release_queue._messages.append(JaxReleaseFeature(request_id="req-1", slot_id=0))
    process._drain_releases()

    assert process.worker.available_live_feature_slots == 3


def test_jax_vlm_process_fcfs_splits_incompatible_prompt_lengths():
    model = FakeJaxSplitModel()
    pool = _shared_pool(max_lanes=4)
    process = JaxVLMProcess(
        model=model,
        request_queue=SimpleQueue([_request("req-1", prompt_len=8), _request("req-2", prompt_len=9), JaxShutdown()]),
        prefix_queue=SimpleQueue(),
        release_queue=SimpleQueue(),
        max_batch_size=4,
        max_wait_ms=0.0,
        max_live_features=4,
        shared_pool=pool,
    )
    process.worker.attach_shared_pool(pool, JaxLaneCredits(lane_ids=(0, 1, 2, 3)))
    process._ae_export_ready = True

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
