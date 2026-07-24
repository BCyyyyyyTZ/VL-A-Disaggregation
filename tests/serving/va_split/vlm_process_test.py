from __future__ import annotations

import queue
import time

import pytest
import torch

from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.vlm_process import VLMProcess
from openpi.serving.va_split.vlm_process import VLMWorker


def _request_observation(*, prompt_len: int = 8, image_size: int = 224) -> dict:
    image = torch.zeros(1, 3, image_size, image_size)
    return {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image.clone(),
            "right_wrist_0_rgb": image.clone(),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        "state": torch.zeros(1, 8),
        "tokenized_prompt": torch.ones(1, prompt_len, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, prompt_len, dtype=torch.bool),
    }


class FakeVLMModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
        self.batch_sizes.append(int(observation.state.shape[0]))
        return PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(observation.state.shape[0], 3, dtype=torch.bool),
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


class SlowBlockingGetQueue(SimpleQueue):
    def __init__(self, messages=(), *, get_delay_s: float):
        super().__init__(messages)
        self._get_delay_s = get_delay_s

    def get(self, timeout=None):
        del timeout
        time.sleep(self._get_delay_s)
        return super().get_nowait()


def _request(request_id: str, *, prompt_len: int = 8, image_size: int = 224) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        observation=_request_observation(prompt_len=prompt_len, image_size=image_size),
        sample_kwargs={"num_steps": 4},
        enqueue_ns=123,
    )


def test_vlm_process_splits_request_queue_wait_and_transfer():
    class OnlyBlockingGetQueue(SimpleQueue):
        def get(self, timeout=None):
            del timeout
            time.sleep(0.01)
            return super().get_nowait()

        def get_nowait(self):
            raise queue.Empty

    request = RequestEnvelope(
        request_id="req-1",
        observation=_request_observation(),
        sample_kwargs={"num_steps": 1},
        enqueue_ns=time.monotonic_ns() - 5_000_000,
    )
    request_queue = OnlyBlockingGetQueue([request, Shutdown()])
    prefix_queue = SimpleQueue()
    process = VLMProcess(
        model=FakeVLMModel(),
        device="cpu",
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=1,
        max_wait_ms=0.0,
    )

    process.run()

    ready = next(item for item in prefix_queue.items if hasattr(item, "timing"))
    assert ready.timing["vlm_request_queue_wait_ms"] >= 4.0
    assert ready.timing["vlm_request_transfer_ms"] >= 8.0
    assert ready.timing["vlm_request_transfer_ms"] <= 50.0


def test_vlm_worker_publishes_prefix_and_releases_live_feature():
    worker = VLMWorker(model=FakeVLMModel(), device="cpu")
    request = RequestEnvelope(
        request_id="req-1",
        observation=_request_observation(),
        sample_kwargs={"num_steps": 4},
        enqueue_ns=1_000_000,
        dequeue_start_ns=2_500_000,
        dequeue_ns=3_000_000,
    )

    ready = worker.handle_request(request)

    assert ready.request_id == "req-1"
    assert ready.num_steps == 4
    assert ready.sample_kwargs == {"num_steps": 4}
    assert ready.timing is not None
    assert ready.timing["vlm_prefix_forward_ms"] >= 0.0
    assert ready.timing["vlm_request_queue_wait_ms"] == 1.5
    assert ready.timing["vlm_request_transfer_ms"] == 0.5
    assert ready.timing["vlm_queue_wait_ms"] >= 0.0
    assert "req-1" in worker.live_features
    torch.testing.assert_close(ready.feature.prefix_pad_masks, torch.ones(1, 3, dtype=torch.bool))

    worker.release(ReleaseFeature(request_id="req-1", slot_id=-1))

    assert worker.live_features == {}


def test_vlm_worker_handle_batch_builds_prefix_once_and_releases_by_refcount():
    model = FakeVLMModel()
    worker = VLMWorker(model=model, device="cpu")
    requests = [
        RequestEnvelope(
            request_id=f"req-{idx}",
            observation=_request_observation(),
            sample_kwargs={"num_steps": 4},
            enqueue_ns=idx,
        )
        for idx in range(2)
    ]

    ready = worker.handle_batch(requests)

    assert model.batch_sizes == [2]
    assert [message.request_id for message in ready] == ["req-0", "req-1"]
    assert all(message.feature.prefix_pad_masks.shape == (1, 3) for message in ready)
    assert all(message.timing is not None and message.timing["vlm_effective_batch"] == 2 for message in ready)
    assert set(worker.live_features) == {"req-0", "req-1"}
    assert len(worker.live_batches) == 1

    worker.release(ReleaseFeature(request_id="req-0", slot_id=-1))

    assert set(worker.live_features) == {"req-1"}
    assert len(worker.live_batches) == 1

    worker.release(ReleaseFeature(request_id="req-1", slot_id=-1))

    assert worker.live_features == {}
    assert worker.live_batches == {}


def test_vlm_worker_enforces_live_feature_capacity_until_release():
    worker = VLMWorker(model=FakeVLMModel(), device="cpu", max_live_features=1)

    worker.handle_request(_request("req-1"))

    with pytest.raises(RuntimeError, match="VLM live prefix feature slots are full"):
        worker.handle_request(_request("req-2"))

    worker.release(ReleaseFeature(request_id="req-1", slot_id=-1))
    ready = worker.handle_request(_request("req-2"))

    assert ready.request_id == "req-2"


def test_vlm_process_fcfs_collector_batches_compatible_ready_requests():
    model = FakeVLMModel()
    request_queue = SimpleQueue([*[_request(f"req-{idx}") for idx in range(5)], Shutdown()])
    prefix_queue = SimpleQueue()
    process = VLMProcess(
        model=model,
        device="cpu",
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=4,
        max_wait_ms=0.0,
    )

    process.run()

    ready = prefix_queue.items[:-1]
    assert model.batch_sizes == [4, 1]
    assert [message.request_id for message in ready] == [f"req-{idx}" for idx in range(5)]
    assert [message.timing["vlm_effective_batch"] for message in ready] == [4.0, 4.0, 4.0, 4.0, 1.0]
    assert isinstance(prefix_queue.items[-1], Shutdown)


def test_vlm_process_prefetches_ready_queue_before_fcfs_wait_window():
    model = FakeVLMModel()
    request_queue = SlowBlockingGetQueue(
        [*[_request(f"req-{idx}") for idx in range(5)], Shutdown()],
        get_delay_s=0.003,
    )
    prefix_queue = SimpleQueue()
    process = VLMProcess(
        model=model,
        device="cpu",
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=4,
        max_wait_ms=1.0,
    )

    process.run()

    ready = prefix_queue.items[:-1]
    assert model.batch_sizes == [4, 1]
    assert [message.request_id for message in ready] == [f"req-{idx}" for idx in range(5)]
    assert [message.timing["vlm_effective_batch"] for message in ready] == [4.0, 4.0, 4.0, 4.0, 1.0]
    assert isinstance(prefix_queue.items[-1], Shutdown)


def test_vlm_process_fcfs_collector_splits_incompatible_prompt_lengths():
    model = FakeVLMModel()
    request_queue = SimpleQueue([_request("req-1", prompt_len=8), _request("req-2", prompt_len=9), Shutdown()])
    prefix_queue = SimpleQueue()
    process = VLMProcess(
        model=model,
        device="cpu",
        request_queue=request_queue,
        prefix_queue=prefix_queue,
        release_queue=SimpleQueue(),
        max_batch_size=4,
        max_wait_ms=0.0,
    )

    process.run()

    ready = prefix_queue.items[:-1]
    assert model.batch_sizes == [1, 1]
    assert [message.request_id for message in ready] == ["req-1", "req-2"]
