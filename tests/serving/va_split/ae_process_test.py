from __future__ import annotations

import dataclasses
import queue
import time
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.ae_process import AEProcess
from openpi.serving.va_split.ae_process import AEWorker
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import WorkerError


class FakeAEModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.batch_sizes: list[int] = []

    def init_denoise_state(
        self, device: str, batch_size: int, noise: torch.Tensor | None, num_steps: int
    ) -> DenoiseState:
        assert device == "cpu"
        if noise is None:
            noise = torch.zeros(batch_size, self.config.action_horizon, self.config.action_dim)
        return DenoiseState(
            x_t=noise,
            step_idx=torch.tensor(0, dtype=torch.int32),
            num_steps=num_steps,
            dt=torch.tensor(-1.0 / num_steps),
        )

    def denoise_one_batch(self, prefix_batch: PrefixFeature, denoise_batch: DenoiseState) -> torch.Tensor:
        self.batch_sizes.append(denoise_batch.x_t.shape[0])
        assert prefix_batch.prefix_pad_masks.shape[0] == denoise_batch.x_t.shape[0]
        return torch.ones_like(denoise_batch.x_t)


class FakeDynamicCacheAEModel(FakeAEModel):
    def denoise_one_batch(self, prefix_batch: PrefixFeature, denoise_batch: DenoiseState) -> torch.Tensor:
        assert isinstance(prefix_batch.past_key_values, DynamicCache)
        key, value = prefix_batch.past_key_values[0]
        assert key.shape == (2, 1, 3, 4)
        assert value.shape == (2, 1, 3, 4)
        return super().denoise_one_batch(prefix_batch, denoise_batch)


class FailingAddPrefixModel(FakeAEModel):
    def init_denoise_state(
        self, device: str, batch_size: int, noise: torch.Tensor | None, num_steps: int
    ) -> DenoiseState:
        raise RuntimeError("bad prefix")


class FailingStepModel(FakeAEModel):
    def denoise_one_batch(self, prefix_batch: PrefixFeature, denoise_batch: DenoiseState) -> torch.Tensor:
        raise RuntimeError("bad step")


class SimpleQueue:
    def __init__(self, messages=()):
        self._messages = list(messages)
        self.items = []

    def get(self):
        return self._messages.pop(0)

    def get_nowait(self):
        if not self._messages:
            raise queue.Empty
        return self._messages.pop(0)

    def put(self, item):
        self.items.append(item)


def _ready(request_id: str, *, num_steps: int = 1) -> PrefixReady:
    return PrefixReady(
        request_id=request_id,
        feature=PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
            state=torch.zeros(1, 2),
        ),
        num_steps=num_steps,
        sample_kwargs={"noise": torch.zeros(1, 2, 1)},
        timing={"vlm_prefix_forward_ms": 1.5},
    )


def _dynamic_cache_ready(request_id: str, fill_value: float) -> PrefixReady:
    cache = DynamicCache()
    cache.update(
        torch.full((1, 1, 3, 4), fill_value),
        torch.full((1, 1, 3, 4), fill_value),
        layer_idx=0,
    )
    return PrefixReady(
        request_id=request_id,
        feature=PrefixFeature(
            past_key_values=cache,
            prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
            state=torch.zeros(1, 2),
        ),
        num_steps=1,
        sample_kwargs={"noise": torch.zeros(1, 2, 1)},
    )


def test_ae_worker_batches_two_ready_requests_for_one_denoise_step():
    model = FakeAEModel()
    worker = AEWorker(model=model, device="cpu", max_batch_size=2)
    worker.add_prefix(_ready("req-1"))
    worker.add_prefix(_ready("req-2"))

    results, releases = worker.step_once()

    assert model.batch_sizes == [2]
    assert [result.request_id for result in results] == ["req-1", "req-2"]
    assert [release.request_id for release in releases] == ["req-1", "req-2"]
    for result in results:
        torch.testing.assert_close(result.actions, -torch.ones(1, 2, 1))
        assert result.timing is not None
        assert result.timing["vlm_prefix_forward_ms"] == 1.5
        assert result.timing["ae_step_ms"] >= 0.0
        assert result.timing["ae_effective_batch"] == 2.0
    assert worker.active == {}


def test_ae_process_records_prefix_transfer_latency():
    prefix_ready = _ready("req-1")
    prefix_ready = dataclasses.replace(
        prefix_ready,
        timing={**(prefix_ready.timing or {}), "_prefix_enqueue_ns": time.monotonic_ns() - 1_000_000},
    )
    process = AEProcess(
        model=FakeAEModel(),
        device="cpu",
        prefix_queue=SimpleQueue([prefix_ready]),
        result_queue=SimpleQueue(),
        release_queue=SimpleQueue(),
        max_batch_size=1,
    )

    process.drain_prefix_ready(block=True)

    request_state = process.worker.active["req-1"]
    assert request_state.timing["prefix_transfer_ms"] >= 0.0
    assert "_prefix_enqueue_ns" not in request_state.timing


def test_ae_worker_batches_huggingface_dynamic_cache_by_batch_dimension():
    model = FakeDynamicCacheAEModel()
    worker = AEWorker(model=model, device="cpu", max_batch_size=2)
    worker.add_prefix(_dynamic_cache_ready("req-1", 1.0))
    worker.add_prefix(_dynamic_cache_ready("req-2", 2.0))

    results, releases = worker.step_once()

    assert [result.request_id for result in results] == ["req-1", "req-2"]
    assert [release.request_id for release in releases] == ["req-1", "req-2"]


def test_ae_worker_does_not_batch_requests_with_different_num_steps():
    model = FakeAEModel()
    worker = AEWorker(model=model, device="cpu", max_batch_size=2)
    worker.add_prefix(_ready("req-1", num_steps=1))
    worker.add_prefix(_ready("req-2", num_steps=2))

    results, releases = worker.step_once()

    assert model.batch_sizes == [1]
    assert [result.request_id for result in results] == ["req-1"]
    assert [release.request_id for release in releases] == ["req-1"]
    assert list(worker.active) == ["req-2"]


def test_ae_process_releases_feature_when_add_prefix_fails():
    prefix_queue = SimpleQueue([_ready("req-1")])
    result_queue = SimpleQueue()
    release_queue = SimpleQueue()
    process = AEProcess(
        model=FailingAddPrefixModel(),
        device="cpu",
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=1,
    )

    process.drain_prefix_ready(block=True)

    assert isinstance(result_queue.items[0], WorkerError)
    assert result_queue.items[0].request_id == "req-1"
    assert release_queue.items[0].request_id == "req-1"


def test_ae_process_releases_active_features_when_step_fails():
    prefix_queue = SimpleQueue()
    result_queue = SimpleQueue()
    release_queue = SimpleQueue()
    process = AEProcess(
        model=FailingStepModel(),
        device="cpu",
        prefix_queue=prefix_queue,
        result_queue=result_queue,
        release_queue=release_queue,
        max_batch_size=2,
    )
    process.worker.add_prefix(_ready("req-1"))
    process.worker.add_prefix(_ready("req-2"))

    process.step_active_once()

    assert [item.request_id for item in result_queue.items] == ["req-1", "req-2"]
    assert all(isinstance(item, WorkerError) for item in result_queue.items)
    assert [item.request_id for item in release_queue.items] == ["req-1", "req-2"]
    assert process.worker.active == {}
