from __future__ import annotations

from dataclasses import dataclass
import queue
import time
import traceback
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError


@dataclass
class AERequestState:
    request_id: str
    feature: PrefixFeature
    x_t: torch.Tensor
    step_idx: int
    num_steps: int
    dt: torch.Tensor
    started_ns: int


def _cat_tree(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, DynamicCache):
        batched_cache = DynamicCache()
        for layer_idx in range(len(first)):
            keys = []
            values_ = []
            for cache in values:
                key, value = cache[layer_idx]
                keys.append(key)
                values_.append(value)
            batched_cache.update(torch.cat(keys, dim=0), torch.cat(values_, dim=0), layer_idx=layer_idx)
        return batched_cache
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, tuple):
        return tuple(_cat_tree([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, list):
        return [_cat_tree([value[index] for value in values]) for index in range(len(first))]
    return first


def _batch_prefix_features(features: list[PrefixFeature]) -> PrefixFeature:
    # Basic tensor-sharing path: assemble per-request caches for each AE step.
    # The slab/lane-pool version should replace this with dense views.
    state = None
    if all(feature.state is not None for feature in features):
        state = torch.cat([feature.state for feature in features if feature.state is not None], dim=0)
    return PrefixFeature(
        past_key_values=_cat_tree([feature.past_key_values for feature in features]),
        prefix_pad_masks=torch.cat([feature.prefix_pad_masks for feature in features], dim=0),
        state=state,
    )


class AEWorker:
    """Runs step-level continuous batching over active AE requests."""

    def __init__(self, model: Any, device: str, max_batch_size: int):
        self._model = model
        self._device = device
        self._max_batch_size = max_batch_size
        self.active: dict[str, AERequestState] = {}

    def add_prefix(self, ready: PrefixReady) -> None:
        sample_kwargs = dict(ready.sample_kwargs)
        noise = sample_kwargs.get("noise")
        if torch.is_tensor(noise):
            noise = noise.to(self._device)
        batch_size = ready.feature.prefix_pad_masks.shape[0]
        denoise_state = self._model.init_denoise_state(self._device, batch_size, noise, ready.num_steps)
        self.active[ready.request_id] = AERequestState(
            request_id=ready.request_id,
            feature=ready.feature,
            x_t=denoise_state.x_t,
            step_idx=int(denoise_state.step_idx.item()),
            num_steps=ready.num_steps,
            dt=denoise_state.dt,
            started_ns=time.monotonic_ns(),
        )

    def select_ready_lanes(self) -> list[AERequestState]:
        return sorted(self.active.values(), key=lambda req: (req.step_idx, req.started_ns))[: self._max_batch_size]

    def step_once(self) -> tuple[list[ActionResult], list[ReleaseFeature]]:
        batch = self.select_ready_lanes()
        if not batch:
            return [], []

        prefix_batch = _batch_prefix_features([request.feature for request in batch])
        x_t = torch.cat([request.x_t for request in batch], dim=0)
        step_idx = torch.tensor([request.step_idx for request in batch], device=self._device, dtype=torch.int32)
        dt = torch.stack([request.dt.to(self._device) for request in batch])
        denoise_batch = DenoiseState(x_t=x_t, step_idx=step_idx, num_steps=batch[0].num_steps, dt=dt)
        v_t = self._model.denoise_one_batch(prefix_batch, denoise_batch)

        results: list[ActionResult] = []
        releases: list[ReleaseFeature] = []
        for row, request in enumerate(batch):
            request.x_t = request.x_t + request.dt * v_t[row : row + 1]
            request.step_idx += 1
            if request.step_idx == request.num_steps:
                results.append(ActionResult(request_id=request.request_id, actions=request.x_t))
                releases.append(ReleaseFeature(request_id=request.request_id, slot_id=-1))
                del self.active[request.request_id]
        return results, releases


class AEProcess:
    """Queue loop for step-level AE continuous batching."""

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        prefix_queue,
        result_queue,
        release_queue,
        max_batch_size: int,
    ):
        self.worker = AEWorker(model=model, device=device, max_batch_size=max_batch_size)
        self._prefix_queue = prefix_queue
        self._result_queue = result_queue
        self._release_queue = release_queue

    def run(self) -> None:
        while True:
            self.drain_prefix_ready(block=not self.worker.active)
            if self.worker.active:
                self.step_active_once()

    def step_active_once(self) -> None:
        try:
            results, releases = self.worker.step_once()
        except Exception as exc:  # pragma: no cover - real model failures are surfaced through queues.
            self._fail_active_requests(error=str(exc), traceback_text=traceback.format_exc())
            return

        for result in results:
            actions = result.actions.detach().cpu() if torch.is_tensor(result.actions) else result.actions
            self._result_queue.put(ActionResult(request_id=result.request_id, actions=actions, timing=result.timing))
        for release in releases:
            self._release_queue.put(release)

    def drain_prefix_ready(self, *, block: bool) -> None:
        while True:
            try:
                message = self._prefix_queue.get() if block else self._prefix_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, Shutdown):
                self._result_queue.put(message)
                raise SystemExit
            if isinstance(message, WorkerError):
                self._result_queue.put(message)
                continue
            if not isinstance(message, PrefixReady):
                self._result_queue.put(WorkerError(request_id=None, error=f"Unexpected AE message: {type(message)}"))
                continue
            try:
                self.worker.add_prefix(message)
            except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                self._result_queue.put(
                    WorkerError(
                        request_id=message.request_id,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                self._release_queue.put(ReleaseFeature(request_id=message.request_id, slot_id=message.slot_id))
            if block:
                block = False

    def _fail_active_requests(self, *, error: str, traceback_text: str) -> None:
        for request_id in list(self.worker.active):
            self._result_queue.put(WorkerError(request_id=request_id, error=error, traceback=traceback_text))
            self._release_queue.put(ReleaseFeature(request_id=request_id, slot_id=-1))
            del self.worker.active[request_id]
