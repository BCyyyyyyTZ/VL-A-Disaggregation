from __future__ import annotations

import queue
import time
import traceback
from typing import Any

import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.timing import synchronize_cuda_if_needed
from openpi.serving.va_split.types import PrefixReady
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.types import Shutdown
from openpi.serving.va_split.types import WorkerError


class VLMWorker:
    """Builds prefix features and keeps producer-side tensor references alive."""

    def __init__(self, model: Any, device: str):
        self._model = model
        self._device = device
        self.live_features: dict[str, PrefixFeature] = {}

    def handle_request(self, request: RequestEnvelope) -> PrefixReady:
        synchronize_cuda_if_needed(self._device)
        start_ns = time.monotonic_ns()
        observation = _model.Observation.from_dict(_move_tensors_to_device(dict(request.observation), self._device))
        feature = self._model.build_prefix_feature(self._device, observation)
        synchronize_cuda_if_needed(self._device)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        self.live_features[request.request_id] = feature
        sample_kwargs = dict(request.sample_kwargs)
        num_steps = int(sample_kwargs.get("num_steps", 10))
        return PrefixReady(
            request_id=request.request_id,
            feature=feature,
            num_steps=num_steps,
            sample_kwargs=sample_kwargs,
            timing={"vlm_prefix_forward_ms": elapsed_ms},
        )

    def release(self, release: ReleaseFeature) -> None:
        self.live_features.pop(release.request_id, None)


class VLMProcess:
    """Queue loop for the VLM worker.

    This class is intentionally small so it can run either in a child process or
    in-process tests. The first implementation forwards per-request CUDA tensors
    through PyTorch multiprocessing, relying on producer-side `live_features`.
    """

    def __init__(
        self,
        *,
        model: Any,
        device: str,
        request_queue,
        prefix_queue,
        release_queue,
    ):
        self.worker = VLMWorker(model=model, device=device)
        self._request_queue = request_queue
        self._prefix_queue = prefix_queue
        self._release_queue = release_queue

    def run(self) -> None:
        while True:
            self._drain_releases()
            try:
                message = self._request_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            if isinstance(message, Shutdown):
                self._prefix_queue.put(message)
                return
            if not isinstance(message, RequestEnvelope):
                self._prefix_queue.put(WorkerError(request_id=None, error=f"Unexpected VLM message: {type(message)}"))
                continue
            try:
                self._prefix_queue.put(self.worker.handle_request(message))
            except Exception as exc:  # pragma: no cover - exercised through integration/runtime failures.
                self._prefix_queue.put(
                    WorkerError(
                        request_id=message.request_id,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )

    def _drain_releases(self) -> None:
        while True:
            try:
                message = self._release_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, ReleaseFeature):
                self.worker.release(message)


def _move_tensors_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value
