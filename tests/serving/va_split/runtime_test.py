# ruff: noqa: SLF001

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.runtime import LocalVASplitRuntime
from openpi.serving.va_split.runtime import ProcessVASplitRuntime
from openpi.serving.va_split.types import ActionResult
from openpi.serving.va_split.types import Shutdown


class FakeSplitModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)
        self.prefix_batch_sizes: list[int] = []

    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
        self.prefix_batch_sizes.append(int(observation.state.shape[0]))
        return PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(observation.state.shape[0], 3, dtype=torch.bool),
            state=observation.state,
        )

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
        assert prefix_batch.prefix_pad_masks.shape[0] == denoise_batch.x_t.shape[0]
        return torch.ones_like(denoise_batch.x_t)


def test_local_va_split_runtime_runs_vlm_then_ae_and_releases_prefix():
    runtime = LocalVASplitRuntime(model=FakeSplitModel(), device="cpu", max_ae_batch_size=2)
    image = torch.zeros(1, 3, 224, 224)
    observation = {
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
        "state": torch.zeros(1, 2),
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    result = runtime.infer(observation, {"num_steps": 1, "noise": torch.zeros(1, 2, 1)})

    assert result.request_id
    torch.testing.assert_close(result.actions, -torch.ones(1, 2, 1))
    assert runtime.vlm_worker.live_features == {}


def test_local_va_split_runtime_infer_batch_builds_prefix_once_and_returns_row_order():
    model = FakeSplitModel()
    runtime = LocalVASplitRuntime(model=model, device="cpu", max_ae_batch_size=2)
    image = torch.zeros(2, 3, 224, 224)
    observation = {
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image.clone(),
            "right_wrist_0_rgb": image.clone(),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
        },
        "state": torch.zeros(2, 2),
        "tokenized_prompt": torch.ones(2, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    result = runtime.infer_batch(observation, {"num_steps": 1, "noise": torch.zeros(2, 2, 1)})

    assert result.request_id
    assert model.prefix_batch_sizes == [2]
    torch.testing.assert_close(result.actions, -torch.ones(2, 2, 1))
    assert result.timing is not None
    assert result.timing["effective_batch"] == 2
    assert result.timing["vlm_effective_batch"] == 2.0
    assert result.timing["ae_effective_batch_mean"] == 2.0
    assert runtime.vlm_worker.live_features == {}


class _ResultQueue:
    def __init__(self, messages):
        self._messages = list(messages)

    def get(self):
        return self._messages.pop(0)


def test_process_runtime_collect_results_records_ae_result_transfer_latency():
    runtime = object.__new__(ProcessVASplitRuntime)
    runtime._result_queue = _ResultQueue(
        [
            ActionResult(
                request_id="req-1",
                actions=torch.zeros(1, 2, 1),
                timing={"_ae_result_enqueue_ns": time.monotonic_ns() - 1_000_000},
            ),
            Shutdown(),
        ]
    )
    runtime._condition = threading.Condition()
    runtime._pending_results = {}
    runtime._pending_errors = {}
    runtime._shutdown_seen = False

    runtime._collect_results()

    timing = runtime._pending_results["req-1"].timing
    assert timing is not None
    assert timing["ae_result_transfer_ms"] >= 0.0
    assert "_ae_result_enqueue_ns" not in timing
