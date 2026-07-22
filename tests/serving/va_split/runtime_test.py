from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.runtime import LocalVASplitRuntime


class FakeSplitModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=1)

    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
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
