from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.models_pytorch.pi0_split_types import DenoiseState
from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.policies import batch_inference
from openpi.policies.policy import Policy


class FakeTorchPolicyModel:
    def __init__(self):
        self.config = SimpleNamespace(action_horizon=2, action_dim=3)
        self.batch_sizes: list[int] = []

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def sample_actions(self, device: str, observation, **kwargs):
        assert device == "cpu"
        self.batch_sizes.append(int(observation.state.shape[0]))
        noise = kwargs.get("noise")
        if noise is None:
            noise = torch.zeros(
                observation.state.shape[0],
                self.config.action_horizon,
                self.config.action_dim,
                dtype=torch.float32,
            )
        state_offset = observation.state[:, :1].reshape(observation.state.shape[0], 1, 1)
        return noise + state_offset


class FakeTimedTorchPolicyModel(FakeTorchPolicyModel):
    def __init__(self):
        super().__init__()
        self.prefix_batch_sizes: list[int] = []
        self.denoise_batch_sizes: list[int] = []

    def sample_actions(self, device: str, observation, **kwargs):
        raise AssertionError("component-timed baseline should use split helper methods")

    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
        self.prefix_batch_sizes.append(int(observation.state.shape[0]))
        return PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(observation.state.shape[0], 1, dtype=torch.bool),
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
        self.denoise_batch_sizes.append(int(denoise_batch.x_t.shape[0]))
        return torch.ones_like(denoise_batch.x_t)


def _model_input_obs_batch() -> dict:
    image = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    return {
        "state": np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image.copy(),
            "right_wrist_0_rgb": image.copy(),
        },
        "image_mask": {
            "base_0_rgb": np.ones((2,), dtype=bool),
            "left_wrist_0_rgb": np.ones((2,), dtype=bool),
            "right_wrist_0_rgb": np.ones((2,), dtype=bool),
        },
        "tokenized_prompt": np.ones((2, 4), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((2, 4), dtype=bool),
    }


def test_policy_infer_batch_calls_pytorch_model_once_and_matches_single_rows():
    model = FakeTorchPolicyModel()
    policy = Policy(model, is_pytorch=True, pytorch_device="cpu")
    obs_batch = _model_input_obs_batch()
    noise_batch = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    batch_out = policy.infer_batch(obs_batch, noise=noise_batch)

    assert model.batch_sizes == [2]
    assert batch_out["actions"].shape == (2, 2, 3)
    assert batch_out["policy_timing"]["effective_batch"] == 2

    row_outputs = [
        policy.infer(sample, noise=noise_batch[row])
        for row, sample in enumerate(batch_inference.split_obs_batch(obs_batch))
    ]
    np.testing.assert_allclose(batch_out["actions"][0], row_outputs[0]["actions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(batch_out["actions"][1], row_outputs[1]["actions"], rtol=1e-4, atol=1e-4)
    assert row_outputs[0]["actions"].shape == (2, 3)


def test_policy_infer_batch_rejects_single_noise_for_multi_row_batch():
    policy = Policy(FakeTorchPolicyModel(), is_pytorch=True, pytorch_device="cpu")

    with pytest.raises(ValueError, match="B > 1 requires noise shape"):
        policy.infer_batch(_model_input_obs_batch(), noise=np.zeros((2, 3), dtype=np.float32))


def test_policy_infer_batch_reports_baseline_vlm_and_ae_component_timing():
    model = FakeTimedTorchPolicyModel()
    policy = Policy(model, is_pytorch=True, pytorch_device="cpu", sample_kwargs={"num_steps": 2})
    noise_batch = np.zeros((2, 2, 3), dtype=np.float32)

    result = policy.infer_batch(_model_input_obs_batch(), noise=noise_batch)

    assert model.prefix_batch_sizes == [2]
    assert model.denoise_batch_sizes == [2, 2]
    np.testing.assert_allclose(result["actions"], -np.ones((2, 2, 3), dtype=np.float32))
    timing = result["policy_timing"]
    assert timing["baseline_vlm_ms"] >= 0.0
    assert timing["baseline_ae_ms"] >= 0.0
    assert timing["baseline_ae_step_ms"] >= 0.0
    assert timing["baseline_ae_steps"] == 2
    assert timing["baseline_effective_batch"] == 2
