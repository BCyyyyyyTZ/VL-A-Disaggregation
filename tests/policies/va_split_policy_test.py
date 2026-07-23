from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.policies import va_split_policy
from openpi.policies.va_split_policy import VASplitPolicy
from openpi.serving.va_split.types import ActionResult


class FakeRuntime:
    def __init__(self):
        self.sample_kwargs = None
        self.shutdown_called = False

    def infer(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        self.sample_kwargs = sample_kwargs
        assert isinstance(observation["state"], torch.Tensor)
        assert observation["state"].device.type == "cpu"
        return ActionResult(
            request_id="req-1",
            actions=torch.ones(1, 2, 3),
            timing={"runtime_ms": 2.0},
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeBatchRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.batch_calls = 0
        self.batch_observation = None

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        self.batch_calls += 1
        self.sample_kwargs = sample_kwargs
        self.batch_observation = observation
        assert isinstance(observation["state"], torch.Tensor)
        return ActionResult(
            request_id="batch-1",
            actions=torch.zeros(observation["state"].shape[0], 2, 3),
            timing={"vlm_effective_batch": float(observation["state"].shape[0]), "ae_effective_batch_mean": 2.0},
        )


def test_va_split_policy_preserves_infer_output_contract():
    runtime = FakeRuntime()
    policy = VASplitPolicy(
        runtime=runtime,
        transforms=(),
        output_transforms=(),
        sample_kwargs={"num_steps": 4},
        metadata={"model": "fake"},
        pytorch_device="cuda",
    )

    result = policy.infer({"state": np.array([0.25, -0.5], dtype=np.float32)})

    assert result["actions"].shape == (2, 3)
    np.testing.assert_allclose(result["actions"], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(result["state"], np.array([0.25, -0.5], dtype=np.float32))
    assert result["policy_timing"]["runtime_ms"] == 2.0
    assert result["policy_timing"]["infer_ms"] >= 0.0
    assert runtime.sample_kwargs == {"num_steps": 4}
    assert policy.metadata == {"model": "fake"}


def test_va_split_policy_infer_batch_uses_runtime_batch_once():
    runtime = FakeBatchRuntime()
    policy = VASplitPolicy(
        runtime=runtime,
        transforms=(),
        output_transforms=(),
        sample_kwargs={"num_steps": 4},
    )

    result = policy.infer_batch(
        {"state": np.asarray([[0.25, -0.5], [0.5, -1.0]], dtype=np.float32)},
        noise=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert runtime.batch_calls == 1
    assert runtime.batch_observation["state"].shape == (2, 2)
    assert runtime.sample_kwargs["noise"].shape == (2, 2, 3)
    assert result["actions"].shape == (2, 2, 3)
    assert result["policy_timing"]["effective_batch"] == 2
    assert result["policy_timing"]["vlm_effective_batch"] == 2.0
    assert result["policy_timing"]["ae_effective_batch_mean"] == 2.0


def test_va_split_policy_shutdown_delegates_to_runtime():
    runtime = FakeRuntime()
    policy = VASplitPolicy(runtime=runtime)

    policy.shutdown()

    assert runtime.shutdown_called is True


@pytest.mark.parametrize(
    ("compile_mode", "expected_compilations"),
    [
        ("max-autotune", [("build_prefix_feature", "max-autotune", True), ("denoise_one_batch", "max-autotune", True)]),
        (None, []),
    ],
)
def test_load_pytorch_model_matches_monolithic_compile_mode(monkeypatch, compile_mode, expected_compilations):
    class FakePaliGemma:
        def to_bfloat16_for_selected_params(self, dtype):
            assert dtype == "bfloat16"

    class FakeModel:
        def __init__(self):
            self.paligemma_with_expert = FakePaliGemma()

        def build_prefix_feature(self, device, observation):
            return device, observation

        def denoise_one_batch(self, prefix_batch, denoise_batch):
            return prefix_batch, denoise_batch

    model = FakeModel()
    train_config = SimpleNamespace(
        model=SimpleNamespace(
            pytorch_compile_mode=compile_mode,
            load_pytorch=lambda config, weight_path: model,
        )
    )
    compilations = []

    def compile_spy(function, *, mode, dynamic):
        compilations.append((function.__name__, mode, dynamic))
        return function

    monkeypatch.setattr(va_split_policy.torch, "compile", compile_spy)

    assert va_split_policy._load_pytorch_model(train_config, "/tmp/model.safetensors") is model  # noqa: SLF001
    assert compilations == expected_compilations
