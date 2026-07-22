from __future__ import annotations

import numpy as np
import torch

from openpi.policies.va_split_policy import VASplitPolicy
from openpi.serving.va_split.types import ActionResult


class FakeRuntime:
    def __init__(self):
        self.sample_kwargs = None

    def infer(self, observation: dict, sample_kwargs: dict) -> ActionResult:
        self.sample_kwargs = sample_kwargs
        assert isinstance(observation["state"], torch.Tensor)
        assert observation["state"].device.type == "cpu"
        return ActionResult(
            request_id="req-1",
            actions=torch.ones(1, 2, 3),
            timing={"runtime_ms": 2.0},
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
