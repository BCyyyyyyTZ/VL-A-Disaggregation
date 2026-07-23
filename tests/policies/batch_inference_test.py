from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.policies import batch_inference


def _raw_obs_batch() -> dict:
    return {
        "observation/state": np.zeros((2, 8), dtype=np.float32),
        "observation/image": np.zeros((2, 224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((2, 224, 224, 3), dtype=np.uint8),
        "prompt": ["task a", "task b"],
    }


def test_infer_obs_batch_size_uses_prompt_batch():
    assert batch_inference.infer_obs_batch_size(_raw_obs_batch()) == 2


def test_infer_obs_batch_size_rejects_string_prompt():
    obs_batch = _raw_obs_batch()
    obs_batch["prompt"] = "task a"

    with pytest.raises(ValueError, match="prompt to be a sequence of strings"):
        batch_inference.infer_obs_batch_size(obs_batch)


def test_apply_input_transform_batch_splits_prompts_and_stacks_torch_tensors():
    seen_prompts: list[str] = []

    def transform(sample: dict) -> dict:
        seen_prompts.append(sample["prompt"])
        return {
            "state": sample["observation/state"],
            "image": {
                "base_0_rgb": sample["observation/image"],
                "left_wrist_0_rgb": sample["observation/wrist_image"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
            "tokenized_prompt": np.full((4,), len(seen_prompts), dtype=np.int32),
            "tokenized_prompt_mask": np.ones((4,), dtype=bool),
        }

    result = batch_inference.apply_input_transform_batch(_raw_obs_batch(), transform, kind="torch", device="cpu")

    assert seen_prompts == ["task a", "task b"]
    assert isinstance(result["state"], torch.Tensor)
    assert result["state"].shape == (2, 8)
    assert result["image"]["base_0_rgb"].shape == (2, 224, 224, 3)
    assert result["image_mask"]["base_0_rgb"].shape == (2,)
    assert result["tokenized_prompt"].shape == (2, 4)
    torch.testing.assert_close(result["tokenized_prompt"][0], torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(result["tokenized_prompt"][1], torch.full((4,), 2, dtype=torch.int32))


def test_apply_output_transform_batch_splits_rows_and_stacks_outputs():
    seen_action_shapes: list[tuple[int, ...]] = []
    outputs = {
        "state": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "actions": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
    }

    def output_transform(sample: dict) -> dict:
        seen_action_shapes.append(sample["actions"].shape)
        return {"actions": sample["actions"] + 1.0, "state": sample["state"]}

    result = batch_inference.apply_output_transform_batch(outputs, output_transform)

    assert seen_action_shapes == [(2, 3), (2, 3)]
    assert result["actions"].shape == (2, 2, 3)
    np.testing.assert_allclose(result["actions"][0], np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0)
    np.testing.assert_allclose(result["state"], np.arange(4, dtype=np.float32).reshape(2, 2))
