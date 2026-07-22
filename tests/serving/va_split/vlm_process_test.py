from __future__ import annotations

import torch

from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.types import ReleaseFeature
from openpi.serving.va_split.types import RequestEnvelope
from openpi.serving.va_split.vlm_process import VLMWorker


def _request_observation() -> dict:
    image = torch.zeros(1, 3, 224, 224)
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
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
    }


class FakeVLMModel:
    def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
        assert device == "cpu"
        return PrefixFeature(
            past_key_values=("kv",),
            prefix_pad_masks=torch.ones(observation.state.shape[0], 3, dtype=torch.bool),
            state=observation.state,
        )


def test_vlm_worker_publishes_prefix_and_releases_live_feature():
    worker = VLMWorker(model=FakeVLMModel(), device="cpu")
    request = RequestEnvelope(
        request_id="req-1",
        observation=_request_observation(),
        sample_kwargs={"num_steps": 4},
        enqueue_ns=123,
    )

    ready = worker.handle_request(request)

    assert ready.request_id == "req-1"
    assert ready.num_steps == 4
    assert ready.sample_kwargs == {"num_steps": 4}
    assert "req-1" in worker.live_features
    torch.testing.assert_close(ready.feature.prefix_pad_masks, torch.ones(1, 3, dtype=torch.bool))

    worker.release(ReleaseFeature(request_id="req-1", slot_id=-1))

    assert worker.live_features == {}
