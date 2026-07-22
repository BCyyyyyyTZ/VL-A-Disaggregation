from types import SimpleNamespace

import torch

from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def _make_torch_observation(config: pi0_config.Pi0Config, *, batch_size: int = 1) -> SimpleNamespace:
    image = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32)
    images = {
        "base_0_rgb": image,
        "left_wrist_0_rgb": image.clone(),
        "right_wrist_0_rgb": image.clone(),
    }
    image_masks = {key: torch.ones(batch_size, dtype=torch.bool) for key in images}
    return SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(batch_size, config.action_dim, dtype=torch.float32),
        tokenized_prompt=torch.ones(batch_size, config.max_token_len, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch_size, config.max_token_len, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def test_split_helpers_match_sample_actions_for_pi05_dummy_model():
    torch.manual_seed(0)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        dtype="float32",
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config).eval()
    hidden_size = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    model.paligemma_with_expert.embed_image = lambda image: torch.zeros(  # type: ignore[method-assign]
        image.shape[0],
        1,
        hidden_size,
        dtype=torch.float32,
        device=image.device,
    )
    observation = _make_torch_observation(config)
    noise = torch.randn(1, config.action_horizon, config.action_dim)

    mono = model.sample_actions("cpu", observation, noise=noise.clone(), num_steps=4)

    prefix = model.build_prefix_feature("cpu", observation)
    denoise_state = model.init_denoise_state("cpu", batch_size=1, noise=noise.clone(), num_steps=4)
    for _ in range(4):
        v_t = model.denoise_one_batch(prefix, denoise_state)
        denoise_state.x_t = denoise_state.x_t + denoise_state.dt * v_t
        denoise_state.step_idx += 1

    torch.testing.assert_close(denoise_state.x_t, mono, rtol=1e-4, atol=1e-4)
