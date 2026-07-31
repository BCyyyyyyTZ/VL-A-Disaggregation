from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config


def _make_observation(config: pi0_config.Pi0Config, *, batch_size: int = 1):
    return config.fake_obs(batch_size=batch_size)


def test_pi05_jax_split_helpers_match_sample_actions_dummy_model():
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
    model = config.create(jax.random.key(0))
    obs = _make_observation(config)
    noise = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    mono = model.sample_actions(jax.random.key(1), obs, noise=noise, num_steps=4)
    prefix = model.build_prefix_feature(None, obs)
    state = model.init_denoise_state(jax.random.key(1), batch_size=1, noise=noise, num_steps=4)
    for _ in range(4):
        v_t = model.denoise_one_batch(prefix, state)
        state = type(state)(
            x_t=state.x_t + state.dt * v_t,
            step_idx=state.step_idx + 1,
            num_steps=state.num_steps,
            dt=state.dt,
        )
    np.testing.assert_allclose(np.asarray(state.x_t), np.asarray(mono), rtol=1e-4, atol=1e-4)


def test_pi0_jax_split_helpers_smoke_dummy_model():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=False,
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        dtype="float32",
        pytorch_compile_mode=None,
    )
    model = config.create(jax.random.key(0))
    obs = _make_observation(config)
    noise = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    prefix = model.build_prefix_feature(None, obs)
    state = model.init_denoise_state(jax.random.key(1), batch_size=1, noise=noise, num_steps=2)
    v_t = model.denoise_one_batch(prefix, state)
    assert v_t.shape == noise.shape
