from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import maybe_jit_monolithic_model
from openpi.serving.va_split_jax.compile import maybe_jit_split_model
from openpi.serving.va_split_jax.compile import planned_warmup_batches
from openpi.serving.va_split_jax.compile import warmup_monolithic_model
from openpi.serving.va_split_jax.compile import warmup_split_model


def test_planned_warmup_batches_clamps_to_prefix_capacity():
    assert planned_warmup_batches(max_batch_size=24, warmup_max_batch_size=32) == (
        (1, 2),
        (4, 1),
        (8, 2),
        (16, 1),
        (20, 2),
        (24, 1),
    )


def test_compile_config_defaults_to_enabled():
    config = JaxCompileConfig()
    assert config.enabled is True
    assert config.warmup_enabled is True
    assert config.warmup_max_batch_size == 32


def test_compile_helpers_use_module_jit(monkeypatch):
    calls = []

    def fake_module_jit(method, *args, **kwargs):
        calls.append((method.__name__, args, kwargs))
        return method

    class FakeModel:
        def build_prefix_feature(self):
            return None

        def denoise_one_batch(self):
            return None

        def sample_actions(self):
            return None

    monkeypatch.setattr("openpi.serving.va_split_jax.compile.nnx_utils.module_jit", fake_module_jit)
    maybe_jit_split_model(FakeModel(), JaxCompileConfig(enabled=True))
    maybe_jit_monolithic_model(FakeModel(), JaxCompileConfig(enabled=True))

    assert [call[0] for call in calls] == ["build_prefix_feature", "denoise_one_batch", "sample_actions"]
    assert calls[-1][2] == {"static_argnames": ("num_steps",)}


class WarmupFakeModel:
    def __init__(self):
        self.split_batches: list[int] = []
        self.mono_batches: list[int] = []

    def build_prefix_feature(self, rng, observation):
        del rng
        batch = int(observation.state.shape[0])
        self.split_batches.append(batch)
        return JaxPrefixFeature(
            past_key_values=(jnp.ones((batch, 3, 2, 4), dtype=jnp.float32),),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=observation.state,
        )

    def init_denoise_state(self, rng, batch_size, noise, num_steps):
        del rng
        return JaxDenoiseState(
            x_t=noise,
            step_idx=jnp.asarray(0, dtype=jnp.int32),
            num_steps=num_steps,
            dt=jnp.asarray(-1.0 / num_steps, dtype=jnp.float32),
        )

    def denoise_one_batch(self, prefix_batch, denoise_batch):
        return jnp.ones_like(denoise_batch.x_t)

    def sample_actions(self, rng, observation, *, noise, num_steps):
        del rng, num_steps
        self.mono_batches.append(int(observation.state.shape[0]))
        return noise


class Observation:
    def __init__(self, batch_size: int):
        self.state = jnp.zeros((batch_size, 8), dtype=jnp.float32)


def test_warmup_split_model_clamps_to_prefix_slots():
    model = WarmupFakeModel()

    stats = warmup_split_model(
        model=model,
        observation_factory=Observation,
        noise_factory=lambda batch: jnp.zeros((batch, 2, 1), dtype=jnp.float32),
        max_vlm_batch_size=8,
        max_ae_batch_size=8,
        max_prefix_slots=4,
        num_steps=2,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8),
    )

    assert stats == {"jax_warmup_batches": 3.0}
    assert model.split_batches == [1, 1, 4]


def test_warmup_monolithic_model_uses_planned_batches():
    model = WarmupFakeModel()

    stats = warmup_monolithic_model(
        model=model,
        observation_factory=Observation,
        noise_factory=lambda batch: jnp.zeros((batch, 2, 1), dtype=jnp.float32),
        max_batch_size=4,
        num_steps=2,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8),
    )

    assert stats == {"jax_warmup_batches": 3.0}
    assert model.mono_batches == [1, 1, 4]
