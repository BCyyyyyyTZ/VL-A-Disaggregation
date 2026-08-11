from __future__ import annotations

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import pytest

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.models.pi0_config import Pi0Config
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import make_pi0_prefix_feature_template
from openpi.serving.va_split_jax.compile import make_prefix_feature_template
from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
from openpi.serving.va_split_jax.compile import maybe_jit_monolithic_model
from openpi.serving.va_split_jax.compile import maybe_jit_split_model
from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
from openpi.serving.va_split_jax.compile import planned_warmup_batches
from openpi.serving.va_split_jax.compile import prune_split_model_for_role
from openpi.serving.va_split_jax.compile import runtime_aligned_denoise_state
from openpi.serving.va_split_jax.compile import warmup_ae_denoise_model
from openpi.serving.va_split_jax.compile import warmup_ae_denoise_on_mapped_slabs
from openpi.serving.va_split_jax.compile import warmup_monolithic_model
from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_lane_pool
from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
from openpi.training.config import get_config


def test_planned_warmup_batches_covers_every_size_up_to_cap():
    assert planned_warmup_batches(max_batch_size=24, warmup_max_batch_size=32) == (
        (1, 2),
        *((batch_size, 1) for batch_size in range(2, 25)),
    )
    assert planned_warmup_batches(max_batch_size=8, warmup_max_batch_size=24) == (
        (1, 2),
        *((batch_size, 1) for batch_size in range(2, 9)),
    )


def test_compile_config_defaults_to_enabled():
    config = JaxCompileConfig()
    assert config.enabled is True
    assert config.warmup_enabled is True
    assert config.compile_ae is True
    assert config.compile_vlm is True
    assert config.warmup_max_batch_size == 20
    assert config.num_steps == 10


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
    maybe_jit_vlm_model(FakeModel(), JaxCompileConfig(enabled=True))
    maybe_jit_ae_model(FakeModel(), JaxCompileConfig(enabled=True))
    maybe_jit_monolithic_model(FakeModel(), JaxCompileConfig(enabled=True))

    assert [call[0] for call in calls] == [
        "build_prefix_feature",
        "denoise_one_batch",
        "build_prefix_feature",
        "denoise_one_batch",
        "build_prefix_feature",
        "denoise_one_batch",
    ]


def test_compile_config_can_disable_vlm_jit_and_warmup(monkeypatch):
    calls = []

    def fake_module_jit(method, *args, **kwargs):
        del args, kwargs
        calls.append(method.__name__)
        return method

    monkeypatch.setattr("openpi.serving.va_split_jax.compile.nnx_utils.module_jit", fake_module_jit)
    model = WarmupFakeModel()
    config = JaxCompileConfig(enabled=True, warmup_enabled=True, compile_vlm=False)

    maybe_jit_vlm_model(model, config)
    stats = warmup_vlm_prefix_model(
        model=model,
        observation_factory=Observation,
        max_vlm_batch_size=4,
        config=config,
    )

    assert calls == []
    assert stats == {"jax_warmup_batches": 0.0}
    assert model.prefix_batches == []


def test_prune_split_model_for_role_removes_only_role_unused_top_level_modules():
    class PaliGemma:
        def __init__(self):
            self.llm = object()
            self.img = object()

    class FakeModel:
        def __init__(self):
            self.PaliGemma = PaliGemma()
            self.action_in_proj = object()
            self.action_out_proj = object()
            self.time_mlp_in = object()
            self.time_mlp_out = object()

    vlm_model = FakeModel()
    prune_split_model_for_role(vlm_model, role="vlm")
    assert hasattr(vlm_model.PaliGemma, "img")
    assert hasattr(vlm_model.PaliGemma, "llm")
    assert not hasattr(vlm_model, "action_in_proj")
    assert not hasattr(vlm_model, "action_out_proj")

    ae_model = FakeModel()
    prune_split_model_for_role(ae_model, role="ae")
    assert not hasattr(ae_model.PaliGemma, "img")
    assert hasattr(ae_model.PaliGemma, "llm")
    assert hasattr(ae_model, "action_in_proj")


def test_prune_split_model_for_role_removes_action_expert_from_vlm_llm_state():
    model = _dummy_pi0_model()

    prune_split_model_for_role(model, role="vlm")

    paths = _state_paths(model)
    assert "PaliGemma/llm/embedder/input_embedding" in paths
    assert "PaliGemma/img/embedding/kernel" in paths
    assert "PaliGemma/llm/layers/attn/q_einsum/w" in paths
    assert "PaliGemma/llm/layers/attn/q_einsum_1/w" not in paths
    assert "PaliGemma/llm/layers/attn/kv_einsum_1/w" not in paths
    assert "PaliGemma/llm/layers/mlp_1/gating_einsum" not in paths
    assert "PaliGemma/llm/final_norm_1/scale" not in paths
    assert not any(path.startswith("action_in_proj/") for path in paths)
    assert not any(path.startswith("action_out_proj/") for path in paths)


def test_prune_split_model_for_role_removes_paligemma_expert_from_ae_llm_state():
    model = _dummy_pi0_model()

    prune_split_model_for_role(model, role="ae")

    paths = _state_paths(model)
    assert not any(path.startswith("PaliGemma/img/") for path in paths)
    assert "PaliGemma/llm/embedder/input_embedding" not in paths
    assert "PaliGemma/llm/layers/attn/q_einsum/w" not in paths
    assert "PaliGemma/llm/layers/attn/kv_einsum/w" not in paths
    assert "PaliGemma/llm/layers/mlp/gating_einsum" not in paths
    assert "PaliGemma/llm/final_norm/scale" not in paths
    assert "PaliGemma/llm/layers/attn/q_einsum_1/w" in paths
    assert "PaliGemma/llm/layers/attn/kv_einsum_1/w" in paths
    assert "PaliGemma/llm/layers/mlp_1/gating_einsum" in paths
    assert "PaliGemma/llm/final_norm_1/scale" in paths
    assert any(path.startswith("action_in_proj/") for path in paths)
    assert any(path.startswith("action_out_proj/") for path in paths)


def _dummy_pi0_model():
    config = Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    return nnx.eval_shape(config.create, jax.random.key(0))


def _state_paths(model) -> set[str]:
    _, state = nnx.split(model)
    return {"/".join(str(part) for part in key_path) for key_path in traverse_util.flatten_dict(state.to_pure_dict())}


def test_runtime_aligned_denoise_state_uses_batch_vectors():
    state = runtime_aligned_denoise_state(
        batch_size=4,
        noise=jnp.zeros((4, 2, 1), dtype=jnp.float32),
        num_steps=5,
    )
    assert state.step_idx.shape == (4,)
    assert state.dt.shape == (4,)
    assert int(state.step_idx[0]) == 0
    assert float(state.dt[0]) == pytest.approx(-0.2)


class WarmupFakeModel:
    def __init__(self):
        self.action_horizon = 2
        self.action_dim = 1
        self.prefix_batches: list[int] = []
        self.denoise_step_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.mono_batches: list[int] = []
        self.seen_prefix_from_slab = False

    def build_prefix_feature(self, rng, observation):
        del rng
        batch = int(observation.state.shape[0])
        self.prefix_batches.append(batch)
        return JaxPrefixFeature(
            past_key_values=(jnp.ones((3, batch, 2, 4), dtype=jnp.float32),),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=observation.state,
        )

    def init_denoise_state(self, rng, batch_size, noise, num_steps):
        del rng
        return runtime_aligned_denoise_state(batch_size=batch_size, noise=noise, num_steps=num_steps)

    def denoise_one_batch(self, prefix_batch, denoise_batch):
        self.denoise_step_shapes.append((tuple(denoise_batch.step_idx.shape), tuple(denoise_batch.dt.shape)))
        if prefix_batch.prefix_pad_masks.shape[0] == denoise_batch.x_t.shape[0]:
            self.seen_prefix_from_slab = True
        return jnp.ones_like(denoise_batch.x_t)

    def sample_actions(self, rng, observation, *, noise, num_steps):
        del rng, num_steps
        self.mono_batches.append(int(observation.state.shape[0]))
        return noise


class Observation:
    def __init__(self, batch_size: int):
        self.state = jnp.zeros((batch_size, 8), dtype=jnp.float32)


def test_warmup_vlm_prefix_model_covers_all_sizes_up_to_vlm_cap():
    model = WarmupFakeModel()

    stats = warmup_vlm_prefix_model(
        model=model,
        observation_factory=Observation,
        max_vlm_batch_size=4,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8),
    )

    assert stats == {"jax_warmup_batches": 5.0}
    assert model.prefix_batches == [1, 1, 2, 3, 4]


def test_make_prefix_feature_template_uses_shapes_without_real_prefix_values():
    model = WarmupFakeModel()

    template = make_prefix_feature_template(model, Observation)

    assert template.past_key_values[0].shape == (3, 1, 2, 4)
    assert template.past_key_values[0].dtype == jnp.float32
    assert template.prefix_pad_masks.shape == (1, 3)
    assert template.prefix_pad_masks.dtype == jnp.bool_
    assert template.state.shape == (1, 8)
    assert jnp.all(template.past_key_values[0] == 0)
    assert jnp.all(template.prefix_pad_masks)


def test_make_pi0_prefix_feature_template_builds_pi05_libero_shapes_without_prefix_forward():
    config = get_config("pi05_libero").model

    template = make_pi0_prefix_feature_template(config)

    assert template.past_key_values[0].shape == (18, 1, 968, 1, 256)
    assert template.past_key_values[1].shape == (18, 1, 968, 1, 256)
    assert template.past_key_values[0].dtype == jnp.bfloat16
    assert template.past_key_values[1].dtype == jnp.bfloat16
    assert template.prefix_pad_masks.shape == (1, 968)
    assert template.prefix_pad_masks.dtype == jnp.bool_
    assert template.state.shape == (1, 32)
    assert template.state.dtype == jnp.float32
    assert jnp.all(template.prefix_pad_masks)


def test_warmup_vlm_prefix_lane_pool_puts_views_and_releases():
    model = WarmupFakeModel()
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.initialize_from_feature(
        JaxPrefixFeature(
            past_key_values=(jnp.zeros((3, 1, 2, 4), dtype=jnp.float32),),
            prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
            state=jnp.zeros((1, 8), dtype=jnp.float32),
        )
    )

    stats = warmup_vlm_prefix_lane_pool(
        model=model,
        observation_factory=Observation,
        prefix_pool=pool,
        max_vlm_batch_size=4,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8),
    )

    assert stats == {"jax_warmup_batches": 5.0}
    assert model.prefix_batches == [1, 1, 2, 3, 4]
    assert pool.active_count == 0


def test_warmup_ae_denoise_model_uses_vector_step_dt_and_direct_prefix():
    model = WarmupFakeModel()

    stats = warmup_ae_denoise_model(
        model=model,
        observation_factory=Observation,
        noise_factory=lambda batch: jnp.zeros((batch, 2, 1), dtype=jnp.float32),
        max_ae_batch_size=2,
        max_prefix_slots=4,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8, num_steps=2),
    )

    assert stats == {"jax_warmup_batches": 5.0}
    assert model.prefix_batches == [1, 1, 2, 3, 4]
    assert model.denoise_step_shapes == [
        ((1,), (1,)),
        ((1,), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
        ((4,), (4,)),
    ]


def test_warmup_ae_denoise_on_mapped_slabs_uses_make_prefix_batch():
    model = WarmupFakeModel()
    seen_slots: list[tuple[int, ...]] = []

    def make_prefix_batch(slot_ids):
        seen_slots.append(tuple(slot_ids))
        batch = len(slot_ids)
        return JaxPrefixFeature(
            past_key_values=(jnp.ones((3, batch, 2, 4), dtype=jnp.float32),),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=jnp.zeros((batch, 8), dtype=jnp.float32),
        )

    stats = warmup_ae_denoise_on_mapped_slabs(
        model=model,
        noise_factory=lambda batch: jnp.zeros((batch, 2, 1), dtype=jnp.float32),
        max_ae_batch_size=2,
        max_prefix_slots=4,
        config=JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=8, num_steps=2),
        make_prefix_batch=make_prefix_batch,
    )

    assert stats == {"jax_warmup_batches": 5.0}
    # Each planned batch runs warmup_steps(=min(num_steps,5)=2) denoise calls.
    assert seen_slots == [(0,), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
    assert len(model.denoise_step_shapes) == 10
    assert model.denoise_step_shapes[0] == ((1,), (1,))
    assert model.denoise_step_shapes[-1] == ((4,), (4,))


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

    # Two small graphs: VLM prefix sizes + AE denoise sizes (each [1]+[1..4] with the
    # first size repeated once by planned_warmup_batches).
    assert stats == {"jax_warmup_batches": 10.0}
    assert model.prefix_batches == [1, 1, 2, 3, 4, 1, 1, 2, 3, 4]
    assert model.mono_batches == []
