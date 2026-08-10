from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.models import pi0_config
from openpi.policies import jax_va_split_policy
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import maybe_jit_ae_model


class _TinyAeModel(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

    def denoise_one_batch(self, prefix_batch: JaxPrefixFeature, denoise_batch: JaxDenoiseState) -> jax.Array:
        del prefix_batch
        return denoise_batch.x_t + self.scale.value


def test_pi0_prefix_template_from_config_does_not_require_image_model_weights():
    config = pi0_config.Pi0Config(pi05=True)

    template = jax_va_split_policy._make_prefix_feature_template_for_model_config(config)  # noqa: SLF001

    key_cache, value_cache = template.past_key_values
    assert key_cache.shape == (18, 1, 968, 1, 256)
    assert value_cache.shape == (18, 1, 968, 1, 256)
    assert key_cache.dtype == jnp.bfloat16
    assert template.prefix_pad_masks.shape == (1, 968)
    assert template.state.shape == (1, config.action_dim)


def test_ae_prefix_template_wrapper_keeps_arrays_out_of_nnx_module_jit():
    model = _TinyAeModel()
    template = JaxPrefixFeature(
        past_key_values=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        prefix_pad_masks=jnp.ones((1, 1), dtype=jnp.bool_),
        state=jnp.zeros((1, 1), dtype=jnp.float32),
    )

    wrapped = jax_va_split_policy._wrap_ae_model_with_prefix_template(model, template)  # noqa: SLF001

    assert wrapped.model is model
    assert wrapped.prefix_template is template
    assert not hasattr(model, "_va_split_prefix_feature_template")
    maybe_jit_ae_model(model, JaxCompileConfig(enabled=True))


def test_multigpu_metadata_treats_missing_policy_metadata_as_empty():
    metadata = jax_va_split_policy._multigpu_policy_metadata(  # noqa: SLF001
        None,
        va_split_runtime="jax-multigpu-split-ipc",
        vlm_devices=("1", "2"),
        ae_device="3",
        cross_card_transfer_strategy="host-staged",
    )

    assert metadata == {
        "va_split_runtime": "jax-multigpu-split-ipc",
        "vlm_devices": ("1", "2"),
        "ae_device": "3",
        "cross_card_transfer_strategy": "host-staged",
    }
