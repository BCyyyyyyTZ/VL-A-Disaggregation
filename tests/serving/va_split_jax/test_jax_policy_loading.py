from __future__ import annotations

import jax.numpy as jnp

from openpi.models import pi0_config
from openpi.policies import jax_va_split_policy


def test_pi0_prefix_template_from_config_does_not_require_image_model_weights():
    config = pi0_config.Pi0Config(pi05=True)

    template = jax_va_split_policy._make_prefix_feature_template_for_model_config(config)  # noqa: SLF001

    key_cache, value_cache = template.past_key_values
    assert key_cache.shape == (18, 1, 968, 1, 256)
    assert value_cache.shape == (18, 1, 968, 1, 256)
    assert key_cache.dtype == jnp.bfloat16
    assert template.prefix_pad_masks.shape == (1, 968)
    assert template.state.shape == (1, config.action_dim)
