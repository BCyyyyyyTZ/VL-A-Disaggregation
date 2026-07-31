from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool


def _feature(fill: float) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=(
            jnp.full((1, 3, 2, 4), fill, dtype=jnp.bfloat16),
            jnp.full((1, 3, 2, 4), fill + 1, dtype=jnp.bfloat16),
        ),
        prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.full((1, 8), fill, dtype=jnp.float32),
    )


def test_vlm_owned_prefix_cache_lane_pool_exports_dense_batch():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    batch = pool.export_batch_view(("req-1", "req-2"))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][0]), np.full((3, 2, 4), 1.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][1]), np.full((3, 2, 4), 2.0, dtype=np.float32))
    assert batch.prefix_pad_masks.shape == (2, 3)
    assert batch.state.shape == (2, 8)


def test_vlm_owned_prefix_cache_lane_pool_compacts_on_release():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    moved = pool.release_lane("req-1")
    assert moved == (1, 0, "req-2")
    batch = pool.export_batch_view(("req-2",))
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 2.0, dtype=np.float32))


def test_vlm_owned_prefix_cache_lane_pool_rejects_sparse_export():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    with pytest.raises(ValueError, match="dense prefix"):
        pool.export_batch_view(("req-2",))
