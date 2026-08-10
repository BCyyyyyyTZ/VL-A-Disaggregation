from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool


def _feature(fill: float) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=(
            jnp.full((3, 1, 2, 4), fill, dtype=jnp.bfloat16),
            jnp.full((3, 1, 2, 4), fill + 1, dtype=jnp.bfloat16),
        ),
        prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.full((1, 8), fill, dtype=jnp.float32),
    )


def _batch_feature(*fills: float) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=(
            jnp.concatenate([jnp.full((3, 1, 2, 4), fill, dtype=jnp.bfloat16) for fill in fills], axis=1),
            jnp.concatenate([jnp.full((3, 1, 2, 4), fill + 1, dtype=jnp.bfloat16) for fill in fills], axis=1),
        ),
        prefix_pad_masks=jnp.ones((len(fills), 3), dtype=jnp.bool_),
        state=jnp.concatenate([jnp.full((1, 8), fill, dtype=jnp.float32) for fill in fills], axis=0),
    )


def test_ae_owned_prefix_cache_lane_pool_exports_dense_batch():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    batch = pool.export_batch_view(("req-1", "req-2"))
    assert batch.past_key_values[0].shape == (3, 2, 2, 4)
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 0]), np.full((3, 2, 4), 1.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 1]), np.full((3, 2, 4), 2.0, dtype=np.float32))
    assert batch.prefix_pad_masks.shape == (2, 3)
    assert batch.state.shape == (2, 8)


def test_ae_owned_prefix_cache_lane_pool_releases_original_physical_lane():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    freed = pool.release_lane("req-1")
    assert freed == 0
    batch = pool.export_batch_view(("req-2",))
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 2.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_write_claim_preserves_physical_lane():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.initialize_from_feature(_feature(0.0))
    pool.write_lane(2, _feature(9.0))
    physical, vacated = pool.claim_written_lane("req-1", 2)
    assert physical == 2
    assert vacated is None
    batch = pool.export_batch_view(("req-1",))
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 9.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_batch_write_claim_preserves_sparse_physical_lanes():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.initialize_from_feature(_feature(0.0))
    pool.write_lanes((1, 2), _batch_feature(4.0, 5.0))

    physical_1, vacated_1 = pool.claim_written_lane("req-1", 1)
    physical_2, vacated_2 = pool.claim_written_lane("req-2", 2)

    assert (physical_1, vacated_1) == (1, None)
    assert (physical_2, vacated_2) == (2, None)
    batch = pool.export_batch_view(("req-1", "req-2"))
    np.testing.assert_allclose(np.asarray(batch.state[0]), np.full((8,), 4.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.state[1]), np.full((8,), 5.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 0]), np.full((3, 2, 4), 4.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 1]), np.full((3, 2, 4), 5.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_gathers_sparse_export_by_request_order():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    batch = pool.export_batch_view(("req-2", "req-1"))
    np.testing.assert_allclose(np.asarray(batch.state[0]), np.full((8,), 2.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.state[1]), np.full((8,), 1.0, dtype=np.float32))
