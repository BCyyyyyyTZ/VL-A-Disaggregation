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


def test_ae_owned_prefix_cache_lane_pool_compacts_on_release():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    freed = pool.release_lane("req-1")
    assert freed == 1
    batch = pool.export_batch_view(("req-2",))
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 2.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_write_claim_densifies():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.initialize_from_feature(_feature(0.0))
    pool.write_lane(2, _feature(9.0))
    dense, vacated = pool.claim_written_lane("req-1", 2)
    assert dense == 0
    assert vacated == 2
    batch = pool.view_prefix_batch(1)
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 9.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_batch_write_claim_densifies():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.initialize_from_feature(_feature(0.0))
    pool.write_lanes((1, 2), _batch_feature(4.0, 5.0))

    dense_1, vacated_1 = pool.claim_written_lane("req-1", 1)
    dense_2, vacated_2 = pool.claim_written_lane("req-2", 2)

    assert (dense_1, vacated_1) == (0, 1)
    assert (dense_2, vacated_2) == (1, 2)
    batch = pool.export_batch_view(("req-1", "req-2"))
    np.testing.assert_allclose(np.asarray(batch.state[0]), np.full((8,), 4.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.state[1]), np.full((8,), 5.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 0]), np.full((3, 2, 4), 4.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][:, 1]), np.full((3, 2, 4), 5.0, dtype=np.float32))


def test_sparse_prefix_cache_lane_pool_keeps_out_of_order_physical_lanes():
    pool = JaxVlmPrefixCacheLanePool(
        max_lanes=4,
        backend=make_default_device_slab_backend(),
        compact_lanes=False,
    )
    pool.initialize_from_feature(_feature(0.0))
    pool.write_lane(1, _feature(4.0))
    dense_1, vacated_1 = pool.claim_written_lane("req-1", 1)
    pool.write_lane(0, _feature(5.0))
    dense_2, vacated_2 = pool.claim_written_lane("req-2", 0)

    assert (dense_1, vacated_1) == (0, None)
    assert (dense_2, vacated_2) == (1, None)
    batch = pool.export_batch_view(("req-1", "req-2"))
    np.testing.assert_allclose(np.asarray(batch.state[0]), np.full((8,), 4.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.state[1]), np.full((8,), 5.0, dtype=np.float32))


def test_sparse_prefix_cache_lane_pool_reuses_first_free_lane_on_put():
    pool = JaxVlmPrefixCacheLanePool(
        max_lanes=4,
        backend=make_default_device_slab_backend(),
        compact_lanes=False,
    )
    pool.initialize_from_feature(_feature(0.0))
    lane_1 = pool.put_lane("req-1", _feature(1.0))
    lane_2 = pool.put_lane("req-2", _feature(2.0))
    freed = pool.release_lane("req-1")
    lane_3 = pool.put_lane("req-3", _feature(3.0))

    assert (lane_1, lane_2, freed, lane_3) == (0, 1, 0, 0)
    batch = pool.export_batch_view(("req-2", "req-3"))
    np.testing.assert_allclose(np.asarray(batch.state[0]), np.full((8,), 2.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.state[1]), np.full((8,), 3.0, dtype=np.float32))


def test_ae_owned_prefix_cache_lane_pool_rejects_sparse_export():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    with pytest.raises(ValueError, match="dense prefix"):
        pool.export_batch_view(("req-2",))
