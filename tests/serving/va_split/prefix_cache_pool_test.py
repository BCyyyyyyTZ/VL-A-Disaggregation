from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_split_types import PrefixFeature
from openpi.serving.va_split.prefix_cache_pool import PrefixCacheLanePool


def _feature(fill_value: float) -> PrefixFeature:
    cache = DynamicCache()
    cache.update(
        torch.full((1, 1, 3, 4), fill_value),
        torch.full((1, 1, 3, 4), fill_value + 10.0),
        layer_idx=0,
    )
    return PrefixFeature(
        past_key_values=cache,
        prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
        state=torch.full((1, 2), fill_value),
    )


def test_prefix_cache_lane_pool_returns_dense_dynamic_cache_views():
    pool = PrefixCacheLanePool(max_lanes=3)
    pool.put_lane(0, _feature(1.0))
    pool.put_lane(1, _feature(2.0))

    feature = pool.view_prefix_batch(2)

    assert isinstance(feature.past_key_values, DynamicCache)
    key, value = feature.past_key_values[0]
    assert key.shape == (2, 1, 3, 4)
    assert value.shape == (2, 1, 3, 4)
    torch.testing.assert_close(key[:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(value[:, 0, 0, 0], torch.tensor([11.0, 12.0]))
    torch.testing.assert_close(feature.state, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))

    key[0, 0, 0, 0] = 7.0
    key_again, _ = pool.view_prefix_batch(2).past_key_values[0]
    assert key_again[0, 0, 0, 0].item() == 7.0


def test_prefix_cache_lane_pool_moves_last_lane_into_hole():
    pool = PrefixCacheLanePool(max_lanes=3)
    pool.put_lane(0, _feature(1.0))
    pool.put_lane(1, _feature(2.0))

    pool.move_lane(1, 0)
    feature = pool.view_prefix_batch(1)

    key, value = feature.past_key_values[0]
    torch.testing.assert_close(key[:, 0, 0, 0], torch.tensor([2.0]))
    torch.testing.assert_close(value[:, 0, 0, 0], torch.tensor([12.0]))
    torch.testing.assert_close(feature.state, torch.tensor([[2.0, 2.0]]))
