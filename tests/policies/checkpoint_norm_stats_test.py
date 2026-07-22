from __future__ import annotations

import numpy as np

from openpi.shared import normalize
from openpi.training import checkpoints


def _stats(value: float) -> dict[str, normalize.NormStats]:
    return {
        "state": normalize.NormStats(
            mean=np.asarray([value], dtype=np.float32),
            std=np.asarray([1.0], dtype=np.float32),
        )
    }


def test_load_checkpoint_norm_stats_prefers_assets_layout(tmp_path):
    normalize.save(tmp_path / "assets" / "physical-intelligence/libero", _stats(1.0))
    normalize.save(tmp_path / "physical-intelligence/libero", _stats(2.0))

    loaded = checkpoints.load_checkpoint_norm_stats(tmp_path, "physical-intelligence/libero")

    np.testing.assert_array_equal(loaded["state"].mean, np.asarray([1.0], dtype=np.float32))


def test_load_checkpoint_norm_stats_falls_back_to_raw_asset_layout(tmp_path):
    normalize.save(tmp_path / "physical-intelligence/libero", _stats(2.0))

    loaded = checkpoints.load_checkpoint_norm_stats(tmp_path, "physical-intelligence/libero")

    np.testing.assert_array_equal(loaded["state"].mean, np.asarray([2.0], dtype=np.float32))
