from __future__ import annotations

import pytest

from openpi.serving.va_split_jax.multigpu_config import JaxMultiGpuVASplitConfig
from openpi.serving.va_split_jax.multigpu_config import parse_device_list


def test_multigpu_config_defaults_to_latency_first_batching():
    cfg = JaxMultiGpuVASplitConfig(vlm_devices=("0", "1"), ae_device="2")

    assert cfg.num_vlm_workers == 2
    assert cfg.max_vlm_wait_ms == 0.0
    assert cfg.max_vlm_batch_size == 8
    assert cfg.max_ae_batch_size == 64
    assert cfg.max_prefix_slots == 48


def test_multigpu_config_rejects_overlapping_devices():
    with pytest.raises(ValueError, match="ae_device must not appear in vlm_devices"):
        JaxMultiGpuVASplitConfig(vlm_devices=("0", "1"), ae_device="1")


def test_parse_device_list_trims_csv():
    assert parse_device_list("GPU-a, 1,,GPU-c") == ("GPU-a", "1", "GPU-c")


def test_multigpu_config_accepts_host_staged_cross_card_transfer_strategy():
    cfg = JaxMultiGpuVASplitConfig(
        vlm_devices=("0", "1"),
        ae_device="2",
        cross_card_transfer_strategy="host-staged",
    )

    assert cfg.cross_card_transfer_strategy == "host-staged"


def test_multigpu_config_rejects_unknown_cross_card_transfer_strategy():
    with pytest.raises(ValueError, match="cross_card_transfer_strategy"):
        JaxMultiGpuVASplitConfig(
            vlm_devices=("0",),
            ae_device="1",
            cross_card_transfer_strategy="unknown",
        )
