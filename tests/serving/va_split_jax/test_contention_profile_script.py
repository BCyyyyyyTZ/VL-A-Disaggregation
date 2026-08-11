from __future__ import annotations

import importlib.util
import pathlib


def _load_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/profile_jax_va_split_contention.py"
    spec = importlib.util.spec_from_file_location("profile_jax_va_split_contention", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_batch_sizes_accepts_commas_and_spaces():
    module = _load_module()

    assert module.parse_batch_sizes("1, 4 8") == (1, 4, 8)


def test_parse_batch_sizes_rejects_non_positive_values():
    module = _load_module()

    try:
        module.parse_batch_sizes("1,0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("parse_batch_sizes accepted a non-positive batch size")


def test_summarize_latencies_reports_percentiles():
    module = _load_module()

    summary = module.summarize_latencies([10.0, 20.0, 30.0, 40.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 25.0
    assert summary["p95_ms"] == 38.5
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 40.0


def test_summarize_concurrent_pair_reports_slowdown_and_overlap():
    module = _load_module()
    vlm = {
        "role": "vlm",
        "batch_size": 8,
        "latencies_ms": [100.0, 120.0],
        "wall_start_ns": 1_000,
        "wall_end_ns": 301_000_000,
    }
    ae = {
        "role": "ae",
        "batch_size": 8,
        "latencies_ms": [5.0, 7.0],
        "wall_start_ns": 101_000,
        "wall_end_ns": 201_000_000,
    }
    solo = {
        "vlm": {8: {"p50_ms": 80.0}},
        "ae": {8: {"p50_ms": 4.0}},
    }

    summary = module.summarize_concurrent_pair(vlm, ae, solo_by_role_batch=solo)

    assert summary["batch_size"] == 8
    assert summary["vlm"]["p50_ms"] == 110.0
    assert summary["vlm_slowdown_vs_solo_p50"] == 1.375
    assert summary["ae"]["p50_ms"] == 6.0
    assert summary["ae_slowdown_vs_solo_p50"] == 1.5
    assert summary["overlap_ms"] == 200.899
    assert summary["overlap_fraction_of_shorter"] == 1.0
