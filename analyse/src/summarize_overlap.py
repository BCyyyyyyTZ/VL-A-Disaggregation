#!/usr/bin/env python3
"""Summarize controlled VLM/AE overlap benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latency(result: dict[str, Any], role: str | None = None) -> float:
    if role is None:
        return float(result.get("measured", {}).get("latency", {}).get("avg_ms", 0.0))
    return float(result.get(role, {}).get("measured", {}).get("latency", {}).get("avg_ms", 0.0))


def _iters(result: dict[str, Any], role: str | None = None) -> int:
    if role is None:
        return int(result.get("measured", {}).get("iterations", 0))
    return int(result.get(role, {}).get("measured", {}).get("iterations", 0))


def _read_case(output_dir: Path, case_id: str) -> dict[str, Any] | None:
    path = output_dir / "cases" / case_id / "result.json"
    return _load(path) if path.exists() else None


def build_summary(output_dir: Path) -> dict[str, Any]:
    rows = []
    batches = sorted(
        {
            int(path.parent.name.split("_b", 1)[1].split("_", 1)[0])
            for path in (output_dir / "cases").glob("*_b*/result.json")
            if "_b" in path.parent.name
        }
    )
    for batch_size in batches:
        ae_batch_size = batch_size
        solo_vlm = _read_case(output_dir, f"solo_vlm_b{batch_size}_ae{ae_batch_size}")
        solo_ae = _read_case(output_dir, f"solo_ae_b{batch_size}_ae{ae_batch_size}")
        concurrent = _read_case(output_dir, f"concurrent_b{batch_size}_ae{ae_batch_size}")
        serial = _read_case(output_dir, f"single-serial_b{batch_size}_ae{ae_batch_size}")
        threads = _read_case(output_dir, f"single-threads_b{batch_size}_ae{ae_batch_size}")
        if not solo_vlm or not solo_ae:
            continue
        vlm_solo = _latency(solo_vlm)
        ae_solo = _latency(solo_ae)
        row = {
            "batch_size": batch_size,
            "ae_batch_size": ae_batch_size,
            "vlm_solo_ms": vlm_solo,
            "ae_solo_ms": ae_solo,
            "ae_solo_over_vlm": ae_solo / vlm_solo if vlm_solo else 0.0,
        }
        if concurrent and concurrent.get("status") == "pass":
            vlm_conc = _latency(concurrent, "vlm")
            ae_conc = _latency(concurrent, "ae")
            actual_cycle = max(vlm_conc, ae_conc)
            ideal_cycle = max(vlm_solo, ae_solo)
            overlap_eff = ((vlm_solo + ae_solo) - actual_cycle) / min(vlm_solo, ae_solo) if min(vlm_solo, ae_solo) else 0.0
            row.update(
                {
                    "vlm_concurrent_ms": vlm_conc,
                    "ae_concurrent_ms": ae_conc,
                    "vlm_slowdown": vlm_conc / vlm_solo if vlm_solo else 0.0,
                    "ae_slowdown": ae_conc / ae_solo if ae_solo else 0.0,
                    "ideal_cycle_ms": ideal_cycle,
                    "actual_cycle_ms": actual_cycle,
                    "ideal_speedup": (vlm_solo + ae_solo) / ideal_cycle if ideal_cycle else 0.0,
                    "actual_speedup_est": (vlm_solo + ae_solo) / actual_cycle if actual_cycle else 0.0,
                    "overlap_efficiency": overlap_eff,
                    "vlm_concurrent_iters": _iters(concurrent, "vlm"),
                    "ae_concurrent_iters": _iters(concurrent, "ae"),
                }
            )
        if serial and serial.get("status") == "pass":
            row["single_serial_combined_ms"] = float(serial.get("measured", {}).get("combined", {}).get("latency", {}).get("avg_ms", 0.0))
        if threads and threads.get("status") == "pass":
            row["single_threads_vlm_ms"] = float(threads.get("measured", {}).get("vlm", {}).get("latency", {}).get("avg_ms", 0.0))
            row["single_threads_ae_ms"] = float(threads.get("measured", {}).get("ae", {}).get("latency", {}).get("avg_ms", 0.0))
            row["single_threads_wall_s"] = float(threads.get("measured", {}).get("wall_s", 0.0))
        rows.append(row)
    return {"rows": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    bs = [row["batch_size"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    ax = axes[0][0]
    ax.plot(bs, [row.get("vlm_solo_ms", 0) for row in rows], marker="o", label="VLM solo")
    ax.plot(bs, [row.get("ae_solo_ms", 0) for row in rows], marker="s", label="AE solo")
    ax.plot(bs, [row.get("vlm_concurrent_ms", 0) for row in rows], marker="o", linestyle="--", label="VLM concurrent")
    ax.plot(bs, [row.get("ae_concurrent_ms", 0) for row in rows], marker="s", linestyle="--", label="AE concurrent")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(bs, [row.get("vlm_slowdown", 0) for row in rows], marker="o", label="VLM slowdown")
    ax.plot(bs, [row.get("ae_slowdown", 0) for row in rows], marker="s", label="AE slowdown")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Concurrent / solo")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(bs, [row.get("ideal_speedup", 0) for row in rows], marker="o", label="Ideal speedup")
    ax.plot(bs, [row.get("actual_speedup_est", 0) for row in rows], marker="s", label="Actual speedup est.")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.plot(bs, [row.get("overlap_efficiency", 0) for row in rows], marker="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Overlap efficiency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "overlap_summary.png")
    fig.savefig(output_dir / "overlap_summary.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analyse/summary/jax_va_overlap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(output_dir)
    (output_dir / "overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "overlap_summary.csv", summary["rows"])
    plot(output_dir, summary["rows"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
