#!/usr/bin/env python3
"""Plot OpenPI roofline points against the measured hardware roof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="roofline/logs/hardware_roofline.json")
    parser.add_argument("--openpi", default="roofline/logs/openpi05_jax_roofline/summary.json")
    parser.add_argument("--output", default="roofline/logs/openpi05_jax_roofline.png")
    parser.add_argument("--title", default="OpenPI0.5 JAX Roofline on RTX 4090")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware = _load_json(args.hardware)
    openpi = _load_json(args.openpi)

    import matplotlib.pyplot as plt
    import numpy as np

    peak_tflops = float(hardware["peak_tflops"])
    peak_tbps = float(hardware["peak_bandwidth_tbps"])
    points = [point for point in openpi.get("points", []) if point.get("achieved_tflops", 0.0) > 0.0]
    if not points:
        raise RuntimeError("No passing OpenPI roofline points found")

    max_x = max(max(point["arithmetic_intensity_flop_per_byte"] for point in points) * 1.25, peak_tflops / peak_tbps * 2.0)
    xs = np.linspace(0.01, max_x, 600)
    ys = np.minimum(peak_tflops, xs * peak_tbps)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    ax.plot(xs, ys, color="black", linewidth=2.0, label="Measured roofline")
    ax.axhline(peak_tflops, color="black", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.axvline(peak_tflops / peak_tbps, color="gray", linewidth=1.0, linestyle=":", alpha=0.8)

    styles = {
        "VLM": {"marker": "o", "color": "#c94f4a", "label": "VLM"},
        "Denoise-5": {"marker": "s", "color": "#4c78a8", "label": "Denoise step=5"},
        "Denoise-10": {"marker": "^", "color": "#59a14f", "label": "Denoise step=10"},
    }
    seen: set[str] = set()
    for family, style in styles.items():
        family_points = [point for point in points if point["name"] == family]
        family_points.sort(key=lambda point: point["batch_size"])
        if not family_points:
            continue
        ax.plot(
            [point["arithmetic_intensity_flop_per_byte"] for point in family_points],
            [point["achieved_tflops"] for point in family_points],
            marker=style["marker"],
            color=style["color"],
            linewidth=1.7,
            markersize=5.5,
            label=style["label"] if family not in seen else None,
        )
        seen.add(family)
        for point in family_points:
            ax.annotate(
                f"BS{point['batch_size']}",
                (point["arithmetic_intensity_flop_per_byte"], point["achieved_tflops"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=style["color"],
            )

    ax.set_title(args.title)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.set_xlim(left=0.0, right=max_x)
    ax.set_ylim(bottom=0.0, top=max(peak_tflops * 1.12, max(point["achieved_tflops"] for point in points) * 1.25))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.02,
        0.96,
        f"PyTorch measured roof: {peak_tflops:.1f} TFLOP/s, {peak_tbps:.2f} TB/s",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
