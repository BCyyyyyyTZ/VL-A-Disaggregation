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
    ridge = peak_tflops / peak_tbps
    points = [point for point in openpi.get("points", []) if point.get("achieved_tflops", 0.0) > 0.0]
    if not points:
        raise RuntimeError("No passing OpenPI roofline points found")

    # Both axes linear: AI is now in a normal range (~10-500), so uniform ticks work.
    max_x = max(point["arithmetic_intensity_flop_per_byte"] for point in points)
    max_y = max(max(point["achieved_tflops"] for point in points), peak_tflops)
    x_left = 0.0
    x_right = max(max_x * 1.15, ridge * 1.35, 50.0)
    y_bottom = 0.0
    y_top = max_y * 1.12

    xs = np.linspace(max(x_left, 1e-6), x_right, 800)
    ys = np.minimum(peak_tflops, xs * peak_tbps)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    ax.plot(xs, ys, color="black", linewidth=2.0, label="Measured roofline")
    ax.axhline(peak_tflops, color="black", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.axvline(ridge, color="gray", linewidth=1.0, linestyle=":", alpha=0.8)

    styles = {
        "VLM": {"marker": "o", "color": "#c94f4a", "label": "VLM"},
        "Denoise-5": {"marker": "s", "color": "#4c78a8", "label": "Denoise step=5"},
        "Denoise-10": {"marker": "^", "color": "#59a14f", "label": "Denoise step=10"},
    }
    # Stagger label offsets so Denoise-5/10 overlapping AI values stay readable.
    label_offsets = {
        "VLM": (6, 6),
        "Denoise-5": (6, -10),
        "Denoise-10": (-28, 6),
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
                xytext=label_offsets.get(family, (5, 5)),
                fontsize=8,
                color=style["color"],
            )

    ax.set_title(args.title)
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    if x_right <= 100:
        x_major = 10.0
    elif x_right <= 300:
        x_major = 25.0
    elif x_right <= 600:
        x_major = 50.0
    else:
        x_major = 100.0
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    y_major = 25.0 if y_top >= 100 else 10.0
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which="major", linestyle="--", linewidth=0.65, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
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
