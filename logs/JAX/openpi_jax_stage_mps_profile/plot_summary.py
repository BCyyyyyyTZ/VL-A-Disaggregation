#!/usr/bin/env python3
"""Plot JAX OpenPI stage MPS profile summary as line charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
SUMMARY_JSON = HERE / "summary.json"
OUT_COMBINED = HERE / "latency_vs_batch.png"
OUT_STAGE = HERE / "stage_latency_vs_batch.png"
OUT_SM100 = HERE / "latency_vs_batch_sm100.png"


def _metric_avg(metrics: dict | None, key: str) -> float:
    if not metrics:
        return 0.0
    return float((metrics.get(key) or {}).get("avg_ms", 0.0))


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for case in payload.get("cases", []):
        if case.get("status") != "pass":
            continue
        metrics = case.get("metrics") or {}
        rows.append(
            {
                "mps_sm": int(case["mps_sm"]),
                "batch_size": int(case["batch_size"]),
                "total": _metric_avg(metrics, "total"),
                "vlm": _metric_avg(metrics, "vlm"),
                "action_head": _metric_avg(metrics, "action_head"),
            }
        )
    rows.sort(key=lambda r: (r["mps_sm"], r["batch_size"]))
    return rows


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def plot_combined(rows: list[dict], out_path: Path) -> None:
    """One figure: total / vlm / action_head vs batch size, lines by MPS SM."""
    _style()
    sms = sorted({r["mps_sm"] for r in rows})
    cmap = plt.get_cmap("viridis")
    colors = {sm: cmap(i / max(len(sms) - 1, 1)) for i, sm in enumerate(sms)}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    metrics = [
        ("total", "Total latency"),
        ("vlm", "VLM (build_prefix_feature)"),
        ("action_head", "Action head (10 denoise steps)"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        for sm in sms:
            xs = [r["batch_size"] for r in rows if r["mps_sm"] == sm]
            ys = [r[key] for r in rows if r["mps_sm"] == sm]
            ax.plot(xs, ys, marker="o", color=colors[sm], label=f"SM {sm}%")
        ax.set_title(title)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency (ms, avg)")
        ax.set_xticks(sorted({r["batch_size"] for r in rows}))

    axes[0].legend(loc="upper left", framealpha=0.9, title="MPS quota")
    fig.suptitle(
        "JAX OpenPI Stage MPS Profile — Latency vs Batch Size",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _tight_ylim(ax, values: list[float], *, pad_frac: float = 0.06) -> None:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1.0)
    pad = span * pad_frac
    ymin = max(0.0, lo - pad)
    ymax = hi + pad
    ax.set_ylim(ymin, ymax)
    # Finer major ticks for the zoomed range.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]))


def plot_sm100(
    rows: list[dict],
    out_path: Path,
    *,
    mps_sm: int = 100,
    power_of_two_only: bool = True,
) -> None:
    """Same 3-panel layout as latency_vs_batch, but only one MPS SM curve with tight y-limits."""
    _style()
    subset = [r for r in rows if r["mps_sm"] == mps_sm]
    if power_of_two_only:
        subset = [
            r
            for r in subset
            if r["batch_size"] > 0 and (r["batch_size"] & (r["batch_size"] - 1)) == 0
        ]
    subset = sorted(subset, key=lambda r: r["batch_size"])
    if not subset:
        raise ValueError(f"No rows for mps_sm={mps_sm}")

    xs = [r["batch_size"] for r in subset]
    x_pos = list(range(len(xs)))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    metrics = [
        ("total", "Total latency", "#1f77b4"),
        ("vlm", "VLM (build_prefix_feature)", "#2ca02c"),
        ("action_head", "Action head (10 denoise steps)", "#d62728"),
    ]

    for ax, (key, title, color) in zip(axes, metrics):
        ys = [r[key] for r in subset]
        ax.plot(x_pos, ys, marker="o", color=color, label=f"SM {mps_sm}%")
        for x, y in zip(x_pos, ys):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color="#333333",
            )
        _tight_ylim(ax, ys)
        ax.set_title(title)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency (ms, avg)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(b) for b in xs])
        ax.set_xlim(-0.35, len(xs) - 1 + 0.35)

    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.suptitle(
        f"JAX OpenPI Stage MPS Profile — SM {mps_sm}% Latency vs Batch Size",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_stage_focus(rows: list[dict], out_path: Path) -> None:
    """Two panels: latency vs SM at fixed batch sizes; AE share vs batch."""
    _style()
    batches = sorted({r["batch_size"] for r in rows})
    sms = sorted({r["mps_sm"] for r in rows})
    cmap = plt.get_cmap("plasma")
    colors = {bs: cmap(i / max(len(batches) - 1, 1)) for i, bs in enumerate(batches)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for bs in batches:
        xs = [r["mps_sm"] for r in rows if r["batch_size"] == bs]
        ys = [r["total"] for r in rows if r["batch_size"] == bs]
        ax.plot(xs, ys, marker="o", color=colors[bs], label=f"bs={bs}")
    ax.set_title("Total latency vs MPS SM quota")
    ax.set_xlabel("MPS SM (%)")
    ax.set_ylabel("Total latency (ms, avg)")
    ax.set_xticks(sms)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    ax = axes[1]
    for sm in sms:
        subset = [r for r in rows if r["mps_sm"] == sm]
        xs = [r["batch_size"] for r in subset]
        ys = [
            100.0 * r["action_head"] / r["total"] if r["total"] > 0 else 0.0
            for r in subset
        ]
        ax.plot(xs, ys, marker="s", label=f"SM {sm}%")
    ax.set_title("Action-head share of total latency")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Action head / total (%)")
    ax.set_xticks(batches)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle(
        "JAX OpenPI Stage MPS Profile — SM Sensitivity & AE Share",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows(SUMMARY_JSON)
    if not rows:
        raise SystemExit(f"No passing cases in {SUMMARY_JSON}")
    plot_combined(rows, OUT_COMBINED)
    plot_stage_focus(rows, OUT_STAGE)
    plot_sm100(rows, OUT_SM100)
    print(f"wrote {OUT_COMBINED}")
    print(f"wrote {OUT_STAGE}")
    print(f"wrote {OUT_SM100}")
    print(f"cases={len(rows)}")


if __name__ == "__main__":
    main()
