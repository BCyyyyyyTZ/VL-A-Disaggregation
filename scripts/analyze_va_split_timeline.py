#!/usr/bin/env python3
"""Analyze VA_SPLIT_TIMELINE_PATH ndjson for VLM/AE wall-clock overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    events.sort(key=lambda e: e["t_ns"])
    return events


def _pair_intervals(events: list[dict], begin: str, end: str) -> list[tuple[int, int, dict]]:
    open_stack: list[dict] = []
    out: list[tuple[int, int, dict]] = []
    for ev in events:
        if ev["event"] == begin:
            open_stack.append(ev)
        elif ev["event"] == end and open_stack:
            start = open_stack.pop(0)
            out.append((int(start["t_ns"]), int(ev["t_ns"]), {**start, **{"end_ms": ev.get("ms")}}))
    return out


def _overlap_ns(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def analyze(events: list[dict]) -> dict:
    vlm_fwd = _pair_intervals([e for e in events if e["proc"] == "vlm"], "vlm_prefix_fwd_begin", "vlm_prefix_fwd_end")
    vlm_write = _pair_intervals([e for e in events if e["proc"] == "vlm"], "vlm_slab_write_begin", "vlm_slab_write_end")
    ae_step = _pair_intervals([e for e in events if e["proc"] == "ae"], "ae_denoise_begin", "ae_denoise_end")

    vlm_gpu = vlm_fwd + vlm_write  # approximate VLM GPU-bound windows

    def overlap_stats(query: list[tuple[int, int, dict]], against: list[tuple[int, int, dict]]) -> dict:
        if not query:
            return {"n": 0}
        overlaps = []
        fracs = []
        for q0, q1, _meta in query:
            dur = max(1, q1 - q0)
            ov = sum(_overlap_ns(q0, q1, a0, a1) for a0, a1, _ in against)
            overlaps.append(ov)
            fracs.append(ov / dur)
        fracs_sorted = sorted(fracs)
        return {
            "n": len(query),
            "overlap_frac_mean": sum(fracs) / len(fracs),
            "overlap_frac_p50": fracs_sorted[len(fracs_sorted) // 2],
            "overlap_frac_p90": fracs_sorted[int(0.9 * (len(fracs_sorted) - 1))],
            "any_overlap_rate": sum(1 for f in fracs if f > 0) / len(fracs),
            "strong_overlap_rate_gt_20pct": sum(1 for f in fracs if f > 0.2) / len(fracs),
            "mean_query_ms": (sum(q1 - q0 for q0, q1, _ in query) / len(query)) / 1e6,
            "mean_overlap_ms": (sum(overlaps) / len(overlaps)) / 1e6,
        }

    # Mutual exclusion proxy: fraction of wall time where both claim to be in GPU work.
    if events:
        t0 = events[0]["t_ns"]
        t1 = events[-1]["t_ns"]
        span = max(1, t1 - t0)
    else:
        span = 1
        t0 = 0

    # Sweep both-busy time via event endpoints.
    points = sorted({t0, t1, *[x for iv in vlm_gpu + ae_step for x in iv[:2]]})
    both_busy = 0
    vlm_busy = 0
    ae_busy = 0
    for i in range(len(points) - 1):
        mid = (points[i] + points[i + 1]) // 2
        dt = points[i + 1] - points[i]
        in_vlm = any(a0 <= mid < a1 for a0, a1, _ in vlm_gpu)
        in_ae = any(a0 <= mid < a1 for a0, a1, _ in ae_step)
        if in_vlm:
            vlm_busy += dt
        if in_ae:
            ae_busy += dt
        if in_vlm and in_ae:
            both_busy += dt

    return {
        "events": len(events),
        "vlm_fwd_intervals": len(vlm_fwd),
        "vlm_write_intervals": len(vlm_write),
        "ae_step_intervals": len(ae_step),
        "vlm_fwd_vs_ae_step": overlap_stats(vlm_fwd, ae_step),
        "vlm_write_vs_ae_step": overlap_stats(vlm_write, ae_step),
        "vlm_gpu_vs_ae_step": overlap_stats(vlm_gpu, ae_step),
        "wall_span_ms": span / 1e6,
        "vlm_busy_frac_of_wall": vlm_busy / span,
        "ae_busy_frac_of_wall": ae_busy / span,
        "both_busy_frac_of_wall": both_busy / span,
        "both_busy_frac_of_vlm_busy": (both_busy / vlm_busy) if vlm_busy else 0.0,
        "both_busy_frac_of_ae_busy": (both_busy / ae_busy) if ae_busy else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(_load_events(args.timeline))
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
