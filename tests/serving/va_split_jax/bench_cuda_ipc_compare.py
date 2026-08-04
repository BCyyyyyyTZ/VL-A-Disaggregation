#!/usr/bin/env python3
"""Compare PyTorch automatic CUDA-IPC vs JAX hand-written DeviceSlab IPC.

Lane payload ≈ one production prefix row:
  shape=(18,1,968,1,256), bf16/uint16 → ~8.9 MiB

Writes JSON under logs/tests/cuda_ipc_compare_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time
import traceback
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.12")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "tests"
LANE_SHAPE = (18, 1, 968, 1, 256)
LANE_BYTES = int(__import__("math").prod(LANE_SHAPE)) * 2


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean_ms": float(statistics.fmean(values)) if values else float("nan"),
        "p50_ms": _pct(values, 0.50),
        "p95_ms": _pct(values, 0.95),
        "min_ms": float(min(values)) if values else float("nan"),
        "max_ms": float(max(values)) if values else float("nan"),
    }


# ------------------------- PyTorch ---------------------------------


def _pt_worker_pair(queue, result_queue, *, iters: int, warmup: int, role: str) -> None:
    import torch

    try:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        if role == "producer":
            total = warmup + iters
            for i in range(total):
                payload = torch.full(LANE_SHAPE, 1.0 + i * 1e-6, device=device, dtype=torch.bfloat16)
                torch.cuda.synchronize()
                queue.put(payload)
                ack = queue.get(timeout=60)
                if ack != "__ack__":
                    raise RuntimeError(f"bad ack {ack!r}")
            return

        # consumer
        local = None
        copy_ms_list: list[float] = []
        get_copy_ms_list: list[float] = []
        total = warmup + iters
        for i in range(total):
            t0 = time.perf_counter_ns()
            src = queue.get(timeout=60)
            t1 = time.perf_counter_ns()
            if local is None:
                local = torch.empty_like(src, device=device)
            local.copy_(src, non_blocking=False)
            torch.cuda.synchronize()
            t2 = time.perf_counter_ns()
            if i >= warmup:
                copy_ms_list.append((t2 - t1) / 1e6)
                get_copy_ms_list.append((t2 - t0) / 1e6)
            queue.put("__ack__")
        result_queue.put(
            {
                "ok": True,
                "copy_ms": copy_ms_list,
                "get_plus_copy_ms": get_copy_ms_list,
            }
        )
    except Exception as exc:  # pragma: no cover
        result_queue.put({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})


def run_pytorch_bench(*, iters: int, warmup: int) -> dict[str, Any]:
    import torch
    import torch.multiprocessing as tmp

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "no CUDA for PyTorch"}

    ctx = tmp.get_context("spawn")
    queue = ctx.Queue(maxsize=2)
    result_queue = ctx.Queue()
    consumer = ctx.Process(
        target=_pt_worker_pair,
        args=(queue, result_queue),
        kwargs={"iters": iters, "warmup": warmup, "role": "consumer"},
    )
    producer = ctx.Process(
        target=_pt_worker_pair,
        args=(queue, result_queue),
        kwargs={"iters": iters, "warmup": warmup, "role": "producer"},
    )
    consumer.start()
    producer.start()
    producer.join(timeout=180)
    consumer.join(timeout=180)
    alive = producer.is_alive() or consumer.is_alive()
    if alive:
        producer.kill()
        consumer.kill()
        return {"status": "error", "reason": "pytorch bench hang"}
    raw = result_queue.get(timeout=5)
    if not raw.get("ok"):
        return {"status": "error", "reason": raw.get("error", "unknown")}
    return {
        "status": "ok",
        "framework": "pytorch",
        "transport": "torch.multiprocessing CUDA-IPC (CUDA tensor pickle) + copy_",
        "lane_shape": list(LANE_SHAPE),
        "lane_bytes": LANE_BYTES,
        "copy_only": _summary(raw["copy_ms"]),
        "get_plus_copy": _summary(raw["get_plus_copy_ms"]),
        "note": "get_plus_copy ≈ VLM->AE per-request: queue CUDA-IPC get + AE-local copy.",
    }


# ------------------------- JAX ---------------------------------


def _jax_consumer(down_q, up_q, result_q, *, iters: int, warmup: int) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax
        import jax.numpy as jnp

        from openpi.serving.va_split_jax.ae_process import _slice_single_lane_array
        from openpi.serving.va_split_jax.device_slab import CudaIpcDeviceSlabBackend
        from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec
        from openpi.serving.va_split_jax.device_slab import _logical_view_for_spec

        print("[jax-consumer] waiting open handle", flush=True)
        msg = down_q.get(timeout=120)
        handle = msg["handle"]
        backend = CudaIpcDeviceSlabBackend()
        t0 = time.perf_counter_ns()
        slab = backend.open_slab(handle)
        slab.array.block_until_ready()
        open_ms = (time.perf_counter_ns() - t0) / 1e6
        print(f"[jax-consumer] opened slab in {open_ms:.1f}ms devices={jax.devices()}", flush=True)
        up_q.put("opened")

        local_spec = DeviceSlabSpec(
            name="local", shape=LANE_SHAPE, dtype="bfloat16", max_lanes=8, lane_axis=1
        )
        local = backend.create_slab(local_spec)

        slice_ms_l: list[float] = []
        bitcast_ms_l: list[float] = []
        copy_ms_l: list[float] = []
        ingest_ms_l: list[float] = []
        prod_ms_l: list[float] = []

        total = warmup + iters
        for i in range(total):
            down_q.get(timeout=180)  # fill_done
            t_a = time.perf_counter_ns()
            sliced = jax.lax.dynamic_slice_in_dim(slab.array, 0, 1, axis=1)
            sliced.block_until_ready()
            t_b = time.perf_counter_ns()
            logical = _logical_view_for_spec(sliced, slab.spec)
            logical.block_until_ready()
            t_c = time.perf_counter_ns()
            local = backend.copy_lane_from_array(local, 0, logical)
            t_d = time.perf_counter_ns()

            t_e = time.perf_counter_ns()
            view = _slice_single_lane_array(backend, slab, 0)
            local = backend.copy_lane_from_array(local, 1, view)
            t_f = time.perf_counter_ns()

            if i >= warmup:
                slice_ms_l.append((t_b - t_a) / 1e6)
                bitcast_ms_l.append((t_c - t_b) / 1e6)
                copy_ms_l.append((t_d - t_c) / 1e6)
                ingest_ms_l.append((t_d - t_a) / 1e6)
                prod_ms_l.append((t_f - t_e) / 1e6)
            up_q.put("__ack__")
            if i < 3 or i == total - 1:
                print(f"[jax-consumer] iter={i} ingest_ms={(t_d - t_a)/1e6:.2f}", flush=True)

        local.close()
        slab.close()
        result_q.put(
            {
                "ok": True,
                "open_ms": open_ms,
                "slice_ms": slice_ms_l,
                "bitcast_ms": bitcast_ms_l,
                "copy_ms": copy_ms_l,
                "ingest_ms": ingest_ms_l,
                "prod_ingest_ms": prod_ms_l,
            }
        )
    except Exception as exc:  # pragma: no cover
        result_q.put({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})


def _jax_producer(down_q, up_q, result_q, *, iters: int, warmup: int) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        import jax.numpy as jnp

        from openpi.serving.va_split_jax.device_slab import CudaIpcDeviceSlabBackend
        from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec

        backend = CudaIpcDeviceSlabBackend()
        spec = DeviceSlabSpec(
            name="ipc", shape=LANE_SHAPE, dtype="bfloat16", max_lanes=8, lane_axis=1
        )
        print("[jax-producer] create slab", flush=True)
        slab = backend.create_slab(spec)
        value = jnp.ones(LANE_SHAPE, dtype=jnp.bfloat16)
        value.block_until_ready()
        slab = backend.copy_lane_from_array(slab, 0, value)
        down_q.put({"handle": slab.handle})
        print("[jax-producer] sent handle, wait opened", flush=True)
        opened = up_q.get(timeout=180)
        if opened != "opened":
            raise RuntimeError(f"expected opened, got {opened!r}")

        total = warmup + iters
        for i in range(total):
            value = jnp.full(LANE_SHAPE, 1.0 + i * 1e-3, dtype=jnp.bfloat16)
            value.block_until_ready()
            slab = backend.copy_lane_from_array(slab, 0, value)
            down_q.put("fill_done")
            ack = up_q.get(timeout=180)
            if ack != "__ack__":
                raise RuntimeError(f"bad ack {ack!r}")
        time.sleep(0.2)
        slab.close()
        print("[jax-producer] done", flush=True)
    except Exception as exc:  # pragma: no cover
        result_q.put({"ok": False, "error": f"producer: {exc}\n{traceback.format_exc()}"})


def run_jax_bench(*, iters: int, warmup: int) -> dict[str, Any]:
    import multiprocessing as mp
    import shutil
    import subprocess

    # Do NOT import jax in the parent before spawn: initializing CUDA in the parent
    # commonly breaks CUDA-IPC / JAX in children.
    if shutil.which("nvidia-smi") is None:
        return {"status": "skipped", "reason": "nvidia-smi missing"}
    try:
        subprocess.check_call(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {"status": "skipped", "reason": "no NVIDIA GPU visible"}

    ctx = mp.get_context("spawn")
    down_q = ctx.Queue(maxsize=1)  # producer -> consumer
    up_q = ctx.Queue(maxsize=1)  # consumer -> producer
    result_q = ctx.Queue()
    consumer = ctx.Process(
        target=_jax_consumer, args=(down_q, up_q, result_q), kwargs={"iters": iters, "warmup": warmup}
    )
    producer = ctx.Process(
        target=_jax_producer, args=(down_q, up_q, result_q), kwargs={"iters": iters, "warmup": warmup}
    )
    consumer.start()
    producer.start()
    producer.join(timeout=420)
    consumer.join(timeout=420)
    if producer.is_alive() or consumer.is_alive():
        err = None
        try:
            err = result_q.get_nowait()
        except Exception:
            pass
        producer.kill()
        consumer.kill()
        return {"status": "error", "reason": "jax bench hang", "child": err}
    if result_q.empty():
        return {"status": "error", "reason": "jax bench produced no result"}
    raw = result_q.get(timeout=5)
    if not raw.get("ok"):
        return {"status": "error", "reason": raw.get("error", "unknown")}
    return {
        "status": "ok",
        "framework": "jax",
        "transport": "numba CUDA-IPC DeviceSlab + slice/bitcast + copy_lane_kernel(+sync)",
        "lane_shape": list(LANE_SHAPE),
        "lane_bytes": LANE_BYTES,
        "open_once_ms": float(raw["open_ms"]),
        "slice": _summary(raw["slice_ms"]),
        "bitcast": _summary(raw["bitcast_ms"]),
        "copy_with_sync": _summary(raw["copy_ms"]),
        "ingest_total": _summary(raw["ingest_ms"]),
        "prod_helper_ingest": _summary(raw["prod_ingest_ms"]),
        "note": (
            "Steady ingest assumes slab already mapped (open_once_ms is startup). "
            "ingest_total = slice + bitcast + copy_lane_from_array."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--log-dir", type=pathlib.Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.log_dir / f"cuda_ipc_compare_{ts}.json"

    report: dict[str, Any] = {
        "timestamp": ts,
        "gpu": args.gpu,
        "iters": args.iters,
        "warmup": args.warmup,
        "lane_shape": list(LANE_SHAPE),
        "lane_bytes": LANE_BYTES,
        "lane_mib": LANE_BYTES / (1024 * 1024),
    }

    print(f"[bench] PyTorch on GPU {args.gpu}", flush=True)
    report["pytorch"] = run_pytorch_bench(iters=args.iters, warmup=args.warmup)
    print(json.dumps(report["pytorch"], indent=2), flush=True)

    print(f"[bench] JAX on GPU {args.gpu}", flush=True)
    report["jax"] = run_jax_bench(iters=args.iters, warmup=args.warmup)
    print(json.dumps(report["jax"], indent=2), flush=True)

    comparison: dict[str, Any] = {"status": "partial"}
    if report["pytorch"].get("status") == "ok" and report["jax"].get("status") == "ok":
        pt = report["pytorch"]["get_plus_copy"]["p50_ms"]
        jx = report["jax"]["ingest_total"]["p50_ms"]
        comparison = {
            "status": "ok",
            "pytorch_get_plus_copy_p50_ms": pt,
            "jax_steady_ingest_p50_ms": jx,
            "jax_minus_pytorch_p50_ms": jx - pt,
            "jax_over_pytorch_ratio": (jx / pt) if pt > 0 else None,
            "fairness_note": (
                "PyTorch metric includes per-request queue CUDA-IPC get; "
                "JAX metric is steady ingest after one-time open (closer to production AE admit)."
            ),
        }
    report["comparison"] = comparison
    text = json.dumps(report, indent=2, sort_keys=True)
    out_path.write_text(text, encoding="utf-8")
    (args.log_dir / "cuda_ipc_compare_latest.json").write_text(text, encoding="utf-8")
    print(f"[bench] wrote {out_path}", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
