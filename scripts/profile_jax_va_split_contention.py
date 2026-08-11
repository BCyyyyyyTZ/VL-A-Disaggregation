#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import time
from typing import Any

import numpy as np


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in re.split(r"[\s,]+", value.strip()) if item)
    if not sizes:
        raise ValueError("batch sizes must contain at least one value")
    if any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must be positive")
    return sizes


def summarize_latencies(values_ms: list[float] | tuple[float, ...]) -> dict[str, float | int | None]:
    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": _round(float(np.mean(values))),
        "p50_ms": _round(float(np.percentile(values, 50))),
        "p95_ms": _round(float(np.percentile(values, 95))),
        "min_ms": _round(float(np.min(values))),
        "max_ms": _round(float(np.max(values))),
    }


def summarize_concurrent_pair(
    vlm_result: dict[str, Any],
    ae_result: dict[str, Any],
    *,
    solo_by_role_batch: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    batch_size = int(vlm_result["batch_size"])
    if int(ae_result["batch_size"]) != batch_size:
        raise ValueError("VLM and AE concurrent results must use the same batch size")
    vlm_summary = summarize_latencies(vlm_result["latencies_ms"])
    ae_summary = summarize_latencies(ae_result["latencies_ms"])
    overlap_ms = _overlap_ms(
        int(vlm_result["wall_start_ns"]),
        int(vlm_result["wall_end_ns"]),
        int(ae_result["wall_start_ns"]),
        int(ae_result["wall_end_ns"]),
    )
    vlm_wall_ms = max(0.0, (int(vlm_result["wall_end_ns"]) - int(vlm_result["wall_start_ns"])) / 1_000_000)
    ae_wall_ms = max(0.0, (int(ae_result["wall_end_ns"]) - int(ae_result["wall_start_ns"])) / 1_000_000)
    shorter_wall_ms = min(vlm_wall_ms, ae_wall_ms)
    return {
        "batch_size": batch_size,
        "vlm": vlm_summary,
        "ae": ae_summary,
        "vlm_slowdown_vs_solo_p50": _slowdown(vlm_summary, solo_by_role_batch["vlm"][batch_size]),
        "ae_slowdown_vs_solo_p50": _slowdown(ae_summary, solo_by_role_batch["ae"][batch_size]),
        "overlap_ms": _round(overlap_ms),
        "overlap_fraction_of_shorter": _round(overlap_ms / shorter_wall_ms) if shorter_wall_ms > 0 else None,
    }


def main() -> None:
    args = _parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    solo_results: list[dict[str, Any]] = []
    concurrent_results: list[dict[str, Any]] = []
    concurrent_raw_results: list[dict[str, Any]] = []

    for role in ("vlm", "ae"):
        for batch_size in batch_sizes:
            print(f"[contention] solo role={role} B={batch_size}", flush=True)
            solo_results.append(_run_child_bench(ctx, role=role, args=args, batch_size=batch_size))

    solo_by_role_batch = _index_solo_results(solo_results)

    for batch_size in batch_sizes:
        print(f"[contention] concurrent B={batch_size}", flush=True)
        vlm_result, ae_result = _run_concurrent_pair(ctx, args=args, batch_size=batch_size)
        concurrent_raw_results.append({"batch_size": batch_size, "vlm": vlm_result, "ae": ae_result})
        concurrent_results.append(
            summarize_concurrent_pair(vlm_result, ae_result, solo_by_role_batch=solo_by_role_batch)
        )

    report = {
        "config": {
            "policy_config": args.policy_config,
            "policy_dir": args.policy_dir,
            "batch_sizes": list(batch_sizes),
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "num_steps": args.num_steps,
            "jax_compile": args.jax_compile,
            "jax_compile_warmup": args.jax_compile_warmup,
            "jax_compile_warmup_max_batch_size": args.jax_compile_warmup_max_batch_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        },
        "solo": _summarize_solo_results(solo_results),
        "concurrent": concurrent_results,
        "raw": {
            "solo": solo_results,
            "concurrent": concurrent_raw_results,
        },
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[contention] wrote {output_path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure JAX V-A split VLM/AE same-GPU contention with role-pruned models."
    )
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--policy-dir", default="/mnt/tianze/models/pi05_libero")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--jax-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-compile-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-compile-warmup-max-batch-size", type=int, default=16)
    parser.add_argument("--output", default="logs/tests/jax-contention/profile.json")
    return parser.parse_args()


def _run_child_bench(ctx, *, role: str, args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    result_queue = ctx.Queue()
    process = ctx.Process(target=_role_worker_once, args=(role, vars(args), batch_size, result_queue), daemon=True)
    process.start()
    result = _get_result(result_queue, timeout_s=900.0)
    process.join(timeout=30.0)
    if process.exitcode not in (0, None):
        raise RuntimeError(f"{role} worker exited with code {process.exitcode}: {result}")
    if result.get("status") != "ok":
        raise RuntimeError(f"{role} worker failed: {result}")
    return result


def _run_concurrent_pair(ctx, *, args: argparse.Namespace, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    start_queue = ctx.Queue()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_role_worker_wait_start,
            args=(role, vars(args), batch_size, start_queue, result_queue),
            daemon=True,
        )
        for role in ("vlm", "ae")
    ]
    for process in processes:
        process.start()

    ready = {_get_result(result_queue, timeout_s=900.0)["role"] for _ in processes}
    if ready != {"vlm", "ae"}:
        raise RuntimeError(f"unexpected ready roles: {ready}")
    start_at_ns = time.monotonic_ns() + 2_000_000_000
    for _ in processes:
        start_queue.put(start_at_ns)

    results = [_get_result(result_queue, timeout_s=900.0) for _ in processes]
    for process in processes:
        process.join(timeout=30.0)
        if process.exitcode not in (0, None):
            raise RuntimeError(f"concurrent worker exited with code {process.exitcode}")
    by_role = {result["role"]: result for result in results}
    return by_role["vlm"], by_role["ae"]


def _role_worker_once(role: str, args_dict: dict[str, Any], batch_size: int, result_queue) -> None:
    try:
        runner = _make_role_runner(role, args_dict, batch_size)
        result_queue.put(runner())
    except Exception as exc:  # pragma: no cover - exercised in manual GPU diagnostics.
        result_queue.put({"status": "error", "role": role, "error": repr(exc)})
        raise


def _role_worker_wait_start(role: str, args_dict: dict[str, Any], batch_size: int, start_queue, result_queue) -> None:
    try:
        runner = _make_role_runner(role, args_dict, batch_size)
        result_queue.put({"status": "ready", "role": role, "batch_size": batch_size})
        start_at_ns = int(start_queue.get())
        while time.monotonic_ns() < start_at_ns:
            time.sleep(0.001)
        result_queue.put(runner())
    except Exception as exc:  # pragma: no cover - exercised in manual GPU diagnostics.
        result_queue.put({"status": "error", "role": role, "error": repr(exc)})
        raise


def _make_role_runner(role: str, args_dict: dict[str, Any], batch_size: int):
    # Import JAX/model code only inside child processes so the parent never initializes CUDA.
    import jax
    import jax.numpy as jnp

    from openpi.models.jax_split_types import JaxDenoiseState
    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.serving.va_split_jax.compile import make_pi0_prefix_feature_template
    from openpi.serving.va_split_jax.compile import maybe_jit_ae_model
    from openpi.serving.va_split_jax.compile import maybe_jit_vlm_model
    from openpi.serving.va_split_jax.compile import planned_warmup_batches
    from openpi.serving.va_split_jax.compile import warmup_vlm_prefix_model
    from openpi.training import config as _config

    train_config = _config.get_config(args_dict["policy_config"])
    compile_config = JaxCompileConfig(
        enabled=bool(args_dict["jax_compile"]),
        warmup_enabled=bool(args_dict["jax_compile_warmup"]),
        compile_ae=True,
        compile_vlm=True,
        warmup_max_batch_size=int(args_dict["jax_compile_warmup_max_batch_size"]),
        num_steps=int(args_dict["num_steps"]),
    )
    model = _load_jax_model(train_config, args_dict["policy_dir"], role=role)
    observation_factory = make_model_observation_factory(model)
    if role == "vlm":
        model = maybe_jit_vlm_model(model, compile_config)
        if compile_config.warmup_enabled:
            warmup_vlm_prefix_model(
                model=model,
                observation_factory=observation_factory,
                max_vlm_batch_size=max(batch_size, compile_config.warmup_max_batch_size),
                config=compile_config,
            )
        observation = observation_factory(batch_size)

        def run_once():
            value = model.build_prefix_feature(None, observation)
            _block_until_ready(value, jax=jax)

    elif role == "ae":
        model = maybe_jit_ae_model(model, compile_config)
        noise_factory = make_model_noise_factory(model)
        prefix_template = make_pi0_prefix_feature_template(train_config.model)
        if compile_config.warmup_enabled:
            for warm_batch_size, repeats in planned_warmup_batches(
                max_batch_size=max(batch_size, compile_config.warmup_max_batch_size),
                warmup_max_batch_size=compile_config.warmup_max_batch_size,
            ):
                prefix = _tile_prefix_feature(prefix_template, warm_batch_size, jnp=jnp)
                state = _make_denoise_state(
                    noise_factory=noise_factory,
                    batch_size=warm_batch_size,
                    num_steps=int(args_dict["num_steps"]),
                    jnp=jnp,
                    state_type=JaxDenoiseState,
                )
                for _ in range(repeats):
                    value = model.denoise_one_batch(prefix, state)
                    _block_until_ready(value, jax=jax)
        prefix = _tile_prefix_feature(prefix_template, batch_size, jnp=jnp)
        noise = noise_factory(batch_size)
        dt = jnp.full((batch_size,), -1.0 / float(args_dict["num_steps"]), dtype=jnp.float32)
        step_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
        state = JaxDenoiseState(x_t=noise, step_idx=step_idx, num_steps=int(args_dict["num_steps"]), dt=dt)

        def run_once():
            value = model.denoise_one_batch(prefix, state)
            _block_until_ready(value, jax=jax)

    else:
        raise ValueError(f"unsupported role: {role}")

    for _ in range(max(0, int(args_dict["warmup_repeats"]))):
        run_once()

    def runner() -> dict[str, Any]:
        latencies_ms: list[float] = []
        wall_start_ns = time.monotonic_ns()
        for _ in range(int(args_dict["repeats"])):
            start_ns = time.monotonic_ns()
            run_once()
            latencies_ms.append((time.monotonic_ns() - start_ns) / 1_000_000)
        wall_end_ns = time.monotonic_ns()
        return {
            "status": "ok",
            "role": role,
            "batch_size": batch_size,
            "latencies_ms": [_round(value) for value in latencies_ms],
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
        }

    return runner


def _make_denoise_state(*, noise_factory, batch_size: int, num_steps: int, jnp, state_type):
    noise = noise_factory(batch_size)
    dt = jnp.full((batch_size,), -1.0 / float(num_steps), dtype=jnp.float32)
    step_idx = jnp.zeros((batch_size,), dtype=jnp.int32)
    return state_type(x_t=noise, step_idx=step_idx, num_steps=num_steps, dt=dt)


def _tile_prefix_feature(feature, batch_size: int, *, jnp):
    from openpi.models.jax_split_types import JaxPrefixFeature

    def tile_past(value):
        return jnp.repeat(value, batch_size, axis=1)

    return JaxPrefixFeature(
        past_key_values=tuple(tile_past(value) for value in feature.past_key_values),
        prefix_pad_masks=jnp.repeat(feature.prefix_pad_masks, batch_size, axis=0),
        state=None if feature.state is None else jnp.repeat(feature.state, batch_size, axis=0),
    )


def _block_until_ready(value: Any, *, jax) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _summarize_solo_results(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {"vlm": {}, "ae": {}}
    for result in results:
        summary[result["role"]][str(result["batch_size"])] = summarize_latencies(result["latencies_ms"])
    return summary


def _index_solo_results(results: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {"vlm": {}, "ae": {}}
    for result in results:
        indexed[result["role"]][int(result["batch_size"])] = summarize_latencies(result["latencies_ms"])
    return indexed


def _get_result(result_queue, *, timeout_s: float) -> dict[str, Any]:
    try:
        return result_queue.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise TimeoutError(f"timed out waiting for contention worker after {timeout_s}s") from exc


def _overlap_ms(start_a_ns: int, end_a_ns: int, start_b_ns: int, end_b_ns: int) -> float:
    return max(0.0, (min(end_a_ns, end_b_ns) - max(start_a_ns, start_b_ns)) / 1_000_000)


def _slowdown(current: dict[str, Any], solo: dict[str, Any]) -> float | None:
    current_p50 = current.get("p50_ms")
    solo_p50 = solo.get("p50_ms")
    if current_p50 is None or solo_p50 in (None, 0):
        return None
    return _round(float(current_p50) / float(solo_p50))


def _round(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
