#!/usr/bin/env python3
"""Measure OpenPI0.5 JAX VLM and denoise roofline points.

Each batch size is measured in a fresh worker process.  Inside a worker, VLM
and AE helpers are compiled with the same frozen-state JAX pattern used by the
repository's JAX split/baseline helpers, then warmed before timed runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    batch_size: int

    @property
    def case_id(self) -> str:
        return f"bs{self.batch_size}"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    def percentile(pct: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * pct
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "avg_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(percentile(0.50)),
        "p95_ms": float(percentile(0.95)),
    }


def _block_until_ready(value: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _flatten_cost_analysis(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = [item for item in raw if isinstance(item, dict)]
    else:
        return {}

    merged: dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)):
                merged[str(key)] = merged.get(str(key), 0.0) + float(value)
    return merged


def _pick_cost_value(cost: dict[str, float], candidates: tuple[str, ...]) -> float:
    lower = {key.lower(): value for key, value in cost.items()}
    for candidate in candidates:
        if candidate in lower:
            return float(lower[candidate])
    for key, value in lower.items():
        if any(candidate in key for candidate in candidates):
            return float(value)
    return 0.0


def _cost_metrics(compiled: Any) -> dict[str, Any]:
    cost = _flatten_cost_analysis(compiled.cost_analysis())
    flops = _pick_cost_value(cost, ("flops",))
    bytes_accessed = _pick_cost_value(cost, ("bytes accessed", "bytes_accessed"))
    return {
        "flops": flops,
        "bytes_accessed": bytes_accessed,
        "raw": cost,
    }


def _static_pi05_roofline_costs(model: Any, *, batch_size: int) -> dict[str, dict[str, float]]:
    """Return analytical OpenPI0.5 VLM/AE FLOPs and estimated traffic bytes.

    XLA cost_analysis is kept in the output for diagnostics, but it can
    undercount scan/remat-heavy models.  These formulas count dominant matmul,
    convolution, and attention FLOPs from the model shapes used by pi05_libero.
    Bytes estimate dominant tensor traffic for those same ops.  This is still
    not a substitute for Nsight DRAM counters, but it avoids the unrealistic AI
    values produced by counting only weights and final inputs/outputs.
    """
    from openpi.models import gemma as _gemma
    from openpi.models import model as _model
    from openpi.models import siglip as _siglip

    dtype_bytes = 2.0
    image_dtype_bytes = 4.0
    num_images = len(_model.IMAGE_KEYS)
    image_h, image_w = _model.IMAGE_RESOLUTION
    patch_h, patch_w = _siglip.decode_variant("So400m/14")["patch_size"]
    image_tokens = (image_h // patch_h) * (image_w // patch_w)

    vision = _siglip.decode_variant("So400m/14")
    vision_width = int(vision["width"])
    vision_depth = int(vision["depth"])
    vision_mlp = int(vision["mlp_dim"])
    vision_heads = int(vision["num_heads"])
    vision_head_dim = vision_width // vision_heads

    pg = _gemma.get_config("gemma_2b")
    ae = _gemma.get_config("gemma_300m")
    prefix_tokens = num_images * image_tokens + int(model.max_token_len)
    suffix_tokens = int(model.action_horizon)
    action_dim = int(model.action_dim)

    def dense_flops(b: int, t: int, in_dim: int, out_dim: int) -> float:
        return 2.0 * b * t * in_dim * out_dim

    def weight_bytes(*shape: int) -> float:
        count = 1
        for dim in shape:
            count *= int(dim)
        return float(count) * dtype_bytes

    def dense_bytes(b: int, t: int, in_dim: int, out_dim: int, *, outputs: int = 1) -> float:
        m = float(b * t)
        return (m * in_dim + in_dim * out_dim * outputs + m * out_dim * outputs) * dtype_bytes

    def mlp_activation_bytes(b: int, t: int, hidden_dim: int) -> float:
        # GELU/gating/product intermediates are materialized unless fully fused.
        return 4.0 * b * t * hidden_dim * dtype_bytes

    def attention_activation_bytes(
        *,
        b: int,
        query_tokens: int,
        key_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> float:
        q_elems = float(b * query_tokens * num_heads * head_dim)
        kv_elems = float(b * key_tokens * num_kv_heads * head_dim)
        out_elems = q_elems
        score_elems = float(b * num_heads * query_tokens * key_tokens)
        # QK writes logits, mask/softmax read and write logits/probs, AV reads
        # probs and V. Logits/probs are FP32 in the JAX implementation.
        return (
            (q_elems + 2.0 * kv_elems + out_elems) * dtype_bytes
            + 5.0 * score_elems * 4.0
        )

    vision_stem_flops = 2.0 * batch_size * image_tokens * patch_h * patch_w * 3 * vision_width
    vision_layer_flops = (
        3.0 * dense_flops(batch_size, image_tokens, vision_width, vision_width)
        + dense_flops(batch_size, image_tokens, vision_width, vision_width)
        + 2.0 * batch_size * vision_heads * image_tokens * image_tokens * vision_head_dim
        + 2.0 * batch_size * vision_heads * image_tokens * image_tokens * vision_head_dim
        + dense_flops(batch_size, image_tokens, vision_width, vision_mlp)
        + dense_flops(batch_size, image_tokens, vision_mlp, vision_width)
    )
    vision_head_flops = dense_flops(batch_size, image_tokens, vision_width, pg.width)
    vision_flops = num_images * (vision_stem_flops + vision_depth * vision_layer_flops + vision_head_flops)

    vision_stem_bytes = (
        batch_size * image_h * image_w * 3 * image_dtype_bytes
        + weight_bytes(patch_h, patch_w, 3, vision_width)
        + batch_size * image_tokens * vision_width * dtype_bytes
    )
    vision_layer_bytes = (
        dense_bytes(batch_size, image_tokens, vision_width, vision_width, outputs=3)
        + attention_activation_bytes(
            b=batch_size,
            query_tokens=image_tokens,
            key_tokens=image_tokens,
            num_heads=vision_heads,
            num_kv_heads=vision_heads,
            head_dim=vision_head_dim,
        )
        + dense_bytes(batch_size, image_tokens, vision_width, vision_width)
        + dense_bytes(batch_size, image_tokens, vision_width, vision_mlp)
        + mlp_activation_bytes(batch_size, image_tokens, vision_mlp)
        + dense_bytes(batch_size, image_tokens, vision_mlp, vision_width)
    )
    vision_head_bytes = dense_bytes(batch_size, image_tokens, vision_width, pg.width)
    vision_bytes = num_images * (vision_stem_bytes + vision_depth * vision_layer_bytes + vision_head_bytes)

    def gemma_layer_flops(config: Any, *, query_tokens: int, key_tokens: int) -> float:
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        return (
            dense_flops(batch_size, query_tokens, config.width, q_dim)
            + 2.0 * dense_flops(batch_size, query_tokens, config.width, kv_dim)
            + 2.0 * batch_size * config.num_heads * query_tokens * key_tokens * config.head_dim
            + 2.0 * batch_size * config.num_heads * query_tokens * key_tokens * config.head_dim
            + dense_flops(batch_size, query_tokens, q_dim, config.width)
            + 2.0 * dense_flops(batch_size, query_tokens, config.width, config.mlp_dim)
            + dense_flops(batch_size, query_tokens, config.mlp_dim, config.width)
        )

    def gemma_layer_weight_bytes(config: Any, *, adarms: bool = False) -> float:
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        total = (
            weight_bytes(config.width, q_dim)
            + 2.0 * weight_bytes(config.width, kv_dim)
            + weight_bytes(q_dim, config.width)
            + 2.0 * weight_bytes(config.width, config.mlp_dim)
            + weight_bytes(config.mlp_dim, config.width)
        )
        if adarms:
            # Two adaptive RMSNorm dense projections per transformer block.
            total += 2.0 * weight_bytes(config.width, config.width * 3)
        return total

    def gemma_layer_bytes(config: Any, *, query_tokens: int, key_tokens: int, adarms: bool = False) -> float:
        q_dim = config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        total = (
            dense_bytes(batch_size, query_tokens, config.width, q_dim)
            + dense_bytes(batch_size, query_tokens, config.width, kv_dim, outputs=2)
            + attention_activation_bytes(
                b=batch_size,
                query_tokens=query_tokens,
                key_tokens=key_tokens,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
            )
            + dense_bytes(batch_size, query_tokens, q_dim, config.width)
            + dense_bytes(batch_size, query_tokens, config.width, config.mlp_dim, outputs=2)
            + mlp_activation_bytes(batch_size, query_tokens, config.mlp_dim)
            + dense_bytes(batch_size, query_tokens, config.mlp_dim, config.width)
        )
        if adarms:
            total += 2.0 * dense_bytes(batch_size, 1, config.width, config.width * 3)
        return total

    prefix_llm_flops = pg.depth * gemma_layer_flops(pg, query_tokens=prefix_tokens, key_tokens=prefix_tokens)
    prefix_kv_bytes = (
        2.0
        * pg.depth
        * batch_size
        * prefix_tokens
        * pg.num_kv_heads
        * pg.head_dim
        * dtype_bytes
    )
    prefix_llm_bytes = (
        pg.depth * gemma_layer_bytes(pg, query_tokens=prefix_tokens, key_tokens=prefix_tokens)
        + prefix_kv_bytes
    )
    vlm_flops = vision_flops + prefix_llm_flops
    vlm_bytes = vision_bytes + prefix_llm_bytes

    ae_time_mlp_flops = dense_flops(batch_size, 1, ae.width, ae.width) * 2.0
    ae_action_in_flops = dense_flops(batch_size, suffix_tokens, action_dim, ae.width)
    ae_action_out_flops = dense_flops(batch_size, suffix_tokens, ae.width, action_dim)
    ae_step_flops = (
        ae_action_in_flops
        + ae_time_mlp_flops
        + ae.depth * gemma_layer_flops(ae, query_tokens=suffix_tokens, key_tokens=prefix_tokens + suffix_tokens)
        + ae_action_out_flops
    )
    ae_prefix_read_bytes = (
        2.0
        * ae.depth
        * batch_size
        * prefix_tokens
        * ae.num_kv_heads
        * ae.head_dim
        * dtype_bytes
    )
    ae_step_bytes = (
        dense_bytes(batch_size, suffix_tokens, action_dim, ae.width)
        + 2.0 * dense_bytes(batch_size, 1, ae.width, ae.width)
        + ae.depth * gemma_layer_bytes(
            ae,
            query_tokens=suffix_tokens,
            key_tokens=prefix_tokens + suffix_tokens,
            adarms=True,
        )
        + dense_bytes(batch_size, suffix_tokens, ae.width, action_dim)
        + ae_prefix_read_bytes
    )
    return {
        "vlm": {"flops": vlm_flops, "bytes_accessed": vlm_bytes},
        "ae_step": {"flops": ae_step_flops, "bytes_accessed": ae_step_bytes},
    }


def _make_compiled_methods(
    model: Any,
    observation: Any,
    denoise_state: Any,
    *,
    prefix: Any | None = None,
    compile_vlm: bool = True,
):
    import jax
    from flax import nnx

    graphdef, state = nnx.split(model)

    def vlm_fn(module_state, obs):
        module = nnx.merge(graphdef, module_state)
        return module.build_prefix_feature(None, obs)

    def ae_fn(module_state, prefix_batch, state_batch):
        module = nnx.merge(graphdef, module_state)
        return module.denoise_one_batch(prefix_batch, state_batch)

    compiled_vlm = None
    if compile_vlm:
        compiled_vlm = jax.jit(vlm_fn).lower(state, observation).compile()
    compiled_ae = None
    if prefix is not None:
        compiled_ae = jax.jit(ae_fn).lower(state, prefix, denoise_state).compile()
    return state, compiled_vlm, compiled_ae


def _measure_worker(args: argparse.Namespace) -> None:
    # Must be set before importing openpi/jaxtyping: lowering passes ArgInfo into
    # typed dataclasses and fails runtime typechecks otherwise.
    os.environ.setdefault("JAXTYPING_DISABLE", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-openpi-roofline")

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    case_id = str(args.case_id)
    try:
        import jax
        import jax.numpy as jnp

        from openpi.models.jax_split_types import JaxDenoiseState
        from openpi.policies.jax_va_split_policy import _load_jax_model
        from openpi.serving.va_split_jax.compile import make_model_noise_factory
        from openpi.serving.va_split_jax.compile import make_model_observation_factory
        from openpi.training import config as _config

        if not jax.devices("gpu"):
            raise RuntimeError("No JAX GPU device is available")

        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        denoise_steps = _parse_csv_ints(args.denoise_steps)
        max_steps = max(denoise_steps)

        train_config = _config.get_config(args.config)
        model = _load_jax_model(train_config, args.checkpoint_dir)
        observation_factory = make_model_observation_factory(model)
        noise_factory = make_model_noise_factory(model)
        observation = observation_factory(batch_size)
        noise = noise_factory(batch_size)
        dt = jnp.full((batch_size,), -1.0 / float(max_steps), dtype=jnp.float32)
        denoise_state0 = JaxDenoiseState(
            x_t=noise,
            step_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            num_steps=max_steps,
            dt=dt,
        )
        _block_until_ready((observation, noise, denoise_state0))

        # Compile VLM first, materialize a real prefix via the compiled path, then
        # compile AE. Avoids eager build_prefix_feature (OOM at large batch) and a
        # throwaway AE compile against an eager dummy prefix.
        state, compiled_vlm, _ = _make_compiled_methods(model, observation, denoise_state0)
        prefix = compiled_vlm(state, observation)
        _block_until_ready(prefix)
        state, _, compiled_ae = _make_compiled_methods(
            model, observation, denoise_state0, prefix=prefix, compile_vlm=False
        )
        vlm_cost = _cost_metrics(compiled_vlm)
        ae_step_cost = _cost_metrics(compiled_ae)
        static_costs = _static_pi05_roofline_costs(model, batch_size=batch_size)
        point_vlm_cost = static_costs["vlm"] if args.cost_source == "static" else vlm_cost
        point_ae_step_cost = static_costs["ae_step"] if args.cost_source == "static" else ae_step_cost

        for _ in range(int(args.compile_warmup)):
            prefix = compiled_vlm(state, observation)
            _block_until_ready(prefix)
            x_t = noise
            for step in range(max_steps):
                step_state = JaxDenoiseState(
                    x_t=x_t,
                    step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
                    num_steps=max_steps,
                    dt=dt,
                )
                v_t = compiled_ae(state, prefix, step_state)
                x_t = x_t + dt.reshape((-1,) + (1,) * (v_t.ndim - 1)) * v_t
            _block_until_ready(x_t)

        vlm_samples: list[float] = []
        for _ in range(int(args.measure_repeats)):
            t0 = time.perf_counter()
            prefix = compiled_vlm(state, observation)
            _block_until_ready(prefix)
            vlm_samples.append((time.perf_counter() - t0) * 1000.0)

        denoise_samples: dict[str, list[float]] = {str(steps): [] for steps in denoise_steps}
        for steps in denoise_steps:
            for _ in range(int(args.measure_repeats)):
                x_t = noise
                t0 = time.perf_counter()
                for step in range(steps):
                    step_state = JaxDenoiseState(
                        x_t=x_t,
                        step_idx=jnp.full((batch_size,), step, dtype=jnp.int32),
                        num_steps=steps,
                        dt=jnp.full((batch_size,), -1.0 / float(steps), dtype=jnp.float32),
                    )
                    v_t = compiled_ae(state, prefix, step_state)
                    x_t = x_t + step_state.dt.reshape((-1,) + (1,) * (v_t.ndim - 1)) * v_t
                _block_until_ready(x_t)
                denoise_samples[str(steps)].append((time.perf_counter() - t0) * 1000.0)

        def point_from_cost(name: str, cost: dict[str, Any], samples_ms: list[float], *, multiplier: int = 1) -> dict[str, Any]:
            flops = float(cost["flops"]) * multiplier
            bytes_accessed = float(cost["bytes_accessed"]) * multiplier
            avg_ms = _summary(samples_ms)["avg_ms"]
            achieved_tflops = flops / (avg_ms / 1000.0) / 1.0e12 if avg_ms > 0.0 else 0.0
            ai = flops / bytes_accessed if bytes_accessed > 0.0 else 0.0
            return {
                "name": name,
                "batch_size": batch_size,
                "flops": flops,
                "bytes_accessed": bytes_accessed,
                "arithmetic_intensity_flop_per_byte": ai,
                "achieved_tflops": achieved_tflops,
                "samples_ms": samples_ms,
                "latency": _summary(samples_ms),
            }

        points = {
            "vlm": point_from_cost("VLM", point_vlm_cost, vlm_samples),
            "denoise": {
                str(steps): point_from_cost(
                    f"Denoise-{steps}",
                    point_ae_step_cost,
                    denoise_samples[str(steps)],
                    multiplier=int(steps),
                )
                for steps in denoise_steps
            },
        }
        payload = {
            "status": "pass",
            "case_id": case_id,
            "batch_size": batch_size,
            "config": args.config,
            "checkpoint_dir": args.checkpoint_dir,
            "compile_warmup": int(args.compile_warmup),
            "measure_repeats": int(args.measure_repeats),
            "elapsed_s": time.perf_counter() - started,
            "model": {
                "action_horizon": int(model.action_horizon),
                "action_dim": int(model.action_dim),
                "max_token_len": int(model.max_token_len),
                "devices": [str(device) for device in jax.devices()],
            },
            "cost_analysis": {
                "vlm": vlm_cost,
                "ae_step": ae_step_cost,
                "note": "Raw XLA compiled executable cost_analysis retained for diagnostics.",
            },
            "static_costs": {
                **static_costs,
                "note": "Analytical pi05_libero FLOPs plus conservative lower-bound bytes; denoise-5/10 multiply one-step cost by step count.",
            },
            "point_cost_source": {
                "source": args.cost_source,
                "note": "static is the default because XLA cost_analysis undercounts scan/remat-heavy OpenPI graphs.",
            },
            "points": points,
            "timing_kind": "compiled_jax_synchronous_gpu_wall_clock_ms_after_warmup",
        }
        _write_json(output_dir / "cases" / case_id / "result.json", payload)
    except Exception as exc:
        payload = {
            "status": "failed",
            "case_id": case_id,
            "batch_size": args.batch_size,
            "elapsed_s": time.perf_counter() - started,
            "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
        _write_json(output_dir / "cases" / case_id / "result.json", payload)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_case(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    case_dir = Path(args.output_dir) / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("JAXTYPING_DISABLE", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT / "src"),
            str(REPO_ROOT / "packages" / "openpi-client" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case-id",
        case.case_id,
        "--batch-size",
        str(case.batch_size),
        "--config",
        args.config,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-dir",
        args.output_dir,
        "--denoise-steps",
        args.denoise_steps,
        "--compile-warmup",
        str(args.compile_warmup),
        "--measure-repeats",
        str(args.measure_repeats),
        "--cost-source",
        args.cost_source,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.case_timeout_s,
        check=False,
    )
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    result_path = case_dir / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    record = {
        "case_id": case.case_id,
        "batch_size": case.batch_size,
        "status": "pass" if completed.returncode == 0 and payload.get("status") == "pass" else "failed",
        "returncode": completed.returncode,
        "elapsed_s": time.perf_counter() - started,
        "result_path": str(result_path),
        "error_message": None,
    }
    if record["status"] != "pass":
        record["error_message"] = payload.get("error_message") or completed.stderr[-4000:]
    return record


def _write_summary(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    points = []
    for record in records:
        result_path = Path(record["result_path"])
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        cases.append({**record, "result": payload if payload.get("status") != "pass" else None})
        if payload.get("status") == "pass":
            points.append(payload["points"]["vlm"])
            points.extend(payload["points"]["denoise"].values())
    summary = {
        "status": "pass" if all(record["status"] == "pass" for record in records) else "partial",
        "counts": {
            "total": len(records),
            "pass": sum(1 for record in records if record["status"] == "pass"),
            "failed": sum(1 for record in records if record["status"] != "pass"),
        },
        "cases": records,
        "points": points,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", default="/home/miliang/model/openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--output-dir", default="roofline/logs/openpi05_jax_roofline")
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64,128")
    parser.add_argument("--denoise-steps", default="5,10")
    parser.add_argument("--compile-warmup", type=int, default=5, help="Untimed post-compile full VLM+AE iterations")
    parser.add_argument("--measure-repeats", type=int, default=20)
    parser.add_argument("--cost-source", choices=("static", "xla"), default="static")
    parser.add_argument("--case-timeout-s", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _measure_worker(args)
        return

    output_dir = Path(args.output_dir)
    records = []
    for batch_size in _parse_csv_ints(args.batch_sizes):
        case = Case(batch_size=batch_size)
        result_path = output_dir / "cases" / case.case_id / "result.json"
        if args.skip_existing and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            records.append(
                {
                    "case_id": case.case_id,
                    "batch_size": batch_size,
                    "status": payload.get("status", "failed"),
                    "returncode": 0 if payload.get("status") == "pass" else 1,
                    "elapsed_s": float(payload.get("elapsed_s", 0.0)),
                    "result_path": str(result_path),
                    "error_message": payload.get("error_message"),
                }
            )
        else:
            records.append(_run_case(args, case))
        print(json.dumps(records[-1], indent=2), flush=True)
        _write_summary(output_dir, records)
    print(json.dumps(_write_summary(output_dir, records), indent=2))


if __name__ == "__main__":
    main()
