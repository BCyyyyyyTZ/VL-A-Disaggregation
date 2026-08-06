#!/usr/bin/env python3
"""Cross-process AE IPC warmup vs timed denoise (JAX_LOG_COMPILES)."""
from __future__ import annotations

import multiprocessing as mp
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_LOG_COMPILES", "1")


def _vlm_side(conn, gpu: str, ckpt: str, max_slots: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jax.numpy as jnp

    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_observation_factory
    from openpi.serving.va_split_jax.compile import maybe_jit_split_model
    from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
    from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool
    from openpi.training import config as _config

    print("[vlm] devices", jax.devices(), flush=True)
    train_config = _config.get_config("pi05_libero")
    model = _load_jax_model(train_config, ckpt)
    cfg = JaxCompileConfig(enabled=True, warmup_enabled=False, num_steps=5)
    model = maybe_jit_split_model(model, cfg)
    obs_factory = make_model_observation_factory(model)
    backend = make_default_device_slab_backend()
    pool = JaxVlmPrefixCacheLanePool(max_lanes=max_slots, backend=backend)

    # Fill all lanes once so AE slices see real written data.
    for lane in range(max_slots):
        prefix = model.build_prefix_feature(None, obs_factory(1))
        pool.put_lane(f"warm-{lane}", prefix)

    handle_tree = pool.export_slab_handle_tree()
    conn.send({"handle_tree": handle_tree, "backend": type(backend).__name__})
    conn.recv()  # wait AE done
    conn.close()


def _ae_side(conn, gpu: str, ckpt: str, max_slots: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_LOG_COMPILES"] = "1"
    import jax
    import jax.numpy as jnp

    from openpi.models.jax_split_types import JaxDenoiseState
    from openpi.policies.jax_va_split_policy import _load_jax_model
    from openpi.serving.va_split_jax.ae_process import _open_slab_tree
    from openpi.serving.va_split_jax.ae_process import _slice_prefix_batch
    from openpi.serving.va_split_jax.compile import JaxCompileConfig
    from openpi.serving.va_split_jax.compile import make_model_noise_factory
    from openpi.serving.va_split_jax.compile import maybe_jit_split_model
    from openpi.serving.va_split_jax.compile import runtime_aligned_denoise_state
    from openpi.serving.va_split_jax.compile import warmup_ae_denoise_on_mapped_slabs
    from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
    from openpi.training import config as _config

    print("[ae] devices", jax.devices(), flush=True)
    msg = conn.recv()
    handle_tree = msg["handle_tree"]
    print("[ae] got handle_tree backend_from_vlm=", msg["backend"], flush=True)

    train_config = _config.get_config("pi05_libero")
    model = _load_jax_model(train_config, ckpt)
    cfg = JaxCompileConfig(enabled=True, warmup_enabled=True, warmup_max_batch_size=24, num_steps=5)
    model = maybe_jit_split_model(model, cfg)
    noise_factory = make_model_noise_factory(model)
    backend = make_default_device_slab_backend()
    mapped = _open_slab_tree(backend, handle_tree)
    print("[ae] opened IPC slabs", flush=True)

    def make_prefix_batch(slot_ids):
        return _slice_prefix_batch(backend, mapped, slot_ids)

    print("[ae] === IPC-mapped warmup B=1..24 ===", flush=True)
    t0 = time.perf_counter()
    stats = warmup_ae_denoise_on_mapped_slabs(
        model=model,
        noise_factory=noise_factory,
        max_ae_batch_size=999,
        max_prefix_slots=max_slots,
        config=cfg,
        make_prefix_batch=make_prefix_batch,
    )
    print(f"[ae] ipc warmup done in {time.perf_counter()-t0:.1f}s stats={stats}", flush=True)

    print("[ae] === timed after IPC warmup ===", flush=True)
    for batch_size in (1, 8, 18, 24):
        prefix = make_prefix_batch(tuple(range(batch_size)))
        noise = noise_factory(batch_size)
        # Path A: runtime_aligned (warmup style)
        state_a = runtime_aligned_denoise_state(batch_size=batch_size, noise=noise, num_steps=5)
        times_a = []
        for _ in range(3):
            t0 = time.perf_counter()
            v = model.denoise_one_batch(prefix, state_a)
            v.block_until_ready()
            times_a.append((time.perf_counter() - t0) * 1000)

        # Path B: step_once construction (concat / asarray / stack)
        xs = [noise[i : i + 1] for i in range(batch_size)]
        dts = [jnp.asarray(-1.0 / 5, dtype=jnp.float32) for _ in range(batch_size)]
        times_b = []
        for _ in range(3):
            t0 = time.perf_counter()
            x_t = jnp.concatenate(xs, axis=0)
            step_idx = jnp.asarray([0 for _ in range(batch_size)], dtype=jnp.int32)
            dt = jnp.stack(dts)
            state_b = JaxDenoiseState(x_t=x_t, step_idx=step_idx, num_steps=5, dt=dt)
            v = model.denoise_one_batch(prefix, state_b)
            v.block_until_ready()
            for row, (x, d) in enumerate(zip(xs, dts)):
                new_x = x + d * v[row : row + 1]
                new_x.block_until_ready()
            times_b.append((time.perf_counter() - t0) * 1000)

        print(
            f"[ae] B={batch_size}: aligned={[round(x,1) for x in times_a]} "
            f"step_once_like={[round(x,1) for x in times_b]}",
            flush=True,
        )

    conn.send("done")
    conn.close()


def main() -> None:
    gpu = os.environ.get("GPU_ID", "5")
    import subprocess

    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    ckpt = "/mnt/tianze/models/pi05_libero"
    max_slots = 24
    ctx = mp.get_context("spawn")
    vlm_conn, ae_conn = ctx.Pipe()
    vlm = ctx.Process(target=_vlm_side, args=(vlm_conn, uuid, ckpt, max_slots), name="diag-vlm")
    ae = ctx.Process(target=_ae_side, args=(ae_conn, uuid, ckpt, max_slots), name="diag-ae")
    vlm.start()
    ae.start()
    vlm.join()
    ae.join()
    if vlm.exitcode != 0 or ae.exitcode != 0:
        raise SystemExit(f"diag failed vlm={vlm.exitcode} ae={ae.exitcode}")


if __name__ == "__main__":
    main()
