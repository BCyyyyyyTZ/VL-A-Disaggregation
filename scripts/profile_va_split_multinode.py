#!/usr/bin/env python3
"""Profile PyTorch or JAX multi-GPU / multi-node VLM–AE split.

Same-machine default (``--transport host_shm``) follows the Triton bench:
N VLM GPUs + one AE GPU, host shared-memory prefix lanes, fair credits,
overlapped D2H and packed noise.

Cross-machine (``--transport tcp``) keeps that schedule and ships prefix rows
over TCP. Start the AE, then each VLM, then the coordinator:

  python scripts/profile_va_split_multinode.py --role ae --backend pytorch ...
  python scripts/profile_va_split_multinode.py --role vlm --worker-index 0 ...
  python scripts/profile_va_split_multinode.py --role coordinator ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpi.serving.va_split_multinode.messages import RunConfig
from openpi.serving.va_split_multinode.runtime import run_benchmark


def _parse_ints(value: str) -> tuple[int, ...]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"expected a comma-separated device list, got {value!r}")
    return tuple(items)


def _parse_rates(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile multi-GPU / multi-node VA-split")
    parser.add_argument("--backend", choices=("pytorch", "jax", "fake"), required=True)
    parser.add_argument("--mode", choices=("ours", "baseline"), default="ours")
    parser.add_argument("--role", choices=("all", "vlm", "ae", "coordinator"), default="all")
    parser.add_argument("--transport", choices=("host_shm", "tcp", "local"), default="host_shm")
    parser.add_argument("--launch", choices=("process", "thread"), default="process")
    parser.add_argument("--vlm-devices", default="0,1")
    parser.add_argument("--ae-device", type=int, default=2)
    parser.add_argument("--baseline-devices", default="")
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--policy-dir", default="")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--rate", type=float, default=16.0)
    parser.add_argument("--rates", default="")
    parser.add_argument("--arrival-scope", choices=("global", "per_worker"), default="global")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--ae-max-batch-size", type=int, default=16)
    parser.add_argument("--max-prefix-slots", type=int, default=256)
    parser.add_argument("--max-wait-ms", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--slo-ms", type=float, default=200.0)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--token-len", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup-max-batch", type=int, default=4)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--ae-addr", default="")
    parser.add_argument("--vlm-addrs", default="")
    parser.add_argument("--result-addr", default="")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--output", default="")
    parser.add_argument("--overlap-d2h", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packed-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-compile", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    rates = _parse_rates(args.rates) if args.rates else [args.rate]
    summaries = []
    for rate in rates:
        cfg = RunConfig(
            backend=args.backend,
            mode=args.mode,
            role=args.role,
            vlm_devices=_parse_ints(args.vlm_devices),
            ae_device=args.ae_device,
            baseline_devices=_parse_ints(args.baseline_devices) if args.baseline_devices else (),
            transport=args.transport,
            launch=args.launch,
            num_requests=args.num_requests,
            rate=rate,
            arrival_scope=args.arrival_scope,
            seed=args.seed,
            max_batch_size=args.max_batch_size,
            ae_max_batch_size=args.ae_max_batch_size,
            max_prefix_slots=args.max_prefix_slots,
            max_wait_ms=args.max_wait_ms,
            num_steps=args.num_steps,
            timeout_s=args.timeout_s,
            startup_timeout_s=args.startup_timeout_s,
            slo_ms=args.slo_ms,
            overlap_d2h=bool(args.overlap_d2h),
            packed_noise=bool(args.packed_noise),
            policy_config=args.policy_config,
            policy_dir=args.policy_dir,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            token_len=args.token_len,
            image_size=args.image_size,
            warmup=bool(args.warmup),
            warmup_max_batch=args.warmup_max_batch,
            jax_compile=bool(args.jax_compile),
            ae_addr=args.ae_addr,
            vlm_addrs=tuple(part.strip() for part in args.vlm_addrs.split(",") if part.strip()),
            result_addr=args.result_addr,
            bind_host=args.bind_host,
            worker_index=args.worker_index,
        )
        payload = run_benchmark(cfg)
        public = {key: value for key, value in payload.items() if key != "results"}
        public["pass_requests"] = payload.get("completed")
        summaries.append(public)
        e2e = public.get("e2e", {})
        print(
            f"backend={public.get('backend')} mode={public.get('mode')} transport={public.get('transport')} "
            f"rate={rate} completed={public.get('completed')} failed={public.get('failed')} "
            f"e2e_p50={e2e.get('p50_ms')} e2e_p95={e2e.get('p95_ms')} "
            f"throughput={public.get('throughput_rps')} goodput={public.get('goodput_rps')}",
            flush=True,
        )
    document = summaries[0] if len(summaries) == 1 else {"rates": summaries}
    text = json.dumps(document, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    main()
