"""Process entry that sets CUDA visibility before importing JAX."""

from __future__ import annotations

import os
from typing import Any

from openpi.serving.va_split_multinode.messages import RunConfig


def entry(
    role: str,
    device: int,
    cfg: RunConfig,
    work_q: Any,
    prefix_q: Any,
    result_q: Any,
    pool_q: Any,
    credit_q: Any,
    worker_index: int,
    num_vlm: int,
) -> None:
    if cfg.backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if device >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["JAX_PLATFORMS"] = "cpu"
    elif cfg.backend == "pytorch":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from openpi.serving.va_split_multinode.workers import run_worker

    run_worker(
        role,
        cfg,
        device=device,
        worker_index=worker_index,
        num_vlm=num_vlm,
        work_q=work_q,
        prefix_q=prefix_q,
        result_q=result_q,
        pool_q=pool_q,
        credit_q=credit_q,
    )
