from __future__ import annotations

from typing import Any


def _child_env_updates(device: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(device),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def run_vlm_worker_entry(
    *args: Any, device: str, env_updates: dict[str, str | None] | None = None, **kwargs: Any
) -> None:
    from openpi.serving.va_split_jax.runtime import _run_jax_vlm_process

    updates: dict[str, str | None] = _child_env_updates(device)
    updates.update(env_updates or {})
    kwargs["env_updates"] = updates
    _run_jax_vlm_process(*args, **kwargs)


def run_ae_worker_entry(
    *args: Any, device: str, env_updates: dict[str, str | None] | None = None, **kwargs: Any
) -> None:
    from openpi.serving.va_split_jax.runtime import _run_jax_ae_process

    updates: dict[str, str | None] = _child_env_updates(device)
    updates.update(env_updates or {})
    kwargs["env_updates"] = updates
    _run_jax_ae_process(*args, **kwargs)
