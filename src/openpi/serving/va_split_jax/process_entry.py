from __future__ import annotations

from typing import Any


def _child_env_updates(device: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(device),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def _apply_env_updates(updates: dict[str, str | None]) -> None:
    import os

    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_vlm_worker_entry(
    *args: Any, device: str, env_updates: dict[str, str | None] | None = None, **kwargs: Any
) -> None:
    updates: dict[str, str | None] = _child_env_updates(device)
    updates.update(env_updates or {})
    _apply_env_updates(updates)

    from openpi.serving.va_split_jax.runtime import _run_jax_vlm_process

    _run_jax_vlm_process(*args, **kwargs)


def run_ae_worker_entry(
    *args: Any, device: str, env_updates: dict[str, str | None] | None = None, **kwargs: Any
) -> None:
    updates: dict[str, str | None] = _child_env_updates(device)
    updates.update(env_updates or {})
    _apply_env_updates(updates)

    from openpi.serving.va_split_jax.runtime import _run_jax_ae_process

    _run_jax_ae_process(*args, **kwargs)
