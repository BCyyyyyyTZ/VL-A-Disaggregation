from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import torch

ArrayKind = Literal["numpy", "jax", "torch"]


def infer_obs_batch_size(obs_batch: dict[str, Any]) -> int:
    prompts = obs_batch.get("prompt")
    if isinstance(prompts, str):
        raise ValueError("infer_batch requires prompt to be a sequence of strings")
    if prompts is not None:
        if not _is_string_sequence(prompts):
            raise ValueError("infer_batch requires prompt to be a sequence of strings")
        return len(prompts)

    leaf = _first_batch_leaf(obs_batch)
    if leaf is None:
        raise ValueError("Cannot infer batch size from observation batch")
    return int(leaf.shape[0])


def split_obs_batch(obs_batch: dict[str, Any]) -> list[dict[str, Any]]:
    batch_size = infer_obs_batch_size(obs_batch)
    return [_slice_batch_value(obs_batch, row, batch_size=batch_size) for row in range(batch_size)]


def stack_transformed_samples(
    samples: Sequence[dict[str, Any]],
    *,
    kind: ArrayKind = "numpy",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot stack an empty sample sequence")
    return _stack_values(list(samples), kind=kind, device=device)


def apply_input_transform_batch(
    obs_batch: dict[str, Any],
    input_transform,
    *,
    kind: Literal["jax", "torch"],
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    samples = [input_transform(sample) for sample in split_obs_batch(obs_batch)]
    return stack_transformed_samples(samples, kind=kind, device=device)


def apply_output_transform_batch(outputs: dict[str, Any], output_transform) -> dict[str, Any]:
    batch_size = infer_output_batch_size(outputs)
    samples = []
    for row in range(batch_size):
        sample = _slice_output_row(outputs, row)
        samples.append(output_transform(sample))
    return stack_transformed_samples(samples, kind="numpy")


def infer_output_batch_size(outputs: dict[str, Any]) -> int:
    actions = outputs.get("actions")
    if actions is None or not hasattr(actions, "shape") or len(actions.shape) == 0:
        raise ValueError("Cannot infer output batch size without batched actions")
    return int(actions.shape[0])


def prepare_batch_noise(
    noise: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    kind: Literal["jax", "torch"],
    device: str | torch.device | None = None,
) -> Any:
    noise_ndim = int(noise.ndim)
    if noise_ndim == 2:
        if batch_size != 1:
            raise ValueError("B > 1 requires noise shape [B, action_horizon, action_dim]")
        noise = noise[None, ...]
    elif noise_ndim != 3:
        raise ValueError("infer_batch noise must have shape [B, action_horizon, action_dim]")

    if int(noise.shape[0]) != batch_size:
        raise ValueError(
            f"infer_batch noise batch dimension {noise.shape[0]} does not match observation batch {batch_size}"
        )

    if kind == "torch":
        if torch.is_tensor(noise):
            return noise.to(device).contiguous() if device is not None else noise.contiguous()
        return (
            torch.from_numpy(np.asarray(noise).copy()).to(device).contiguous()
            if device is not None
            else torch.from_numpy(np.asarray(noise).copy()).contiguous()
        )
    return jnp.asarray(noise)


def _first_batch_leaf(value: Any) -> Any | None:
    if isinstance(value, Mapping):
        for item in value.values():
            leaf = _first_batch_leaf(item)
            if leaf is not None:
                return leaf
        return None
    if _is_string_sequence(value) or isinstance(value, str):
        return None
    if hasattr(value, "shape") and len(value.shape) > 0:
        return value
    return None


def _slice_batch_value(value: Any, row: int, *, batch_size: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_batch_value(item, row, batch_size=batch_size) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if _is_string_sequence(value):
        if len(value) != batch_size:
            raise ValueError(f"Prompt batch length {len(value)} does not match inferred batch size {batch_size}")
        return value[row]
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        if len(value.shape) == 0:
            return value
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"Leaf batch dimension {value.shape[0]} does not match inferred batch size {batch_size}")
        return value[row]
    if hasattr(value, "shape") and len(value.shape) > 0:
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"Leaf batch dimension {value.shape[0]} does not match inferred batch size {batch_size}")
        return value[row]
    return value


def _slice_output_row(value: Any, row: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_output_row(item, row) for key, item in value.items()}
    if torch.is_tensor(value):
        return np.asarray(value[row].detach().cpu())
    if hasattr(value, "shape") and len(value.shape) > 0:
        return np.asarray(value[row])
    return value


def _stack_values(values: list[Any], *, kind: ArrayKind, device: str | torch.device | None) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack_values([value[key] for value in values], kind=kind, device=device) for key in first}
    if isinstance(first, str):
        return list(values)
    if first is None:
        return None
    if torch.is_tensor(first):
        stacked = torch.stack(values, dim=0)
        if kind == "numpy":
            return np.asarray(stacked.detach().cpu()).copy()
        if kind == "jax":
            return jnp.asarray(np.asarray(stacked.detach().cpu()).copy())
        return stacked.to(device).contiguous() if device is not None else stacked.contiguous()

    array = np.asarray(values).copy()
    if array.dtype.kind in {"O", "U", "S"}:
        return array
    if kind == "numpy":
        return array
    if kind == "jax":
        return jnp.asarray(array)
    return (
        torch.from_numpy(array).to(device).contiguous() if device is not None else torch.from_numpy(array).contiguous()
    )


def _is_string_sequence(value: Any) -> bool:
    if isinstance(value, str):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.dtype.kind in {"O", "U", "S"} and all(isinstance(item, str) for item in value)
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)
