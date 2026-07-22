from __future__ import annotations

from collections.abc import Sequence
import functools
import os
import pathlib
import time
from typing import Any

import jax
import numpy as np
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.policies import policy as _policy
from openpi.serving.va_split.runtime import ProcessVASplitRuntime
from openpi.serving.va_split.types import ActionResult
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class VASplitPolicy(_policy.BasePolicy):
    """OpenPI policy wrapper that delegates PyTorch inference to a VLM/AE split runtime."""

    supports_concurrent_infer = True

    def __init__(
        self,
        *,
        runtime: Any,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
    ):
        self._runtime = runtime
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._pytorch_device = pytorch_device

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x))[None, ...], inputs)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_tensor = torch.from_numpy(noise)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor[None, ...]
            sample_kwargs["noise"] = noise_tensor

        start_time = time.monotonic()
        runtime_result = self._runtime.infer(inputs, sample_kwargs)
        infer_ms = (time.monotonic() - start_time) * 1000

        if isinstance(runtime_result, ActionResult):
            actions = runtime_result.actions
            runtime_timing = dict(runtime_result.timing or {})
        else:
            actions = runtime_result
            runtime_timing = {}

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()) if torch.is_tensor(x) else x, outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": infer_ms,
            **runtime_timing,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def reset(self) -> None:
        if hasattr(self._runtime, "reset"):
            self._runtime.reset()

    def shutdown(self) -> None:
        if hasattr(self._runtime, "shutdown"):
            self._runtime.shutdown()

    def close(self) -> None:
        self.shutdown()


def _load_pytorch_model(train_config: _config.TrainConfig, weight_path: str):
    model = train_config.model.load_pytorch(train_config, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model


def _mps_env_updates(sm_percent: int) -> dict[str, str | None]:
    if sm_percent > 0:
        return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(sm_percent)}
    return {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": None}


def create_trained_va_split_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    max_ae_batch_size: int = 8,
    ae_sm_percent: int = 20,
    vlm_sm_percent: int = 0,
) -> VASplitPolicy:
    """Create a PyTorch VA split policy from a trained checkpoint."""
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weight_path):
        raise ValueError("VA split policy requires a PyTorch checkpoint containing model.safetensors.")

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    runtime = ProcessVASplitRuntime(
        model_factory=functools.partial(_load_pytorch_model, train_config, weight_path),
        device=pytorch_device,
        max_ae_batch_size=max_ae_batch_size,
        vlm_env_updates=_mps_env_updates(vlm_sm_percent),
        ae_env_updates=_mps_env_updates(ae_sm_percent),
    )
    return VASplitPolicy(
        runtime=runtime,
        transforms=[
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        pytorch_device=pytorch_device,
    )
