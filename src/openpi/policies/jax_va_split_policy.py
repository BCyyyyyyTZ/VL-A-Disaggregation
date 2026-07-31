from __future__ import annotations

from collections.abc import Sequence
import functools
import os
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import batch_inference as _batch
from openpi.policies import policy as _policy
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.runtime import JaxProcessVASplitRuntime
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class JaxVASplitPolicy(_policy.BasePolicy):
    supports_concurrent_infer = True

    def __init__(
        self,
        *,
        runtime: Any,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._runtime = runtime
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_array = jnp.asarray(noise)
            if noise_array.ndim == 2:
                noise_array = noise_array[None, ...]
            sample_kwargs["noise"] = noise_array

        start_time = time.monotonic()
        runtime_result = self._runtime.infer(inputs, sample_kwargs)
        infer_ms = (time.monotonic() - start_time) * 1000
        if isinstance(runtime_result, JaxActionResult):
            actions = runtime_result.actions
            runtime_timing = dict(runtime_result.timing or {})
        else:
            actions = runtime_result
            runtime_timing = {}
        runtime_timing = {**self._runtime_compile_timing(), **runtime_timing}

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": infer_ms, **runtime_timing}
        return outputs

    def infer_batch(self, obs_batch: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = _batch.apply_input_transform_batch(
            obs_batch,
            self._input_transform,
            kind="jax",
        )
        batch_size = int(inputs["state"].shape[0])

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            sample_kwargs["noise"] = _batch.prepare_batch_noise(
                noise,
                batch_size=batch_size,
                kind="jax",
            )

        start_time = time.monotonic()
        runtime_result = self._runtime.infer_batch(inputs, sample_kwargs)
        infer_ms = (time.monotonic() - start_time) * 1000
        if isinstance(runtime_result, JaxActionResult):
            actions = runtime_result.actions
            runtime_timing = dict(runtime_result.timing or {})
        else:
            actions = runtime_result
            runtime_timing = {}
        runtime_timing = {**self._runtime_compile_timing(), **runtime_timing}

        outputs = _batch.apply_output_transform_batch(
            {
                "state": inputs["state"],
                "actions": actions,
            },
            self._output_transform,
        )
        outputs["policy_timing"] = {
            "infer_ms": infer_ms,
            "effective_batch": batch_size,
            "policy_effective_batch": batch_size,
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

    def _runtime_compile_timing(self) -> dict[str, float]:
        compile_timing = getattr(self._runtime, "compile_timing", None)
        if isinstance(compile_timing, dict):
            return dict(compile_timing)
        return {}


def _load_jax_model(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path | str):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        if (checkpoint_dir / "model.safetensors").exists():
            raise ValueError(
                "JAX VA split policy requires a JAX checkpoint containing a params/ directory; "
                "found only PyTorch model.safetensors. Download or provide a JAX checkpoint."
            )
        raise ValueError(f"JAX checkpoint params directory not found: {params_dir}")
    return train_config.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))


def _mps_env_updates(sm_percent: int) -> dict[str, str | None]:
    updates: dict[str, str | None] = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    if sm_percent > 0:
        updates["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(sm_percent)
    else:
        updates["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = None
    return updates


def create_trained_jax_va_split_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str = "/data1/miliang/models/RLinf-Pi05-LIBERO-SFT",
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    max_ae_batch_size: int = 8,
    max_vlm_batch_size: int = 8,
    max_vlm_wait_ms: float = 2.0,
    ae_sm_percent: int = 20,
    vlm_sm_percent: int = 0,
    result_timeout_s: float = 120.0,
    jax_compile: bool = True,
    jax_compile_warmup: bool = True,
    jax_compile_warmup_max_batch_size: int = 32,
) -> JaxVASplitPolicy:
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        if (checkpoint_dir / "model.safetensors").exists():
            raise ValueError(
                "JAX VA split policy requires a JAX checkpoint containing params/. "
                "The provided checkpoint appears to be PyTorch-only (model.safetensors)."
            )
        raise ValueError(f"JAX checkpoint params directory not found: {params_dir}")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_checkpoint_norm_stats(checkpoint_dir, data_config.asset_id)

    runtime = JaxProcessVASplitRuntime(
        model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir),
        max_ae_batch_size=max_ae_batch_size,
        max_vlm_batch_size=max_vlm_batch_size,
        max_vlm_wait_ms=max_vlm_wait_ms,
        result_timeout_s=result_timeout_s,
        vlm_env_updates=_mps_env_updates(vlm_sm_percent),
        ae_env_updates=_mps_env_updates(ae_sm_percent),
        compile_config=JaxCompileConfig(
            enabled=jax_compile,
            warmup_enabled=jax_compile_warmup,
            warmup_max_batch_size=jax_compile_warmup_max_batch_size,
        ),
    )
    return JaxVASplitPolicy(
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
    )
