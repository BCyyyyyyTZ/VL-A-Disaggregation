from __future__ import annotations

from collections.abc import Sequence
import functools
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
from openpi.serving.va_split_jax.multigpu_config import JaxMultiGpuVASplitConfig
from openpi.serving.va_split_jax.runtime import JaxMultiGpuProcessVASplitRuntime
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
        policy_stage_start = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        transform_start = time.monotonic()
        inputs = self._input_transform(inputs)
        input_transform_ms = (time.monotonic() - transform_start) * 1000
        image_stage_start = time.monotonic()
        inputs = _normalize_uint8_images_for_vlm_ipc(inputs)
        parent_image_stage_ms = (time.monotonic() - image_stage_start) * 1000
        batch_stage_start = time.monotonic()
        inputs = jax.tree.map(lambda x: np.asarray(x)[None, ...], inputs)
        input_batch_stage_ms = (time.monotonic() - batch_stage_start) * 1000

        sample_kwargs_stage_start = time.monotonic()
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_array = np.asarray(noise).copy()
            if noise_array.ndim == 2:
                noise_array = noise_array[None, ...]
            sample_kwargs["noise"] = noise_array
        sample_kwargs_stage_ms = (time.monotonic() - sample_kwargs_stage_start) * 1000
        policy_input_stage_ms = (time.monotonic() - policy_stage_start) * 1000

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
        outputs["policy_timing"] = {
            "infer_ms": infer_ms,
            "policy_input_stage_ms": policy_input_stage_ms,
            "policy_input_transform_ms": input_transform_ms,
            "policy_input_batch_stage_ms": input_batch_stage_ms,
            "policy_sample_kwargs_stage_ms": sample_kwargs_stage_ms,
            "policy_observation_from_dict_ms": 0.0,
            "vlm_parent_image_stage_ms": parent_image_stage_ms,
            **runtime_timing,
        }
        return outputs

    def infer_batch(self, obs_batch: dict, *, noise: np.ndarray | None = None) -> dict:
        policy_stage_start = time.monotonic()
        transform_start = time.monotonic()
        inputs = _batch.apply_input_transform_batch(
            obs_batch,
            self._input_transform,
            kind="numpy",
        )
        input_transform_ms = (time.monotonic() - transform_start) * 1000
        image_stage_start = time.monotonic()
        inputs = _normalize_uint8_images_for_vlm_ipc(inputs)
        parent_image_stage_ms = (time.monotonic() - image_stage_start) * 1000
        batch_size = int(inputs["state"].shape[0])

        sample_kwargs_stage_start = time.monotonic()
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            sample_kwargs["noise"] = _batch.prepare_batch_noise(
                noise,
                batch_size=batch_size,
                kind="numpy",
            )
        sample_kwargs_stage_ms = (time.monotonic() - sample_kwargs_stage_start) * 1000
        policy_input_stage_ms = (time.monotonic() - policy_stage_start) * 1000

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
            "policy_input_stage_ms": policy_input_stage_ms,
            "policy_input_transform_ms": input_transform_ms,
            "policy_input_batch_stage_ms": 0.0,
            "policy_sample_kwargs_stage_ms": sample_kwargs_stage_ms,
            "policy_observation_from_dict_ms": 0.0,
            "vlm_parent_image_stage_ms": parent_image_stage_ms,
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


def _normalize_uint8_images_for_vlm_ipc(inputs: dict[str, Any]) -> dict[str, Any]:
    images = inputs.get("image")
    if not isinstance(images, dict):
        return inputs

    staged_images = {}
    changed = False
    for key, value in images.items():
        arr = np.asarray(value)
        if arr.dtype == np.uint8:
            normalized = arr.astype(np.float32)
            normalized *= 2.0 / 255.0
            normalized -= 1.0
            staged_images[key] = np.ascontiguousarray(normalized)
            changed = True
        else:
            staged_images[key] = np.asarray(value)

    if not changed:
        return inputs
    staged = dict(inputs)
    staged["image"] = staged_images
    return staged


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
    checkpoint_dir: pathlib.Path | str = "/mnt/tianze/models/pi05_libero",
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
    jax_compile_warmup_max_batch_size: int | None = None,
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

    warmup_max_batch_size = (
        jax_compile_warmup_max_batch_size
        if jax_compile_warmup_max_batch_size is not None
        else max_vlm_batch_size * 3
    )
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
            warmup_max_batch_size=warmup_max_batch_size,
            num_steps=int((sample_kwargs or {}).get("num_steps", 10)),
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

def create_trained_jax_multigpu_va_split_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str = "/mnt/tianze/models/pi05_libero",
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
    vlm_devices: str | Sequence[str],
    ae_device: str,
    max_ae_batch_size: int = 64,
    max_vlm_batch_size: int = 8,
    max_vlm_wait_ms: float = 1.0,
    result_timeout_s: float = 120.0,
    jax_compile: bool = True,
    jax_compile_warmup: bool = True,
    jax_compile_warmup_max_batch_size: int | None = None,
) -> JaxVASplitPolicy:
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        if (checkpoint_dir / "model.safetensors").exists():
            raise ValueError(
                "JAX multi-GPU VA split policy requires a JAX checkpoint containing params/. "
                "The provided checkpoint appears to be PyTorch-only (model.safetensors)."
            )
        raise ValueError(f"JAX checkpoint params directory not found: {params_dir}")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_checkpoint_norm_stats(checkpoint_dir, data_config.asset_id)

    config = JaxMultiGpuVASplitConfig(
        vlm_devices=vlm_devices,
        ae_device=ae_device,
        max_vlm_batch_size=max_vlm_batch_size,
        max_vlm_wait_ms=max_vlm_wait_ms,
        max_ae_batch_size=max_ae_batch_size,
    )
    warmup_max_batch_size = (
        jax_compile_warmup_max_batch_size
        if jax_compile_warmup_max_batch_size is not None
        else int(config.max_prefix_slots or max_vlm_batch_size * 3)
    )
    runtime = JaxMultiGpuProcessVASplitRuntime(
        model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir),
        config=config,
        result_timeout_s=result_timeout_s,
        vlm_env_updates=_mps_env_updates(0),
        ae_env_updates=_mps_env_updates(0),
        compile_config=JaxCompileConfig(
            enabled=jax_compile,
            warmup_enabled=jax_compile_warmup,
            warmup_max_batch_size=warmup_max_batch_size,
            num_steps=int((sample_kwargs or {}).get("num_steps", 10)),
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
        metadata={
            **train_config.policy_metadata,
            "va_split_runtime": "jax-multigpu-split-ipc",
            "vlm_devices": tuple(config.vlm_devices),
            "ae_device": config.ae_device,
        },
    )

