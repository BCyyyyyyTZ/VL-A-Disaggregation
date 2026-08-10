from __future__ import annotations

from collections.abc import Sequence
import functools
import pathlib
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.policies import batch_inference as _batch
from openpi.policies import policy as _policy
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import ModelWithPrefixTemplate
from openpi.serving.va_split_jax.compile import make_model_observation_factory
from openpi.serving.va_split_jax.compile import make_prefix_feature_template
from openpi.serving.va_split_jax.compile import prune_split_model_for_role
from openpi.serving.va_split_jax.multigpu_config import JaxMultiGpuVASplitConfig
from openpi.serving.va_split_jax.runtime import JaxMultiGpuProcessVASplitRuntime
from openpi.serving.va_split_jax.runtime import JaxProcessVASplitRuntime
from openpi.serving.va_split_jax.types import JaxActionResult
from openpi.shared import array_typing as at
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


def _load_jax_model(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    role: str | None = None,
    device_index: int | None = None,
):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        if (checkpoint_dir / "model.safetensors").exists():
            raise ValueError(
                "JAX VA split policy requires a JAX checkpoint containing a params/ directory; "
                "found only PyTorch model.safetensors. Download or provide a JAX checkpoint."
            )
        raise ValueError(f"JAX checkpoint params directory not found: {params_dir}")

    sharding = _single_device_sharding(device_index) if device_index is not None else None
    if role is None:
        return train_config.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16, sharding=sharding))

    model, expected_params = _role_pruned_model_and_expected_params(train_config.model, role=role)
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16, sharding=sharding, target=expected_params)
    params = ocp.transform_utils.intersect_trees(expected_params, params)
    at.check_pytree_equality(expected=expected_params, got=params, check_shapes=True, check_dtypes=False)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    loaded = nnx.merge(graphdef, state)
    if role == "ae":
        return _wrap_ae_model_with_prefix_template(
            loaded,
            _make_prefix_feature_template_for_model_config(train_config.model),
        )
    return loaded


def _wrap_ae_model_with_prefix_template(model: Any, template: JaxPrefixFeature) -> ModelWithPrefixTemplate:
    return ModelWithPrefixTemplate(model=model, prefix_template=template)


def _multigpu_policy_metadata(
    policy_metadata: dict[str, Any] | None,
    *,
    va_split_runtime: str,
    vlm_devices: tuple[str, ...],
    ae_device: str,
    cross_card_transfer_strategy: str,
) -> dict[str, Any]:
    return {
        **(policy_metadata or {}),
        "va_split_runtime": va_split_runtime,
        "vlm_devices": vlm_devices,
        "ae_device": ae_device,
        "cross_card_transfer_strategy": cross_card_transfer_strategy,
    }


def _single_device_sharding(device_index: int) -> jax.sharding.SingleDeviceSharding:
    devices = jax.devices()
    if device_index < 0 or device_index >= len(devices):
        raise ValueError(f"JAX device_index={device_index} is out of range for visible devices: {devices}")
    return jax.sharding.SingleDeviceSharding(devices[device_index])


def _role_pruned_model_and_expected_params(
    model_config: _model.BaseModelConfig,
    *,
    role: str,
) -> tuple[_model.BaseModel, at.Params]:
    model = nnx.eval_shape(model_config.create, jax.random.key(0))
    model = prune_split_model_for_role(model, role=role)
    _, state = nnx.split(model)
    return model, state.to_pure_dict()


def _make_prefix_feature_template_for_model_config(model_config: _model.BaseModelConfig):
    if hasattr(model_config, "paligemma_variant"):
        paligemma_config = _gemma.get_config(model_config.paligemma_variant)
        observation_spec, _ = model_config.inputs_spec(batch_size=1)
        siglip_patch_size = 14
        image_tokens = 0
        for image_spec in observation_spec.images.values():
            height, width = image_spec.shape[1:3]
            if height % siglip_patch_size or width % siglip_patch_size:
                raise ValueError(f"Image shape {image_spec.shape} is not divisible by SigLIP patch size 14")
            image_tokens += (height // siglip_patch_size) * (width // siglip_patch_size)
        text_tokens = 0
        if observation_spec.tokenized_prompt is not None:
            text_tokens = int(observation_spec.tokenized_prompt.shape[1])
        prefix_tokens = int(image_tokens + text_tokens)
        dtype = jnp.dtype(getattr(model_config, "dtype", jnp.bfloat16))
        kv_shape = (
            int(paligemma_config.depth),
            1,
            prefix_tokens,
            int(paligemma_config.num_kv_heads),
            int(paligemma_config.head_dim),
        )
        return JaxPrefixFeature(
            past_key_values=(jnp.zeros(kv_shape, dtype=dtype), jnp.zeros(kv_shape, dtype=dtype)),
            prefix_pad_masks=jnp.ones((1, prefix_tokens), dtype=jnp.bool_),
            state=jnp.zeros((1, int(model_config.action_dim)), dtype=jnp.float32),
        )

    model = nnx.eval_shape(model_config.create, jax.random.key(0))
    observation_factory = make_model_observation_factory(model)
    return make_prefix_feature_template(model, observation_factory)


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
        vlm_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="vlm", device_index=0),
        ae_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="ae", device_index=0),
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
    cross_card_transfer_strategy: str = "device-direct",
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
        cross_card_transfer_strategy=cross_card_transfer_strategy,
    )
    warmup_max_batch_size = (
        jax_compile_warmup_max_batch_size
        if jax_compile_warmup_max_batch_size is not None
        else int(config.max_prefix_slots or max_vlm_batch_size * 3)
    )
    runtime = JaxMultiGpuProcessVASplitRuntime(
        model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir),
        vlm_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="vlm", device_index=1),
        ae_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="ae", device_index=0),
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
        metadata=_multigpu_policy_metadata(
            train_config.policy_metadata,
            va_split_runtime="jax-multigpu-split-ipc",
            vlm_devices=tuple(config.vlm_devices),
            ae_device=config.ae_device,
            cross_card_transfer_strategy=config.cross_card_transfer_strategy,
        ),
    )
