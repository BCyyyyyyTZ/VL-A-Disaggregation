from __future__ import annotations

from collections.abc import Sequence
import functools
import pathlib
import time
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import batch_inference as _batch
from openpi.policies import policy as _policy
from openpi.serving.va_split_jax.compile import DEFAULT_WARMUP_MAX_BATCH_SIZE
from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import make_pi0_prefix_feature_template
from openpi.serving.va_split_jax.compile import prune_split_model_for_role
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


def _load_jax_model(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path | str, *, role: str | None = None):
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    params_dir = checkpoint_dir / "params"
    if not params_dir.exists():
        if (checkpoint_dir / "model.safetensors").exists():
            raise ValueError(
                "JAX VA split policy requires a JAX checkpoint containing a params/ directory; "
                "found only PyTorch model.safetensors. Download or provide a JAX checkpoint."
            )
        raise ValueError(f"JAX checkpoint params directory not found: {params_dir}")
    if role is not None:
        return _load_role_pruned_jax_model(train_config, params_dir, role=role)
    return train_config.model.load(_model.restore_params(params_dir, dtype=jnp.bfloat16))


def _load_role_pruned_jax_model(train_config: _config.TrainConfig, params_dir: pathlib.Path, *, role: str):
    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    model = prune_split_model_for_role(model, role=role)
    graphdef, state = nnx.split(model)
    target_params = state.to_pure_dict()
    params = _restore_params_for_target_state(params_dir, target_params, dtype=jnp.bfloat16)
    at.check_pytree_equality(expected=target_params, got=params, check_shapes=True, check_dtypes=False)
    state.replace_by_pure_dict(params)
    return nnx.merge(graphdef, state)


def _restore_params_for_target_state(
    params_dir: pathlib.Path,
    target_params: at.Params,
    *,
    dtype: jnp.dtype | None,
) -> at.Params:
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_dir)
        metadata_params = metadata["params"]
        item = {"params": _make_role_restore_item(metadata_params, target_params)}
        restore_args = jax.tree.map(
            lambda value: ocp.PLACEHOLDER
            if value is ocp.PLACEHOLDER
            else ocp.ArrayRestoreArgs(sharding=sharding, restore_type=jax.Array, dtype=dtype),
            item,
            is_leaf=lambda value: value is ocp.PLACEHOLDER,
        )
        params = ckptr.restore(
            params_dir,
            ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
        )["params"]

    flat_params = traverse_util.flatten_dict(params)
    if flat_params and all(key_path[-1] == "value" for key_path in flat_params):
        flat_params = {key_path[:-1]: value for key_path, value in flat_params.items()}
        params = traverse_util.unflatten_dict(flat_params)
    return ocp.transform_utils.intersect_trees(target_params, params)


def _make_role_restore_item(metadata_params: at.Params, target_params: at.Params) -> at.Params:
    flat_metadata = traverse_util.flatten_dict(metadata_params)
    target_key_paths = set(traverse_util.flatten_dict(target_params))
    metadata_has_value_suffix = bool(flat_metadata) and all(key_path[-1] == "value" for key_path in flat_metadata)
    item = {}
    for key_path, value in flat_metadata.items():
        normalized_key_path = key_path[:-1] if metadata_has_value_suffix and key_path[-1] == "value" else key_path
        item[key_path] = value if normalized_key_path in target_key_paths else ocp.PLACEHOLDER
    return traverse_util.unflatten_dict(item)


def _make_jax_prefix_feature_template(train_config: _config.TrainConfig):
    return make_pi0_prefix_feature_template(train_config.model)


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
    jax_compile_ae: bool = True,
    jax_compile_vlm: bool = True,
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
        else min(max_vlm_batch_size * 3, DEFAULT_WARMUP_MAX_BATCH_SIZE)
    )
    runtime = JaxProcessVASplitRuntime(
        model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir),
        vlm_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="vlm"),
        ae_model_factory=functools.partial(_load_jax_model, train_config, checkpoint_dir, role="ae"),
        prefix_template_factory=functools.partial(_make_jax_prefix_feature_template, train_config),
        max_ae_batch_size=max_ae_batch_size,
        max_vlm_batch_size=max_vlm_batch_size,
        max_vlm_wait_ms=max_vlm_wait_ms,
        result_timeout_s=result_timeout_s,
        vlm_env_updates=_mps_env_updates(vlm_sm_percent),
        ae_env_updates=_mps_env_updates(ae_sm_percent),
        compile_config=JaxCompileConfig(
            enabled=jax_compile,
            warmup_enabled=jax_compile_warmup,
            compile_ae=jax_compile_ae,
            compile_vlm=jax_compile_vlm,
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
