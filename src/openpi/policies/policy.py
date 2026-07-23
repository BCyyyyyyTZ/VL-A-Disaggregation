from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import batch_inference as _batch
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._is_pytorch_model:
            actions, component_timing = _sample_pytorch_actions_with_component_timing(
                self._model,
                self._sample_actions,
                self._pytorch_device,
                observation,
                sample_kwargs,
            )
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            component_timing = {}
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            **component_timing,
        }
        return outputs

    def infer_batch(self, obs_batch: dict, *, noise: np.ndarray | None = None) -> dict:
        if not self._is_pytorch_model:
            raise NotImplementedError("Policy.infer_batch currently supports PyTorch models only")

        inputs = _batch.apply_input_transform_batch(
            obs_batch,
            self._input_transform,
            kind="torch",
            device=self._pytorch_device,
        )
        batch_size = int(inputs["state"].shape[0])

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            sample_kwargs["noise"] = _batch.prepare_batch_noise(
                noise,
                batch_size=batch_size,
                kind="torch",
                device=self._pytorch_device,
            )

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, component_timing = _sample_pytorch_actions_with_component_timing(
            self._model,
            self._sample_actions,
            self._pytorch_device,
            observation,
            sample_kwargs,
        )
        model_time = time.monotonic() - start_time

        outputs = _batch.apply_output_transform_batch(
            {
                "state": inputs["state"],
                "actions": actions,
            },
            self._output_transform,
        )
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "effective_batch": batch_size,
            "policy_effective_batch": batch_size,
            **component_timing,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _sample_pytorch_actions_with_component_timing(
    model: Any,
    sample_actions,
    device: str,
    observation: _model.Observation,
    sample_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if not _supports_pytorch_component_timing(model, sample_kwargs):
        return sample_actions(device, observation, **sample_kwargs), {}

    batch_size = int(observation.state.shape[0])
    num_steps = int(sample_kwargs.get("num_steps", 10))
    noise = sample_kwargs.get("noise")

    with torch.no_grad():
        _sync_torch_device(device)
        vlm_start = time.monotonic()
        prefix_feature = model.build_prefix_feature(device, observation)
        _sync_torch_device(device)
        vlm_ms = (time.monotonic() - vlm_start) * 1000

        denoise_state = model.init_denoise_state(device, batch_size, noise, num_steps)
        ae_step_ms: list[float] = []
        for _ in range(num_steps):
            _sync_torch_device(device)
            ae_step_start = time.monotonic()
            v_t = model.denoise_one_batch(prefix_feature, denoise_state)
            denoise_state = dataclasses.replace(
                denoise_state,
                x_t=denoise_state.x_t + denoise_state.dt * v_t,
                step_idx=denoise_state.step_idx + 1,
            )
            _sync_torch_device(device)
            ae_step_ms.append((time.monotonic() - ae_step_start) * 1000)

    ae_ms = sum(ae_step_ms)
    timing = {
        "baseline_vlm_ms": vlm_ms,
        "baseline_ae_ms": ae_ms,
        "baseline_ae_step_ms": ae_ms / len(ae_step_ms) if ae_step_ms else 0.0,
        "baseline_ae_steps": float(num_steps),
        "baseline_effective_batch": float(batch_size),
    }
    return denoise_state.x_t, timing


def _supports_pytorch_component_timing(model: Any, sample_kwargs: dict[str, Any]) -> bool:
    if set(sample_kwargs) - {"noise", "num_steps"}:
        return False
    return all(
        callable(getattr(model, name, None))
        for name in ("build_prefix_feature", "init_denoise_state", "denoise_one_batch")
    )


def _sync_torch_device(device: str) -> None:
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
