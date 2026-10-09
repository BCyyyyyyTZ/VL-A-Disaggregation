"""Framework backends for multi-GPU / multi-node VA-split.

Each backend exposes the same VLM export and AE step surface. Prefix rows are
plain numpy dicts so host shared memory and TCP carry one payload format.
"""

from __future__ import annotations

import os
import time
from typing import Any
from typing import Protocol

import numpy as np

from openpi.serving.va_split_multinode.messages import RunConfig


class PrefixExport(Protocol):
    def materialize(self) -> list[dict[str, np.ndarray]]: ...


class _ImmediateExport:
    def __init__(self, rows: list[dict[str, np.ndarray]]) -> None:
        self._rows = rows

    def materialize(self) -> list[dict[str, np.ndarray]]:
        return self._rows


def synthetic_observation(cfg: RunConfig, *, batch_size: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    image_shape = (batch_size, cfg.image_size, cfg.image_size, 3)
    images = {key: rng.uniform(-1.0, 1.0, size=image_shape).astype(np.float32) for key in keys}
    masks = {key: np.ones((batch_size,), dtype=bool) for key in keys}
    masks["right_wrist_0_rgb"] = np.zeros((batch_size,), dtype=bool)
    return {
        "image": images,
        "image_mask": masks,
        "state": np.zeros((batch_size, cfg.state_dim), dtype=np.float32),
        "tokenized_prompt": np.ones((batch_size, cfg.token_len), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((batch_size, cfg.token_len), dtype=bool),
    }


def unbatch_observation(observation: dict[str, Any]) -> list[dict[str, Any]]:
    from openpi.serving.va_split_multinode.host_lane_pool import slice_observation

    state = np.asarray(observation["state"])
    return [slice_observation(observation, index) for index in range(int(state.shape[0]))]


def fit_batched_observation(observation: dict[str, Any], *, token_len: int, state_dim: int) -> dict[str, Any]:
    fitted = dict(observation)
    state = np.asarray(fitted["state"], dtype=np.float32)
    if state.shape[-1] < state_dim:
        pad = np.zeros((*state.shape[:-1], state_dim - state.shape[-1]), dtype=np.float32)
        state = np.concatenate([state, pad], axis=-1)
    elif state.shape[-1] > state_dim:
        state = state[..., :state_dim]
    batch = int(state.shape[0])
    fitted["state"] = state
    fitted["tokenized_prompt"] = np.ones((batch, token_len), dtype=np.int32)
    fitted["tokenized_prompt_mask"] = np.ones((batch, token_len), dtype=bool)
    return fitted


def _drop_noise(batched: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value for key, value in batched.items() if key != "noise"}


class FakeBackend:
    """Deterministic stand-in: velocity is zero, so actions stay at the input noise."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.packed_noise = bool(cfg.packed_noise)
        self.slab = np.zeros((cfg.max_prefix_slots, cfg.action_horizon, cfg.action_dim), dtype=np.float32)
        self.prev_ids: list[str] = []
        self.copy_count = 0
        self.token_len = cfg.token_len
        self.state_dim = cfg.state_dim
        self.horizon = cfg.action_horizon
        self.action_dim = cfg.action_dim

    def example_row(self) -> dict[str, np.ndarray]:
        state = np.zeros((self.state_dim,), dtype=np.float32)
        return {
            "pad": np.ones((4,), dtype=np.int32),
            "state": state,
            "k0": np.zeros((2, self.state_dim), dtype=np.float32),
            "v0": np.zeros((2, self.state_dim), dtype=np.float32),
            "noise": np.zeros((self.horizon, self.action_dim), dtype=np.float32),
        }

    def warmup_vlm(self) -> None:
        return None

    def warmup_ae(self) -> None:
        if not self.cfg.warmup:
            return
        max_batch = max(1, min(self.cfg.warmup_max_batch, self.cfg.ae_max_batch_size, self.cfg.max_prefix_slots))
        for batch in range(1, max_batch + 1):
            _ = np.zeros((batch, self.horizon, self.action_dim), dtype=np.float32)

    def forward_vlm(self, observations: list[dict[str, Any]]) -> PrefixExport:
        rows: list[dict[str, np.ndarray]] = []
        for observation in observations:
            state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            rows.append(
                {
                    "pad": np.ones((4,), dtype=np.int32),
                    "state": state,
                    "k0": np.broadcast_to(state, (2, state.shape[0])).copy(),
                    "v0": np.zeros((2, state.shape[0]), dtype=np.float32),
                }
            )
        return _ImmediateExport(rows)

    def import_prefix(self, batched: dict[str, np.ndarray]) -> None:
        del batched

    def admit_noise(self, lane_id: int, noise: np.ndarray) -> None:
        self.slab[int(lane_id)] = np.asarray(noise, dtype=np.float32)

    def copy_noise(self, lane_id: int) -> np.ndarray:
        return np.array(self.slab[int(lane_id)], copy=True)

    def step_batch(self, records: list[Any]) -> dict[str, float]:
        started = time.monotonic_ns()
        for row, record in enumerate(records):
            if (not self.packed_noise) or row >= len(self.prev_ids) or self.prev_ids[row] != record.request_id:
                self.copy_count += 1
                _ = self.slab[int(record.lane_id)]
        pack_ms = (time.monotonic_ns() - started) / 1e6
        forward_started = time.monotonic_ns()
        survivors: list[str] = []
        for record in records:
            record.step_idx += 1
            if record.step_idx < record.num_steps:
                survivors.append(record.request_id)
        self.prev_ids = survivors
        forward_ms = (time.monotonic_ns() - forward_started) / 1e6
        return {"ae_pack_ms": pack_ms, "ae_forward_ms": forward_ms, "ae_unpack_ms": 0.0}

    def sample_batch(
        self,
        observations: list[dict[str, Any]],
        noises: list[np.ndarray],
        num_steps: int,
    ) -> list[np.ndarray]:
        del observations, num_steps
        return [np.array(noise, copy=True) for noise in noises]


def _torch_to_host(tensor: Any) -> np.ndarray:
    import torch

    tensor = tensor.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        # NumPy has no bfloat16. Keep the bits in int16 and view them back on import.
        return tensor.view(torch.int16).numpy()
    return tensor.numpy()


def _host_to_torch(array: np.ndarray, device: str, *, bfloat16: bool) -> Any:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if bfloat16:
        tensor = tensor.view(torch.bfloat16)
    return tensor.to(device)


def _torch_feature_to_rows(feature: Any) -> list[dict[str, np.ndarray]]:
    pad = feature.prefix_pad_masks.detach()
    batch = int(pad.shape[0])
    cache = feature.past_key_values
    rows: list[dict[str, np.ndarray]] = []
    for index in range(batch):
        row: dict[str, np.ndarray] = {"pad": _torch_to_host(pad[index])}
        if feature.state is not None:
            row["state"] = _torch_to_host(feature.state[index])
        for layer_idx in range(len(cache)):
            key, value = cache[layer_idx]
            row[f"k{layer_idx}"] = _torch_to_host(key[index])
            row[f"v{layer_idx}"] = _torch_to_host(value[index])
        rows.append(row)
    return rows


def _torch_feature_from_batch(batched: dict[str, np.ndarray], device: str) -> Any:
    import torch
    from transformers.cache_utils import DynamicCache

    from openpi.models_pytorch.pi0_split_types import PrefixFeature

    cache = DynamicCache()
    layer = 0
    while f"k{layer}" in batched:
        key = _host_to_torch(batched[f"k{layer}"], device, bfloat16=batched[f"k{layer}"].dtype == np.int16)
        value = _host_to_torch(batched[f"v{layer}"], device, bfloat16=batched[f"v{layer}"].dtype == np.int16)
        cache.update(key, value, layer_idx=layer)
        layer += 1
    if layer == 0:
        raise ValueError("prefix batch is missing KV layers")
    pad = torch.as_tensor(np.ascontiguousarray(batched["pad"]), device=device)
    state = None
    if "state" in batched:
        state = torch.as_tensor(np.ascontiguousarray(batched["state"]), device=device)
    return PrefixFeature(past_key_values=cache, prefix_pad_masks=pad, state=state)


class _TorchExport:
    def __init__(self, feature: Any, device_index: int) -> None:
        import torch

        self.feature = feature
        self.device_index = device_index
        self.event = None
        self.copy_stream = None
        if device_index >= 0 and torch.cuda.is_available():
            self.copy_stream = torch.cuda.Stream(device=device_index)
            self.event = torch.cuda.Event()
            self.event.record(torch.cuda.current_stream(device_index))

    def materialize(self) -> list[dict[str, np.ndarray]]:
        import torch

        if self.event is None or self.copy_stream is None:
            return _torch_feature_to_rows(self.feature)
        torch.cuda.set_device(self.device_index)
        self.copy_stream.wait_event(self.event)
        with torch.cuda.stream(self.copy_stream):
            rows = _torch_feature_to_rows(self.feature)
        self.copy_stream.synchronize()
        return rows


class TorchBackend:
    def __init__(self, cfg: RunConfig, role: str, device: int) -> None:
        import torch

        from openpi.policies.va_split_policy import _load_pytorch_model
        from openpi.training import config as _config

        if device >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is not available")
            if device >= torch.cuda.device_count():
                raise ValueError(f"gpu={device} out of range for device_count={torch.cuda.device_count()}")
            torch.cuda.set_device(device)
            self.device = f"cuda:{device}"
        else:
            self.device = "cpu"
        self.device_index = device
        self.cfg = cfg
        self.role = role
        self.packed_noise = bool(cfg.packed_noise)
        if not cfg.policy_dir:
            raise ValueError("pytorch backend requires policy_dir")
        weight_path = os.path.join(cfg.policy_dir, "model.safetensors")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)
        train_config = _config.get_config(cfg.policy_config)
        model_role = None if role == "full" else role
        model = _load_pytorch_model(train_config, weight_path, role=model_role)
        self.model = model.to(self.device).eval()
        self.token_len = int(self.model.config.max_token_len)
        self.state_dim = int(self.model.config.action_dim)
        self.horizon = int(self.model.config.action_horizon)
        self.action_dim = int(self.model.config.action_dim)
        self._prefix = None
        self._slab: Any = None
        self._packed: Any = None
        self.prev_ids: list[str] = []
        self._example: dict[str, np.ndarray] | None = None

    def example_row(self) -> dict[str, np.ndarray]:
        if self._example is None:
            raise RuntimeError("call warmup_vlm before reading the prefix example row")
        return self._example

    def _observation_batch(self, observations: list[dict[str, Any]]) -> Any:
        import torch

        from openpi.models import model as _model
        from openpi.serving.va_split_multinode.host_lane_pool import stack_observations

        stacked = stack_observations(observations)
        stacked = fit_batched_observation(stacked, token_len=self.token_len, state_dim=self.state_dim)
        tensor_tree = _tree_to_torch(stacked, self.device)
        return _model.Observation.from_dict(tensor_tree), torch

    def warmup_vlm(self) -> None:
        limit = 1 if not self.cfg.warmup else max(1, min(self.cfg.warmup_max_batch, self.cfg.max_batch_size))
        last_rows: list[dict[str, np.ndarray]] | None = None
        for batch in range(1, limit + 1):
            observation = synthetic_observation(self.cfg, batch_size=batch, seed=1000 + batch)
            last_rows = self.forward_vlm(unbatch_observation(observation)).materialize()
        if not last_rows:
            raise RuntimeError("VLM warmup produced no prefix rows")
        example = dict(last_rows[0])
        example["noise"] = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        self._example = example

    def warmup_ae(self) -> None:
        import torch

        from openpi.models_pytorch.pi0_split_types import DenoiseState

        if self._example is None:
            self._example = {
                "pad": np.ones((8,), dtype=np.int32),
                "state": np.zeros((self.state_dim,), dtype=np.float32),
                "noise": np.zeros((self.horizon, self.action_dim), dtype=np.float32),
            }
        if not self.cfg.warmup:
            self._ensure_slab()
            return
        limit = max(1, min(self.cfg.warmup_max_batch, self.cfg.ae_max_batch_size))
        self._ensure_slab()
        for batch in range(1, limit + 1):
            rows = []
            for _ in range(batch):
                row = {key: value for key, value in self._example.items() if key != "noise"}
                rows.append(row)
            from openpi.serving.va_split_multinode.host_lane_pool import stack_rows

            self.import_prefix(stack_rows(rows))
            noise = torch.zeros((batch, self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            step_idx = torch.arange(batch, device=self.device, dtype=torch.int32) % max(1, self.cfg.num_steps)
            dt = torch.full((batch,), -1.0 / max(1, self.cfg.num_steps), dtype=torch.float32, device=self.device)
            denoise = DenoiseState(x_t=noise, step_idx=step_idx, num_steps=self.cfg.num_steps, dt=dt)
            _ = self.model.denoise_one_batch(self._prefix, denoise)
        if self.device_index >= 0 and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def forward_vlm(self, observations: list[dict[str, Any]]) -> PrefixExport:
        import torch

        observation, _torch = self._observation_batch(observations)
        with torch.no_grad():
            feature = self.model.build_prefix_feature(self.device, observation)
        return _TorchExport(feature, self.device_index)

    def import_prefix(self, batched: dict[str, np.ndarray]) -> None:
        self._prefix = _torch_feature_from_batch(_drop_noise(batched), self.device)

    def _ensure_slab(self) -> None:
        import torch

        if self._slab is None:
            lanes = int(self.cfg.max_prefix_slots)
            self._slab = torch.empty((lanes, self.horizon, self.action_dim), dtype=torch.float32, device=self.device)
            cap = int(self.cfg.ae_max_batch_size)
            self._packed = torch.empty((cap, self.horizon, self.action_dim), dtype=torch.float32, device=self.device)

    def admit_noise(self, lane_id: int, noise: np.ndarray) -> None:
        import torch

        self._ensure_slab()
        tensor = torch.as_tensor(np.asarray(noise, dtype=np.float32), device=self.device)
        self._slab[int(lane_id)].copy_(tensor, non_blocking=True)

    def copy_noise(self, lane_id: int) -> np.ndarray:
        return self._slab[int(lane_id)].detach().to(device="cpu").numpy()

    def step_batch(self, records: list[Any]) -> dict[str, float]:
        import torch

        from openpi.models_pytorch.pi0_split_types import DenoiseState

        if self._prefix is None:
            raise RuntimeError("AE prefix is not loaded")
        self._ensure_slab()
        assert self._packed is not None
        batch = len(records)
        started = time.monotonic_ns()
        for row, record in enumerate(records):
            if (not self.packed_noise) or row >= len(self.prev_ids) or self.prev_ids[row] != record.request_id:
                self._packed[row].copy_(self._slab[int(record.lane_id)], non_blocking=True)
        pack_ms = (time.monotonic_ns() - started) / 1e6
        forward_started = time.monotonic_ns()
        step_idx = torch.tensor([int(record.step_idx) for record in records], dtype=torch.int32, device=self.device)
        num_steps = torch.tensor([int(record.num_steps) for record in records], dtype=torch.float32, device=self.device)
        dt = -1.0 / num_steps
        denoise = DenoiseState(
            x_t=self._packed[:batch],
            step_idx=step_idx,
            num_steps=int(records[0].num_steps),
            dt=dt,
        )
        with torch.no_grad():
            velocity = self.model.denoise_one_batch(self._prefix, denoise)
            updated = self._packed[:batch] + dt[:, None, None] * velocity
        forward_ms = (time.monotonic_ns() - forward_started) / 1e6
        unpack_started = time.monotonic_ns()
        survivors: list[str] = []
        write_row = 0
        for row, record in enumerate(records):
            self._slab[int(record.lane_id)].copy_(updated[row], non_blocking=True)
            record.step_idx += 1
            if record.step_idx < record.num_steps:
                if self.packed_noise and write_row != row:
                    self._packed[write_row].copy_(updated[row], non_blocking=True)
                survivors.append(record.request_id)
                write_row += 1
        self.prev_ids = survivors
        unpack_ms = (time.monotonic_ns() - unpack_started) / 1e6
        return {"ae_pack_ms": pack_ms, "ae_forward_ms": forward_ms, "ae_unpack_ms": unpack_ms}

    def sample_batch(
        self,
        observations: list[dict[str, Any]],
        noises: list[np.ndarray],
        num_steps: int,
    ) -> list[np.ndarray]:
        import torch

        observation, _torch = self._observation_batch(observations)
        noise = torch.as_tensor(np.stack(noises).astype(np.float32), device=self.device)
        with torch.no_grad():
            actions = self.model.sample_actions(self.device, observation, noise=noise, num_steps=num_steps)
        cpu = actions.detach().to(device="cpu").numpy()
        return [cpu[index] for index in range(cpu.shape[0])]


def _jax_to_host(value: Any) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.dtype == jnp.bfloat16:
        bits = jax.lax.bitcast_convert_type(array, jnp.uint16)
        return np.asarray(bits)
    return np.asarray(array)


def _host_to_jax(array: np.ndarray) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(array)
    if array.dtype == np.uint16:
        return jax.lax.bitcast_convert_type(values, jnp.bfloat16)
    return values


def _jax_feature_to_rows(feature: Any) -> list[dict[str, np.ndarray]]:
    key, value = feature.past_key_values
    key_np = _jax_to_host(key)
    value_np = _jax_to_host(value)
    pad = _jax_to_host(feature.prefix_pad_masks)
    state = None if feature.state is None else _jax_to_host(feature.state)
    rows: list[dict[str, np.ndarray]] = []
    for index in range(int(pad.shape[0])):
        row: dict[str, np.ndarray] = {"pad": pad[index]}
        if state is not None:
            row["state"] = state[index]
        for layer in range(int(key_np.shape[0])):
            row[f"k{layer}"] = key_np[layer, index]
            row[f"v{layer}"] = value_np[layer, index]
        rows.append(row)
    return rows


def _jax_feature_from_batch(batched: dict[str, np.ndarray]) -> Any:
    import jax.numpy as jnp

    from openpi.models.jax_split_types import JaxPrefixFeature

    keys = []
    values = []
    layer = 0
    while f"k{layer}" in batched:
        keys.append(_host_to_jax(batched[f"k{layer}"]))
        values.append(_host_to_jax(batched[f"v{layer}"]))
        layer += 1
    if layer == 0:
        raise ValueError("prefix batch is missing KV layers")
    state = _host_to_jax(batched["state"]) if "state" in batched else None
    return JaxPrefixFeature(
        past_key_values=(jnp.stack(keys, axis=0), jnp.stack(values, axis=0)),
        prefix_pad_masks=_host_to_jax(batched["pad"]),
        state=state,
    )


class _JaxExport:
    def __init__(self, feature: Any) -> None:
        self.feature = feature

    def materialize(self) -> list[dict[str, np.ndarray]]:
        import jax

        jax.block_until_ready(self.feature.past_key_values)
        jax.block_until_ready(self.feature.prefix_pad_masks)
        if self.feature.state is not None:
            jax.block_until_ready(self.feature.state)
        return _jax_feature_to_rows(self.feature)


class JaxBackend:
    def __init__(self, cfg: RunConfig, role: str, device: int) -> None:
        import jax
        import jax.numpy as jnp

        from openpi.policies.jax_va_split_policy import _load_jax_model
        from openpi.training import config as _config

        del device
        if not cfg.policy_dir:
            raise ValueError("jax backend requires policy_dir")
        train_config = _config.get_config(cfg.policy_config)
        model_role = None if role == "full" else role
        model = _load_jax_model(train_config, cfg.policy_dir, role=model_role)
        if cfg.jax_compile:
            from openpi.shared import nnx_utils

            if role in {"vlm", "full"}:
                model.build_prefix_feature = nnx_utils.module_jit(model.build_prefix_feature)
            if role in {"ae", "full"}:
                model.denoise_one_batch = nnx_utils.module_jit(model.denoise_one_batch)
            if role == "full":
                model.sample_actions = nnx_utils.module_jit(model.sample_actions)
        self.model = model
        self.cfg = cfg
        self.role = role
        self.packed_noise = bool(cfg.packed_noise)
        self.jnp = jnp
        self.jax = jax
        self.token_len = int(train_config.model.max_token_len)
        self.state_dim = int(train_config.model.action_dim)
        self.horizon = int(train_config.model.action_horizon)
        self.action_dim = int(train_config.model.action_dim)
        self._prefix = None
        self.slab = np.zeros((cfg.max_prefix_slots, self.horizon, self.action_dim), dtype=np.float32)
        self._packed = np.zeros((cfg.ae_max_batch_size, self.horizon, self.action_dim), dtype=np.float32)
        self.prev_ids: list[str] = []
        self._example: dict[str, np.ndarray] | None = None

    def example_row(self) -> dict[str, np.ndarray]:
        if self._example is None:
            raise RuntimeError("call warmup_vlm before reading the prefix example row")
        return self._example

    def _observation_batch(self, observations: list[dict[str, Any]]) -> Any:
        from openpi.models import model as _model
        from openpi.serving.va_split_multinode.host_lane_pool import stack_observations

        stacked = fit_batched_observation(
            stack_observations(observations),
            token_len=self.token_len,
            state_dim=self.state_dim,
        )

        def _to_jax(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: _to_jax(item) for key, item in value.items()}
            return self.jnp.asarray(value)

        return _model.Observation.from_dict(_to_jax(stacked))

    def warmup_vlm(self) -> None:
        limit = 1 if not self.cfg.warmup else max(1, min(self.cfg.warmup_max_batch, self.cfg.max_batch_size))
        last_rows: list[dict[str, np.ndarray]] | None = None
        for batch in range(1, limit + 1):
            observation = synthetic_observation(self.cfg, batch_size=batch, seed=2000 + batch)
            last_rows = self.forward_vlm(unbatch_observation(observation)).materialize()
        if not last_rows:
            raise RuntimeError("VLM warmup produced no prefix rows")
        example = dict(last_rows[0])
        example["noise"] = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        self._example = example

    def warmup_ae(self) -> None:
        from openpi.models.jax_split_types import JaxDenoiseState
        from openpi.serving.va_split_multinode.host_lane_pool import stack_rows

        if not self.cfg.warmup:
            return
        if self._example is None:
            raise RuntimeError("AE warmup needs a prefix example row from the host pool")
        limit = max(1, min(self.cfg.warmup_max_batch, self.cfg.ae_max_batch_size))
        for batch in range(1, limit + 1):
            rows = [{key: value for key, value in self._example.items() if key != "noise"} for _ in range(batch)]
            self.import_prefix(stack_rows(rows))
            noise = self.jnp.zeros((batch, self.horizon, self.action_dim), dtype=self.jnp.float32)
            step_idx = self.jnp.arange(batch, dtype=self.jnp.int32) % max(1, self.cfg.num_steps)
            dt = self.jnp.full((batch,), -1.0 / max(1, self.cfg.num_steps), dtype=self.jnp.float32)
            denoise = JaxDenoiseState(x_t=noise, step_idx=step_idx, num_steps=self.cfg.num_steps, dt=dt)
            velocity = self.model.denoise_one_batch(self._prefix, denoise)
            self.jax.block_until_ready(velocity)

    def forward_vlm(self, observations: list[dict[str, Any]]) -> PrefixExport:
        observation = self._observation_batch(observations)
        feature = self.model.build_prefix_feature(None, observation)
        return _JaxExport(feature)

    def import_prefix(self, batched: dict[str, np.ndarray]) -> None:
        self._prefix = _jax_feature_from_batch(_drop_noise(batched))

    def admit_noise(self, lane_id: int, noise: np.ndarray) -> None:
        self.slab[int(lane_id)] = np.asarray(noise, dtype=np.float32)

    def copy_noise(self, lane_id: int) -> np.ndarray:
        return np.array(self.slab[int(lane_id)], copy=True)

    def step_batch(self, records: list[Any]) -> dict[str, float]:
        from openpi.models.jax_split_types import JaxDenoiseState

        if self._prefix is None:
            raise RuntimeError("AE prefix is not loaded")
        started = time.monotonic_ns()
        batch = len(records)
        for row, record in enumerate(records):
            if (not self.packed_noise) or row >= len(self.prev_ids) or self.prev_ids[row] != record.request_id:
                self._packed[row] = self.slab[int(record.lane_id)]
        pack_ms = (time.monotonic_ns() - started) / 1e6
        forward_started = time.monotonic_ns()
        step_idx = self.jnp.asarray([int(record.step_idx) for record in records], dtype=self.jnp.int32)
        num_steps = self.jnp.asarray([int(record.num_steps) for record in records], dtype=self.jnp.float32)
        dt = -1.0 / num_steps
        denoise = JaxDenoiseState(
            x_t=self.jnp.asarray(self._packed[:batch]),
            step_idx=step_idx,
            num_steps=int(records[0].num_steps),
            dt=dt,
        )
        velocity = self.model.denoise_one_batch(self._prefix, denoise)
        updated = np.asarray(denoise.x_t + dt[:, None, None] * velocity)
        forward_ms = (time.monotonic_ns() - forward_started) / 1e6
        unpack_started = time.monotonic_ns()
        survivors: list[str] = []
        write_row = 0
        for row, record in enumerate(records):
            self.slab[int(record.lane_id)] = updated[row]
            record.step_idx += 1
            if record.step_idx < record.num_steps:
                if self.packed_noise and write_row != row:
                    self._packed[write_row] = updated[row]
                survivors.append(record.request_id)
                write_row += 1
        self.prev_ids = survivors
        unpack_ms = (time.monotonic_ns() - unpack_started) / 1e6
        return {"ae_pack_ms": pack_ms, "ae_forward_ms": forward_ms, "ae_unpack_ms": unpack_ms}

    def sample_batch(
        self,
        observations: list[dict[str, Any]],
        noises: list[np.ndarray],
        num_steps: int,
    ) -> list[np.ndarray]:
        observation = self._observation_batch(observations)
        noise = self.jnp.asarray(np.stack(noises).astype(np.float32))
        actions = self.model.sample_actions(self.jax.random.key(0), observation, num_steps=num_steps, noise=noise)
        self.jax.block_until_ready(actions)
        cpu = np.asarray(actions)
        return [cpu[index] for index in range(cpu.shape[0])]


def _tree_to_torch(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, dict):
        return {key: _tree_to_torch(item, device) for key, item in value.items()}
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    return tensor.to(device)


def build_backend(cfg: RunConfig, role: str, device: int) -> Any:
    if cfg.backend == "fake":
        return FakeBackend(cfg)
    if cfg.backend == "pytorch":
        return TorchBackend(cfg, role, device)
    if cfg.backend == "jax":
        return JaxBackend(cfg, role, device)
    raise ValueError(f"unsupported backend {cfg.backend}")
