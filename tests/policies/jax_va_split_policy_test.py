from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.policies import jax_va_split_policy
from openpi.policies.jax_va_split_policy import JaxVASplitPolicy
from openpi.serving.va_split_jax.types import JaxActionResult


class FakeRuntime:
    def __init__(self):
        self.sample_kwargs = None
        self.observation = None
        self.shutdown_called = False
        self.compile_timing = {"jax_warmup_batches": 3.0}

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        self.sample_kwargs = sample_kwargs
        self.observation = observation
        assert isinstance(observation["state"], np.ndarray)
        return JaxActionResult(
            request_id="req-1",
            actions=jnp.ones((1, 2, 3), dtype=jnp.float32),
            timing={"runtime_ms": 2.0},
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeBatchRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.batch_calls = 0
        self.batch_observation = None

    def infer_batch(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        self.batch_calls += 1
        self.sample_kwargs = sample_kwargs
        self.batch_observation = observation
        assert isinstance(observation["state"], np.ndarray)
        return JaxActionResult(
            request_id="batch-1",
            actions=jnp.zeros((observation["state"].shape[0], 2, 3), dtype=jnp.float32),
            timing={"vlm_effective_batch": float(observation["state"].shape[0]), "ae_effective_batch_mean": 2.0},
        )


def test_jax_va_split_policy_preserves_infer_output_contract():
    runtime = FakeRuntime()
    policy = JaxVASplitPolicy(
        runtime=runtime,
        transforms=(),
        output_transforms=(),
        sample_kwargs={"num_steps": 4},
        metadata={"model": "fake"},
    )

    result = policy.infer({"state": np.array([0.25, -0.5], dtype=np.float32)})

    assert result["actions"].shape == (2, 3)
    np.testing.assert_allclose(result["actions"], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(result["state"], np.array([0.25, -0.5], dtype=np.float32))
    assert result["policy_timing"]["runtime_ms"] == 2.0
    assert result["policy_timing"]["jax_warmup_batches"] == 3.0
    assert result["policy_timing"]["infer_ms"] >= 0.0
    assert runtime.sample_kwargs == {"num_steps": 4}
    assert policy.metadata == {"model": "fake"}
    assert policy.supports_concurrent_infer is True


def test_jax_va_split_policy_normalizes_uint8_images_before_runtime_ipc():
    runtime = FakeRuntime()
    policy = JaxVASplitPolicy(runtime=runtime)

    policy.infer(
        {
            "state": np.array([0.25, -0.5], dtype=np.float32),
            "image": {
                "base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.full((2, 2, 3), 255, dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }
    )

    assert runtime.observation is not None
    base = runtime.observation["image"]["base_0_rgb"]
    wrist = runtime.observation["image"]["left_wrist_0_rgb"]
    assert base.dtype == np.float32
    assert wrist.dtype == np.float32
    np.testing.assert_allclose(base, np.full((1, 2, 2, 3), -1.0, dtype=np.float32))
    np.testing.assert_allclose(wrist, np.full((1, 2, 2, 3), 1.0, dtype=np.float32))


def test_jax_va_split_policy_infer_batch_uses_runtime_batch_once():
    runtime = FakeBatchRuntime()
    policy = JaxVASplitPolicy(
        runtime=runtime,
        transforms=(),
        output_transforms=(),
        sample_kwargs={"num_steps": 4},
    )

    result = policy.infer_batch(
        {"state": np.asarray([[0.25, -0.5], [0.5, -1.0]], dtype=np.float32)},
        noise=np.zeros((2, 2, 3), dtype=np.float32),
    )

    assert runtime.batch_calls == 1
    assert runtime.batch_observation["state"].shape == (2, 2)
    assert isinstance(runtime.sample_kwargs["noise"], np.ndarray)
    assert runtime.sample_kwargs["noise"].shape == (2, 2, 3)
    assert result["actions"].shape == (2, 2, 3)
    assert result["policy_timing"]["effective_batch"] == 2
    assert result["policy_timing"]["vlm_effective_batch"] == 2.0
    assert result["policy_timing"]["ae_effective_batch_mean"] == 2.0


def test_jax_va_split_policy_shutdown_delegates_to_runtime():
    runtime = FakeRuntime()
    policy = JaxVASplitPolicy(runtime=runtime)

    policy.shutdown()

    assert runtime.shutdown_called is True


def test_load_jax_model_rejects_pytorch_only_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    train_config = SimpleNamespace(model=SimpleNamespace())

    with pytest.raises(ValueError, match="JAX checkpoint"):
        jax_va_split_policy._load_jax_model(train_config, tmp_path)  # noqa: SLF001


def test_restore_params_for_target_state_skips_extra_value_suffix_leaves(tmp_path):
    params_dir = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            params_dir,
            {
                "params": {
                    "keep": {"value": jnp.ones((2,), dtype=jnp.float32)},
                    "drop": {"value": jnp.ones((3,), dtype=jnp.float32)},
                }
            },
        )

    restored = jax_va_split_policy._restore_params_for_target_state(  # noqa: SLF001
        params_dir,
        {"keep": jnp.zeros((2,), dtype=jnp.float32)},
        dtype=jnp.bfloat16,
    )

    assert sorted(restored) == ["keep"]
    assert restored["keep"].shape == (2,)
    assert restored["keep"].dtype == jnp.bfloat16
