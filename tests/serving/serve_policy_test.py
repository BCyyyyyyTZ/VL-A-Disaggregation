# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses

import pytest

from scripts import serve_policy


class _FakePolicy:
    def __init__(self):
        self.metadata = {"model": "fake"}
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeServer:
    def __init__(self, *, policy, host, port, metadata, enable_policy_batch):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata
        self.enable_policy_batch = enable_policy_batch

    def serve_forever(self) -> None:
        raise RuntimeError("server stopped")


def test_main_closes_policy_when_server_exits(monkeypatch):
    policy = _FakePolicy()
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: policy)
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _FakeServer)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda _hostname: "127.0.0.1")

    with pytest.raises(RuntimeError, match="server stopped"):
        serve_policy.main(serve_policy.Args(enable_policy_batch=False))

    assert policy.shutdown_called is True
    assert (
        serve_policy.websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=8000,
            metadata=policy.metadata,
            enable_policy_batch=False,
        ).enable_policy_batch
        is False
    )


def test_create_checkpoint_policy_forwards_pytorch_device_to_va_split(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    calls = {}
    monkeypatch.setattr(serve_policy._config, "get_config", lambda config: FakeTrainConfig())

    def fake_create_trained_va_split_policy(train_config, checkpoint_dir, **kwargs):
        calls["train_config"] = train_config
        calls["checkpoint_dir"] = checkpoint_dir
        calls.update(kwargs)
        return _FakePolicy()

    monkeypatch.setattr(
        serve_policy._va_split_policy,
        "create_trained_va_split_policy",
        fake_create_trained_va_split_policy,
    )

    policy = serve_policy._create_checkpoint_policy(
        serve_policy.Args(va_split=True, pytorch_device="cuda:0"),
        serve_policy.Checkpoint(config="pi05_libero", dir="/tmp/checkpoint"),
    )

    assert isinstance(policy, _FakePolicy)
    assert calls["train_config"].model.pytorch_compile_mode is None
    assert calls["checkpoint_dir"] == "/tmp/checkpoint"
    assert calls["pytorch_device"] == "cuda:0"
    assert calls["max_vlm_batch_size"] == 8
    assert calls["max_vlm_wait_ms"] == 2.0


def test_create_checkpoint_policy_allows_explicit_pytorch_compile_opt_in(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    calls = {}
    monkeypatch.setattr(serve_policy._config, "get_config", lambda config: FakeTrainConfig())

    def fake_create_trained_va_split_policy(train_config, checkpoint_dir, **kwargs):
        calls["train_config"] = train_config
        return _FakePolicy()

    monkeypatch.setattr(
        serve_policy._va_split_policy,
        "create_trained_va_split_policy",
        fake_create_trained_va_split_policy,
    )

    serve_policy._create_checkpoint_policy(
        serve_policy.Args(va_split=True, pytorch_compile_mode="default"),
        serve_policy.Checkpoint(config="pi05_libero", dir="/tmp/checkpoint"),
    )

    assert calls["train_config"].model.pytorch_compile_mode == "default"


def test_create_checkpoint_policy_disables_pytorch_compile_by_default_for_va_split(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    calls = {}
    monkeypatch.setattr(serve_policy._config, "get_config", lambda config: FakeTrainConfig())

    def fake_create_trained_va_split_policy(train_config, checkpoint_dir, **kwargs):
        calls["train_config"] = train_config
        return _FakePolicy()

    monkeypatch.setattr(
        serve_policy._va_split_policy,
        "create_trained_va_split_policy",
        fake_create_trained_va_split_policy,
    )

    serve_policy._create_checkpoint_policy(
        serve_policy.Args(va_split=True),
        serve_policy.Checkpoint(config="pi05_libero", dir="/tmp/checkpoint"),
    )

    assert calls["train_config"].model.pytorch_compile_mode is None


def test_create_checkpoint_policy_disables_pytorch_compile_by_default_for_baseline(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        pytorch_compile_mode: str | None = "max-autotune"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    calls = {}
    monkeypatch.setattr(serve_policy._config, "get_config", lambda config: FakeTrainConfig())

    def fake_create_trained_policy(train_config, checkpoint_dir, **kwargs):
        calls["train_config"] = train_config
        return _FakePolicy()

    monkeypatch.setattr(serve_policy._policy_config, "create_trained_policy", fake_create_trained_policy)

    serve_policy._create_checkpoint_policy(
        serve_policy.Args(va_split=False),
        serve_policy.Checkpoint(config="pi05_libero", dir="/tmp/checkpoint"),
    )

    assert calls["train_config"].model.pytorch_compile_mode is None
