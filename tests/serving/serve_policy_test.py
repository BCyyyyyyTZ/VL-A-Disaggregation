# ruff: noqa: SLF001

from __future__ import annotations

import pytest

from scripts import serve_policy


class _FakePolicy:
    def __init__(self):
        self.metadata = {"model": "fake"}
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeServer:
    def __init__(self, *, policy, host, port, metadata):
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata

    def serve_forever(self) -> None:
        raise RuntimeError("server stopped")


def test_main_closes_policy_when_server_exits(monkeypatch):
    policy = _FakePolicy()
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: policy)
    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _FakeServer)
    monkeypatch.setattr(serve_policy.socket, "gethostname", lambda: "test-host")
    monkeypatch.setattr(serve_policy.socket, "gethostbyname", lambda _hostname: "127.0.0.1")

    with pytest.raises(RuntimeError, match="server stopped"):
        serve_policy.main(serve_policy.Args())

    assert policy.shutdown_called is True


def test_create_checkpoint_policy_forwards_pytorch_device_to_va_split(monkeypatch):
    calls = {}
    monkeypatch.setattr(serve_policy._config, "get_config", lambda config: f"train-{config}")

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
    assert calls["train_config"] == "train-pi05_libero"
    assert calls["checkpoint_dir"] == "/tmp/checkpoint"
    assert calls["pytorch_device"] == "cuda:0"
