from __future__ import annotations

import asyncio

from openpi.serving import websocket_policy_server


class FakePolicy:
    supports_concurrent_infer = False

    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": obs["value"]}


class FakeConcurrentPolicy(FakePolicy):
    supports_concurrent_infer = True


class FakeBatchPolicy(FakePolicy):
    def __init__(self):
        super().__init__()
        self.batch_calls = 0

    def infer_batch(self, obs):
        self.batch_calls += 1
        return {"actions": obs["value"]}


def test_infer_policy_keeps_default_policies_on_event_loop(monkeypatch):
    policy = FakePolicy()

    async def fail_to_thread(*_args, **_kwargs):
        raise AssertionError("default policies should not be offloaded")

    monkeypatch.setattr(asyncio, "to_thread", fail_to_thread)

    result = asyncio.run(websocket_policy_server.infer_policy_async(policy, {"value": 3}))

    assert result == {"actions": 3}
    assert policy.calls == 1


def test_infer_policy_offloads_concurrent_policies(monkeypatch):
    policy = FakeConcurrentPolicy()
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(websocket_policy_server.infer_policy_async(policy, {"value": 4}))

    assert result == {"actions": 4}
    assert len(calls) == 1


def test_infer_policy_uses_batch_path_for_prompt_batches():
    policy = FakeBatchPolicy()

    result = asyncio.run(websocket_policy_server.infer_policy_async(policy, {"prompt": ["a", "b"], "value": [1, 2]}))

    assert result == {"actions": [1, 2]}
    assert policy.calls == 0
    assert policy.batch_calls == 1


def test_infer_policy_batch_path_can_be_disabled():
    policy = FakeBatchPolicy()

    result = asyncio.run(
        websocket_policy_server.infer_policy_async(
            policy,
            {"prompt": ["a", "b"], "value": [1, 2]},
            enable_policy_batch=False,
        )
    )

    assert result == {"actions": [1, 2]}
    assert policy.calls == 1
    assert policy.batch_calls == 0
