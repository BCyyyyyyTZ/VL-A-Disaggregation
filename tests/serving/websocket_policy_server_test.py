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
