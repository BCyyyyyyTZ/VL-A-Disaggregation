from __future__ import annotations

from openpi.serving.va_split_jax.prefix_transfer import PrefixTransferTicket
from openpi.serving.va_split_jax.prefix_transfer import wait_for_prefix_ticket


class FakeWaitable:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


class FakeSynchronizable:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


def test_prefix_transfer_ticket_waits_once_on_waitable():
    raw = FakeWaitable()

    wait_ms = wait_for_prefix_ticket(PrefixTransferTicket(kind="fake", payload=raw))

    assert wait_ms >= 0.0
    assert raw.wait_calls == 1


def test_prefix_transfer_ticket_supports_synchronize_payload():
    raw = FakeSynchronizable()

    wait_for_prefix_ticket(PrefixTransferTicket(kind="fake", payload=raw))

    assert raw.sync_calls == 1


def test_prefix_transfer_ticket_waits_composite_payloads():
    left = FakeWaitable()
    right = FakeSynchronizable()

    wait_for_prefix_ticket(PrefixTransferTicket(kind="composite", payload=(left, right)))

    assert left.wait_calls == 1
    assert right.sync_calls == 1
