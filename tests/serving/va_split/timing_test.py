from __future__ import annotations

import torch

from openpi.serving.va_split import timing


def test_synchronize_cuda_if_needed_is_noop_for_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))

    timing.synchronize_cuda_if_needed("cpu")

    assert calls == []


def test_synchronize_cuda_if_needed_syncs_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))

    timing.synchronize_cuda_if_needed("cuda:2")

    assert calls == [torch.device("cuda:2")]
