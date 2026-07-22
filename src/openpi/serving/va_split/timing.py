from __future__ import annotations

import torch


def synchronize_cuda_if_needed(device: str | torch.device) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)
