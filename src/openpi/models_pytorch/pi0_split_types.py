from dataclasses import dataclass

import torch


@dataclass
class PrefixFeature:
    past_key_values: object
    prefix_pad_masks: torch.Tensor
    state: torch.Tensor | None


@dataclass
class DenoiseState:
    x_t: torch.Tensor
    step_idx: torch.Tensor
    num_steps: int
    dt: torch.Tensor
