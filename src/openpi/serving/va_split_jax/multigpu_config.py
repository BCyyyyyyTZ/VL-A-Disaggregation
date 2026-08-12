from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

CrossCardTransferStrategy: TypeAlias = Literal["device-direct", "host-staged"]
_VALID_CROSS_CARD_TRANSFER_STRATEGIES = frozenset({"device-direct", "host-staged"})


def parse_device_list(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def normalize_cross_card_transfer_strategy(value: str) -> CrossCardTransferStrategy:
    strategy = str(value).strip()
    if strategy not in _VALID_CROSS_CARD_TRANSFER_STRATEGIES:
        valid = ", ".join(sorted(_VALID_CROSS_CARD_TRANSFER_STRATEGIES))
        raise ValueError(f"cross_card_transfer_strategy must be one of: {valid}; got {value!r}")
    return cast(CrossCardTransferStrategy, strategy)


@dataclass(frozen=True, slots=True)
class JaxMultiGpuVASplitConfig:
    vlm_devices: tuple[str, ...]
    ae_device: str
    max_vlm_batch_size: int = 8
    max_vlm_wait_ms: float = 0.0
    max_ae_batch_size: int = 64
    max_prefix_admits_per_drain: int | None = 1
    max_prefix_slots: int | None = None
    start_method: str = "spawn"
    cross_card_transfer_strategy: CrossCardTransferStrategy = "device-direct"

    def __post_init__(self) -> None:
        vlm_devices = parse_device_list(self.vlm_devices)
        ae_device = str(self.ae_device).strip()
        transfer_strategy = normalize_cross_card_transfer_strategy(self.cross_card_transfer_strategy)
        object.__setattr__(self, "vlm_devices", vlm_devices)
        object.__setattr__(self, "ae_device", ae_device)
        object.__setattr__(self, "cross_card_transfer_strategy", transfer_strategy)
        if not vlm_devices:
            raise ValueError("vlm_devices must contain at least one device")
        if not ae_device:
            raise ValueError("ae_device must be non-empty")
        if len(set(vlm_devices)) != len(vlm_devices):
            raise ValueError("vlm_devices must be unique")
        if ae_device in set(vlm_devices):
            raise ValueError("ae_device must not appear in vlm_devices")
        if self.max_vlm_batch_size <= 0:
            raise ValueError("max_vlm_batch_size must be positive")
        if self.max_vlm_wait_ms < 0:
            raise ValueError("max_vlm_wait_ms must be non-negative")
        if self.max_ae_batch_size <= 0:
            raise ValueError("max_ae_batch_size must be positive")
        if self.max_prefix_admits_per_drain is not None and self.max_prefix_admits_per_drain <= 0:
            raise ValueError("max_prefix_admits_per_drain must be positive when set")
        max_prefix_slots = self.max_prefix_slots
        if max_prefix_slots is None:
            max_prefix_slots = max(1, len(vlm_devices)) * max(self.max_vlm_batch_size, 1) * 3
        if max_prefix_slots <= 0:
            raise ValueError("max_prefix_slots must be positive")
        object.__setattr__(self, "max_prefix_slots", int(max_prefix_slots))

    @property
    def num_vlm_workers(self) -> int:
        return len(self.vlm_devices)

    @property
    def vlm_worker_ids(self) -> tuple[str, ...]:
        return tuple(f"vlm-{idx}" for idx in range(self.num_vlm_workers))
