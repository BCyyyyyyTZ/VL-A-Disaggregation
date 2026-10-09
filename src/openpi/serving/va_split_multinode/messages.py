from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WorkItem:
    request_id: str
    observation: dict[str, Any]
    noise: np.ndarray
    num_steps: int
    scheduled_at_s: float
    enqueue_ns: int


@dataclass(slots=True)
class PrefixMsg:
    request_id: str
    lane_id: int
    num_steps: int
    scheduled_at_s: float
    enqueue_ns: int
    vlm_done_ns: int
    vlm_batch_size: int
    vlm_batch_start_ns: int = 0
    vlm_compute_done_ns: int = 0
    vlm_export_done_ns: int = 0
    payload: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class RunConfig:
    backend: str = "fake"
    mode: str = "ours"
    role: str = "all"
    vlm_devices: tuple[int, ...] = (0,)
    ae_device: int = 1
    baseline_devices: tuple[int, ...] = ()
    transport: str = "host_shm"
    launch: str = "process"
    num_requests: int = 8
    rate: float = 16.0
    arrival_scope: str = "global"
    seed: int = 0
    max_batch_size: int = 8
    ae_max_batch_size: int = 16
    max_prefix_slots: int = 256
    max_wait_ms: float = 1.0
    num_steps: int = 5
    timeout_s: float = 600.0
    startup_timeout_s: float = 600.0
    slo_ms: float = 200.0
    overlap_d2h: bool = True
    packed_noise: bool = True
    policy_config: str = "pi05_libero"
    policy_dir: str = ""
    state_dim: int = 32
    action_dim: int = 32
    action_horizon: int = 50
    token_len: int = 200
    image_size: int = 224
    warmup: bool = True
    warmup_max_batch: int = 4
    jax_compile: bool = False
    ae_addr: str = ""
    vlm_addrs: tuple[str, ...] = ()
    result_addr: str = ""
    bind_host: str = "0.0.0.0"
    worker_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_transport(self) -> str:
        if self.transport == "local" and self.launch == "process":
            return "host_shm"
        return self.transport

    def validate(self) -> None:
        if self.backend not in {"fake", "pytorch", "jax"}:
            raise ValueError(f"unsupported backend {self.backend}")
        if self.mode not in {"ours", "baseline"}:
            raise ValueError(f"unsupported mode {self.mode}")
        if self.role not in {"all", "vlm", "ae", "coordinator"}:
            raise ValueError(f"unsupported role {self.role}")
        if self.max_batch_size <= 0 or self.ae_max_batch_size <= 0 or self.max_prefix_slots <= 0:
            raise ValueError("batch and prefix capacities must be positive")
        if self.num_requests < 0:
            raise ValueError("num_requests must be non-negative")
        if self.mode == "ours" and not self.vlm_devices:
            raise ValueError("ours mode requires at least one VLM device")
        physical = [d for d in (*self.vlm_devices, self.ae_device) if d >= 0]
        if self.mode == "ours" and len(physical) != len(set(physical)):
            raise ValueError("VLM and AE physical devices must be distinct")
        if self.mode == "baseline":
            devices = self.baseline_devices or self.vlm_devices
            if not devices:
                raise ValueError("baseline mode requires devices")
            baseline_physical = [d for d in devices if d >= 0]
            if len(baseline_physical) != len(set(baseline_physical)):
                raise ValueError("baseline physical devices must be distinct")
