# Multi-GPU V-A Disaggregation 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现并评估 JAX OpenPI 的多卡 VLM-AE disaggregation，优先验证同机多张 4090 下 `N VLM : 1 AE` 相比 VLM+AE 顺序执行的 monolithic compile baseline 是否能改善 mean latency、E2E p95、TTFA、TPA、throughput 和 goodput，同时保留 A100 对照实验。

**架构：** 第一版实现 `2/3 VLM workers -> 1 AE worker`。AE 进程在 AE GPU 上拥有 prefix slab，多个 VLM 进程在各自 GPU 上运行 prefix/prefill，并通过异步 contiguous batch copy 写入 AE-owned slab；router 按 VLM backlog 和 credit 状态把 Poisson 到达请求分配给 VLM worker。`max_batch_size` 只表示上限，worker 不为了凑满 batch 主动等待；每轮只消费当时已经排队且 credit 足够的请求。跨卡传输必须尽量与 AE 当前 denoise step overlap，AE 只在 admit 对应 lane 前等待该 lane 的 ready ticket，不做整 slab 或整 GPU 同步。

**技术栈：** Python multiprocessing spawn、JAX/XLA、CUDA IPC/Numba slab copy、现有 `src/openpi/serving/va_split_jax` runtime、`scripts/profile_va_split.py` Poisson benchmark。

---

## 关键结论和设计约束

现有 A100 compile profile 显示，单卡 baseline compile 已经很强：r8 baseline `82.514ms` E2E mean，ours 同卡 split `85.493ms`；r16 baseline `115.856ms`，ours `134.744ms`；r32 baseline `324.648ms`，ours `447.435ms`。r32 差距主要来自 split 队列等待和同卡 VLM/AE 资源争用，ours r32 的 `va_split_queue_wait_mean_ms=152.430`。

4090 stage profile 显示 VLM 是瓶颈，AE/action head 明显更轻。SM100 下 VLM bs8 为 `217.677ms`，bs16 为 `414.197ms`，bs24 为 `606.502ms`，bs32 增至 `1194.706ms` 且吞吐降至约 `26.8 req/s`。因此 4090 VLM 默认 sweep `max_vlm_batch_size=4,8,16,24`，第一候选为 `4/8`，只把 `16/24` 作为高压吞吐对照。AE 默认 sweep `max_ae_batch_size=24,32,48,64`，因为 AE 侧更适合 continuous batching，较大的上限有利于同时处理不同 denoise step 的 active requests；最终选择必须由 `TPA_mean/p95`、`action_latency_mean/p95` 和 AE GPU 利用率共同决定。

传输路径是 mean latency 的主要风险。实现必须优先满足三条约束：VLM 对一个 FCFS batch 使用 contiguous batch copy 写 AE-owned slab，避免逐 lane copy；VLM copy 在专用 CUDA stream 上异步发起，并把 per-batch 或 per-lane ready ticket 交给 AE；AE 在每次 denoise step 之间 drain prefix ready，但只在 admit 具体 lane 前等待该 lane ready，不等待整 slab，也不调用 device-wide synchronize。

本计划参考 disaggregated serving 的经典思路：DistServe/Splitwise 的 prefill/decode 分离、ORCA/vLLM 的 continuous batching、Triton dynamic batching 的队列上限思想。这里不照搬 LLM decode KV cache 调度，而是把 VLM prefix 和 AE denoise 视为两个独立服务阶段，各自独立设置资源比例和 batch 上限。

## 文件结构

- 创建：`src/openpi/serving/va_split_jax/multigpu_config.py`
  - 职责：多卡 split runtime 的配置、设备列表解析、batch/slot 参数校验。
- 创建：`src/openpi/serving/va_split_jax/multigpu_router.py`
  - 职责：router 选择 VLM worker；维护每个 VLM 的 inflight/backlog 估计和 source worker 映射。
- 创建：`src/openpi/serving/va_split_jax/process_entry.py`
  - 职责：在子进程内先设置 `CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`，再导入 JAX worker 代码，避免多卡进程绑定错误。
- 创建：`src/openpi/serving/va_split_jax/prefix_transfer.py`
  - 职责：封装异步 prefix slab 写入、contiguous batch copy、ready ticket、per-lane admit wait 和 fallback 同步路径。
- 修改：`src/openpi/serving/va_split_jax/types.py`
  - 职责：给 prefix ready/release/result 消息携带 `source_worker_id` 和 `prefix_ready_ticket`，让 AE release credit 能回到正确 VLM，并让 AE 只等待对应 lane 的传输完成。
- 修改：`src/openpi/serving/va_split_jax/vlm_process.py`
  - 职责：VLM worker 附加 `source_worker_id`；batch 收集语义改成可显式配置的 no-wait 模式；prefix 写入改为 contiguous async transfer。
- 修改：`src/openpi/serving/va_split_jax/ae_process.py`
  - 职责：AE 记录每个 active request 的 source worker，并把 release credit 发回正确 release queue；每步 denoise 前后 drain prefix ready，保持不同 denoise step 请求的 continuous batching。
- 修改：`src/openpi/serving/va_split_jax/runtime.py`
  - 职责：新增 `JaxMultiGpuVASplitRuntime`，启动 1 个 AE 进程和 N 个 VLM 进程，汇总结果。
- 修改：`src/openpi/policies/jax_va_split_policy.py`
  - 职责：新增 `create_trained_jax_multi_gpu_va_split_policy(...)` 工厂。
- 修改：`scripts/profile_va_split.py`
  - 职责：新增 `mode="jax-multigpu-split-ipc"`，增加设备和多卡 batch 参数，输出多卡指标。
- 修改：`scripts/run_profile_va_split_jax.sh`
  - 职责：支持多 GPU UUID 列表，不再把整个 profile 限制到单个 `CUDA_VISIBLE_DEVICES`。
- 创建：`tests/serving/va_split_jax/test_multigpu_config.py`
- 创建：`tests/serving/va_split_jax/test_multigpu_router.py`
- 创建：`tests/serving/va_split_jax/test_multigpu_runtime_unit.py`
- 修改：`tests/serving/va_split_jax/test_profile_script.py`

---

### 任务 1：新增多卡配置对象

**文件：**
- 创建：`src/openpi/serving/va_split_jax/multigpu_config.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_config.py`

- [ ] **步骤 1：编写失败的测试**

```python
from openpi.serving.va_split_jax.multigpu_config import JaxMultiGpuVASplitConfig


def test_multigpu_config_defaults_to_no_wait_batching():
    cfg = JaxMultiGpuVASplitConfig(vlm_devices=("0", "1"), ae_device="2")

    assert cfg.num_vlm_workers == 2
    assert cfg.max_vlm_wait_ms == 0.0
    assert cfg.max_vlm_batch_size == 8
    assert cfg.max_ae_batch_size == 64
    assert cfg.max_prefix_slots == 48


def test_multigpu_config_rejects_overlapping_devices():
    try:
        JaxMultiGpuVASplitConfig(vlm_devices=("0", "1"), ae_device="1")
    except ValueError as exc:
        assert "ae_device must not appear in vlm_devices" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_config.py -q`

预期：FAIL，报错包含 `No module named 'openpi.serving.va_split_jax.multigpu_config'`。

- [ ] **步骤 3：实现配置对象**

```python
from __future__ import annotations

from dataclasses import dataclass


def parse_device_list(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        devices = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        devices = tuple(str(item).strip() for item in value if str(item).strip())
    if not devices:
        raise ValueError("vlm_devices must contain at least one device")
    return devices


@dataclass(frozen=True, slots=True)
class JaxMultiGpuVASplitConfig:
    vlm_devices: tuple[str, ...]
    ae_device: str
    max_vlm_batch_size: int = 8
    max_ae_batch_size: int = 64
    max_vlm_wait_ms: float = 0.0
    max_prefix_slots: int | None = None
    result_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        vlm_devices = parse_device_list(self.vlm_devices)
        object.__setattr__(self, "vlm_devices", vlm_devices)
        if self.ae_device in vlm_devices:
            raise ValueError("ae_device must not appear in vlm_devices")
        if self.max_vlm_batch_size <= 0:
            raise ValueError("max_vlm_batch_size must be positive")
        if self.max_ae_batch_size <= 0:
            raise ValueError("max_ae_batch_size must be positive")
        if self.max_vlm_wait_ms < 0:
            raise ValueError("max_vlm_wait_ms must be non-negative")
        if self.max_prefix_slots is None:
            slots = len(vlm_devices) * self.max_vlm_batch_size * 3
            object.__setattr__(self, "max_prefix_slots", slots)
        elif self.max_prefix_slots <= 0:
            raise ValueError("max_prefix_slots must be positive")

    @property
    def num_vlm_workers(self) -> int:
        return len(self.vlm_devices)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_config.py -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/openpi/serving/va_split_jax/multigpu_config.py tests/serving/va_split_jax/test_multigpu_config.py
git commit -m "feat: add JAX multi-gpu split config"
```

---

### 任务 2：实现 no-wait VLM router

**文件：**
- 创建：`src/openpi/serving/va_split_jax/multigpu_router.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_router.py`

- [ ] **步骤 1：编写失败的测试**

```python
from openpi.serving.va_split_jax.multigpu_router import JaxVlmRouteState
from openpi.serving.va_split_jax.multigpu_router import LeastBacklogVlmRouter


def test_router_chooses_worker_with_lowest_backlog_then_inflight():
    router = LeastBacklogVlmRouter(("vlm-0", "vlm-1", "vlm-2"))
    router.update("vlm-0", JaxVlmRouteState(inflight=8, queued=2, available_credits=16))
    router.update("vlm-1", JaxVlmRouteState(inflight=4, queued=0, available_credits=16))
    router.update("vlm-2", JaxVlmRouteState(inflight=4, queued=1, available_credits=16))

    assert router.choose_worker().worker_id == "vlm-1"


def test_router_skips_workers_without_credit():
    router = LeastBacklogVlmRouter(("vlm-0", "vlm-1"))
    router.update("vlm-0", JaxVlmRouteState(inflight=0, queued=0, available_credits=0))
    router.update("vlm-1", JaxVlmRouteState(inflight=2, queued=3, available_credits=1))

    assert router.choose_worker().worker_id == "vlm-1"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_router.py -q`

预期：FAIL，报错包含 `No module named 'openpi.serving.va_split_jax.multigpu_router'`。

- [ ] **步骤 3：实现 router**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JaxVlmRouteState:
    inflight: int = 0
    queued: int = 0
    available_credits: int = 0

    @property
    def backlog_score(self) -> tuple[int, int]:
        return (self.queued + self.inflight, -self.available_credits)


@dataclass(frozen=True, slots=True)
class JaxVlmRouteDecision:
    worker_id: str


class LeastBacklogVlmRouter:
    def __init__(self, worker_ids: tuple[str, ...]):
        if not worker_ids:
            raise ValueError("worker_ids must not be empty")
        self._worker_ids = worker_ids
        self._states = {worker_id: JaxVlmRouteState() for worker_id in worker_ids}
        self._cursor = 0

    def update(self, worker_id: str, state: JaxVlmRouteState) -> None:
        if worker_id not in self._states:
            raise KeyError(worker_id)
        self._states[worker_id] = state

    def choose_worker(self) -> JaxVlmRouteDecision:
        candidates = [
            (self._states[worker_id].backlog_score, index, worker_id)
            for index, worker_id in enumerate(self._worker_ids)
            if self._states[worker_id].available_credits > 0
        ]
        if not candidates:
            index = self._cursor % len(self._worker_ids)
            self._cursor += 1
            return JaxVlmRouteDecision(worker_id=self._worker_ids[index])
        _, _, worker_id = min(candidates)
        return JaxVlmRouteDecision(worker_id=worker_id)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_router.py -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/openpi/serving/va_split_jax/multigpu_router.py tests/serving/va_split_jax/test_multigpu_router.py
git commit -m "feat: add JAX multi-gpu VLM router"
```

---

### 任务 3：给消息补充 source worker id

**文件：**
- 修改：`src/openpi/serving/va_split_jax/types.py`
- 修改：`src/openpi/serving/va_split_jax/vlm_process.py`
- 修改：`src/openpi/serving/va_split_jax/ae_process.py`
- 测试：`tests/serving/va_split_jax/test_vlm_process.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_runtime_unit.py`

- [ ] **步骤 1：编写失败的测试**

```python
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.models.jax_split_types import JaxPrefixSlotHandle


def test_prefix_ready_carries_source_worker_id():
    ready = JaxPrefixReady(
        request_id="req-1",
        slot_handle=JaxPrefixSlotHandle(slot_id=3, batch_rows=1),
        num_steps=5,
        sample_kwargs={},
        source_worker_id="vlm-1",
    )

    assert ready.source_worker_id == "vlm-1"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_runtime_unit.py::test_prefix_ready_carries_source_worker_id -q`

预期：FAIL，报错包含 `unexpected keyword argument 'source_worker_id'`。

- [ ] **步骤 3：修改消息类型**

在 `src/openpi/serving/va_split_jax/types.py` 中把 `JaxPrefixReady` 和 `JaxReleaseFeature` 改为：

```python
@dataclass(frozen=True, slots=True)
class JaxPrefixReady:
    """VLM finished writing prefix KV into AE-owned slab lane ``slot_handle.slot_id``."""

    request_id: str
    slot_handle: JaxPrefixSlotHandle
    num_steps: int
    sample_kwargs: dict[str, Any]
    timing: dict[str, float] | None = None
    source_worker_id: str | None = None


@dataclass(frozen=True, slots=True)
class JaxReleaseFeature:
    """AE finished a request; ``slot_id`` is the recycled physical lane credit for VLM."""

    request_id: str
    slot_id: int
    source_worker_id: str | None = None
```

- [ ] **步骤 4：让 VLM 填充 source worker**

给 `JaxVLMProcess.__init__` 和 `JaxVLMWorker` 增加 `source_worker_id: str | None = None`。在创建 `JaxPrefixReady` 的位置加入：

```python
source_worker_id=self._source_worker_id,
```

已有单卡调用传 `None`，保持兼容。

- [ ] **步骤 5：让 AE release 保留 source worker**

在 `JaxAERequestState` 增加字段：

```python
source_worker_id: str | None
```

在 `JaxAEWorker.add_prefix()` 创建 state 时设置：

```python
source_worker_id=ready.source_worker_id,
```

在 `JaxAEWorker.step_once()` 创建 release 时设置：

```python
JaxReleaseFeature(
    request_id=request.request_id,
    slot_id=freed_by_request[request.request_id],
    source_worker_id=request.source_worker_id,
)
```

- [ ] **步骤 6：运行单元测试**

运行：

```bash
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest \
  tests/serving/va_split_jax/test_multigpu_runtime_unit.py::test_prefix_ready_carries_source_worker_id \
  tests/serving/va_split_jax/test_vlm_process.py -q
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add src/openpi/serving/va_split_jax/types.py src/openpi/serving/va_split_jax/vlm_process.py src/openpi/serving/va_split_jax/ae_process.py tests/serving/va_split_jax/test_vlm_process.py tests/serving/va_split_jax/test_multigpu_runtime_unit.py
git commit -m "feat: tag JAX split messages with source worker"
```

---

### 任务 4：新增延迟导入的子进程入口

**文件：**
- 创建：`src/openpi/serving/va_split_jax/process_entry.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_runtime_unit.py`

- [ ] **步骤 1：编写失败的测试**

```python
from openpi.serving.va_split_jax.process_entry import _child_env_updates


def test_child_env_updates_pin_single_visible_device():
    env = _child_env_updates("GPU-abc")

    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_runtime_unit.py::test_child_env_updates_pin_single_visible_device -q`

预期：FAIL，报错包含 `No module named 'openpi.serving.va_split_jax.process_entry'`。

- [ ] **步骤 3：实现延迟导入入口**

```python
from __future__ import annotations

import os
from typing import Any


def _child_env_updates(device: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(device),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }


def _apply_child_env(device: str, extra_env: dict[str, str | None] | None = None) -> None:
    updates: dict[str, str | None] = dict(_child_env_updates(device))
    updates.update(extra_env or {})
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_vlm_worker_entry(*, device: str, kwargs: dict[str, Any]) -> None:
    extra_env = kwargs.pop("env_updates", None)
    _apply_child_env(device, extra_env)
    from openpi.serving.va_split_jax.runtime import _run_jax_vlm_process

    _run_jax_vlm_process(**kwargs)


def run_ae_worker_entry(*, device: str, kwargs: dict[str, Any]) -> None:
    extra_env = kwargs.pop("env_updates", None)
    _apply_child_env(device, extra_env)
    from openpi.serving.va_split_jax.runtime import _run_jax_ae_process

    _run_jax_ae_process(**kwargs)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_runtime_unit.py::test_child_env_updates_pin_single_visible_device -q`

预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/openpi/serving/va_split_jax/process_entry.py tests/serving/va_split_jax/test_multigpu_runtime_unit.py
git commit -m "feat: add delayed JAX child process entrypoints"
```

---


### 任务 5A：异步 prefix transfer 和 per-lane readiness

**文件：**
- 创建：`src/openpi/serving/va_split_jax/prefix_transfer.py`
- 修改：`src/openpi/serving/va_split_jax/device_slab.py`
- 修改：`src/openpi/serving/va_split_jax/prefix_cache_pool.py`
- 修改：`src/openpi/serving/va_split_jax/vlm_process.py`
- 修改：`src/openpi/serving/va_split_jax/ae_process.py`
- 测试：`tests/serving/va_split_jax/test_prefix_transfer.py`
- 测试：`tests/serving/va_split_jax/test_vlm_process.py`
- 测试：`tests/serving/va_split_jax/test_ae_process.py`

- [ ] **步骤 1：编写失败的 transfer abstraction 测试**

```python
from openpi.serving.va_split_jax.prefix_transfer import PrefixTransferTicket
from openpi.serving.va_split_jax.prefix_transfer import wait_for_prefix_ticket


class FakeTicket:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


def test_prefix_transfer_ticket_waits_once_per_admitted_lane():
    raw = FakeTicket()
    ticket = PrefixTransferTicket(kind="fake", payload=raw)

    wait_for_prefix_ticket(ticket)

    assert raw.wait_calls == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_prefix_transfer.py::test_prefix_transfer_ticket_waits_once_per_admitted_lane -q`

预期：FAIL，报错包含 `No module named 'openpi.serving.va_split_jax.prefix_transfer'`。

- [ ] **步骤 3：实现 ticket 类型和等待 helper**

创建 `prefix_transfer.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PrefixTransferTicket:
    kind: str
    payload: Any = None


def wait_for_prefix_ticket(ticket: PrefixTransferTicket | None) -> float:
    if ticket is None or ticket.kind == "synchronous":
        return 0.0
    import time

    start_ns = time.monotonic_ns()
    payload = ticket.payload
    if hasattr(payload, "wait"):
        payload.wait()
    elif hasattr(payload, "synchronize"):
        payload.synchronize()
    else:
        raise RuntimeError(f"unsupported prefix transfer ticket kind {ticket.kind!r}")
    return (time.monotonic_ns() - start_ns) / 1_000_000
```

- [ ] **步骤 4：给 DeviceSlabBackend 增加 async batch copy API**

在 `DeviceSlabBackend` 增加默认同步实现：

```python
def copy_batch_from_array_async(self, slab: DeviceSlab, lane_start: int, value: jax.Array) -> tuple[DeviceSlab, PrefixTransferTicket]:
    slab = self.copy_batch_from_array(slab, lane_start, value, sync=True)
    return slab, PrefixTransferTicket(kind="synchronous")
```

在 `CudaIpcDeviceSlabBackend` 中实现专用 stream 异步 copy：

```python
def copy_batch_from_array_async(self, slab: DeviceSlab, lane_start: int, value: jax.Array) -> tuple[DeviceSlab, PrefixTransferTicket]:
    _validate_batch_update(slab, lane_start, value)
    value.block_until_ready()
    device_ordinal = int(slab.handle.device_ordinal)
    _select_numba_device(device_ordinal)
    stream = self._write_stream(device_ordinal)
    dst = cuda.from_cuda_array_interface(_cuda_array_interface_for_array(slab.array), owner=slab)
    src = cuda.from_cuda_array_interface(_cuda_array_interface_for_array(value), owner=value)
    _copy_batch_with_kernel(dst, src, lane_start, slab.spec, stream=stream)
    event = cuda.event(timing=False)
    event.record(stream=stream)
    return slab, PrefixTransferTicket(kind="cuda-event", payload=event)
```

如果当前 Numba 版本不支持跨进程 event pickle，保持 `PrefixTransferTicket(kind="synchronous")` fallback，但仍必须用 contiguous batch copy，并把 fallback 情况记录到 `vlm_slab_write_async=0.0`。

- [ ] **步骤 5：让 prefix_cache_pool 返回 batch ticket**

新增函数：

```python
def write_feature_batch_to_slab_tree_async(backend, slabs, lane_ids, feature) -> tuple[Any, PrefixTransferTicket]:
    lane_start = lane_ids[0]
    if tuple(lane_ids) != tuple(range(lane_start, lane_start + len(lane_ids))):
        raise ValueError("async prefix transfer requires contiguous lane ids")
    updated = _copy_tree_batch_async(backend, slabs, feature, lane_start)
    return updated.slab_tree, updated.ticket
```

`_copy_tree_batch_async` 对每个 leaf 使用 `copy_batch_from_array_async`。第一版每个 batch 返回一个 ticket；如果树里有多个 leaf，返回 `PrefixTransferTicket(kind="composite", payload=(ticket1, ticket2, ...))`，`wait_for_prefix_ticket` 逐个等待。不要在 VLM 侧调用 `sync_write_stream()`。

- [ ] **步骤 6：VLM 发送 PrefixReady 时附加 ticket**

在 `JaxPrefixReady` 增加字段：

```python
prefix_ready_ticket: PrefixTransferTicket | None = None
```

在 `JaxVLMWorker._write_feature_to_lanes(...)` 使用 async writer，并在每个 row 的 ready timing 里记录：

```python
"vlm_slab_write_async": 1.0,
"vlm_slab_write_ticket_shared": 1.0,
```

同一个 contiguous batch 的所有 rows 可以共享同一个 ticket；AE admit 每个 lane 时等待同一个 ticket 是安全的。为了避免重复等待造成额外开销，AE 需要维护 `id(ticket)` 的已等待集合。

- [ ] **步骤 7：AE 只在 admit lane 前等待 ticket**

在 `JaxAEWorker.add_prefix()` 中，在 `claim_written_lane(...)` 前加入：

```python
wait_ms = self._wait_prefix_ready_once(ready.prefix_ready_ticket)
timing["prefix_ready_wait_ms"] = wait_ms
```

新增 helper：

```python
def _wait_prefix_ready_once(self, ticket: PrefixTransferTicket | None) -> float:
    if ticket is None:
        return 0.0
    key = id(ticket)
    if key in self._waited_prefix_tickets:
        return 0.0
    wait_ms = wait_for_prefix_ticket(ticket)
    self._waited_prefix_tickets.add(key)
    return wait_ms
```

AE 不允许在 `drain_prefix_ready()` 里等待所有 pending prefix；只有即将 admit 的 lane 才等待。这样 VLM transfer 可以和 AE 当前 `step_once()` overlap。

- [ ] **步骤 8：保持 AE continuous batching**

添加测试：

```python
def test_ae_select_ready_lanes_allows_mixed_step_indices(fake_ae_worker):
    fake_ae_worker._lanes[0].step_idx = 0
    fake_ae_worker._lanes[1].step_idx = 2
    fake_ae_worker._active_count = 2

    selected = fake_ae_worker.select_ready_lanes()

    assert [lane.step_idx for lane in selected] == [0, 2]
```

预期：AE 不按 `step_idx` 分桶；每轮选当前 dense active lanes，`JaxDenoiseState.step_idx` 是向量。这样不同 denoise step 的请求可以 continuous batch 到同一个 AE kernel 中。

- [ ] **步骤 9：运行 focused tests**

运行：

```bash
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest \
  tests/serving/va_split_jax/test_prefix_transfer.py \
  tests/serving/va_split_jax/test_vlm_process.py \
  tests/serving/va_split_jax/test_ae_process.py -q
```

预期：PASS。GPU 不可用时，CUDA event 分支测试自动 skip，但同步 fallback 和 no global sync 语义测试必须 PASS。

- [ ] **步骤 10：Commit**

```bash
git add src/openpi/serving/va_split_jax/prefix_transfer.py src/openpi/serving/va_split_jax/device_slab.py src/openpi/serving/va_split_jax/prefix_cache_pool.py src/openpi/serving/va_split_jax/vlm_process.py src/openpi/serving/va_split_jax/ae_process.py src/openpi/serving/va_split_jax/types.py tests/serving/va_split_jax/test_prefix_transfer.py tests/serving/va_split_jax/test_vlm_process.py tests/serving/va_split_jax/test_ae_process.py

git commit -m "feat: overlap prefix transfer with AE denoise"
```

---

### 任务 5：实现多 VLM 单 AE runtime

本任务依赖任务 5A。runtime 的 correctness 不只看请求是否完成，还要看 VLM prefix transfer 是否能与 AE denoise step 并行；任何把 AE process 阻塞到 VLM copy 完成、或在 VLM 侧整 stream 同步后才发送 ready 的实现，都不满足本计划目标。

**文件：**
- 修改：`src/openpi/serving/va_split_jax/runtime.py`
- 修改：`src/openpi/serving/va_split_jax/ae_process.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_runtime_unit.py`

- [ ] **步骤 1：编写失败的测试**

```python
from openpi.serving.va_split_jax.runtime import _release_queue_index
from openpi.serving.va_split_jax.types import JaxReleaseFeature


def test_release_queue_index_routes_to_source_worker():
    release = JaxReleaseFeature(request_id="req-1", slot_id=7, source_worker_id="vlm-2")
    worker_ids = ("vlm-0", "vlm-1", "vlm-2")

    assert _release_queue_index(release, worker_ids) == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_multigpu_runtime_unit.py::test_release_queue_index_routes_to_source_worker -q`

预期：FAIL，报错包含 `cannot import name '_release_queue_index'`。

- [ ] **步骤 3：让 AE 支持多个 release queues**

在 `JaxAEProcess.__init__` 中允许 `release_queue` 是单个 queue 或 `dict[str | None, queue]`。新增 helper：

```python
def _put_release_message(release_queues, message) -> None:
    if not isinstance(release_queues, dict):
        release_queues.put(message)
        return
    worker_id = getattr(message, "source_worker_id", None)
    queue_obj = release_queues.get(worker_id)
    if queue_obj is None:
        raise RuntimeError(f"Missing release queue for source worker {worker_id!r}")
    queue_obj.put(message)
```

把 `self._release_queue.put(release)` 和 pending credits 的 put 替换为 `_put_release_message(...)`。`JaxPrefixSlabReady` 和 initial `JaxLaneCredits` 在 bootstrap 阶段需要广播给所有 VLM queues。

- [ ] **步骤 4：实现 release queue helper**

在 `runtime.py` 中增加：

```python
def _release_queue_index(release: JaxReleaseFeature, worker_ids: tuple[str, ...]) -> int:
    if release.source_worker_id is None:
        return 0
    try:
        return worker_ids.index(release.source_worker_id)
    except ValueError as exc:
        raise RuntimeError(f"Unknown VLM worker id {release.source_worker_id!r}") from exc
```

- [ ] **步骤 5：实现 `JaxMultiGpuVASplitRuntime` skeleton**

`JaxMultiGpuVASplitRuntime` 必须：

```python
class JaxMultiGpuVASplitRuntime:
    def __init__(self, *, model_factory, config, compile_config=None, warmup_timeout_s=None):
        self._config = config
        self._worker_ids = tuple(f"vlm-{idx}" for idx in range(config.num_vlm_workers))
        self._request_queues = []
        self._release_queues = []
        self._prefix_queue = None
        self._result_queue = None
        self._processes = []
        self._pending_results = {}
        self._pending_errors = {}
        self._closed = False
        self._start_processes(model_factory=model_factory, compile_config=compile_config, warmup_timeout_s=warmup_timeout_s)

    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        worker_id = self._choose_vlm_worker()
        request_id = str(uuid.uuid4())
        self._enqueue_request(worker_id, request_id, observation, sample_kwargs)
        return self._wait_for_result(request_id)
```

第一版 `_choose_vlm_worker()` 使用任务 2 的 `LeastBacklogVlmRouter`，worker backlog 初始由 runtime 内部 inflight counter 维护。每次 request 入队对应 worker 的 `queued += 1`，收到 result 后 `inflight -= 1`。后续可从 worker 心跳细化，但第一版不引入心跳协议。

- [ ] **步骤 6：运行 focused tests**

运行：

```bash
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest \
  tests/serving/va_split_jax/test_multigpu_runtime_unit.py \
  tests/serving/va_split_jax/test_multigpu_router.py -q
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add src/openpi/serving/va_split_jax/runtime.py src/openpi/serving/va_split_jax/ae_process.py tests/serving/va_split_jax/test_multigpu_runtime_unit.py
git commit -m "feat: add JAX multi-gpu VA split runtime"
```

---

### 任务 6：接入 policy factory 和 profile mode

**文件：**
- 修改：`src/openpi/policies/jax_va_split_policy.py`
- 修改：`scripts/profile_va_split.py`
- 修改：`tests/serving/va_split_jax/test_profile_script.py`

- [ ] **步骤 1：编写失败的测试**

```python
from scripts.profile_va_split import Args


def test_profile_args_include_multigpu_mode_and_devices():
    args = Args(mode="jax-multigpu-split-ipc", vlm_devices="0,1", ae_device="2")

    assert args.mode == "jax-multigpu-split-ipc"
    assert args.vlm_devices == "0,1"
    assert args.ae_device == "2"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_profile_script.py::test_profile_args_include_multigpu_mode_and_devices -q`

预期：FAIL，报错包含 mode literal 或 dataclass 字段不存在。

- [ ] **步骤 3：新增 policy factory**

在 `jax_va_split_policy.py` 增加：

```python
def create_trained_jax_multi_gpu_va_split_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str = "/mnt/tianze/models/pi05_libero",
    *,
    vlm_devices: tuple[str, ...],
    ae_device: str,
    max_ae_batch_size: int = 64,
    max_vlm_batch_size: int = 8,
    max_vlm_wait_ms: float = 0.0,
    result_timeout_s: float = 120.0,
    jax_compile: bool = True,
    jax_compile_warmup: bool = True,
    jax_compile_warmup_max_batch_size: int | None = None,
    **policy_kwargs,
) -> JaxVASplitPolicy:
    config = JaxMultiGpuVASplitConfig(
        vlm_devices=vlm_devices,
        ae_device=ae_device,
        max_ae_batch_size=max_ae_batch_size,
        max_vlm_batch_size=max_vlm_batch_size,
        max_vlm_wait_ms=max_vlm_wait_ms,
        result_timeout_s=result_timeout_s,
    )
    compile_config = JaxCompileConfig(
        enabled=jax_compile,
        warmup_enabled=jax_compile_warmup,
        warmup_max_batch_size=jax_compile_warmup_max_batch_size or max(config.max_ae_batch_size, config.max_vlm_batch_size),
        num_steps=int(policy_kwargs.get("num_steps", 5)),
    )
    runtime = JaxMultiGpuVASplitRuntime(
        model_factory=lambda: _load_jax_model(train_config, checkpoint_dir),
        config=config,
        compile_config=compile_config,
        warmup_timeout_s=result_timeout_s,
    )
    return JaxVASplitPolicy(runtime=runtime, **_policy_transform_kwargs(train_config, checkpoint_dir, **policy_kwargs))
```

如果 `_policy_transform_kwargs(...)` 不存在，则把现有 `create_trained_jax_va_split_policy(...)` 中构造 transforms/output_transforms/metadata 的重复逻辑先抽成该 helper，再复用。

- [ ] **步骤 4：新增 profile args 和 mode 分支**

在 `scripts/profile_va_split.py`：

```python
Mode = Literal["monolithic", "split-no-mps", "split-mps", "jax-monolithic", "jax-split-ipc", "jax-multigpu-split-ipc"]

@dataclasses.dataclass
class Args:
    ...
    vlm_devices: str = ""
    ae_device: str = ""
```

在 policy 构造分支中加入：

```python
if args.mode == "jax-multigpu-split-ipc":
    if not args.vlm_devices or not args.ae_device:
        raise ValueError("--vlm-devices and --ae-device are required for jax-multigpu-split-ipc")
    return _jax_va_split_policy.create_trained_jax_multi_gpu_va_split_policy(
        train_config,
        checkpoint_dir=args.policy.dir,
        vlm_devices=tuple(item.strip() for item in args.vlm_devices.split(",") if item.strip()),
        ae_device=args.ae_device,
        max_ae_batch_size=args.max_ae_batch_size,
        max_vlm_batch_size=args.max_vlm_batch_size,
        max_vlm_wait_ms=args.max_vlm_wait_ms,
        result_timeout_s=args.timeout_s,
        jax_compile=args.jax_compile,
        jax_compile_warmup=args.jax_compile_warmup,
        jax_compile_warmup_max_batch_size=args.jax_compile_warmup_max_batch_size,
        num_steps=args.num_steps,
    )
```

- [ ] **步骤 5：运行 tests**

运行：

```bash
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest \
  tests/serving/va_split_jax/test_profile_script.py \
  tests/serving/va_split_jax/test_multigpu_config.py \
  tests/serving/va_split_jax/test_multigpu_runtime_unit.py -q
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add src/openpi/policies/jax_va_split_policy.py scripts/profile_va_split.py tests/serving/va_split_jax/test_profile_script.py
git commit -m "feat: expose JAX multi-gpu split profile mode"
```

---

### 任务 7：更新 JAX profile shell wrapper

**文件：**
- 修改：`scripts/run_profile_va_split_jax.sh`
- 测试：`tests/serving/va_split_jax/test_profile_script.py`

- [ ] **步骤 1：编写失败的测试**

```python
from pathlib import Path


def test_run_profile_jax_script_supports_multigpu_envs():
    script = Path("scripts/run_profile_va_split_jax.sh").read_text(encoding="utf-8")

    assert "VLM_DEVICES" in script
    assert "AE_DEVICE" in script
    assert "--vlm-devices" in script
    assert "--ae-device" in script
```

- [ ] **步骤 2：运行测试验证失败**

运行：`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_profile_script.py::test_run_profile_jax_script_supports_multigpu_envs -q`

预期：FAIL，缺少 `VLM_DEVICES` 或 `--vlm-devices`。

- [ ] **步骤 3：修改脚本参数**

在脚本中加入：

```bash
: "${VLM_DEVICES:=}"
: "${AE_DEVICE:=}"

if [[ "${MODE}" == "jax-multigpu-split-ipc" ]]; then
  if [[ -z "${VLM_DEVICES}" || -z "${AE_DEVICE}" ]]; then
    echo "MODE=jax-multigpu-split-ipc requires VLM_DEVICES and AE_DEVICE" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--vlm-devices "${VLM_DEVICES}" --ae-device "${AE_DEVICE}")
fi
```

单卡 `jax-split-ipc` 路径继续保留现有 `CUDA_VISIBLE_DEVICES="${GPU_UUID}"`。多卡模式不要把父进程限制到单 GPU；子进程通过 `process_entry.py` 绑定各自设备。

- [ ] **步骤 4：运行 shell syntax 和测试**

运行：

```bash
bash -n scripts/run_profile_va_split_jax.sh
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_profile_script.py::test_run_profile_jax_script_supports_multigpu_envs -q
```

预期：两个命令都 PASS。

- [ ] **步骤 5：Commit**

```bash
git add scripts/run_profile_va_split_jax.sh tests/serving/va_split_jax/test_profile_script.py
git commit -m "feat: add multi-gpu JAX profile wrapper args"
```

---

### 任务 8：跨卡传输探针

**文件：**
- 修改：`tests/serving/va_split_jax/bench_cuda_ipc_compare.py`

- [ ] **步骤 1：添加设备参数测试入口**

在 bench 脚本 parser 中加入：

```python
parser.add_argument("--producer-gpu", type=str, default="")
parser.add_argument("--consumer-gpu", type=str, default="")
```

当两个参数均为空时继续使用现有 `--gpu` 单卡行为；当传入两个参数时，producer 子进程设置 `CUDA_VISIBLE_DEVICES=producer_gpu`，consumer 子进程设置 `CUDA_VISIBLE_DEVICES=consumer_gpu`。

- [ ] **步骤 2：记录 P2P 和 transport 指标**

在 report 中增加：

```python
report["topology"] = {
    "producer_gpu": args.producer_gpu or str(args.gpu),
    "consumer_gpu": args.consumer_gpu or str(args.gpu),
    "lane_bytes": LANE_BYTES,
}
```

保留现有 `copy_ms`、`get_plus_copy_ms`、`ingest_ms`。新增 `payload_gib_per_s = lane_bytes / mean_ms`，方便判断 8.9 MiB prefix 是否可被 VLM/AE 计算覆盖。

- [ ] **步骤 3：运行单卡和跨卡 probe**

运行：

```bash
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python tests/serving/va_split_jax/bench_cuda_ipc_compare.py --gpu 0 --iters 20 --warmup 5
/mnt/tianze/VL-A-Disaggregation/.venv/bin/python tests/serving/va_split_jax/bench_cuda_ipc_compare.py --producer-gpu 0 --consumer-gpu 1 --iters 20 --warmup 5
```

预期：生成 `logs/tests/cuda_ipc_compare_latest.json`。如果跨卡 CUDA IPC 失败，错误写入 JSON，并在任务 10 的 profile 矩阵中把 4090 方案标记为需要 host bounce fallback。

- [ ] **步骤 4：Commit**

```bash
git add tests/serving/va_split_jax/bench_cuda_ipc_compare.py
git commit -m "test: add cross-gpu CUDA IPC probe"
```

---

### 任务 9：端到端 smoke profile

**文件：**
- 修改：`scripts/profile_va_split.py`
- 修改：`src/openpi/serving/va_split_jax/runtime.py`
- 测试：实际 GPU smoke log

- [ ] **步骤 1：运行 2VLM:1AE 小请求 smoke**

运行：

```bash
MODE=jax-multigpu-split-ipc \
VLM_DEVICES=0,1 \
AE_DEVICE=2 \
NUM_REQUESTS=16 \
REQUEST_RATE_HZ=8 \
MAX_INFLIGHT=16 \
NUM_STEPS=5 \
MAX_VLM_BATCH_SIZE=8 \
MAX_AE_BATCH_SIZE=48 \
MAX_VLM_WAIT_MS=0 \
RUN_TS=smoke-multigpu-2v1a \
scripts/run_profile_va_split_jax.sh
```

预期：profile 成功结束，`completed_requests=16`、`failed_requests=0`，summary 中包含 `vlm_effective_batch_mean`、`ae_effective_batch_mean`、`vlm_slab_write_total_ms`、`vlm_slab_write_async_mean`、`prefix_ready_wait_mean_ms`、`va_split_transfer_ms`。`prefix_ready_wait_mean_ms` 应显著小于 `vlm_slab_write_total_mean_ms`，否则 transfer 没有被 AE denoise 覆盖。

- [ ] **步骤 2：如果 smoke 失败，定位失败阶段**

检查：

```bash
rg -n "RuntimeError|TimeoutError|Traceback|CUDA|IPC|failed" logs/JAX/smoke-multigpu-2v1a
```

预期：无异常。如果有异常，按错误阶段修正任务 3-7 的实现，并重新运行任务 9 步骤 1。

- [ ] **步骤 3：Commit smoke 修正**

```bash
git add src/openpi/serving/va_split_jax scripts/profile_va_split.py scripts/run_profile_va_split_jax.sh tests/serving/va_split_jax
git commit -m "fix: stabilize JAX multi-gpu split smoke profile"
```

---

### 任务 10：4090 优先 profile 矩阵

**文件：**
- 输出：`logs/JAX/multigpu-4090-2v1a-*`
- 输出：`logs/JAX/multigpu-4090-3v1a-*`
- 输出：`logs/JAX/baseline-4090-replicas-*`

- [ ] **步骤 1：建立公平 baseline**

同 GPU 数量比较必须包含：

```bash
# 1 GPU 单卡 monolithic compile baseline
MODE=jax-monolithic NUM_REQUESTS=200 REQUEST_RATE_HZ=16 RUN_TS=baseline-4090-1gpu-r16 scripts/run_profile_va_split_jax.sh

# 3 GPU monolithic 多副本 baseline：3 个独立 profile worker，每个 worker 1 张 4090，router round-robin 分发请求
# 如果现有脚本不支持多副本 baseline，先用任务 2 的 router 复用实现 JaxReplicaBaselineRuntime。
```

预期：报告中必须同时给出 single-GPU baseline 和 same-GPU-count replica baseline。2VLM:1AE 使用 3 张卡时，主要和 3-replica monolithic baseline 比；single-GPU baseline 只用于说明扩容趋势。

- [ ] **步骤 2：运行 2VLM:1AE sweep**

参数矩阵：

```text
VLM_DEVICES: 0,1
AE_DEVICE: 2
request_rate_hz: 16,24,32,48,64
max_vlm_batch_size: 4,8,16,24
max_ae_batch_size: 24,32,48,64
max_vlm_wait_ms: 0
num_requests: 200
num_steps: 5
```

预期：每个 case 记录 mean/p50/p95/p99 的 E2E、action latency、TTFA、TPA、throughput、SLO goodput，并记录 `vlm_slab_write_total_ms`、`vlm_slab_write_async`、`prefix_ready_wait_ms`、`prefix_admit_wait_ms`、`ae_idle_wait_ms`、`va_split_transfer_ms`、`vlm_effective_batch_mean`、`ae_effective_batch_mean`。

- [ ] **步骤 3：运行 3VLM:1AE sweep**

参数矩阵：

```text
VLM_DEVICES: 0,1,2
AE_DEVICE: 3
request_rate_hz: 32,48,64,80,96
max_vlm_batch_size: 4,8,16
max_ae_batch_size: 32,48,64
max_vlm_wait_ms: 0
num_requests: 200
num_steps: 5
```

预期：如果 `ae_effective_batch_mean` 上升但 `TPA_p95` 和 `action_latency_p95` 不劣化超过 20%，3VLM:1AE 进入候选；如果 `prefix_transfer_ms_p95` 或 `ae_queue_wait_p95` 明显升高，3VLM:1AE 仅作为容量上限记录。

- [ ] **步骤 4：判定 4090 收益**

收益标准：

```text
高负载 r32+：multi-gpu split 的 SLO goodput >= same-GPU-count monolithic baseline 的 0.9x，并且 action_latency_p95 更低，才算有服务化收益。
低负载 r8/r16：multi-gpu split 不要求胜过 baseline；如果 E2E mean 劣化超过 15%，文档中明确低负载不推荐。
跨卡传输：`prefix_ready_wait_p95_ms` 应小于 VLM prefix_forward p50 的 10%；`vlm_slab_write_total_p95_ms` 可以更高，但必须被 AE denoise overlap 覆盖。如果 `prefix_ready_wait_p95_ms` 接近 `vlm_slab_write_total_p95_ms`，说明 transfer 仍在关键路径上，4090 PCIe 传输风险过高。
```

- [ ] **步骤 5：生成汇总表**

创建 `logs/JAX/multigpu-4090-summary.md`，包含：

```text
case, gpu_budget, request_rate_hz, max_vlm_batch_size, max_ae_batch_size,
throughput_rps, slo_goodput_rps,
e2e_mean_ms, e2e_p50_ms, e2e_p95_ms,
action_mean_ms, action_p50_ms, action_p95_ms,
ttfa_mean_ms, ttfa_p95_ms, tpa_mean_ms, tpa_p95_ms,
vlm_effective_batch_mean, ae_effective_batch_mean,
vlm_prefix_forward_mean_ms, ae_step_mean_ms,
va_split_queue_wait_mean_ms, va_split_transfer_mean_ms, vlm_slab_write_total_mean_ms, prefix_ready_wait_mean_ms, ae_idle_wait_mean_ms
```

---

### 任务 11：A100 对照 profile

**文件：**
- 输出：`logs/JAX/multigpu-a100-2v1a-*`
- 输出：`logs/JAX/multigpu-a100-summary.md`

- [ ] **步骤 1：运行 A100 2VLM:1AE 对照**

参数矩阵：

```text
VLM_DEVICES: <two A100 device ids or UUIDs>
AE_DEVICE: <one A100 device id or UUID>
request_rate_hz: 16,32,48,64
max_vlm_batch_size: 8,16,24,32
max_ae_batch_size: 16,24,32,48
max_vlm_wait_ms: 0
num_requests: 200
num_steps: 5
```

预期：A100 因 NVLink/PCIe 和数据中心驱动特性可能比 4090 更稳定。关键对照是现有 A100 `baseline-compile` 和 `ours-compile-80-80`，尤其 r32 的 `end_to_end_latency_p95_ms` 和 `slo_goodput_requests_per_second`。

- [ ] **步骤 2：判定 A100 收益**

收益标准：

```text
相对现有 A100 ours r32：E2E p95 从 723.790ms 降低至少 25%，goodput 从 2.700 rps 提升至少 2x。
相对现有 A100 baseline r32：E2E p95 不高于 521.845ms，goodput 接近或超过 4.264 rps。
```

- [ ] **步骤 3：Commit profile 文档**

```bash
git add logs/JAX/multigpu-a100-summary.md logs/JAX/multigpu-4090-summary.md
git commit -m "docs: summarize multi-gpu VA split profiles"
```

---

### 任务 12：最终分析文档

**文件：**
- 创建：`docs/multi_gpu_va_disaggregation_eval.md`

- [ ] **步骤 1：写结论文档**

文档必须包含：

```markdown
# Multi-GPU V-A Disaggregation Evaluation

## Recommendation

## Hardware Topology

## Baselines

## 4090 Results

## A100 Results

## Batch Size Findings

## Transfer Overhead

## When To Use Multi-GPU Split

## When To Prefer Monolithic Compile
```

`Batch Size Findings` 必须明确：`max_batch_size` 是上限，不表示等待凑满；有效 batch 由当前 backlog、lane credit 和 active AE lanes 决定。

- [ ] **步骤 2：写推荐规则**

推荐规则使用当前 profile 数值填入，例如：

```text
4090: 若 request_rate_hz <= 16，优先 monolithic compile；若 request_rate_hz >= 32 且 P2P/IPC transfer p95 < 10ms，优先尝试 2VLM:1AE with max_vlm_batch_size=4/8, max_ae_batch_size=32/48/64。
A100: 若多卡传输稳定且 r32 goodput 超过 same-GPU-count monolithic baseline，则保留 2VLM:1AE；否则使用 monolithic replicas。
```

- [ ] **步骤 3：Commit**

```bash
git add docs/multi_gpu_va_disaggregation_eval.md
git commit -m "docs: add multi-gpu VA disaggregation evaluation"
```

---

## 自检

规格覆盖度：
- 多 VLM 卡配 1 AE 卡：任务 1-7。
- VLM/AE 分卡调度：任务 2、5、6。
- max batch size 语义：计划头部、任务 6、任务 10、任务 12。
- 跨卡传输和覆盖开销：任务 5A、8、10、12。
- baseline 设计：任务 10、11。
- profile 指标：任务 10、11、12。
- 4090 优先和 A100 对照：任务 10、11。

占位符扫描：
- 本计划没有未完成占位标记或空泛执行项。
- 每个任务都有具体文件、命令和预期结果。

类型一致性：
- `source_worker_id` 出现在 `JaxPrefixReady`、`JaxReleaseFeature`、`JaxAERequestState`、runtime routing helper 中；`prefix_ready_ticket` 出现在 `JaxPrefixReady` 和 `prefix_transfer.py` 中。
- `max_vlm_wait_ms=0` 明确表示 no-wait batching，和现有 VLM `_collect_fcfs_batch` 语义一致。
- `JaxMultiGpuVASplitConfig.max_prefix_slots` 默认值为 `num_vlm_workers * max_vlm_batch_size * 3`，与现有单卡 split 的 prefix capacity 规则一致；VLM 默认 `max_batch_size=8`，AE 默认 `max_batch_size=64`。
