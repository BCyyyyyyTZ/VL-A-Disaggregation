# JAX V-A Split Device-Slab IPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 PyTorch V-A disaggregation 调度算法的基础上，实现一个尽可能同构的 JAX 版本，首版强制采用 VLM/AE 两进程，并以前置门禁实验验证 JAX 跨进程 device-slab IPC 是否可行。

**Architecture:** JAX 版沿用当前 PyTorch 版的调度边界：VLM FCFS batching、AE step-level continuous batching、prefix/KV cache lane slab。与当前 PyTorch 路径不同，JAX 首版要求 prefix/KV feature 在 GPU 上只保留一份：VLM 进程独占维护 device-side lane pool，AE 进程启动后只对整块 slab 做一次 CUDA IPC map，之后每个 denoise batch 仅用 slot metadata 在已映射 slab 上切片成 dense view。首个任务必须先验证 JAX/XLA device buffer 能否跨进程以 handle/view 方式共享；若门禁失败，暂停后续实现并回到方案讨论，不继续做 ours runtime。

**Tech Stack:** Python 3.11、JAX 0.5.3、Flax NNX、OpenPI `Pi0`/`Pi0.5`、`multiprocessing`、CUDA IPC 或 JAX device buffer handle、Numba CUDA IPC、pytest、tyro、`/data1/miliang/RLinf/openpi_libero/bin/python`。

---



## 设计约束

- 参考实现是当前 PyTorch 版：
  - `src/openpi/serving/va_split/vlm_process.py`
  - `src/openpi/serving/va_split/ae_process.py`
  - `src/openpi/serving/va_split/prefix_cache_pool.py`
  - `src/openpi/serving/va_split/runtime.py`
  - `src/openpi/policies/va_split_policy.py`
  - `scripts/profile_va_split.py`
- JAX 版需要尽量同构：
  - `VLMWorker` 负责 prefix prefill，并保活 producer-side prefix/KV cache。
  - `VLMWorker` 负责唯一的 prefix/KV lane pool、slot 分配、release 后 lane compaction、request 到 slot 的映射更新。
  - `AEWorker` 负责 active request table、continuous batching、每步 denoise 前选择当前 VLM dense prefix 中仍 active 的 request ids，并在已映射 slab 上创建 slice / dense batch view。
  - `ProcessRuntime` 仍是 request queue、prefix metadata queue、result queue、release queue。
- 首版 ours 必须是两个进程：VLM 进程和 AE 进程。
- 进程间不能通过 Python Queue 传输大量 KV cache / feature tensor。
- 首选 device-side shared slab：VLM 写入并维护唯一 prefix/KV lane pool，AE 在启动握手时 map/open 同一 device allocation，后续读取本地已映射 slab 的 slice / dense batch view；AE 不能创建第二份 GPU prefix/KV storage。
- IPC view 生命周期约束：
  - AE 进程启动并与 VLM 完成 slab 握手后，对 VLM 拥有的整块 prefix/KV device slab 只执行一次 `open_slab` / CUDA IPC map。
  - AE 长期持有该映射，之后每个 denoise step 只根据 active `slot_ids` 和 `JaxSlotMoved` 后的新 slot 映射在已映射 slab 上做 slice / dense batch view。
  - 每个 denoise step 不得重新 export/import IPC handle；若实现退化成逐步重新 open IPC，视为违反本 plan。
  - `prefix_transfer_ms` 只反映首次 map 与 control/metadata 成本，不应包含逐步 IPC reopen 成本。
  - VLM 在 lane 写入和 compaction 后必须通过 `block_until_ready` 或 CUDA event 保证 AE 可见的同步语义。
  - slab 生命周期结束时 AE 再关闭映射。
- VLM compaction 后如果移动了仍在 AE active table 中的请求，必须通过 control/update message 通知 AE 更新 request 对应的 slot；否则 AE 会使用旧 slot 读错 feature。
- CUDA MPS 与 SM 配额语义与 PyTorch 版对齐：
  - profile shell 脚本启动隔离的 MPS 服务，使用独立 `CUDA_MPS_PIPE_DIRECTORY` 和 `CUDA_MPS_LOG_DIRECTORY`。
  - VLM/AE 两个子进程都支持设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。
  - `sm_percent=0` 表示不设置上限，并清理继承的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，与当前 PyTorch `build_mps_process_envs()` 语义一致。
- JAX VLM/AE 两个子进程都必须设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，禁止两个进程启动时各自预占大块 GPU 显存池；显存改为随模型加载、prefix/KV slab 和推理激活按需申请。
- 门禁测试失败时停止，不实现 fallback ours。
- fallback 只能作为后续讨论对象；本 plan 不把 host shared-memory copy 路线作为 ours 继续推进。
- JAX split API 理论上支持 OpenPI0 和 OpenPI0.5；测试与 profile 默认跑 Pi0.5。
- JAX compile 是可选项，默认开启，并尽量与 PyTorch 版 compile/warmup 行为同构：
  - ours split 路径分别编译 VLM 侧 `build_prefix_feature` 和 AE 侧 `denoise_one_batch` 单步 hot path。
  - baseline 路径编译完整 VLA `sample_actions` 图。
  - NNX 模型方法必须用 `openpi.shared.nnx_utils.module_jit` 包装，不能裸用 `jax.jit(model.bound_method)` 或 `nnx.jit` 包装 bound method；现有 OpenPI 已说明裸 NNX jit 路线会带来额外内存占用，可能导致 OOM 或 baseline/ours 对比失真。
  - 开启 compile 时必须执行 shape warmup；关闭 compile 时只做少量 functional warmup。
- JAX checkpoint：
  - 最终应使用 JAX 版 checkpoint。
  - 在 JAX checkpoint 下载完成前，脚本中的 `POLICY_DIR` 可以先用 `/data1/miliang/models/RLinf-Pi05-LIBERO-SFT` 占位，但实现必须把 checkpoint 路径作为 CLI 参数，不写死该占位路径。
- 调度默认值与 PyTorch 版对齐：
  - `max_vlm_wait_ms` 默认 `2.0`，短窗口 FCFS 凑批。
  - VLM 窗口外可 prefetch backlog；达到 `max_vlm_batch_size` 立即 prefix forward，凑不满到窗口截止也 forward，不无限等待。
  - `max_prefix_slots` 默认 `max_vlm_batch_size * 3`，作为 VLM-owned lane pool 槽位上限。
  - `max_ae_batch_size` 默认 `8`，AE 每步最多选择该数量的 active requests。
- 测试代码放在 `tests/serving/va_split_jax/`。
- 测试输出和实验日志放在 `logs/tests/`。



## 文件结构

- Create: `tests/serving/va_split_jax/test_jax_device_ipc_gate.py`
  - JAX 跨进程 device-slab IPC 门禁测试。该文件必须先落地并单独通过，才允许继续后续任务。
- Create: `src/openpi/models/jax_split_types.py`
  - 定义 JAX 版 `JaxPrefixFeature`、`JaxDenoiseState`、`JaxPrefixSlotHandle` 等小型数据结构。
- Modify: `src/openpi/models/pi0.py`
  - 从 `sample_actions` 中抽出 `build_prefix_feature`、`init_denoise_state`、`denoise_one_batch`。
- Create: `tests/models/test_pi0_jax_split.py`
  - 验证 Pi0.5 dummy config 下 split helper 与 `sample_actions` 数值一致；保留 Pi0 链路 smoke test。
- Create: `src/openpi/serving/va_split_jax/types.py`
  - JAX runtime 队列消息类型，只传 request metadata、slot handle、shape/dtype、timing，不传大 tensor。
- Create: `src/openpi/serving/va_split_jax/device_slab.py`
  - device-slab 抽象和通过门禁后选定的 IPC backend。
- Create: `src/openpi/serving/va_split_jax/prefix_cache_pool.py`
  - JAX prefix lane pool，由 VLM 进程独占维护；负责唯一 feature storage、batch view export、release compaction。
- Create: `src/openpi/serving/va_split_jax/vlm_process.py`
  - JAX VLM worker/process，同构 FCFS batching，并承担 prefix pool ownership、slot update 广播。
- Create: `src/openpi/serving/va_split_jax/ae_process.py`
  - JAX AE worker/process，同构 continuous batching；AE 只维护 denoise state 和 slot handle，不保存 prefix/KV feature。
- Create: `src/openpi/serving/va_split_jax/runtime.py`
  - JAX local/process runtime；process runtime 是 ours 主线。
- Create: `src/openpi/serving/va_split_jax/launcher.py`
  - JAX VLM/AE 子进程环境构造；MPS/SM 配额语义同构 PyTorch，额外设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- Create: `src/openpi/serving/va_split_jax/timing.py`
  - 复用 PyTorch timing key 语义，避免 profile 统计分叉。
- Create: `src/openpi/serving/va_split_jax/__init__.py`
  - package marker 和有限导出。
- Create: `src/openpi/policies/jax_va_split_policy.py`
  - JAX split policy wrapper，输入输出 contract 对齐现有 policy。
- Create: `src/openpi/serving/va_split_jax/compile.py`
  - JAX compile 和 warmup helper；区分 ours split 与 baseline monolithic。
- Modify: `scripts/profile_va_split.py`
  - 增加 JAX mode：`jax-monolithic`、`jax-split-ipc`。
- Create: `scripts/run_profile_va_split_jax.sh`
  - 默认 Pi0.5、默认 `/data1/miliang/RLinf/openpi_libero/bin/python`、默认日志进 `logs/`。
- Create: `tests/serving/va_split_jax/*_test.py`
  - 覆盖 IPC handle、lane pool、VLM batching、AE continuous batching、runtime shutdown/error。
- Create: `tests/policies/jax_va_split_policy_test.py`
  - 覆盖 policy contract、batch infer、timing。

---



## Task 1: JAX Device-Slab IPC 门禁测试

**Files:**

- Create: `tests/serving/va_split_jax/test_jax_device_ipc_gate.py`
- Output: `logs/tests/jax_device_ipc_gate.json`

**Current Status:** 门禁测试已经落地并通过：

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_jax_device_ipc_gate.py -q
```

Observed: `1 passed`。

已验证实现路径：

- producer 使用 JAX `unsafe_buffer_pointer()` 获取 device allocation pointer。
- producer 使用 Numba CUDA IPC handle 导出 allocation。
- consumer 使用 `cuda.open_ipc_array` 打开同一 device allocation。
- consumer 检查 JAX 导入后的 pointer alias。
- 日志写入 `logs/tests/jax_device_ipc_gate.json`。

- [x] **Step 1: 创建门禁测试文件**

写入下面的测试骨架。它先不依赖最终 runtime，只验证跨进程 device buffer 共享的事实。测试必须优先尝试 device-side handle；如果当前环境只有 CPU 或无法获得 device IPC handle，测试应 `pytest.skip` 并写出明确原因；如果有 CUDA 设备但 handle 共享失败，测试必须 FAIL。当前仓库已有通过版本时，不要把它退回 skeleton；后续实现应复用已通过的 JAX `unsafe_buffer_pointer()` + Numba CUDA IPC handle 路线。

```python
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
import pathlib
import queue
import time
from typing import Any

import jax
import numpy as np
import pytest


LOG_PATH = pathlib.Path("logs/tests/jax_device_ipc_gate.json")


@dataclasses.dataclass(frozen=True, slots=True)
class GateReport:
    status: str
    reason: str
    producer_devices: list[str]
    consumer_devices: list[str]
    payload_shape: tuple[int, ...]
    payload_dtype: str
    elapsed_ms: float
    transport: str


def _write_report(report: GateReport) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True), encoding="utf-8")


def _device_summary() -> list[str]:
    return [str(device) for device in jax.devices()]


def _has_cuda_device() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def _producer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    start_ns = time.monotonic_ns()
    try:
        payload = jnp.arange(1024, dtype=jnp.float32).reshape(16, 64)
        payload.block_until_ready()
        # 第一版门禁只接受真正的 device-side IPC handle。
        # 具体 handle 导出函数由后续 Step 3 填入，不允许退化成 np.asarray(payload)。
        handle = _export_device_ipc_handle(payload)
        control_queue.put(
            {
                "handle": handle,
                "shape": tuple(payload.shape),
                "dtype": str(payload.dtype),
                "producer_devices": _device_summary(),
                "start_ns": start_ns,
            }
        )
        ack = result_queue.get(timeout=30)
        if ack != "ok":
            raise RuntimeError(f"consumer returned {ack!r}")
    except Exception as exc:
        control_queue.put({"error": repr(exc), "producer_devices": _device_summary(), "start_ns": start_ns})
        raise


def _consumer(control_queue: mp.Queue, result_queue: mp.Queue) -> None:
    message = control_queue.get(timeout=30)
    if "error" in message:
        result_queue.put(f"producer-error: {message['error']}")
        return
    try:
        array = _import_device_ipc_handle(message["handle"], message["shape"], message["dtype"])
        actual = np.asarray(array)
        expected = np.arange(1024, dtype=np.float32).reshape(16, 64)
        np.testing.assert_allclose(actual, expected)
        result_queue.put("ok")
    except Exception as exc:
        result_queue.put(f"consumer-error: {exc!r}")


def _export_device_ipc_handle(array: jax.Array) -> Any:
    raise NotImplementedError("device IPC export is not wired yet")


def _import_device_ipc_handle(handle: Any, shape: tuple[int, ...], dtype: str) -> jax.Array:
    raise NotImplementedError("device IPC import is not wired yet")


def test_jax_device_ipc_gate_requires_cross_process_device_handle():
    start = time.monotonic()
    devices = _device_summary()
    if not _has_cuda_device():
        report = GateReport(
            status="skipped",
            reason="No CUDA JAX device is visible; device IPC gate requires GPU.",
            producer_devices=devices,
            consumer_devices=[],
            payload_shape=(16, 64),
            payload_dtype="float32",
            elapsed_ms=0.0,
            transport="none",
        )
        _write_report(report)
        pytest.skip(report.reason)

    ctx = mp.get_context("spawn")
    control_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    producer = ctx.Process(target=_producer, args=(control_queue, result_queue))
    consumer = ctx.Process(target=_consumer, args=(control_queue, result_queue))
    producer.start()
    consumer.start()
    producer.join(timeout=45)
    consumer.join(timeout=45)

    elapsed_ms = (time.monotonic() - start) * 1000
    status = "passed" if producer.exitcode == 0 and consumer.exitcode == 0 else "failed"
    report = GateReport(
        status=status,
        reason=f"producer_exit={producer.exitcode}, consumer_exit={consumer.exitcode}",
        producer_devices=devices,
        consumer_devices=devices,
        payload_shape=(16, 64),
        payload_dtype="float32",
        elapsed_ms=elapsed_ms,
        transport="device-ipc",
    )
    _write_report(report)

    assert producer.exitcode == 0
    assert consumer.exitcode == 0
```

- [x] **Step 2: 运行测试，确认当前状态是 FAIL 或 SKIP**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_jax_device_ipc_gate.py -q
```

Expected:

- 如果 JAX 只看到 CPU：`SKIPPED`，`logs/tests/jax_device_ipc_gate.json` 中 `status=skipped`。
- 如果 JAX 看到 CUDA：`FAILED`，失败点是 `_export_device_ipc_handle` 的 `NotImplementedError`。

- [x] **Step 3: 实现可用的 device-side IPC handle**

在同一测试文件内临时实现 `_export_device_ipc_handle` 与 `_import_device_ipc_handle`。当前已通过的实现路线为 JAX `unsafe_buffer_pointer()` + Numba CUDA IPC handle。允许的实现路线按优先级排序：

1. JAX/XLA 若暴露跨进程 device buffer handle，则直接使用该 handle。
2. 若 JAX 只暴露 DLPack，但 DLPack capsule 不能跨进程 pickle，则实现一个最小 CUDA IPC adapter，只导出 CUDA IPC mem handle、shape、dtype、strides、device ordinal 和同步事件。
3. 不允许把 `np.asarray(array)`、host shared memory、pickle 大数组作为通过条件。

临时 adapter 的接口必须收敛到：

```python
@dataclasses.dataclass(frozen=True, slots=True)
class DeviceIpcHandle:
    transport: str
    device_ordinal: int
    shape: tuple[int, ...]
    dtype: str
    handle_bytes: bytes
    event_handle_bytes: bytes | None
```

- [x] **Step 4: 运行门禁测试，必须 PASS 才继续**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_jax_device_ipc_gate.py -q
```

Expected:

- CUDA 环境下：`1 passed`。
- `logs/tests/jax_device_ipc_gate.json` 中 `status=passed`，`transport` 不是 `host-copy`。

- [x] **Step 5: 门禁失败时暂停**

如果 Step 4 不能在 CUDA 环境 PASS，停止执行本 plan。不要继续创建 JAX split runtime。需要带着 `logs/tests/jax_device_ipc_gate.json`、pytest 失败信息和所尝试的 IPC backend 回到方案讨论。

- [x] **Step 6: Commit**

只有 Step 4 PASS 后提交。

```bash
git add tests/serving/va_split_jax/test_jax_device_ipc_gate.py logs/tests/jax_device_ipc_gate.json
git commit -m "test: add JAX device IPC gate"
```

---



## Task 2: JAX Split Types 与 Pi0/Pi0.5 Split API

**Files:**

- Create: `src/openpi/models/jax_split_types.py`
- Modify: `src/openpi/models/pi0.py`
- Test: `tests/models/test_pi0_jax_split.py`

**Current Status:** split helper 测试已经通过：

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/models/test_pi0_jax_split.py -q
```

Observed: `2 passed`。

- [x] **Step 1: 写 split types**

Create `src/openpi/models/jax_split_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True, slots=True)
class JaxPrefixFeature:
    past_key_values: Any
    prefix_pad_masks: jax.Array
    state: jax.Array | None


@dataclass(frozen=True, slots=True)
class JaxDenoiseState:
    x_t: jax.Array
    step_idx: jax.Array
    num_steps: int
    dt: jax.Array


@dataclass(frozen=True, slots=True)
class JaxPrefixSlotHandle:
    slot_id: int
    batch_rows: int
    prefix_shape_tree: Any
    prefix_dtype_tree: Any


@dataclass(frozen=True, slots=True)
class JaxPrefixSlabHandleTree:
    max_lanes: int
    prefix_shape_tree: Any
    prefix_dtype_tree: Any
    slab_handle_tree: Any
```

- [x] **Step 2: 在** `Pi0` **上新增** `build_prefix_feature`

在 `src/openpi/models/pi0.py` 中加入 import：

```python
from openpi.models.jax_split_types import JaxDenoiseState
from openpi.models.jax_split_types import JaxPrefixFeature
```

在 `Pi0` 类中新增：

```python
@at.typecheck
def build_prefix_feature(self, rng: at.KeyArrayLike | None, observation: _model.Observation) -> JaxPrefixFeature:
    observation = _model.preprocess_observation(rng, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return JaxPrefixFeature(
        past_key_values=kv_cache,
        prefix_pad_masks=prefix_mask,
        state=observation.state,
    )
```

- [x] **Step 3: 在** `Pi0` **上新增** `init_denoise_state`

```python
@at.typecheck
def init_denoise_state(
    self,
    rng: at.KeyArrayLike,
    batch_size: int,
    noise: at.Float[at.Array, "b ah ad"] | None,
    num_steps: int,
) -> JaxDenoiseState:
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)
    step_idx = jnp.asarray(0, dtype=jnp.int32)
    return JaxDenoiseState(x_t=noise, step_idx=step_idx, num_steps=num_steps, dt=dt)
```

- [x] **Step 4: 在** `Pi0` **上新增** `denoise_one_batch`

```python
@at.typecheck
def denoise_one_batch(
    self,
    prefix_batch: JaxPrefixFeature,
    denoise_batch: JaxDenoiseState,
) -> at.Float[at.Array, "b ah ad"]:
    batch_size = denoise_batch.x_t.shape[0]
    timestep = 1.0 + denoise_batch.step_idx.astype(jnp.float32) * denoise_batch.dt
    timestep = jnp.broadcast_to(timestep, (batch_size,))
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        _model.Observation(
            images={},
            image_masks={},
            state=prefix_batch.state,
        ),
        denoise_batch.x_t,
        timestep,
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_batch.prefix_pad_masks, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_batch.prefix_pad_masks, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=prefix_batch.past_key_values,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    return self.action_out_proj(suffix_out[:, -self.action_horizon :])
```

注意：如果 Pi0 非 Pi0.5 路线需要 `embed_suffix` 的 image/state assumptions，必须保持 `state` 来自 `prefix_batch.state`；测试至少覆盖 Pi0 smoke。

- [x] **Step 5: 重写** `sample_actions` **使用 split helper**

把 `sample_actions` 中 prefix prefill 和 loop 内 denoise 逻辑替换为：

```python
prefix_feature = self.build_prefix_feature(None, observation)
denoise_state = self.init_denoise_state(rng, batch_size, noise, int(num_steps))

def step(state: JaxDenoiseState) -> JaxDenoiseState:
    v_t = self.denoise_one_batch(prefix_feature, state)
    return JaxDenoiseState(
        x_t=state.x_t + state.dt * v_t,
        step_idx=state.step_idx + 1,
        num_steps=state.num_steps,
        dt=state.dt,
    )

def cond(state: JaxDenoiseState) -> jax.Array:
    return state.step_idx < state.num_steps

return jax.lax.while_loop(cond, step, denoise_state).x_t
```

- [x] **Step 6: 写 Pi0.5 数值一致性测试**

Create `tests/models/test_pi0_jax_split.py`:

```python
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config


def _make_observation(config: pi0_config.Pi0Config, *, batch_size: int = 1):
    return config.fake_obs(batch_size=batch_size)


def test_pi05_jax_split_helpers_match_sample_actions_dummy_model():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        dtype="float32",
        pytorch_compile_mode=None,
    )
    model = config.create(jax.random.key(0))
    obs = _make_observation(config)
    noise = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    mono = model.sample_actions(jax.random.key(1), obs, noise=noise, num_steps=4)
    prefix = model.build_prefix_feature(None, obs)
    state = model.init_denoise_state(jax.random.key(1), batch_size=1, noise=noise, num_steps=4)
    for _ in range(4):
        v_t = model.denoise_one_batch(prefix, state)
        state = type(state)(
            x_t=state.x_t + state.dt * v_t,
            step_idx=state.step_idx + 1,
            num_steps=state.num_steps,
            dt=state.dt,
        )
    np.testing.assert_allclose(np.asarray(state.x_t), np.asarray(mono), rtol=1e-4, atol=1e-4)


def test_pi0_jax_split_helpers_smoke_dummy_model():
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=False,
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        dtype="float32",
        pytorch_compile_mode=None,
    )
    model = config.create(jax.random.key(0))
    obs = _make_observation(config)
    noise = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    prefix = model.build_prefix_feature(None, obs)
    state = model.init_denoise_state(jax.random.key(1), batch_size=1, noise=noise, num_steps=2)
    v_t = model.denoise_one_batch(prefix, state)
    assert v_t.shape == noise.shape
```

- [x] **Step 7: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/models/test_pi0_jax_split.py -q
```

Expected: `2 passed`。

- [x] **Step 8: Commit**

```bash
git add src/openpi/models/jax_split_types.py src/openpi/models/pi0.py tests/models/test_pi0_jax_split.py
git commit -m "feat: expose JAX Pi0 split helpers"
```

---



## Task 3: Device Slab Backend 固化

**Files:**

- Create: `src/openpi/serving/va_split_jax/device_slab.py`
- Test: `tests/serving/va_split_jax/test_device_slab.py`

- [x] **Step 1: 写 backend 接口**

Create `src/openpi/serving/va_split_jax/device_slab.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True, slots=True)
class DeviceSlabSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    max_lanes: int


@dataclass(frozen=True, slots=True)
class DeviceSlabHandle:
    spec: DeviceSlabSpec
    transport: str
    device_ordinal: int
    handle_bytes: bytes
    ready_event_bytes: bytes | None = None


class DeviceSlab:
    def __init__(self, spec: DeviceSlabSpec, array: jax.Array, handle: DeviceSlabHandle):
        self.spec = spec
        self.array = array
        self.handle = handle


class DeviceSlabBackend:
    transport = "device-ipc"

    def create_slab(self, spec: DeviceSlabSpec) -> DeviceSlab:
        """Allocate a process-shareable device slab described by spec."""
        raise NotImplementedError

    def open_slab(self, handle: DeviceSlabHandle) -> DeviceSlab:
        """Open a producer-created device slab from another process."""
        raise NotImplementedError

    def copy_lane_from_array(self, slab: DeviceSlab, lane_id: int, value: jax.Array) -> DeviceSlab:
        update = jax.lax.dynamic_update_slice_in_dim(slab.array, value, lane_id, axis=0)
        return DeviceSlab(slab.spec, update, slab.handle)

    def view_batch(self, slab: DeviceSlab, batch_size: int) -> jax.Array:
        return jax.lax.dynamic_slice_in_dim(slab.array, 0, batch_size, axis=0)

    def slice_lanes(self, slab: DeviceSlab, slot_ids: tuple[int, ...]) -> jax.Array:
        """Return a dense-prefix device-side view for already-mapped lanes without reopening IPC."""
        if slot_ids != tuple(range(len(slot_ids))):
            raise ValueError("The first implementation only permits dense-prefix slot ids")
        return self.view_batch(slab, len(slot_ids))
```

- [x] **Step 2: 将 Task 1 通过的 handle 逻辑搬进 backend**

实现一个具体类，例如 `CudaIpcDeviceSlabBackend(DeviceSlabBackend)`。该实现必须复用 Task 1 已经通过的 export/import 逻辑；如果 Task 1 没有得到可工作的 device IPC handle，这一步不能开始。要求：

- `create_slab()` 只在 VLM 进程调用，分配 `(max_lanes, *shape[1:])` 的 device array。
- `open_slab()` 在 AE 进程启动握手时打开 VLM 导出的整块 device allocation；同一 slab 生命周期内只允许调用一次。
- `copy_lane_from_array()` 不能把完整 slab 拉回 host。
- `view_batch()` 返回 JAX array view 或等价 device-side slice。
- `slice_lanes()` 在 AE 进程已经映射的 slab 上按 dense-prefix slot ids 取 view；不得重新 open IPC。
- 第一版优先选择当前 VLM pool dense prefix 中仍 active 的请求，避免为了 view 构造额外 prefix/KV copy。
- AE 进程不能调用 `copy_lane_from_array()`，也不能创建 prefix/KV slab。

- [x] **Step 3: 写单进程 slab 测试**

Create `tests/serving/va_split_jax/test_device_slab.py`:

```python
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend


def test_device_slab_put_lane_and_view_batch():
    backend = make_default_device_slab_backend()
    slab = backend.create_slab(DeviceSlabSpec(name="x", shape=(1, 2, 3), dtype="float32", max_lanes=4))
    slab = backend.copy_lane_from_array(slab, 0, jnp.ones((1, 2, 3), dtype=jnp.float32))
    slab = backend.copy_lane_from_array(slab, 1, jnp.full((1, 2, 3), 2.0, dtype=jnp.float32))
    batch = backend.view_batch(slab, 2)
    np.testing.assert_allclose(np.asarray(batch[0]), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch[1]), np.full((2, 3), 2.0, dtype=np.float32))
```

- [x] **Step 4: 写跨进程 slab 测试**

把 Task 1 的门禁逻辑复用成 `test_device_slab_handle_opens_in_consumer_process()`，但使用 `DeviceSlabBackend` 的 public API。

- [x] **Step 5: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_device_slab.py tests/serving/va_split_jax/test_jax_device_ipc_gate.py -q
```

Expected: 全部 PASS。

- [x] **Step 6: Commit**

```bash
git add src/openpi/serving/va_split_jax/device_slab.py tests/serving/va_split_jax/test_device_slab.py tests/serving/va_split_jax/test_jax_device_ipc_gate.py
git commit -m "feat: add JAX device slab IPC backend"
```

---



## Task 4: VLM-Owned JAX Prefix Cache Lane Pool

**Files:**

- Create: `src/openpi/serving/va_split_jax/prefix_cache_pool.py`
- Test: `tests/serving/va_split_jax/test_prefix_cache_pool.py`

- [ ] **Step 1: 实现 VLM 独占的 JAX tensor tree lane pool**

Create `src/openpi/serving/va_split_jax/prefix_cache_pool.py`:

```python
from __future__ import annotations

from typing import Any

import jax

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import DeviceSlab
from openpi.serving.va_split_jax.device_slab import DeviceSlabBackend
from openpi.serving.va_split_jax.device_slab import DeviceSlabSpec


class JaxVlmPrefixCacheLanePool:
    def __init__(self, *, max_lanes: int, backend: DeviceSlabBackend):
        if max_lanes <= 0:
            raise ValueError("max_lanes must be positive")
        self.max_lanes = max_lanes
        self._backend = backend
        self._past_slabs: Any | None = None
        self._prefix_pad_masks: DeviceSlab | None = None
        self._state: DeviceSlab | None = None
        self._has_state: bool | None = None
        self._request_to_lane: dict[str, int] = {}
        self._lane_to_request: list[str | None] = [None for _ in range(max_lanes)]
        self._active_count = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    def put_lane(self, request_id: str, feature: JaxPrefixFeature) -> int:
        if self._active_count >= self.max_lanes:
            raise RuntimeError(f"VLM prefix lane pool is full ({self.max_lanes} active requests)")
        lane_id = self._active_count
        self._validate_lane_id(lane_id)
        self._ensure_initialized(feature)
        self._past_slabs = _copy_tree_lane(self._backend, self._past_slabs, feature.past_key_values, lane_id)
        assert self._prefix_pad_masks is not None
        self._prefix_pad_masks = self._backend.copy_lane_from_array(
            self._prefix_pad_masks, lane_id, feature.prefix_pad_masks
        )
        if self._has_state:
            assert self._state is not None
            assert feature.state is not None
            self._state = self._backend.copy_lane_from_array(self._state, lane_id, feature.state)
        self._request_to_lane[request_id] = lane_id
        self._lane_to_request[lane_id] = request_id
        self._active_count += 1
        return lane_id

    def release_lane(self, request_id: str) -> tuple[int, int, str] | None:
        lane_id = self._request_to_lane.pop(request_id, None)
        if lane_id is None:
            return None
        last_lane = self._active_count - 1
        moved_request_id = self._lane_to_request[last_lane]
        self._lane_to_request[lane_id] = None
        if lane_id != last_lane:
            if moved_request_id is None:
                raise RuntimeError(f"Cannot compact empty VLM lane {last_lane}")
            self._move_lane(last_lane, lane_id)
            self._lane_to_request[lane_id] = moved_request_id
            self._request_to_lane[moved_request_id] = lane_id
        self._lane_to_request[last_lane] = None
        self._active_count -= 1
        if lane_id != last_lane and moved_request_id is not None:
            return last_lane, lane_id, moved_request_id
        return None

    def export_batch_view(self, request_ids: tuple[str, ...]) -> JaxPrefixFeature:
        if not request_ids:
            raise ValueError("request_ids must be non-empty")
        lane_ids = tuple(self._request_to_lane[request_id] for request_id in request_ids)
        if lane_ids != tuple(range(len(request_ids))):
            raise ValueError(
                "The first implementation exports only the dense prefix of the VLM lane pool; "
                "AE batch selection must use the request ids currently occupying lanes [0, batch_size)."
            )
        return self.view_prefix_batch(len(request_ids))

    def export_slab_handle_tree(self) -> dict[str, Any]:
        if self._past_slabs is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot export prefix slab handles before initialization")
        return _export_slab_handle_tree(self._past_slabs, self._prefix_pad_masks, self._state)

    def _move_lane(self, src_lane: int, dst_lane: int) -> None:
        batch = self.view_prefix_batch(src_lane + 1)
        row = _row_view_tree(batch.past_key_values, src_lane)
        assert self._prefix_pad_masks is not None
        mask_row = self._backend.view_batch(self._prefix_pad_masks, src_lane + 1)[src_lane : src_lane + 1]
        state_row = None
        if self._state is not None:
            state_row = self._backend.view_batch(self._state, src_lane + 1)[src_lane : src_lane + 1]
        assert self._past_slabs is not None
        self._past_slabs = _copy_tree_lane(self._backend, self._past_slabs, row, dst_lane)
        self._prefix_pad_masks = self._backend.copy_lane_from_array(self._prefix_pad_masks, dst_lane, mask_row)
        if self._state is not None:
            assert state_row is not None
            self._state = self._backend.copy_lane_from_array(self._state, dst_lane, state_row)

    def view_prefix_batch(self, batch_size: int) -> JaxPrefixFeature:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.max_lanes:
            raise ValueError(f"batch_size {batch_size} exceeds max lanes {self.max_lanes}")
        if self._past_slabs is None or self._prefix_pad_masks is None:
            raise RuntimeError("Cannot view prefix batch before initialization")
        return JaxPrefixFeature(
            past_key_values=_view_tree_batch(self._backend, self._past_slabs, batch_size),
            prefix_pad_masks=self._backend.view_batch(self._prefix_pad_masks, batch_size),
            state=self._backend.view_batch(self._state, batch_size) if self._state is not None else None,
        )

    def _ensure_initialized(self, feature: JaxPrefixFeature) -> None:
        if self._past_slabs is None:
            self._past_slabs = _make_tree_slabs(self._backend, "past", feature.past_key_values, self.max_lanes)
        if self._prefix_pad_masks is None:
            self._prefix_pad_masks = _make_slab(self._backend, "prefix_pad_masks", feature.prefix_pad_masks, self.max_lanes)
        has_state = feature.state is not None
        if self._has_state is None:
            self._has_state = has_state
            if has_state:
                assert feature.state is not None
                self._state = _make_slab(self._backend, "state", feature.state, self.max_lanes)
        elif self._has_state != has_state:
            raise ValueError("Cannot mix prefix features with and without state")

    def _validate_lane_id(self, lane_id: int) -> None:
        if lane_id < 0 or lane_id >= self.max_lanes:
            raise ValueError(f"lane_id {lane_id} outside prefix lane pool capacity {self.max_lanes}")


def _make_slab(backend: DeviceSlabBackend, name: str, value: jax.Array, max_lanes: int) -> DeviceSlab:
    return backend.create_slab(DeviceSlabSpec(name=name, shape=tuple(value.shape), dtype=str(value.dtype), max_lanes=max_lanes))


def _make_tree_slabs(backend: DeviceSlabBackend, prefix: str, value: Any, max_lanes: int) -> Any:
    if isinstance(value, jax.Array):
        return _make_slab(backend, prefix, value, max_lanes)
    if isinstance(value, tuple):
        return tuple(_make_tree_slabs(backend, f"{prefix}.{idx}", item, max_lanes) for idx, item in enumerate(value))
    if isinstance(value, list):
        return [_make_tree_slabs(backend, f"{prefix}.{idx}", item, max_lanes) for idx, item in enumerate(value)]
    raise TypeError(f"Unsupported JAX prefix tree node: {type(value)}")


def _copy_tree_lane(backend: DeviceSlabBackend, slabs: Any, value: Any, lane_id: int) -> Any:
    if isinstance(slabs, DeviceSlab):
        return backend.copy_lane_from_array(slabs, lane_id, value)
    if isinstance(slabs, tuple):
        return tuple(_copy_tree_lane(backend, slab, item, lane_id) for slab, item in zip(slabs, value, strict=True))
    if isinstance(slabs, list):
        return [_copy_tree_lane(backend, slab, item, lane_id) for slab, item in zip(slabs, value, strict=True)]
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _view_tree_batch(backend: DeviceSlabBackend, slabs: Any, batch_size: int) -> Any:
    if isinstance(slabs, DeviceSlab):
        return backend.view_batch(slabs, batch_size)
    if isinstance(slabs, tuple):
        return tuple(_view_tree_batch(backend, item, batch_size) for item in slabs)
    if isinstance(slabs, list):
        return [_view_tree_batch(backend, item, batch_size) for item in slabs]
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _export_slab_handle_tree(
    past_slabs: Any,
    prefix_pad_masks: DeviceSlab | None,
    state: DeviceSlab | None,
) -> dict[str, Any]:
    if prefix_pad_masks is None:
        raise RuntimeError("Cannot export prefix slab handles before initialization")
    return {
        "past_key_values": _export_slab_tree_handles(past_slabs),
        "prefix_pad_masks": prefix_pad_masks.handle,
        "state": state.handle if state is not None else None,
    }


def _export_slab_tree_handles(slabs: Any) -> Any:
    if isinstance(slabs, DeviceSlab):
        return slabs.handle
    if isinstance(slabs, tuple):
        return tuple(_export_slab_tree_handles(item) for item in slabs)
    if isinstance(slabs, list):
        return [_export_slab_tree_handles(item) for item in slabs]
    raise TypeError(f"Unsupported JAX slab tree node: {type(slabs)}")


def _row_view_tree(value: Any, row: int) -> Any:
    if isinstance(value, jax.Array):
        return value[row : row + 1]
    if isinstance(value, tuple):
        return tuple(_row_view_tree(item, row) for item in value)
    if isinstance(value, list):
        return [_row_view_tree(item, row) for item in value]
    raise TypeError(f"Unsupported JAX prefix tree node: {type(value)}")
```

- [ ] **Step 2: 写 VLM-owned lane pool 测试**

Create `tests/serving/va_split_jax/test_prefix_cache_pool.py`:

```python
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from openpi.models.jax_split_types import JaxPrefixFeature
from openpi.serving.va_split_jax.device_slab import make_default_device_slab_backend
from openpi.serving.va_split_jax.prefix_cache_pool import JaxVlmPrefixCacheLanePool


def _feature(fill: float) -> JaxPrefixFeature:
    return JaxPrefixFeature(
        past_key_values=(jnp.full((1, 3, 2, 4), fill, dtype=jnp.bfloat16), jnp.full((1, 3, 2, 4), fill + 1, dtype=jnp.bfloat16)),
        prefix_pad_masks=jnp.ones((1, 3), dtype=jnp.bool_),
        state=jnp.full((1, 8), fill, dtype=jnp.float32),
    )


def test_vlm_owned_prefix_cache_lane_pool_exports_dense_batch():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    batch = pool.export_batch_view(("req-1", "req-2"))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][0]), np.full((3, 2, 4), 1.0, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(batch.past_key_values[0][1]), np.full((3, 2, 4), 2.0, dtype=np.float32))
    assert batch.prefix_pad_masks.shape == (2, 3)
    assert batch.state.shape == (2, 8)


def test_vlm_owned_prefix_cache_lane_pool_compacts_on_release():
    pool = JaxVlmPrefixCacheLanePool(max_lanes=4, backend=make_default_device_slab_backend())
    pool.put_lane("req-1", _feature(1.0))
    pool.put_lane("req-2", _feature(2.0))
    moved = pool.release_lane("req-1")
    assert moved == (1, 0, "req-2")
    batch = pool.export_batch_view(("req-2",))
    np.testing.assert_allclose(np.asarray(batch.state), np.full((1, 8), 2.0, dtype=np.float32))
```

- [ ] **Step 3: 运行测试**



Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_prefix_cache_pool.py -q
```

Expected: `2 passed`。

- [ ] **Step 4: Commit**

```bash
git add src/openpi/serving/va_split_jax/prefix_cache_pool.py tests/serving/va_split_jax/test_prefix_cache_pool.py
git commit -m "feat: add JAX prefix cache lane pool"
```

---



## Task 5: JAX VLM/AE Message Types 与 Timing

**Files:**

- Create: `src/openpi/serving/va_split_jax/types.py`
- Create: `src/openpi/serving/va_split_jax/timing.py`
- Create: `src/openpi/serving/va_split_jax/__init__.py`
- Test: `tests/serving/va_split_jax/test_types.py`

- [ ] **Step 1: 写消息类型**

Create `src/openpi/serving/va_split_jax/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpi.models.jax_split_types import JaxPrefixSlabHandleTree
from openpi.models.jax_split_types import JaxPrefixSlotHandle


@dataclass(frozen=True, slots=True)
class JaxRequestEnvelope:
    request_id: str
    observation: dict[str, Any]
    sample_kwargs: dict[str, Any]
    enqueue_ns: int
    dequeue_ns: int | None = None
    dequeue_start_ns: int | None = None


@dataclass(frozen=True, slots=True)
class JaxBatchRequestEnvelope:
    batch_id: str
    request_ids: tuple[str, ...]
    observation: dict[str, Any]
    sample_kwargs: dict[str, Any]
    enqueue_ns: int
    dequeue_ns: int | None = None
    dequeue_start_ns: int | None = None


@dataclass(frozen=True, slots=True)
class JaxPrefixReady:
    request_id: str
    slot_handle: JaxPrefixSlotHandle
    num_steps: int
    sample_kwargs: dict[str, Any]
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxActionResult:
    request_id: str
    actions: Any
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxReleaseFeature:
    request_id: str
    slot_id: int


@dataclass(frozen=True, slots=True)
class JaxSlotMoved:
    request_id: str
    old_slot_id: int
    new_slot_id: int


@dataclass(frozen=True, slots=True)
class JaxPrefixSlabReady:
    slab: JaxPrefixSlabHandleTree
    timing: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class JaxDenoiseBatchSlots:
    request_ids: tuple[str, ...]
    slot_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class JaxWorkerError:
    request_id: str | None
    error: str
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class JaxShutdown:
    pass
```

- [ ] **Step 2: 写 timing helper**

Create `src/openpi/serving/va_split_jax/timing.py`，复制 PyTorch 版 `src/openpi/serving/va_split/timing.py` 的 public function 名称，并把 CUDA sync 改成 JAX：

```python
from __future__ import annotations

import queue
import time
from typing import Any

import jax


def synchronize_jax_if_needed() -> None:
    # JAX 没有全局 synchronize；调用方应对关键 array block_until_ready。
    for device in jax.devices():
        del device


def timed_queue_get(q, *args, **kwargs) -> tuple[Any, int, int]:
    start_ns = time.monotonic_ns()
    try:
        message = q.get(*args, **kwargs)
    except queue.Empty:
        raise
    end_ns = time.monotonic_ns()
    return message, start_ns, end_ns


def queue_wait_and_transfer_ms(
    *,
    enqueue_ns: float | int | None,
    get_start_ns: int | None,
    get_end_ns: int | None,
) -> tuple[float, float]:
    if enqueue_ns is None or get_start_ns is None or get_end_ns is None:
        return 0.0, 0.0
    return (
        max(0.0, (float(get_start_ns) - float(enqueue_ns)) / 1_000_000),
        max(0.0, (float(get_end_ns) - float(get_start_ns)) / 1_000_000),
    )
```

- [ ] **Step 3: 写 package marker**

Create `src/openpi/serving/va_split_jax/__init__.py`:

```python
"""JAX V-A split serving runtime."""
```

- [ ] **Step 4: 写 round-trip 测试**

Create `tests/serving/va_split_jax/test_types.py`:

```python
from __future__ import annotations

import multiprocessing as mp
import time

from openpi.models.jax_split_types import JaxPrefixSlotHandle
from openpi.serving.va_split_jax.types import JaxPrefixReady
from openpi.serving.va_split_jax.types import JaxRequestEnvelope


def test_jax_prefix_ready_round_trips_metadata_only():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    message = JaxPrefixReady(
        request_id="req-1",
        slot_handle=JaxPrefixSlotHandle(slot_id=3, batch_rows=1, prefix_shape_tree=("shape",), prefix_dtype_tree=("dtype",)),
        num_steps=10,
        sample_kwargs={"num_steps": 10},
    )
    q.put(message)
    received = q.get(timeout=5)
    assert received.request_id == "req-1"
    assert received.slot_handle.slot_id == 3


def test_jax_slot_moved_round_trips_after_vlm_compaction():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put(JaxSlotMoved(request_id="req-2", old_slot_id=1, new_slot_id=0))
    received = q.get(timeout=5)
    assert received.request_id == "req-2"
    assert received.old_slot_id == 1
    assert received.new_slot_id == 0


def test_jax_denoise_batch_slots_round_trips_metadata_only():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    q.put(JaxDenoiseBatchSlots(request_ids=("req-1", "req-2"), slot_ids=(0, 1)))
    received = q.get(timeout=5)
    assert received.request_ids == ("req-1", "req-2")
    assert received.slot_ids == (0, 1)


def test_jax_request_envelope_has_enqueue_timestamp():
    request = JaxRequestEnvelope(request_id="req-1", observation={}, sample_kwargs={}, enqueue_ns=time.monotonic_ns())
    assert request.enqueue_ns > 0
```

- [ ] **Step 5: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_types.py -q
```

Expected: `2 passed`。

- [ ] **Step 6: Commit**

```bash
git add src/openpi/serving/va_split_jax/types.py src/openpi/serving/va_split_jax/timing.py src/openpi/serving/va_split_jax/__init__.py tests/serving/va_split_jax/test_types.py
git commit -m "feat: add JAX VA split message types"
```

---



## Task 6: JAX VLM Process

**Files:**

- Create: `src/openpi/serving/va_split_jax/vlm_process.py`
- Test: `tests/serving/va_split_jax/test_vlm_process.py`

- [ ] **Step 1: 实现 VLMWorker**

VLMWorker 同构 PyTorch `VLMWorker` 的 FCFS batching，但 JAX 版额外拥有唯一 prefix/KV lane pool。输出 `JaxPrefixReady` 只带 `slot_handle`，真实 prefix 始终只保留在 VLM 进程的 device slab 中。

核心接口：

```python
class JaxVLMWorker:
    def __init__(self, *, model, max_live_features: int, prefix_pool):
        self._model = model
        self._max_live_features = max_live_features
        self._prefix_pool = prefix_pool

    @property
    def available_live_feature_slots(self) -> int:
        return self._max_live_features - self._prefix_pool.active_count

    def handle_request(self, request: JaxRequestEnvelope) -> JaxPrefixReady:
        return self.handle_batch([request])[0]

    def release(self, release: JaxReleaseFeature) -> JaxSlotMoved | None:
        moved = self._prefix_pool.release_lane(release.request_id)
        if moved is None:
            return None
        old_slot_id, new_slot_id, moved_request_id = moved
        return JaxSlotMoved(request_id=moved_request_id, old_slot_id=old_slot_id, new_slot_id=new_slot_id)
```

`handle_batch()` 必须：

- stack observation，逻辑对齐 `_stack_request_observations`。
- 调 `model.build_prefix_feature(None, observation)`。
- 对每个 row 写入 VLM-owned prefix lane。
- 返回对应 row 的 `JaxPrefixReady(slot_handle=...)`。
- slot handle 中只包含 slot id、shape/dtype tree 和必要的 view/open metadata；不包含 prefix/KV array。

- [ ] **Step 2: 实现 JaxVLMProcess queue loop**

同构 `src/openpi/serving/va_split/vlm_process.py`：

- 支持 `JaxRequestEnvelope`。
- 支持 `JaxBatchRequestEnvelope`。
- 第一次 prefix pool 初始化后，VLM 通过 control queue 发送一次 `JaxPrefixSlabReady`，AE 用它执行一次 `open_slab` 并长期持有映射。
- 支持 `JaxShutdown`。
- 支持 release queue drain。
- release 后由 VLM 执行 lane compaction；若移动了仍活跃请求，VLM 通过 prefix/control queue 发送 `JaxSlotMoved` 给 AE。
- VLM 不为每个 denoise step 导出新的 IPC handle。
- 支持 FCFS batching：`max_batch_size`、`max_wait_ms`。
- live slot 满时 backlog，不丢请求。

默认值与 PyTorch 版对齐：

```python
max_vlm_batch_size = 8
max_vlm_wait_ms = 2.0
max_ae_batch_size = 8
max_prefix_slots = max_vlm_batch_size * 3
```

FCFS 行为必须与当前 PyTorch `VLMProcess._collect_fcfs_batch()` 对齐：

- 第一条 request 出队后启动短窗口，窗口长度为 `max_vlm_wait_ms`。
- 窗口内只拼 compatibility key 一致的请求。
- 达到 `max_vlm_batch_size` 立即运行 prefix。
- 窗口截止仍未凑满也立即运行 prefix。
- 窗口外可 prefetch backlog，遇到不兼容 request 放回 backlog 头部，不跳过。
- live slot 不足时 defer 当前 message 并短 sleep，不丢请求。

- [ ] **Step 3: 写 VLM 测试 fake model**

Create `tests/serving/va_split_jax/test_vlm_process.py`，fake model 返回小型 JAX prefix tree：

```python
class FakeJaxSplitModel:
    def __init__(self):
        self.prefix_batch_sizes = []

    def build_prefix_feature(self, rng, observation):
        batch = int(observation.state.shape[0])
        self.prefix_batch_sizes.append(batch)
        return JaxPrefixFeature(
            past_key_values=(jnp.ones((batch, 3, 2, 4), dtype=jnp.float32), jnp.ones((batch, 3, 2, 4), dtype=jnp.float32)),
            prefix_pad_masks=jnp.ones((batch, 3), dtype=jnp.bool_),
            state=observation.state,
        )
```

- [ ] **Step 4: 覆盖 batch 与 release**

测试内容：

- 两个兼容 request 在 `max_wait_ms` 内形成一个 batch。
- `prefix_batch_sizes == [2]`。
- prefix_queue 中有两个 `JaxPrefixReady`。
- release 后 slot 回收。
- release 导致 compaction 时，prefix/control queue 收到 `JaxSlotMoved(request_id="...", old_slot_id=..., new_slot_id=...)`。

- [ ] **Step 5: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_vlm_process.py -q
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/openpi/serving/va_split_jax/vlm_process.py tests/serving/va_split_jax/test_vlm_process.py
git commit -m "feat: add JAX VLM split worker"
```

---



## Task 7: JAX AE Process

**Files:**

- Create: `src/openpi/serving/va_split_jax/ae_process.py`
- Test: `tests/serving/va_split_jax/test_ae_process.py`

- [ ] **Step 1: 实现 AERequestState**

```python
from dataclasses import dataclass

import jax


@dataclass
class JaxAERequestState:
    request_id: str
    active_lane_id: int
    prefix_slot_id: int
    x_t: jax.Array
    step_idx: int
    num_steps: int
    dt: jax.Array
    started_ns: int
    timing: dict[str, float]
    ae_step_ms: list[float]
    ae_batch_sizes: list[int]
    lane_compact_ms: float = 0.0
```

- [ ] **Step 2: 实现 JaxAEWorker**

同构 PyTorch `AEWorker` 的 continuous batching，但不维护 prefix/KV lane pool：

- `add_prefix(ready)` 只登记 request、prefix slot id、denoise state，不复制 prefix/KV feature。
- `apply_slot_moved(message)` 根据 VLM 的 `JaxSlotMoved` 更新 active request 的 `prefix_slot_id`。
- `select_ready_lanes()` 取 active table 中 prefix slot id 为 `[0, batch_size)` 的 dense prefix 请求；第一版不从稀疏 slot gather，避免 AE 侧构造 feature copy。
- `step_once(batch_slots)` 使用 `JaxDenoiseBatchSlots.slot_ids` 在已映射 slab 上切片出 prefix batch view，然后做一轮 denoise。
- 完成请求立即返回 `JaxActionResult` 和 `JaxReleaseFeature`。
- `_remove_active_lane()` 只 compact AE 本地 denoise state table，不移动 prefix/KV feature。

- [ ] **Step 3: 实现 JaxAEProcess**

同构 PyTorch `AEProcess`：

- prefix_queue 只接收 metadata。
- 启动握手时等待 `JaxPrefixSlabReady`，对每个 prefix/KV slab 只调用一次 `open_slab`。
- 遇到 `JaxSlotMoved` 时只更新 active request 的 slot id。
- 每次 denoise batch 前，AE 本地生成 `JaxDenoiseBatchSlots`，在已映射 slab 上做 slice / dense batch view；不得重新 open IPC。
- result_queue 输出 action，可用 host copy，因为 action 体量小。
- release_queue 通知 VLM 释放 slot；VLM 完成 compaction 后通过 `JaxSlotMoved` 更新 AE。
- error 时清空 active 并 release。

- [ ] **Step 4: 写 AE continuous batching 测试**

Create `tests/serving/va_split_jax/test_ae_process.py`，覆盖：

- 两个 ready request 一起进入一次 denoise batch。
- `num_steps=2` 时每个请求得到两次 step。
- 先完成的 request 释放 slot 后，VLM compaction 发送 `JaxSlotMoved`，AE 更新对应 active request 的 `prefix_slot_id`。
- `ae_effective_batch`、`prefix_pool_write_ms`、`prefix_pool_compact_ms`、`prefix_slab_map_ms` 有值。

- [ ] **Step 5: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_ae_process.py -q
```

Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add src/openpi/serving/va_split_jax/ae_process.py tests/serving/va_split_jax/test_ae_process.py
git commit -m "feat: add JAX AE continuous batching worker"
```

---



## Task 8: JAX Process Runtime

**Files:**

- Create: `src/openpi/serving/va_split_jax/launcher.py`
- Create: `src/openpi/serving/va_split_jax/runtime.py`
- Test: `tests/serving/va_split_jax/test_runtime.py`

- [ ] **Step 1: 实现 local runtime 仅用于数值对齐**

`JaxLocalVASplitRuntime` 只用于数值和调度单元测试，不作为 ours profile 主线。它必须复用 `JaxVLMWorker` 与 `JaxAEWorker`，执行流程如下：

```python
class JaxLocalVASplitRuntime:
    def infer(self, observation: dict, sample_kwargs: dict) -> JaxActionResult:
        request_id = str(uuid.uuid4())
        ready = self.vlm_worker.handle_request(
            JaxRequestEnvelope(
                request_id=request_id,
                observation=observation,
                sample_kwargs=dict(sample_kwargs),
                enqueue_ns=time.monotonic_ns(),
            )
        )
        self.ae_worker.add_prefix(ready)
        while request_id in self.ae_worker.active:
            batch_request_ids = self.ae_worker.select_ready_request_ids()
            prefix_batch = self.vlm_worker.export_batch_view(batch_request_ids)
            results, releases = self.ae_worker.step_once(prefix_batch)
            for release in releases:
                moved = self.vlm_worker.release(release)
                if moved is not None:
                    self.ae_worker.apply_slot_moved(moved)
            for result in results:
                if result.request_id == request_id:
                    return result
        raise RuntimeError(f"Request {request_id} finished without an action result")
```

- [ ] **Step 2: 实现 JAX launcher 环境**

Create `src/openpi/serving/va_split_jax/launcher.py`：

```python
from __future__ import annotations

import os


def build_jax_mps_process_envs(
    *,
    cuda_visible_devices: str,
    mps_pipe_dir: str,
    mps_log_dir: str,
    ae_sm_percent: int,
    vlm_sm_percent: int,
) -> tuple[dict[str, str], dict[str, str]]:
    base_env = os.environ.copy()
    common_env = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CUDA_MPS_PIPE_DIRECTORY": mps_pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": mps_log_dir,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    ae_env = base_env | common_env
    if ae_sm_percent != 0:
        ae_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ae_sm_percent)
    else:
        ae_env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

    vlm_env = base_env | common_env
    if vlm_sm_percent != 0:
        vlm_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(vlm_sm_percent)
    else:
        vlm_env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    return ae_env, vlm_env
```

测试必须覆盖：

- AE/VLM env 都包含 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- `ae_sm_percent=20` 时 AE env 设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20`。
- `vlm_sm_percent=0` 时 VLM env 不包含 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，与 PyTorch 语义一致。

- [ ] **Step 3: 实现 process runtime**

`JaxProcessVASplitRuntime` 同构 PyTorch `ProcessVASplitRuntime`：

- `spawn` 两个子进程。
- request queue 传 observation 和 sample kwargs。
- prefix queue 只传 `JaxPrefixReady` metadata。
- prefix/control queue 传 `JaxPrefixReady`、一次性的 `JaxPrefixSlabReady`、`JaxSlotMoved` 和 lightweight `JaxDenoiseBatchSlots` metadata；不传逐步 IPC handle。
- result queue 传 small action result。
- release queue 传 slot release。
- result collector thread 聚合结果。
- `infer()` 与 `infer_batch()` 对齐 PyTorch runtime contract。
- `shutdown()` 发 `JaxShutdown` 并 join/terminate。
- 子进程启动前使用 `build_jax_mps_process_envs()` 构造 env，并在 child target 内先应用 env，再 import/初始化 JAX model。

- [ ] **Step 4: 确保模型加载在子进程内发生**

进程 target：

```python
def _run_jax_vlm_process(model_factory, request_queue, prefix_queue, release_queue, runtime_config):
    model = model_factory()
    JaxVLMProcess(...).run()


def _run_jax_ae_process(model_factory, prefix_queue, result_queue, release_queue, runtime_config):
    model = model_factory()
    JaxAEProcess(...).run()
```

不要在 parent 中创建 JAX model 后 pickle 给 child。

- [ ] **Step 5: 写 runtime 测试**

测试用 fake model：

- `infer()` 返回 shape 正确 action。
- `infer_batch()` 保持 row order。
- result timing 包含 `vlm_effective_batch`、`ae_effective_batch_mean`。
- `shutdown()` 后再 `infer()` 抛 `RuntimeError`。

- [ ] **Step 6: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_runtime.py -q
```

Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/openpi/serving/va_split_jax/launcher.py src/openpi/serving/va_split_jax/runtime.py tests/serving/va_split_jax/test_runtime.py
git commit -m "feat: add JAX VA split process runtime"
```

---



## Task 9: JAX VA Split Policy

**Files:**

- Create: `src/openpi/policies/jax_va_split_policy.py`
- Test: `tests/policies/jax_va_split_policy_test.py`

- [ ] **Step 1: 实现 policy wrapper**

接口对齐 `VASplitPolicy`：

- `supports_concurrent_infer = True`
- `infer(obs, noise=None)`
- `infer_batch(obs_batch, noise=None)`
- `metadata`
- `reset`
- `shutdown`
- `close`

JAX 输入保持 JAX/NumPy，不转 PyTorch。

- [ ] **Step 2: 实现** `create_trained_jax_va_split_policy`

要求：

- 默认加载 JAX checkpoint。
- 默认 Pi0.5 profile config 使用 `pi05_libero`。
- `checkpoint_dir` 当前可用 `/data1/miliang/models/RLinf-Pi05-LIBERO-SFT` 作为占位默认值；实现时必须保留 CLI 可覆盖能力，并在加载阶段检测是否为 JAX checkpoint。
- 如果目录只有 PyTorch `model.safetensors` 而没有 JAX checkpoint state，JAX policy 创建应给出清晰错误，提示需要下载 JAX 版 checkpoint；不要静默走 PyTorch 权重或自动转换。
- 构造 `JaxProcessVASplitRuntime`。
- transforms / norm_stats 路线与 `create_trained_va_split_policy` 对齐。

- [ ] **Step 3: 写 policy contract 测试**

覆盖：

- 单请求输出 `actions` shape。
- batch 请求只调用 runtime batch 一次。
- `noise` batch 形状正确。
- timing 透传。
- shutdown 委托 runtime。

- [ ] **Step 4: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/policies/jax_va_split_policy_test.py -q
```

Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add src/openpi/policies/jax_va_split_policy.py tests/policies/jax_va_split_policy_test.py
git commit -m "feat: add JAX VA split policy"
```

---



## Task 10: JAX Compile 与 Warmup

**Files:**

- Create: `src/openpi/serving/va_split_jax/compile.py`
- Modify: `src/openpi/policies/jax_va_split_policy.py`
- Modify: `scripts/profile_va_split.py`
- Test: `tests/serving/va_split_jax/test_compile_warmup.py`

- [ ] **Step 1: 定义 compile 配置**

Create `src/openpi/serving/va_split_jax/compile.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from openpi.shared import nnx_utils


JAX_COMPILE_WARMUP_BATCH_PLAN: tuple[tuple[int, int], ...] = (
    (1, 2),
    (4, 1),
    (8, 2),
    (16, 1),
    (20, 2),
    (24, 1),
    (32, 1),
)


@dataclass(frozen=True, slots=True)
class JaxCompileConfig:
    enabled: bool = True
    warmup_enabled: bool = True
    warmup_max_batch_size: int = 32


def maybe_jit_split_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled:
        return model
    model.build_prefix_feature = nnx_utils.module_jit(model.build_prefix_feature)
    model.denoise_one_batch = nnx_utils.module_jit(model.denoise_one_batch)
    return model


def maybe_jit_monolithic_model(model: Any, config: JaxCompileConfig) -> Any:
    if not config.enabled:
        return model
    model.sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    return model


def planned_warmup_batches(*, max_batch_size: int, warmup_max_batch_size: int) -> tuple[tuple[int, int], ...]:
    cap = min(max_batch_size, warmup_max_batch_size)
    return tuple((batch_size, repeats) for batch_size, repeats in JAX_COMPILE_WARMUP_BATCH_PLAN if batch_size <= cap)
```

说明：

- `maybe_jit_split_model()` 用于 ours：VLM 子进程编译 `build_prefix_feature`，AE 子进程编译 `denoise_one_batch`。即使 helper 暂时一起包装，也必须在子进程内分别触发 warmup，避免 parent 编译结果不可复用。
- `maybe_jit_monolithic_model()` 用于 baseline：编译完整 `sample_actions`。
- 所有 NNX model method compile 必须走 `nnx_utils.module_jit`，禁止裸 `jax.jit(model.bound_method)` 或 `nnx.jit(model.bound_method)`。
- 默认 `enabled=True`，CLI 可关闭。

- [ ] **Step 2: 实现 split warmup helper**

在 `compile.py` 增加：

```python
def warmup_split_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_vlm_batch_size: int,
    max_ae_batch_size: int,
    max_prefix_slots: int,
    num_steps: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    if not config.warmup_enabled:
        return {"jax_warmup_batches": 0.0}
    batches = planned_warmup_batches(
        max_batch_size=max(max_vlm_batch_size, max_ae_batch_size, max_prefix_slots),
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    warmed = 0
    for batch_size, repeats in batches:
        if batch_size > max_prefix_slots:
            continue
        for _ in range(repeats):
            obs = observation_factory(batch_size)
            noise = noise_factory(batch_size)
            prefix = model.build_prefix_feature(None, obs)
            denoise_state = model.init_denoise_state(jax.random.key(0), batch_size, noise, num_steps)
            v_t = model.denoise_one_batch(prefix, denoise_state)
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, v_t)
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}
```

要求：

- VLM warmup 至少覆盖 `max_vlm_batch_size` 以内的 prefix batch shape。
- AE warmup 至少覆盖 `max_ae_batch_size` 以内的 single-step denoise shape。
- warmup 上限受 `max_prefix_slots` 限制，行为对齐 PyTorch `COMPILE_WARMUP_BATCH_PLAN` 的 clamp 逻辑。

- [ ] **Step 3: 实现 monolithic baseline warmup helper**

在 `compile.py` 增加：

```python
def warmup_monolithic_model(
    *,
    model: Any,
    observation_factory,
    noise_factory,
    max_batch_size: int,
    num_steps: int,
    config: JaxCompileConfig,
) -> dict[str, float]:
    if not config.warmup_enabled:
        return {"jax_warmup_batches": 0.0}
    batches = planned_warmup_batches(
        max_batch_size=max_batch_size,
        warmup_max_batch_size=config.warmup_max_batch_size,
    )
    warmed = 0
    for batch_size, repeats in batches:
        for _ in range(repeats):
            obs = observation_factory(batch_size)
            noise = noise_factory(batch_size)
            actions = model.sample_actions(jax.random.key(0), obs, noise=noise, num_steps=num_steps)
            actions.block_until_ready()
            warmed += 1
    return {"jax_warmup_batches": float(warmed)}
```

baseline warmup 编译完整 VLA 推理图，不拆 VLM/AE。

- [ ] **Step 4: 接入 policy/runtime 创建**

在 `create_trained_jax_va_split_policy()` 增加参数：

```python
jax_compile: bool = True
jax_compile_warmup: bool = True
jax_compile_warmup_max_batch_size: int = 32
```

要求：

- `jax-split-ipc`：VLM/AE 子进程内加载模型后调用 `maybe_jit_split_model()`；进程启动后或接受正式请求前执行 split warmup。
- `jax-monolithic`：创建 baseline policy 时调用 `maybe_jit_monolithic_model()` 并执行 monolithic warmup。
- `jax_compile=False` 时不调用 `jax.jit`，warmup 只跑 1 个 batch size 1 的 functional request，避免首次正式请求才初始化模型。

- [ ] **Step 5: 扩展 profile CLI**

在 `scripts/profile_va_split.py` 的 args 增加：

```python
jax_compile: bool = True
jax_compile_warmup: bool = True
jax_compile_warmup_max_batch_size: int = 32
```

要求：

- `--no-jax-compile` 能关闭 JIT。
- `--no-jax-compile-warmup` 能关闭 compile warmup。
- JSON summary 写入：
  - `jax_compile_enabled`
  - `jax_compile_warmup_enabled`
  - `jax_warmup_batches`

- [ ] **Step 6: 写 compile/warmup 测试**

Create `tests/serving/va_split_jax/test_compile_warmup.py`:

```python
from __future__ import annotations

from openpi.serving.va_split_jax.compile import JaxCompileConfig
from openpi.serving.va_split_jax.compile import maybe_jit_monolithic_model
from openpi.serving.va_split_jax.compile import maybe_jit_split_model
from openpi.serving.va_split_jax.compile import planned_warmup_batches


def test_planned_warmup_batches_clamps_to_prefix_capacity():
    assert planned_warmup_batches(max_batch_size=24, warmup_max_batch_size=32) == (
        (1, 2),
        (4, 1),
        (8, 2),
        (16, 1),
        (20, 2),
        (24, 1),
    )


def test_compile_config_defaults_to_enabled():
    config = JaxCompileConfig()
    assert config.enabled is True
    assert config.warmup_enabled is True
    assert config.warmup_max_batch_size == 32


def test_compile_helpers_use_module_jit(monkeypatch):
    calls = []

    def fake_module_jit(method, *args, **kwargs):
        calls.append((method.__name__, args, kwargs))
        return method

    class FakeModel:
        def build_prefix_feature(self):
            return None

        def denoise_one_batch(self):
            return None

        def sample_actions(self):
            return None

    monkeypatch.setattr("openpi.serving.va_split_jax.compile.nnx_utils.module_jit", fake_module_jit)
    maybe_jit_split_model(FakeModel(), JaxCompileConfig(enabled=True))
    maybe_jit_monolithic_model(FakeModel(), JaxCompileConfig(enabled=True))

    assert [call[0] for call in calls] == ["build_prefix_feature", "denoise_one_batch", "sample_actions"]
    assert calls[-1][2] == {"static_argnames": ("num_steps",)}
```

- [ ] **Step 7: 运行测试**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest tests/serving/va_split_jax/test_compile_warmup.py -q
```

Expected: `3 passed`。

- [ ] **Step 8: Commit**

```bash
git add src/openpi/serving/va_split_jax/compile.py src/openpi/policies/jax_va_split_policy.py scripts/profile_va_split.py tests/serving/va_split_jax/test_compile_warmup.py
git commit -m "feat: add JAX VA split compile warmup"
```

---



## Task 11: Profile 与脚本

**Files:**

- Modify: `scripts/profile_va_split.py`
- Create: `scripts/run_profile_va_split_jax.sh`
- Test: `tests/serving/va_split_jax/test_profile_script.py`

- [ ] **Step 1: 扩展 mode**

在 `scripts/profile_va_split.py` 中把 mode 扩展为：

```python
Mode = Literal["monolithic", "split-no-mps", "split-mps", "jax-monolithic", "jax-split-ipc"]
```

- [ ] **Step 2: 增加 JAX policy 创建分支**

在创建 policy 的函数中：

- `jax-monolithic` 使用现有 JAX `Policy`。
- `jax-split-ipc` 使用 `create_trained_jax_va_split_policy`。
- 默认 config/dir 仍可通过 CLI 覆盖。
- profile 默认只跑 Pi0.5。

- [ ] **Step 3: 确保 summary key 可比较**

JAX split 输出 timing key 与 PyTorch 版保持一致：

- `vlm_prefix_forward_ms`
- `vlm_effective_batch`
- `ae_step_ms`
- `ae_step_total_ms`
- `ae_effective_batch`
- `ae_effective_batch_mean`
- `prefix_queue_wait_ms`
- `prefix_transfer_ms`
- `prefix_pool_write_ms`
- `prefix_pool_compact_ms`
- `prefix_slab_map_ms`
- `prefix_pool_overhead_ms`
- `va_split_queue_wait_ms`
- `va_split_transfer_ms`

为便于复用现有 summary 代码，第一版可以把 `prefix_pool_write_ms` 同时写入兼容 key `prefix_lane_ingest_ms`，把 `prefix_pool_compact_ms` 同时写入 `prefix_lane_compact_ms`，把 `prefix_pool_overhead_ms` 同时写入 `prefix_lane_overhead_ms`；报告中优先使用 `prefix_pool_*` 名称。

- [ ] **Step 4: 写 JAX profile 脚本**

Create `scripts/run_profile_va_split_jax.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-jax-split-ipc}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/data1/miliang/models/RLinf-Pi05-LIBERO-SFT}"
LOG_ROOT="${LOG_ROOT:-/data1/miliang/VL-A-Disaggregation/logs}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/jax-${RUN_TS}}"
MPS_PIPE_DIR="${MPS_PIPE_DIR:-${RUN_LOG_DIR}/mps-pipe}"
MPS_LOG_DIR="${MPS_LOG_DIR:-${RUN_LOG_DIR}}"
AE_SM_PERCENT="${AE_SM_PERCENT:-20}"
VLM_SM_PERCENT="${VLM_SM_PERCENT:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-128}"
REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-16}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
TIMEOUT_S="${TIMEOUT_S:-60}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-8}"
MAX_VLM_BATCH_SIZE="${MAX_VLM_BATCH_SIZE:-8}"
MAX_VLM_WAIT_MS="${MAX_VLM_WAIT_MS:-2.0}"
JAX_COMPILE="${JAX_COMPILE:-1}"
JAX_COMPILE_WARMUP="${JAX_COMPILE_WARMUP:-1}"
JAX_COMPILE_WARMUP_MAX_BATCH_SIZE="${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE:-32}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
PYTHON_BIN="${PYTHON_BIN:-/data1/miliang/RLinf/openpi_libero/bin/python}"

MPS_STARTED=0

cleanup() {
  if [[ "${MPS_STARTED}" -eq 1 ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RUN_LOG_DIR}"

if [[ "${MODE}" == "jax-split-ipc" ]]; then
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_va_split.py"
  --policy.config "${POLICY_CONFIG}"
  --policy.dir "${POLICY_DIR}"
  --mode "${MODE}"
  --num-requests "${NUM_REQUESTS}"
  --request-rate-hz "${REQUEST_RATE_HZ}"
  --max-inflight "${MAX_INFLIGHT}"
  --seed "${SEED}"
  --num-steps "${NUM_STEPS}"
  --timeout-s "${TIMEOUT_S}"
  --max-ae-batch-size "${MAX_AE_BATCH_SIZE}"
  --max-vlm-batch-size "${MAX_VLM_BATCH_SIZE}"
  --max-vlm-wait-ms "${MAX_VLM_WAIT_MS}"
  --ae-sm-percent "${AE_SM_PERCENT}"
  --vlm-sm-percent "${VLM_SM_PERCENT}"
  --jax-compile-warmup-max-batch-size "${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
  --json-output "${JSON_OUTPUT}"
)

if [[ "${JAX_COMPILE}" == "0" || "${JAX_COMPILE}" == "false" ]]; then
  cmd+=(--no-jax-compile)
fi

if [[ "${JAX_COMPILE_WARMUP}" == "0" || "${JAX_COMPILE_WARMUP}" == "false" ]]; then
  cmd+=(--no-jax-compile-warmup)
fi

echo "Running JAX V-A profile: mode=${MODE} gpu=${GPU_ID} policy_dir=${POLICY_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  logs:   ${RUN_LOG_DIR}"
echo "  json:   ${JSON_OUTPUT}"
echo "  compile: enabled=${JAX_COMPILE} warmup=${JAX_COMPILE_WARMUP} warmup_max_batch=${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
echo "  mps:    pipe=${MPS_PIPE_DIR} ae_sm=${AE_SM_PERCENT} vlm_sm=${VLM_SM_PERCENT}"
"${cmd[@]}" "$@"
```

- [ ] **Step 5: 写脚本测试**

测试：

- `scripts/run_profile_va_split_jax.sh` 存在并可执行。
- 默认 `MODE=jax-split-ipc`。
- 默认 `POLICY_CONFIG=pi05_libero`。
- 默认 Python 是 `/data1/miliang/RLinf/openpi_libero/bin/python`。
- 默认 `JAX_COMPILE=1` 且 `JAX_COMPILE_WARMUP=1`。
- `jax-split-ipc` 默认启动隔离 MPS 服务。
- 默认 `AE_SM_PERCENT=20`、`VLM_SM_PERCENT=0`；`0` 表示不设置上限。
- 脚本导出 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。

- [ ] **Step 6: 运行 profile smoke**

Run:

```bash
GPU_ID=0 NUM_REQUESTS=4 REQUEST_RATE_HZ=2 MAX_INFLIGHT=2 TIMEOUT_S=120 scripts/run_profile_va_split_jax.sh
```

Expected:

- `logs/jax-*/profile.json` 写出。
- `summary.ok_count == 4`。
- timing 中存在 `vlm_prefix_forward_mean_ms` 和 `ae_step_mean_ms`。
- summary 中 `jax_compile_enabled == 1`，`jax_compile_warmup_enabled == 1`，`jax_warmup_batches > 0`。

- [ ] **Step 7: Commit**

```bash
git add scripts/profile_va_split.py scripts/run_profile_va_split_jax.sh tests/serving/va_split_jax/test_profile_script.py
git commit -m "feat: add JAX VA split profiling"
```

---



## Task 12: End-to-End 验证矩阵

**Files:**

- Modify: `docs/handoff.md`
- Output: `logs/tests/jax_va_split_validation.json`

- [ ] **Step 1: 运行单元测试集合**

Run:

```bash
PYTHONPATH=src:packages/openpi-client/src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest \
  tests/models/test_pi0_jax_split.py \
  tests/serving/va_split_jax \
  tests/policies/jax_va_split_policy_test.py \
  -q
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行 profile smoke**

Run:

```bash
GPU_ID=0 NUM_REQUESTS=8 REQUEST_RATE_HZ=4 MAX_INFLIGHT=4 TIMEOUT_S=120 scripts/run_profile_va_split_jax.sh
```

Expected: profile JSON 写入 `logs/jax-*/profile.json`。

- [ ] **Step 3: 写验证 JSON**

Create `logs/tests/jax_va_split_validation.json`:

```json
{
  "ipc_gate": "passed",
  "pi05_split_numeric": "passed",
  "pi0_split_smoke": "passed",
  "runtime_process": "passed",
  "profile_smoke": "passed",
  "jax_compile_default": "enabled",
  "jax_compile_warmup": "passed",
  "default_profile_model": "pi05"
}
```

- [ ] **Step 4: 更新 handoff**

在 `docs/handoff.md` 追加一个短节：

```markdown
## JAX V-A Split IPC

- Plan: `docs/plan_jax_va_split_ipc.md`
- Gate test: `tests/serving/va_split_jax/test_jax_device_ipc_gate.py`
- Gate output: `logs/tests/jax_device_ipc_gate.json`
- Runtime package: `src/openpi/serving/va_split_jax`
- Profile script: `scripts/run_profile_va_split_jax.sh`
- Default profile target: Pi0.5 / `pi05_libero`
- Default compile: enabled
- Ours compile target: VLM `build_prefix_feature` and AE `denoise_one_batch`
- Baseline compile target: monolithic `sample_actions`
```

- [ ] **Step 5: Commit**

```bash
git add docs/handoff.md logs/tests/jax_va_split_validation.json
git commit -m "docs: record JAX VA split validation"
```

---



## 自检清单

- [ ] Task 1 是硬门禁；失败时明确暂停，不继续实现 runtime。
- [ ] 没有把 host shared memory copy 写成 ours 的通过路径。
- [ ] JAX split API 覆盖 Pi0.5 数值一致性和 Pi0 smoke。
- [ ] 两进程 runtime 中 prefix/control queue 只传 metadata/view handle，不传 KV cache 大 tensor。
- [ ] Prefix/KV feature 在 GPU 上只保留一份，由 VLM-owned lane pool 持有。
- [ ] AE continuous batching 不依赖 VLM batch 边界，但第一版 batch selection 只选择 VLM pool dense prefix 中仍 active 的请求，避免 AE 侧 gather/copy feature。
- [ ] VLM release compaction 后用 `JaxSlotMoved` 更新 AE active table，避免 stale slot handle。
- [ ] JAX compile 默认开启，但可通过 `--no-jax-compile` 关闭。
- [ ] JAX compile 使用 `nnx_utils.module_jit`，不使用裸 `jax.jit(model.bound_method)` 或 `nnx.jit(model.bound_method)`。
- [ ] Ours split compile 分别覆盖 VLM prefix 和 AE single-step；baseline compile 覆盖完整 VLA `sample_actions`。
- [ ] compile warmup 默认开启，使用与 PyTorch 版同构的 batch plan，并按 `max_prefix_slots` / warmup max batch size clamp。
- [ ] JAX split IPC 采用启动握手时一次 `open_slab` / CUDA IPC map，denoise step 不重新 open IPC。
- [ ] MPS 脚本启动隔离 MPS 服务，AE/VLM SM 配额支持 `0` 表示不设上限。
- [ ] VLM/AE 子进程都设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- [ ] JAX checkpoint 路径可配置；当前 PyTorch checkpoint 路径只作为占位默认值。
- [ ] profile 默认 Pi0.5，且 timing key 与 PyTorch 版可比较。
- [ ] 所有命令默认使用 `/data1/miliang/RLinf/openpi_libero/bin/python`。
