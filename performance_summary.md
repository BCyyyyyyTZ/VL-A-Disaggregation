# V-A 性能汇总：Ours vs Baseline

数据来源目录：`tianze/logs/V-A/`。所有实验均为 **200 requests**，SLO = **200 ms**。

指标说明：

- **E2E 延迟**：`end_to_end_latency_*_ms`（越低越好）
- **吞吐**：`throughput_requests_per_second`（越高越好）
- **Goodput**：`slo_goodput_requests_per_second`（满足 SLO 的有效吞吐，越高越好）
- **提升百分比**：
  - 延迟：`(baseline - ours) / baseline × 100%`（正值表示 ours 更快）
  - 吞吐/Goodput：`(ours - baseline) / baseline × 100%`（正值表示 ours 更高）

实验配置对应关系：


| 组别                     | 路径                                    | 说明               |
| ---------------------- | ------------------------------------- | ---------------- |
| Baseline               | `r{8,16,32}_baseline/`                | 非 Torch baseline |
| Baseline-Torch         | `baseline-torch/r{8,16,32}_baseline/` | Torch baseline   |
| Ours (sm0-0)           | `sm0-0/r{8,16,32}_ours/`              | VA-split，sm0=0   |
| Ours (sm0-0-torch-new) | `sm0-0-torch-new/r{8,16,32}_ours/`    | Torch 版 ours     |
| Ours (sm0-20)          | `sm0-20/compare200_r{8,16,32}_ours_*` | VA-split，sm0=20  |


## 1. 绝对性能数据

### Rate = 8 req/s


| 实验                   | E2E mean (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Goodput (req/s) | SLO good |
| -------------------- | ------------- | ------------ | ------------ | ------------------ | --------------- | -------- |
| Baseline             | 262.01        | 268.77       | 367.60       | 7.05               | 1.71            | 48/200   |
| Baseline-Torch       | 246.16        | 249.53       | 344.06       | 7.05               | 2.06            | 58/200   |
| Ours sm0-0           | 212.66        | 197.67       | 285.40       | 7.06               | 3.80            | 107/200  |
| Ours sm0-0-torch-new | 113.77        | 103.00       | 159.56       | 7.08               | 6.96            | 196/200  |
| Ours sm0-20          | 221.61        | 209.21       | 300.28       | 7.06               | 2.98            | 84/200   |


### Rate = 16 req/s


| 实验                   | E2E mean (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Goodput (req/s) | SLO good |
| -------------------- | ------------- | ------------ | ------------ | ------------------ | --------------- | -------- |
| Baseline             | 357.00        | 350.93       | 500.45       | 13.91              | 0.64            | 9/200    |
| Baseline-Torch       | 337.51        | 337.40       | 468.63       | 13.90              | 0.78            | 11/200   |
| Ours sm0-0           | 293.17        | 281.76       | 443.98       | 13.98              | 1.63            | 23/200   |
| Ours sm0-0-torch-new | 156.34        | 138.87       | 276.52       | 14.08              | 11.65           | 164/200  |
| Ours sm0-20          | 298.37        | 288.49       | 429.34       | 13.97              | 1.21            | 17/200   |


### Rate = 32 req/s


| 实验                   | E2E mean (ms) | E2E p50 (ms) | E2E p95 (ms) | Throughput (req/s) | Goodput (req/s) | SLO good |
| -------------------- | ------------- | ------------ | ------------ | ------------------ | --------------- | -------- |
| Baseline             | 1354.66       | 1390.34      | 2092.47      | 21.99              | 0.14            | 1/200    |
| Baseline-Torch       | 1291.18       | 1320.71      | 2015.65      | 22.18              | 0.14            | 1/200    |
| Ours sm0-0           | 852.56        | 904.41       | 1134.00      | 25.48              | 0.00            | 0/200    |
| Ours sm0-0-torch-new | 312.13        | 284.68       | 525.73       | 27.46              | 6.25            | 44/200   |
| Ours sm0-20          | 903.68        | 951.18       | 1190.81      | 25.23              | 0.14            | 1/200    |


## 2. Ours vs Baseline（非 Torch）

对比对象：`r*_baseline` vs `sm0-0` / `sm0-20`。

### sm0-0 vs Baseline


| Rate | E2E mean 提升 | E2E p50 提升 | E2E p95 提升 | Throughput 提升 | Goodput 提升 |
| ---- | ----------- | ---------- | ---------- | ------------- | ---------- |
| r8   | +18.8%      | +26.5%     | +22.4%     | +0.1%         | +122.9%    |
| r16  | +17.9%      | +19.7%     | +11.3%     | +0.4%         | +155.6%    |
| r32  | +37.1%      | +35.0%     | +45.8%     | +15.9%        | -100.0%    |


### sm0-20 vs Baseline


| Rate | E2E mean 提升 | E2E p50 提升 | E2E p95 提升 | Throughput 提升 | Goodput 提升 |
| ---- | ----------- | ---------- | ---------- | ------------- | ---------- |
| r8   | +15.4%      | +22.2%     | +18.3%     | +0.1%         | +75.0%     |
| r16  | +16.4%      | +17.8%     | +14.2%     | +0.4%         | +88.9%     |
| r32  | +33.3%      | +31.6%     | +43.1%     | +14.8%        | +0.0%      |


## 3. Ours vs Baseline-Torch

对比对象：`baseline-torch` vs `sm0-0-torch-new`。


| Rate | E2E mean 提升 | E2E p50 提升 | E2E p95 提升 | Throughput 提升 | Goodput 提升 |
| ---- | ----------- | ---------- | ---------- | ------------- | ---------- |
| r8   | +53.8%      | +58.7%     | +53.6%     | +0.4%         | +237.9%    |
| r16  | +53.7%      | +58.8%     | +41.0%     | +1.3%         | +1390.9%   |
| r32  | +75.8%      | +78.4%     | +73.9%     | +23.8%        | +4300.0%   |


## 4. 关键结论

### sm0-0-torch-new vs Baseline-Torch（Torch 对照，提升最显著）

- **r8**：E2E mean +53.8%，Throughput +0.4%，Goodput +237.9%（246.16→113.77 ms；7.05→7.08 req/s；2.06→6.96 req/s）
- **r16**：E2E mean +53.7%，Throughput +1.3%，Goodput +1390.9%（337.51→156.34 ms；13.90→14.08 req/s；0.78→11.65 req/s）
- **r32**：E2E mean +75.8%，Throughput +23.8%，Goodput +4300.0%（1291.18→312.13 ms；22.18→27.46 req/s；0.14→6.25 req/s）

### sm0-0 vs Baseline

- **r8**：E2E mean +18.8%，Throughput +0.1%，Goodput +122.9%（262.01→212.66 ms；7.05→7.06 req/s；1.71→3.80 req/s）
- **r16**：E2E mean +17.9%，Throughput +0.4%，Goodput +155.6%（357.00→293.17 ms；13.91→13.98 req/s；0.64→1.63 req/s）
- **r32**：E2E mean +37.1%，Throughput +15.9%，Goodput -100.0%（1354.66→852.56 ms；21.99→25.48 req/s；0.14→0.00 req/s）

### sm0-20 vs Baseline

- **r8**：E2E mean +15.4%，Throughput +0.1%，Goodput +75.0%（262.01→221.61 ms；7.05→7.06 req/s；1.71→2.98 req/s）
- **r16**：E2E mean +16.4%，Throughput +0.4%，Goodput +88.9%（357.00→298.37 ms；13.91→13.97 req/s；0.64→1.21 req/s）
- **r32**：E2E mean +33.3%，Throughput +14.8%，Goodput +0.0%（1354.66→903.68 ms；21.99→25.23 req/s；0.14→0.14 req/s）

观察：

1. **Torch 对照（sm0-0-torch-new）** 在三个速率上均显著降低 E2E 延迟，并大幅提升 Goodput；高负载 r32 下吞吐提升也最明显。
2. **sm0-0 / sm0-20** 相对非 Torch Baseline：低负载时吞吐接近目标速率，提升主要体现在 **E2E 延迟下降** 与 **Goodput 提升**；高负载 r32 下吞吐约有 **+15%** 量级提升，E2E 延迟约下降 **33%~37%**。
3. 多数配置在 r32 下 Goodput 仍很低（SLO=200ms 较紧），仅 `sm0-0-torch-new` 在 r32 仍保持可用 Goodput。
4. `sm0-0` 与 `sm0-20` 性能接近，`sm0-0` 略优。

---

## 附录：原始数值速查表


| 实验              | Rate | E2E mean | E2E p50 | E2E p95 | Throughput | Goodput | SLO good |
| --------------- | ---- | -------- | ------- | ------- | ---------- | ------- | -------- |
| Baseline        | 8    | 262.01   | 268.77  | 367.60  | 7.05       | 1.71    | 48       |
| Baseline        | 16   | 357.00   | 350.93  | 500.45  | 13.91      | 0.64    | 9        |
| Baseline        | 32   | 1354.66  | 1390.34 | 2092.47 | 21.99      | 0.14    | 1        |
| Baseline-Torch  | 8    | 246.16   | 249.53  | 344.06  | 7.05       | 2.06    | 58       |
| Baseline-Torch  | 16   | 337.51   | 337.40  | 468.63  | 13.90      | 0.78    | 11       |
| Baseline-Torch  | 32   | 1291.18  | 1320.71 | 2015.65 | 22.18      | 0.14    | 1        |
| sm0-0           | 8    | 212.66   | 197.67  | 285.40  | 7.06       | 3.80    | 107      |
| sm0-0           | 16   | 293.17   | 281.76  | 443.98  | 13.98      | 1.63    | 23       |
| sm0-0           | 32   | 852.56   | 904.41  | 1134.00 | 25.48      | 0.00    | 0        |
| sm0-0-torch-new | 8    | 113.77   | 103.00  | 159.56  | 7.08       | 6.96    | 196      |
| sm0-0-torch-new | 16   | 156.34   | 138.87  | 276.52  | 14.08      | 11.65   | 164      |
| sm0-0-torch-new | 32   | 312.13   | 284.68  | 525.73  | 27.46      | 6.25    | 44       |
| sm0-20          | 8    | 221.61   | 209.21  | 300.28  | 7.06       | 2.98    | 84       |
| sm0-20          | 16   | 298.37   | 288.49  | 429.34  | 13.97      | 1.21    | 17       |
| sm0-20          | 32   | 903.68   | 951.18  | 1190.81 | 25.23      | 0.14    | 1        |


---

## 5. Ours 数据传输链路分析

本节只讨论 **跨进程/设备数据搬运本身**，**不包括**：队列等待（`*_queue_wait`）、VLM forward、AE denoise step。  
延迟取自 `sm0-0-torch-new` 在 **r32** 下的 p50：该速率下各 hop 的 `*_queue_wait` p50 > 0，说明多数请求是「先入队再 get」，`*_transfer` 更接近纯传输；低负载（r8/r16）下 consumer 常提前堵在 `get()`，会把等生产者的时间算进 transfer，**不可直接当纯传输**。

### 5.1 链路总览

```text
Client/Parent                     VLM process                      AE process
    |                                 |                                |
    | ① RequestEnvelope               |                                |
    |  (~2 MB, CPU pickle)            |                                |
    |-------------------------------->|                                |
    |                                 |  VLM prefix forward (计算,不计)   |
    |                                 |                                |
    |                                 | ② PrefixReady                  |
    |                                 |  (~18 MB, CUDA IPC)            |
    |                                 |------------------------------->|
    |                                 |                                | ③ lane ingest
    |                                 |                                |  (~18 MB, 同卡 memcpy)
    |                                 |                                |  AE step (计算,不计)
    | ④ ActionResult                  |                                |
    |  (~1 KB, GPU→CPU + pickle)      |                                |
    |<----------------------------------------------------------------|
    |                                 | ⑤ ReleaseFeature (极小,无计时)   |
    |                                 |<-------------------------------|
```

实现上四条队列均为 `torch.multiprocessing`（`spawn`）的 `Queue`：  
观测/actions 走 CPU 侧序列化；`PrefixFeature` 中的 CUDA tensor 依赖 PyTorch 队列的 CUDA IPC reduction（VLM 侧 `live_features` 保活 storage，直到 AE→VLM 的 `ReleaseFeature`）。

### 5.2 各跳：方法、数据量、纯传输延迟

配置背景：`pi05_libero`，单请求 B=1，`action_horizon=10`，`action_dim=32`，`max_token_len=200`，PaliGemma/gemma_2b（18 层，`num_kv_heads=1`，`head_dim=256`，bf16 KV）。  
prefix 序列长度约 `S ≈ 3×256 + 200 = 968`（3 路 SigLIP patch + language tokens）。


| #   | 跳                    | 方向           | 传输方法                                                              | 主要 payload                                                             | 估计数据量                                | 纯传输延迟 (p50) | 占 E2E (r32 mean 312ms) | 占纯传输合计 |
| --- | -------------------- | ------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------ | ----------- | ---------------------- | ------ |
| ①   | `vlm_request`        | Client → VLM | `torch.mp.Queue`，**CPU tensor pickle**（入队前已是 CPU）                 | `RequestEnvelope`：3×图像 `[1,3,224,224]` float32 + state/tokens          | **~1.8–2.0 MB**                      | **~1.5 ms** | 0.5%                   | 37%    |
| ②   | `prefix`             | VLM → AE     | `torch.mp.Queue`，CUDA tensor **隐式 CUDA IPC**（无显式 `share_memory_`） | `PrefixReady.feature`：`DynamicCache` KV + `prefix_pad_masks` + `state` | **~17.8 MB**（KV：`18×2×1×968×256×2B`） | **~1.1 ms** | 0.3%                   | 27%    |
| ③   | `prefix_lane_ingest` | AE 进程内       | 同卡 `Tensor.copy_(non_blocking)` + `cuda.synchronize`              | 将单行 PrefixFeature 写入预分配 lane pool                                      | **~17.8 MB**（device→device）          | **~0.7 ms** | 0.2%                   | 17%    |
| ④   | `ae_result`          | AE → Client  | AE 侧 `detach().cpu()` 后 `torch.mp.Queue` **CPU pickle**           | `ActionResult.actions` `[1,10,32]` float32                             | **~1.3 KB**                          | **~0.7 ms** | 0.2%                   | 17%    |
| ⑤   | `release`            | AE → VLM     | `torch.mp.Queue`                                                  | `ReleaseFeature(request_id, slot_id)`                                  | 可忽略                                  | 无独立指标       | —                      | —      |
|     | **合计（①–④）**          |              |                                                                   |                                                                        |                                      | **~4.0 ms** | **~1.3%**              | 100%   |


对应 profile 字段（r32 / `sm0-0-torch-new`）：


| 字段                        | p50 (ms) | mean (ms) | p95 (ms) | 该速率 `queue_wait` p50   |
| ------------------------- | -------- | --------- | -------- | ---------------------- |
| `vlm_request_transfer_ms` | 1.48     | 2.18      | 4.07     | 55.0（已入队，transfer 较干净） |
| `prefix_transfer_ms`      | 1.06     | 21.36     | 123.65   | 6.5                    |
| `ae_result_transfer_ms`   | 0.68     | 36.71     | 192.73   | 1.0                    |
| `prefix_lane_ingest_ms`   | —        | 0.67      | —        | N/A（非 queue）           |


> **读数注意**：mean / p95 仍可能含偶发调度尖刺或少数「提前 get」样本，故纯传输以 **p50** 为准。低负载下 `prefix`/`ae_result` 的 mean 可到几十～上百 ms，主要是口径把空等算进了 transfer，**不是** ~18MB IPC 的真实耗时。

### 5.3 相对计算与结论

同一次 r32 运行的计算侧（仅作对比，不计入传输）：

- VLM prefix forward mean ≈ **144 ms**
- AE step mean ≈ **12 ms**
- 纯传输合计 ≈ **4 ms** → 约为 VLM 计算的 **~3%**，E2E 的 **~1%**

结论：

1. **数据量最大的是 VLM→AE 的 PrefixFeature（~18MB KV）**，以及 AE 侧同等大小的 lane ingest；但在排除等待后，这两段典型各自只有 **~1ms 量级**。
2. **Client→VLM 观测（~2MB）** 纯传输约 **1.5ms**，在纯传输合计里占比最高，但仍远小于计算。
3. **AE→Client actions（~1KB）** 数据量极小；p50 ~0.7ms 更多是 queue get/调度开销，不是带宽瓶颈。
4. 当前 ours 的 E2E 瓶颈不在「字节拷贝」，而在 **排队 + VLM 计算**；传输优化（共享显存、减少 IPC 次数）对 E2E 的上限收益大约只有数毫秒，除非同时改善长尾（p95 仍可达百毫秒级）。

### 5.4 其他 ours 配置交叉验证（纯传输 p50）

在 `queue_wait` p50 > 0 的速率下，数量级一致：


| 配置              | Rate | vlm_req p50 | prefix p50 | ae_result p50 | lane_ingest mean |
| --------------- | ---- | ----------- | ---------- | ------------- | ---------------- |
| sm0-0-torch-new | 32   | 1.48        | 1.06       | 0.68          | 0.67             |
| sm0-0           | 16   | 1.36        | 2.33       | 76.09*        | 0.65             |
| sm0-0           | 32   | 1.33        | 0.90       | 0.62          | 0.65             |


 `sm0-0` r16 的 `ae_result` 仍明显偏高且 `ae_result_queue_wait` p50=0，属于「提前阻塞在 result get」的口径污染，不纳入纯传输估计。

---

## 附录：Ours V–A Disaggregation 整体流程图（文字稿）

布局意图（可直接照着做 PPT）：

- **最左侧**：细竖条 `Parent`，只负责发请求 / 收回结果
- **中间**：大框 `VLM Process`；框顶标 **FCFS batching**，框内用容器名表示数据流
- **右侧**：大框 `AE Process`；框顶标 **continuous batching**，框内用容器名表示数据流
- 箭头旁只标：**内容 · 延迟（r32 p50）**；不标 method、不标数据量

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    Ours: V–A Disaggregation Overview                                │
│                 (transfer latency @ r32 / sm0-0-torch-new, p50)                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

 ┌──┐
 │P │      ┌──────────────────────────────┐      ┌──────────────────────────────┐
 │a │      │        VLM Process           │      │         AE Process           │
 │r │      │     ★ FCFS batching          │      │  ★ continuous batching       │
 │e │      │     (prefix / KV build)      │      │  (mixed-timestep denoise)    │
 │n │      │                              │      │                              │
 │t │  ①   │  ┌────────────────────────┐  │  ②   │  ┌────────────────────────┐  │
 │  │─────►│  │ RequestEnvelope        │  │─────►│  │ PrefixReady            │  │
 │  │~1.5ms│  │ obs + state + tokens   │  │~1.1ms│  │ KV + masks + state     │  │
 │/ │      │  └───────────┬────────────┘  │      │  └───────────┬────────────┘  │
 │  │      │              ▼               │      │              ▼               │
 │C │      │  ┌────────────────────────┐  │      │  ┌────────────────────────┐  │
 │l │      │  │ FCFS collector         │  │      │  │ ③ PrefixFeature        │  │
 │i │      │  │ form VLM batch         │  │      │  │ → lane pool (~0.7 ms)  │  │
 │e │      │  └───────────┬────────────┘  │      │  └───────────┬────────────┘  │
 │n │      │              ▼               │      │              ▼               │
 │t │      │  ┌────────────────────────┐  │      │  ┌────────────────────────┐  │
 │  │      │  │ prefix forward         │  │      │  │ continuous batch      │  │
 │  │      │  │ → PrefixFeature        │  │      │  │ denoise (mixed-t)     │  │
 │  │      │  │   (KV + masks)         │  │      │  │ → ActionResult        │  │
 │  │      │  └───────────┬────────────┘  │      │  └───────────┬────────────┘  │
 │  │      │              │ ②             │      │              │ ④             │
 │  │  ④   │              └───────────────┼──────┼──┐           │               │
 │  │◄─────┼──────────────────────────────┼──────┼──┼───────────┘               │
 │  │~0.7ms│                              │      │  │  ActionResult             │
 │  │      │                              │      │  │  actions[B,H,D] ~0.7 ms   │
 │  │      │  ┌────────────────────────┐  │  ⑤   │  ▼                           │
 │  │      │  │ ReleaseFeature         │◄─┼──────┼──┐ ┌──────────────────────┐  │
 │  │      │  │ free live_features     │  │ ~0ms │  └─│ ReleaseFeature       │  │
 │  │      │  └────────────────────────┘  │      │    │ req_id / slot_id     │  │
 │  │      │                              │      │    └──────────────────────┘  │
 └──┘      └──────────────────────────────┘      └──────────────────────────────┘

  Parent 竖条
  ──────────
  • 发 ① RequestEnvelope
  • 收 ④ ActionResult

  调度如何体现在图上
  ──────────────────
  • VLM 框顶 / collector 块：FCFS batching（到齐一批再做一次 prefix forward）
  • AE 框顶 / denoise 块：continuous / mixed-timestep batching
    （不同请求可处于不同 denoising step，同一步一起 launch）

  主路径
  ──────
  VLM: RequestEnvelope → FCFS batch → PrefixFeature → PrefixReady → ReleaseFeature
  AE : PrefixReady → PrefixFeature(lane) → continuous denoise → ActionResult → ReleaseFeature

  Transfer total (①–④) ≈ 4.0 ms  ≈ 1.3% of E2E (312 ms @ r32)
```

### PPT 画法提示

1. **Parent**：最左侧窄竖条；只留 ① 出、④ 入。
2. **VLM / AE**：两个等高大框；框标题下各加一行调度标签（FCFS / continuous）。
3. 框内步骤只用 **容器名**（RequestEnvelope / PrefixFeature / PrefixReady / ActionResult / ReleaseFeature）。
4. ②⑤ 画在两框之间；③ 只在 AE 框内；计算块用浅虚线，与传输箭头区分。
