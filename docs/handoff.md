# V-A Disaggregation 调度逻辑交接

本文档面向后续需要把当前 OpenPI 上的 V-A disaggregation 调度方案迁移到其他 VLA 模型的 agent。重点说明当前实现的调度思想、代码边界、模型侧最小接口，以及迁移时容易踩坑的位置。

## 1. 当前目标

传统 OpenPI PyTorch 推理路径里，`sample_actions()` 在同一个进程和同一条调用链中顺序完成：

1. VLM prefix forward：处理视觉、语言、状态输入，生成可复用的 prefix KV cache、mask、state。
2. AE denoising：多步连续去噪，最终得到动作序列。

当前实现将这两段拆开成两个进程：

- VLM worker 只负责构建 prefix feature，并尽快把结果交给 AE。
- AE worker 负责维护多个正在去噪的请求，在每个 denoise step 做 continuous batching。

核心收益来自两点：

- VLM 与 AE 不再共享同一个 batch 边界。VLM 使用短等待窗口做 FCFS batching，控制 prefix latency。
- AE 不再被单个 VLM batch 限制。它把来自不同时间、不同 VLM batch 的 prefix 放进 active lanes，每一步从活跃请求中取最多 `max_ae_batch_size` 条一起跑（一般不设`max_ae_batch_size`上限，可以设置为999）。

## 2. 主要代码位置

- `src/openpi/models_pytorch/pi0_pytorch.py`
  - OpenPI 模型侧 split API。
  - `build_prefix_feature()` 对应 VLM prefix。
  - `init_denoise_state()` 初始化 AE denoise loop。
  - `denoise_one_batch()` 执行一个 batched denoise step。

- `src/openpi/models_pytorch/pi0_split_types.py`
  - split runtime 与模型之间的最小数据结构。
  - `PrefixFeature` 包含 `past_key_values`、`prefix_pad_masks`、`state`。
  - `DenoiseState` 包含 `x_t`、`step_idx`、`num_steps`、`dt`。

- `src/openpi/serving/va_split/types.py`
  - 进程间消息协议。
  - 关键消息包括 `RequestEnvelope`、`BatchRequestEnvelope`、`PrefixReady`、`ActionResult`、`ReleaseFeature`、`WorkerError`、`Shutdown`。

- `src/openpi/serving/va_split/vlm_process.py`
  - VLM worker 和进程循环。
  - 实现 FCFS batching、兼容性检查、prefix feature 保活、release 回收。

- `src/openpi/serving/va_split/ae_process.py`
  - AE worker 和进程循环。
  - 实现 prefix admission、lane pool 写入、step-level continuous batching、完成后释放。

- `src/openpi/serving/va_split/prefix_cache_pool.py`
  - AE 侧 prefix lane pool。
  - 将单请求 prefix cache 规整到预分配 lanes，并提供连续 batch view。

- `src/openpi/serving/va_split/runtime.py`
  - runtime 封装。
  - `ProcessVASplitRuntime` 启动 VLM/AE 两个子进程，用队列连接。
  - `LocalVASplitRuntime` 用于本进程测试相同逻辑。

- `src/openpi/policies/va_split_policy.py`
  - OpenPI policy wrapper。
  - 负责输入输出 transforms、checkpoint 加载、runtime 创建。

- `scripts/serve_policy.py`
  - 服务入口。
  - `--va-split` 开启该 runtime。

- `scripts/run_va_split_mps.sh`
  - MPS 启动和 profiling 脚本。
  - 管理 GPU UUID、MPS pipe/log 目录、SM 配额、profile 参数。

## 3. 整体数据流

```text
Client / websocket server
        |
        v
VASplitPolicy.infer() 或 infer_batch()
        |
        v
ProcessVASplitRuntime
        |
        | RequestEnvelope / BatchRequestEnvelope
        v
VLMProcess
        |
        | PrefixReady
        v
AEProcess
        |
        | ActionResult
        v
runtime result collector
        |
        v
VASplitPolicy output transforms
```

AE 完成某个 request 后，会额外发送：

```text
AEProcess -> ReleaseFeature -> VLMProcess
```

这个 release 很重要。跨进程传 CUDA tensor 时，PyTorch multiprocessing 传的是 CUDA IPC 句柄和元数据，VLM 侧仍必须保活底层 storage。VLM 在 `live_features` 和 `live_batches` 中持有引用，直到收到 `ReleaseFeature` 后再删除。

## 4. VLM 调度策略

VLM 侧策略在 `VLMProcess.run()` + `_collect_fcfs_batch()` 中，核心是 **FCFS + 短等待窗口**，而不是「凑满才开干」。

组 batch 的实际顺序：

1. 主循环先非阻塞 `_prefetch_request_backlog()`，把 request queue 里**已经在等的**请求捞进 `_backlog`；再取一条作为本轮 batch head（可能来自 backlog，也可能刚从 queue 取到）。
2. 进入 `_collect_fcfs_batch(head)` 后，再次非阻塞 prefetch，尽量把队列里剩余已等待请求收进 `_backlog`。
3. **此时才**启动 `max_vlm_wait_ms` 计时（代码默认 2ms；脚本里常见设为 1ms）。
4. 在 batch 未满时循环取下一条候选：
   - **优先从 `_backlog` 取**（已等待 / 被退回的请求），不阻塞；
   - 只有 backlog 空了，才对 request queue 做带 timeout 的阻塞 `get`，timeout 为窗口剩余时间。
5. 兼容则并入当前 batch；不兼容则 `appendleft` 回 `_backlog`，留给下一轮（严格 FCFS，不跳过队头去凑兼容请求）。
6. 凑满 `min(max_vlm_batch_size, 可用 live feature slots)`，或窗口到期 / 队列空，就立刻做一次 prefix forward。

因此：

- **已在排队的请求会优先进入当前 batch**，不会被「再等一会儿看有没有更新的」插队逻辑打乱。
- **`max_vlm_wait_ms` 只约束「还要不要继续阻塞等新请求」**；prefetch / 消费 backlog 发生在窗口外或不等待，**不把已等待时间再算进 wait 窗口**。
- 能满则满；凑不满也不死等。`max_vlm_wait_ms = 0` 时完全非阻塞，有多少做多少。

兼容性由 `_request_compatibility_key()` 决定：

- observation 的树结构、tensor batchless shape、dtype 必须一致。
- sample kwargs 的 key 必须一致。
- `noise` 可以按 batch 维拼接；其他 tensor kwargs 要求值相同。
- 非 tensor kwargs 要求值相同。

VLM 侧 batch 不是为了让 AE 吃满，而是为了在短窗口内合并已到达请求、减少 prefix 重复开销。VLM 更偏 compute-bound，窗口过大会增加队头等待和 prefix forward 延迟。

VLM 完成 batch prefix 后，会把 batched `PrefixFeature` 按 row 切成单请求 view：

- `_prefix_feature_row_view()` 使用 `narrow(0, row, 1)` 保留 batch 维。
- `DynamicCache` 会逐层取 key/value 的 batch row。
- 每个 `PrefixReady` 对应一个 request id，但底层 storage 可来自同一个 batched prefix。

## 5. AE 调度策略

AE 侧策略在 `AEWorker.step_once()` 中：

- `add_prefix()` 将一个 `PrefixReady` 接收到本地 lane。
- 每个 request 对应一个 `AERequestState`，保存 `x_t`、`step_idx`、`dt`、`num_steps` 和 lane id。
- 每次 step 选择 dense lane table 前 `min(active_count, max_ae_batch_size)` 条。
- 将这些 lane 的 prefix view 和 denoise state 拼成一个 batch，调用 `model.denoise_one_batch()`。
- 每个 request 独立推进 `step_idx += 1`。
- 达到 `num_steps` 的 request 立即返回 `ActionResult`，并触发 `ReleaseFeature`。

这就是 continuous batching：AE batch 的单位是 denoise step，不是完整请求。不同请求可以来自不同 VLM batch，也可以处于不同进度。当前实现默认把活跃 lanes 维护成 dense prefix table，因此 step 时可以直接取 `[0:batch_size]` 的 view。

注意当前 `DenoiseState.num_steps` 在 `step_once()` 中使用 `batch[0].num_steps` 传入模型，但 timestep 实际由 per-row `step_idx` 和 `dt` 计算。迁移到其他模型时，如果一个 AE batch 混合不同 `num_steps`，模型侧应依赖 per-row `dt` 或扩展 `DenoiseState`，不要假设 `num_steps` 全 batch 相同。

## 6. Prefix lane pool

`PrefixCacheLanePool` 的目的是把来自 VLM 的单 row prefix 规整到 AE 进程内的一组预分配 lanes：

- 第一次 `put_lane()` 时根据真实 feature shape lazy 初始化。
- 对 HuggingFace `DynamicCache` 走 `_DynamicCachePool`，为每层 key/value 分配 `(max_lanes, ...)` tensor。
- 对普通 tensor tree 走 `_TensorTreePool`。
- 对非 tensor payload 只有 fallback，真实 Pi0/Pi0.5 不应依赖 fallback。
- `view_prefix_batch(batch_size)` 返回前 `batch_size` 个 lanes 的 narrow view。
- request 完成后 `_remove_lane()` 会用最后一个 active lane 填补空洞，保持 lane table dense。

迁移时，prefix payload 最好满足两个条件：

- 所有 tensor 的第 0 维是 batch 维。
- 单请求 feature 的 batch 维必须是 1。

注：目前在openpi上的实现是会在 VLM 和 AE 侧的 lane pool 保存两份同样的 feature，这里希望继续优化（针对openpi），希望最终只有一份 feature 存在，然后 AE 侧借助 IPC 等手段直接复用 VLM 生成的 prefix feature 导出视图进行denoise

所以对于把当前调度方法拓展到其他模型，在VLM和AE之间共享信息时，要保证一个基本原则：**尽量做到原地读写，减少跨进程通信开销；尽量减少数据额外复制，减少显存峰值**

## 7. Runtime 和进程模型

`ProcessVASplitRuntime` 启动两个子进程：

- VLM process：加载一份模型，执行 prefix forward。
- AE process：加载一份模型，执行 denoise steps。

队列：

- `_request_queue`：runtime 到 VLM。
- `_prefix_queue`：VLM 到 AE。
- `_result_queue`：AE 到 runtime collector thread。
- `_release_queue`：AE 到 VLM。

`infer()` 为单请求生成 UUID。`infer_batch()` 为 batch 生成一个 batch id，并为每个 row 生成 `batch_id:row` request id，最后按原 row 顺序拼回 actions。

`max_prefix_slots` 默认是 `max_vlm_batch_size * 3`。它同时约束：

- VLM 侧最多保活多少个 prefix feature。
- AE 侧最多接收多少个 active prefix lane。

如果 AE lanes 满了，AE 不再 drain prefix queue。VLM 如果 live feature slots 满了，会把 request 放回 backlog 并短 sleep，等 release 后继续。

## 8. MPS 和资源隔离

当前部署推荐通过 NVIDIA MPS 让 VLM/AE 两个 CUDA client 共享同一张 GPU：

- `scripts/run_va_split_mps.sh` 负责启动 `nvidia-cuda-mps-control -d`。
- 用 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，避免 MPS 下 GPU index 重映射造成误判。
- `ae_sm_percent` 默认 0，含义是清除 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，不限制 AE。
- `vlm_sm_percent` 默认 0，含义是清除 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，不限制 VLM。

`src/openpi/serving/va_split/launcher.py` 也提供 `build_mps_process_envs()`，但当前 policy 创建路径主要通过 `ProcessVASplitRuntime(..., vlm_env_updates=..., ae_env_updates=...)` 在子进程内设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。

## 9. 迁移到其他 VLA 的最小模型接口

目标 VLA 模型需要提供与 OpenPI 当前 split runtime 兼容的三个方法：

```python
@torch.no_grad()
def build_prefix_feature(self, device: str, observation) -> PrefixFeature:
    ...

@torch.no_grad()
def init_denoise_state(
    self,
    device: str,
    batch_size: int,
    noise: torch.Tensor | None,
    num_steps: int,
) -> DenoiseState:
    ...

@torch.no_grad()
def denoise_one_batch(
    self,
    prefix_batch: PrefixFeature,
    denoise_batch: DenoiseState,
) -> torch.Tensor:
    ...
```

语义要求：

- `build_prefix_feature()` 必须只做可复用 prefix 部分，不推进 action denoise。
- `PrefixFeature.past_key_values` 必须能被按 batch row 切分，也能被 AE lane pool 合并。
- `PrefixFeature.prefix_pad_masks` shape 为 `(B, prefix_len)`。
- `PrefixFeature.state` 如果 AE suffix 需要状态，就放在这里；否则可以是 `None`。
- `init_denoise_state()` 返回初始 `x_t` 和每步更新所需的 timestep 状态。
- `denoise_one_batch()` 只跑一小步，并返回与 `x_t` 同 shape 的速度或增量，当前 AE 逻辑会执行 `x_t = x_t + dt * v_t`。
- 如果目标模型不是 flow matching / Euler update，需要同步修改 `AEWorker.step_once()` 的状态推进方式。

## 10. 迁移步骤建议

1. 找到目标 VLA 原始推理函数，把它拆成 prefix 阶段和 action 阶段。
2. 明确 prefix 阶段输出中哪些内容可跨 denoise steps 复用。
3. 将这些内容封装成 `PrefixFeature`，确保 batch 维在第 0 维。
4. 将 action loop 的单步逻辑封装成 `denoise_one_batch()`。
5. 将 action loop 的初始状态封装成 `init_denoise_state()`。
6. 如果 cache 类型不是 `DynamicCache` 或纯 tensor tree，扩展 `prefix_cache_pool.py`。
7. 写 fake-model 单元测试，先不依赖真实 GPU 和真实 checkpoint。
8. 写真实模型一致性测试：固定 noise，对比 monolithic 输出和 split 输出。
9. 再 profile `max_vlm_batch_size`、`max_vlm_wait_ms`、`max_ae_batch_size`、`max_prefix_slots`、MPS SM 配额。

## 11. Profile 与启动

迁移到其他模型时，不必深挖 OpenPI 测试清单；先复现同一套 **workload + 指标 + 启动方式**，再对比 baseline（monolithic）与 ours（VA-split + MPS）。

实现参考openpi中的：`scripts/profile_va_split.py`、`scripts/run_va_split_mps.sh`。

### 11.1 Profile 主流程

1. **选模式**
   - baseline：`PROFILE_MODE=monolithic`（同进程串行 VLM→AE，不开 VA-split）
   - ours：`PROFILE_MODE=split-mps`（双进程 + 启动 NVIDIA MPS；可选 AE/VLM SM 配额）

2. **构造合成请求（泊松到达）**  
   `make_poisson_arrival_times()`：
   - 给定 `NUM_REQUESTS`、`REQUEST_RATE_HZ`、`SEED`
   - 间隔 ~ `Exponential(1 / rate)`，再 `cumsum` 得到每个请求的 `scheduled_at_s`
   - 即目标到达过程为速率 `REQUEST_RATE_HZ` 的泊松过程  
   `make_synthetic_libero_requests()` 再为每个到达时刻生成假观测（图像 / state / prompt）与可选固定 noise。

3. **按时刻提交**  
   异步 runner 睡到 `scheduled_at_s`，受 `MAX_INFLIGHT` 限制并发：
   - ours：通常单请求 `infer`，由服务端 VLM FCFS / AE continuous 自行组批
   - baseline：可用客户端 FCFS 组批（`BATCH_SIZE` / `max_vlm_wait_ms`）再 `infer_batch`

4. **Warmup（正式计时前，不计进 profile 样本）**  
   入口：`_warmup_before_timed_workload()`。目的是把 CUDA context、权重上设备、首次 kernel 启动、以及（若开启）`torch.compile` 的编译/cudagraph 捕获从正式统计里剔掉。  
   Warmup 用的是**同一批已构造好的合成请求**（通常取 `requests[0]` 或按需拼成固定 batch），走与正式跑相同的 `infer` / `infer_batch` 路径，但 **不写入 traces / summary**。

   **分支逻辑（互斥，按优先级）：**

   | 条件 | 行为 |
   |---|---|
   | `PYTORCH_COMPILE_MODE` 已设置 | 先走 **compile shape warmup**；Torch split 会枚举 `B=1..max_vlm_batch*3`，完成后再进入 timed profile |
   | `mode=monolithic` 且 `BATCH_SIZE>1` | 按 FCFS 规则从合成请求拼出 batched requests，对**第一个 batch** 调用 `infer_batch`，重复 `WARMUP_REQUESTS` 次 |
   | 其余（含 `split-mps` 默认、或 monolithic `BATCH_SIZE=1`） | 对 `requests[0]` 单请求 `infer`，重复 `WARMUP_REQUESTS` 次 |

   **`WARMUP_REQUESTS`（默认 2）**  
   - 控制「非 compile」路径下预热调用次数。  
   - 设为 `0` 可关掉普通 warmup（但仍可能因 compile 模式走 shape plan）。  
   - 次数太少：正式样本前几条仍可能偏慢；太多：拉长启动时间，一般 2～4 足够。

   **Compile shape warmup（仅当 compile 实际生效）**  
   `torch.compile` 会按输入 shape（尤其 batch 维）产生专用图；只暖少量 batch 时，正式 profile 一旦遇到未覆盖的 batch，仍会在计时段内首次 compile。当前逻辑会枚举所有 batch shape：

   ```
   (batch_size, repeats): (1,1), (2,1), ..., (cap,1)
   ```

   - 每个 batch 用合成请求循环拼出固定大小（不够则取模复用），再 `infer_batch` 一次。  
   - 为避免 PyTorch/Dynamo 默认 `recompile_limit/cache_size_limit=8` 在第 9 个 shape 后退回 eager，`va_split_policy` 会在子进程 `torch.compile` 前把 limit 抬高到 `PYTORCH_COMPILE_RECOMPILE_LIMIT`（默认 `128`）。  
   - 之后 split 模式会再跑 E2E pipeline warmup，用 `WARMUP_CONCURRENT_INFLIGHT`（MPS 脚本默认 `MAX_VLM_BATCH_SIZE`）把 FCFS 单请求热路径也打热，避免首个 rate 吃到队列/IPC handoff 或 scheduler 冷启动。  
   - `cap` 来自 `profile_warmup_max_batch_size()`：
     - **monolithic**：`min(32, BATCH_SIZE)`
     - **VA-split（split-mps）**：`max_vlm_batch_size*3`，与 `ProcessVASplitRuntime` 默认 prefix/live-feature 槽位容量一致。  
   - 因此默认 VLM batch=8 时，split compile warmup cap=24，会覆盖 `B=1..24`；正式 profile 的不同实际 batch size 不再现场触发 compile。

   **和正式 workload 的关系**  
   - Warmup **不改变**泊松到达时刻，也不占用 `NUM_REQUESTS` 计数。  
   - 正式跑开始前应已完成：模型加载、（可选）compile、以及上述 warmup。  
   - 对比实验时：baseline / ours 应使用**相同**的 `WARMUP_REQUESTS` 与 `PYTORCH_COMPILE_MODE`（开或都关），否则首包延迟不可比。

5. **跑完汇总**  
   写出 `profile.json`（含 `summary` + 每请求 trace）和 `profile.log`。

### 11.2 统计量：两边都要 vs 仅 ours

下列名称与 `profile.json` → `summary` / `policy_timing` 对齐。迁移时至少保证 **表中「双方」字段** 可对比；ours 特有字段用于解释解耦收益。

#### A. Baseline 与 Ours 都要统计

| 类别 | 字段 / 含义 |
|---|---|
| E2E 延迟 | `end_to_end_latency_{mean,p50,p95}_ms`：`completed - scheduled`（含排队） |
| 吞吐 | `throughput_requests_per_second` |
| Goodput | `slo_goodput_requests_per_second`，`slo_good_requests`（默认 `slo_ms=200`：E2E≤SLO 才计入） |
| 到达/提交 | `target_request_rate_hz`，`submit_lateness_*_ms`，`inflight_peak` |
| 阶段计算 | **baseline**：`baseline_vlm_ms`、`baseline_ae_ms`（及 step 细分）→ summary 里 `baseline_vlm_latency_*` / `baseline_ae_latency_*`；**ours**：`vlm_prefix_forward_ms`、`ae_step_ms`（及累计） |
| 等待（通用） | `infer_queue_wait_ms`（若有）；提交晚于到达的 lateness |

#### B. 仅 Ours（VA-split）有的统计

| 类别 | 字段 / 含义 |
|---|---|
| 跨进程传输 | `vlm_request_transfer_ms`，`prefix_transfer_ms`（含 CUDA IPC），`ae_result_transfer_ms`；合计 `va_split_transfer_ms` |
| 跨进程排队 | `vlm_request_queue_wait_ms`，`prefix_queue_wait_ms`，`ae_result_queue_wait_ms`；合计 `va_split_queue_wait_ms` |
| VLM 调度 | `vlm_queue_wait_ms`，`vlm_effective_batch` |
| AE / lane | `prefix_admit_wait_ms`，`prefix_lane_ingest_ms`，`prefix_lane_compact_ms`，`ae_effective_batch` |

#### C. 迁移时建议补齐（当前 OpenPI profile 尚未按进程拆开峰值显存）

对比解耦是否多占 KV 时，建议额外打点：

- **整卡峰值**：`torch.cuda.max_memory_allocated()`（或 NVML used memory）在跑中的峰值  
- **VLM 进程峰值 / AE 进程峰值**（ours 双进程分别采样；baseline 只有单进程）  
- 与「活跃 prefix 双份（VLM live + AE lane pool）」相关的分析对齐  

现有脚本只有 SM / mem **利用率**均值，**没有**现成的 per-process peak memory 字段；移植时请自行加上。

### 11.3 启动方式（仿 `run_va_split_mps.sh`，只保留 baseline / ours+MPS）

迁移后建议保留一个同类脚本：统一日志目录、GPU UUID 绑定、ours 时拉起 MPS，再调 profile。OpenPI 上可直接用现脚本，两种对照如下。

**Ours（VA-split + MPS）**

```bash
RUN_MODE=profile \
PROFILE_MODE=split-mps \
GPU_ID=1 \
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/data1/miliang/models/RLinf-Pi05-LIBERO-SFT \
NUM_REQUESTS=200 \
REQUEST_RATE_HZ=8 \
MAX_INFLIGHT=999 \
NUM_STEPS=5 \
MAX_AE_BATCH_SIZE=999 \
MAX_VLM_BATCH_SIZE=8 \
MAX_VLM_WAIT_MS=1.0 \
AE_SM_PERCENT=20 \
VLM_SM_PERCENT=0 \
PYTORCH_COMPILE_MODE=default \
bash scripts/run_va_split_mps.sh
```

脚本会：解析 GPU UUID → 写 `logs/V-A/<RUN_TS>/` → `nvidia-cuda-mps-control -d` → 调 `profile_va_split.py --mode split-mps`。

**Baseline（monolithic，不启 MPS）**

```bash
RUN_MODE=profile \
PROFILE_MODE=monolithic \
GPU_ID=1 \
POLICY_CONFIG=pi05_libero \
POLICY_DIR=/data1/miliang/models/RLinf-Pi05-LIBERO-SFT \
NUM_REQUESTS=200 \
REQUEST_RATE_HZ=8 \
MAX_INFLIGHT=999 \
NUM_STEPS=5 \
BATCH_SIZE=8 \
PYTORCH_COMPILE_MODE=default \
bash scripts/run_va_split_mps.sh
```

`PROFILE_MODE=monolithic` 时脚本不启 MPS，并加 `--no-require-mps-env`。

**移植到新模型时的最小脚本职责**

1. `PROFILE_MODE=monolithic|split-mps` 二选一（ours 必须起 MPS）。  
2. 环境变量对齐上表：`REQUEST_RATE_HZ`、`NUM_REQUESTS`、`MAX_INFLIGHT`、`SLO`（若有）、batch / SM 配额。  
3. 日志：`gpu_binding.log`、`profile.log`、`profile.json`；ours 另有 MPS `control.log` / `server.log`。  
4. 先 smoke（`NUM_REQUESTS=1`）再正式 200 req × 多 rate（8/16/32）对照。

### 11.4 调参旋钮（启动时常用）

| 变量 | 作用 |
|---|---|
| `REQUEST_RATE_HZ` | 泊松到达速率 |
| `NUM_REQUESTS` | 请求数 |
| `MAX_INFLIGHT` | 客户端最大未完成数 |
| `MAX_VLM_BATCH_SIZE` / `MAX_VLM_WAIT_MS` | VLM FCFS 上限与短等待窗口 |
| `MAX_AE_BATCH_SIZE` | AE 每步最大 lane 数（可用很大值表示基本不封顶） |
| `AE_SM_PERCENT` / `VLM_SM_PERCENT` | MPS 线程百分比；`0` 表示不设限额 |
| `PYTORCH_COMPILE_MODE` | 如 `default`；空则关闭。开启后 Torch split 会先枚举 `B=1..cap` 做 compile warmup |
| `WARMUP_REQUESTS` | E2E warmup 的最小请求数，默认 `2`；`0` 可减少普通 warmup |
| `WARMUP_CONCURRENT_INFLIGHT` | E2E warmup 的并发 burst，MPS 脚本默认 `MAX_VLM_BATCH_SIZE` |
| `NUM_STEPS` | AE 去噪步数 |

功能验证可先 `split-no-mps`；正式对比用 **`monolithic` vs `split-mps`** 即可。
