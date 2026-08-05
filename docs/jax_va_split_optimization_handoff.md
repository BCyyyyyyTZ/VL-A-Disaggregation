# JAX V-A Split 优化交接文档

日期：2026-08-05

本文档总结当前 JAX 版 VLM-AE 分离服务的优化进度、实验发现和后续方向。旧版英文流水账已经压缩为当前可接手状态；如需追溯更早细节，优先查看 git diff 和 `logs/tests/` 下的 standalone bench。

## 当前结论

当前 JAX OpenPI 单卡 VLM-AE 分离在工程上已经修掉了多个明显 bug，但在最新 profile 下仍没有超过 JAX baseline。主要原因不是 batch 攒不起来，也不是 AE/VLM 小图没有编译，而是：

1. JAX baseline compile 后已经很接近单进程单卡热执行下限。
2. 当前模型中 VLM 是绝对主导，AE/action head 很轻，理论 pipeline 收益较小。
3. split 后 VLM 和 AE 两个 JAX process 同卡并发，在 MPS 下造成 GPU queueing/竞争，AE 被明显拖慢。
4. split 引入 IPC、prefix slab、result queue、两个 XLA client 等额外代价，当前代价大于同卡 overlap 能省下的时间。

因此当前判断是：

- **同卡 V-A 分离不是理论上不可行**，在 AE 占比更高、调度更强、或单进程/多流实现更好的情况下可能有收益。
- **但对当前 JAX OpenPI + 5-step serving + compiled baseline，当前双进程 MPS split 不是一个自然能带来推理加速/goodput 提升的方案。**
- 后续如果继续尝试，应重点做调度和架构验证，而不是继续抠 AE Python 小开销。

## 最新关键实验

### 1. JAX full profile 对比

主要日志：

```text
logs/JAX/baseline-compile/r{1,8,16,32}_baseline/profile.json
logs/JAX/ours-compile/r{1,8,16,32}_ours/profile.json
logs/JAX/ours-compile-80-20/r{8,16,32}_ours/profile.json
logs/JAX/ours-compile-80-80/r{8,16,32}_ours/profile.json
```

r32 关键结果：

```text
baseline:
  end_to_end_latency_mean_ms       324.65
  action_latency_mean_ms           204.01
  slo_goodput_requests_per_second    4.26
  effective_batch_mean               5.96
  effective_batch_p50/p95/max      7 / 8 / 8
  baseline_vlm_latency_mean_ms     168.55
  baseline_ae_step_latency_mean_ms   4.27
  baseline_ae_latency_mean_ms       21.35

ours sm0/0:
  end_to_end_latency_mean_ms       457.30
  action_latency_mean_ms           456.54
  slo_goodput_requests_per_second    2.56
  effective_batch_mean               6.30
  effective_batch_p50/p95/max      8 / 8 / 8
  vlm_prefix_forward_mean_ms       183.20
  ae_step_mean_ms                   13.16
  ae total, 5 steps                 65.79
  va_split_queue_wait_mean_ms      168.83

ours 80/20:
  end_to_end_latency_mean_ms       472.18
  action_latency_mean_ms           471.44
  slo_goodput_requests_per_second    3.41
  effective_batch_mean               6.33
  vlm_prefix_forward_mean_ms       184.15
  ae_step_mean_ms                   16.81
  ae total, 5 steps                 84.05
  va_split_queue_wait_mean_ms      161.26

ours 80/80:
  end_to_end_latency_mean_ms       447.43
  action_latency_mean_ms           446.68
  slo_goodput_requests_per_second    2.70
  effective_batch_mean               6.46
  effective_batch_p50/p95/max      8 / 8 / 8
  vlm_prefix_forward_mean_ms       186.16
  ae_step_mean_ms                   12.93
  ae total, 5 steps                 64.65
  va_split_queue_wait_mean_ms      152.43
```

结论：

- r32 batch 现在已经能攒起来，`p50/p95/max` 都到 8；当前瓶颈不是组 batch。
- 80/80 比 80/20 好，说明 AE 不能只给 20% SM。
- 80/80 只小幅优于 sm0/0，仍明显差于 baseline；SM quota 不是根因修复。
- r32 split 的 AE 从 baseline `4.27 ms/step` 变成约 `13 ms/step`，是当前最大异常之一。

### 2. JAX stage MPS profile

主要日志：

```text
logs/JAX/openpi_jax_stage_mps_profile/summary.md
logs/JAX/openpi_jax_stage_mps_profile/summary.json
```

这个实验单独测一个 JAX OpenPI 进程在不同 MPS SM 配额和 batch size 下的 stage 热执行时间：

- `vlm`: `build_prefix_feature`
- `action_head`: 多步 `denoise_one_batch`
- stage 边界用 `block_until_ready` 同步计时
- 这里的 `action_head` 是 `num_steps=10`

sm100 关键结果：

```text
batch  total_ms  vlm_ms  action_head_ms
1        69.83    40.59       28.96
4       160.89   121.17       39.40
8       262.21   217.68       44.10
16      464.98   414.20       50.48
24      665.25   606.50       58.39
```

配额敏感性：

```text
bs8:
  sm100: total 262.21, vlm 217.68, action_head 44.10
  sm80 : total 292.52, vlm 248.84, action_head 43.31
  sm60 : total 362.49, vlm 317.48, action_head 44.52
  sm40 : total 519.29, vlm 465.52, action_head 53.22
  sm20 : total 1025.79, vlm 942.61, action_head 82.83
```

由此得到的判断：

- VLM 对 SM 配额非常敏感，是大头。
- action head/AE 单独跑时很轻，且在 60/80/100 SM 下差异不大。
- 当前 serving 用 `num_steps=5`，stage 的 action head 约可折半估算；bs8 下 AE 理论上大约 `22 ms/5 steps`。
- 这与 JAX baseline r32 的 `baseline_ae_latency_mean_ms ~= 21.35 ms` 对齐，说明 baseline AE 已经接近 stage 下限。
- r32 split 的 AE `~65 ms/5 steps` 明显不是 AE 图本身慢，而是 full split 并发环境里的 GPU queueing/竞争。

理论上，如果 bs8 下 VLM `~218 ms`、5-step AE `~22 ms`，理想 pipeline 且无竞争的上界收益大约：

```text
(VLM + AE) / max(VLM, AE) ~= (218 + 22) / 218 ~= 1.10x
```

即高 batch 下理论收益只有约 10%。batch 越大，VLM 占比越高，AE 占比越低，同卡 V-A 分离的收益空间越小。

### 3. Standalone AE/VLM bench

主要日志：

```text
logs/tests/AE-test/
logs/tests/VLM-test/
```

当前保留的关键 AE 结果：

```text
logs/tests/AE-test/ae_worker_bench_20260805_023426.json

B=1:
  action_ms_mean   18.84
  ae_step_ms        3.74

B=8:
  action_ms_mean   25.28
  ae_step_ms        5.01
```

AE direct prefix vs slab prefix 诊断：

```text
logs/tests/AE-test/ae_direct_vs_slab_20260805_021614.json
B=8:
  direct_sync_step_ms  5.17
  slab_sync_step_ms    5.03

logs/tests/AE-test/ae_direct_vs_slab_20260805_021830.json
B=24:
  direct_sync_step_ms  6.86
  slab_sync_step_ms    6.75
```

结论：

- split slab view 本身不是 AE full profile 变慢的原因。
- standalone AE B=8 约 `5 ms/step`，与 stage profile 的单独执行结果一致。
- full r32 split 的 `13 ms/step` 是并发/排队环境造成的。

当前保留的 VLM 结果：

```text
logs/tests/VLM-test/vlm_worker_bench_20260804_223515.json

B=1:
  vlm_prefix_forward_ms     39.79
  worker_total_ms_mean      43.04

B=4:
  vlm_prefix_forward_ms    121.39
  worker_total_ms_mean     126.18

B=8:
  vlm_prefix_forward_ms    217.64
  worker_total_ms_mean     223.53
```

结论：

- VLM worker overhead 已经较小，主要耗时就是 prefix forward。
- full r32 split 的 VLM `183-186 ms` 对 batch mean `6.3-6.5` 来说并不离谱，但仍比 baseline r32 `168.55 ms` 慢十几 ms。

## 已完成的主要代码优化

### VLM 侧

- 修复/优化 batch 聚合逻辑：高负载下现在可以达到接近满 batch，r32 `p50/p95/max=8`。
- VLM child 增加输入阶段细分 timing：
  - `vlm_sample_kwargs_stage_ms`
  - `vlm_observation_stack_ms`
  - `vlm_to_jax_tree_ms`
  - `vlm_observation_from_dict_ms`
  - image dtype / uint8 normalize 相关计数
- 父进程提前处理 uint8 image normalization，避免 child `Observation.from_dict()` 在关键路径上做重转换。
- `_to_jax_tree()` 避免对已有 `jax.Array` 做 `np.asarray -> jnp.asarray` 的 device-host-device 往返。
- VLM contiguous prefix slab batch write 已接入，减少逐 row slab 写入开销。
- VLM 不再为了单 row shape metadata 做不必要 KV slice。

### AE 侧

- AE 使用 exact-active dense denoise state，避免每 step 从 per-request rows 重新拼 batch。
- AE 缓存 prefix view，active lane composition 不变时复用 `view_prefix_batch(batch_size)`。
- AE 在 sample kwargs 已有 noise 时跳过 `init_denoise_state()`，直接构造初始 `x_t/step_idx/dt`。
- whole-batch completion 走 bulk clear。
- 修复 AE result path：完成请求时返回轻量 batch-row reference，按 batch 一次 `np.asarray(batch)`，再在 host 侧切 row，避免 per-row JAX slice/materialization。
- 新增/汇总 timing：
  - `ae_init_denoise_ms`
  - `ae_init_denoise_noise_fast_path`
  - `ae_prefix_view_ms`
  - `ae_prefix_view_cache_hit`
  - `ae_state_batch_stage_ms`
  - `ae_denoise_enqueue_ms`
  - `ae_update_stage_ms`
  - `ae_complete_block_ms`
  - `ae_result_slice_ms`
  - `ae_result_device_get_ms`

### 编译和模型加载

- split runtime 中 VLM process 只 jit VLM prefix，AE process 只 jit denoise。
- role-specific pruning：
  - VLM 删除 AE-only 的 action/time/state projection 模块。
  - AE 删除 `PaliGemma.img`。
  - `PaliGemma.llm` 两边都保留，因为 VLM 生成 prefix KV，AE 消费 KV 做 suffix denoise。
- AE process 先启动并导出 prefix slab，再启动 VLM warmup，降低同时 restore/warmup 的峰值显存压力。
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` 已在 split/MPS 环境中设置，避免两个 JAX 进程各自预占大块显存。

## 当前已知问题

### 1. full split 下 AE 被严重拖慢

证据：

- baseline r32 AE: `4.27 ms/step`
- standalone AE B=8: `5.01 ms/step`
- stage sm100-bs8: `44.10 ms / 10 steps ~= 4.41 ms/step`
- full split r32 80/80: `12.93 ms/step`

判断：

- 不是 slab view 本身。
- 不是 result device get，最新 r32 里 `ae_result_device_get_mean_ms` 只有小量级。
- 不是 AE Python state construction，standalone 已经压到很低。
- 最可能是 VLM/AE 两个 JAX process 同卡并发导致 GPU queueing/MPS 调度竞争。

### 2. VLM 是主要计算大头

stage profile 显示 batch 越大，VLM 占比越高。bs8 sm100 下 VLM 占 total 约 83%；bs24 下约 91%。

这意味着即使 AE 完美 overlap，收益空间也有限。当前 split 多出来的损失超过了理论 overlap 收益。

### 3. MPS quota 不是简单解

- 80/20 明显饿死 AE，r32 AE step 变到 `16.81 ms`。
- 80/80 有改善，但仍差于 baseline。
- stage profile 显示 VLM 到 80 SM 会慢一些，AE 在 60/80/100 下差别不大；但 full serving 中的主要问题是两个进程同时争 GPU，而不是单个阶段被固定配额限制。

### 4. 当前 profile 还缺少并发 overlap 的直接证据

代码里已有 `VA_SPLIT_TIMELINE_PATH` 和：

```text
scripts/analyze_va_split_timeline.py
```

但当前日志没有现成 timeline ndjson。现在“同卡并发导致 slowdown”主要由以下间接证据支持：

- standalone/stage AE 正常，full split AE 异常变慢。
- r32 split batch 已满，但 action/E2E/goodput 仍差。
- 80/80 只轻微缓解，不能把 AE 拉回 stage 下限。

后续如需定论，应打开 timeline 或 Nsight Systems，量化：

- VLM prefix forward 与 AE denoise 的 overlap fraction
- AE step 是否长时间等待在 GPU queue
- 两个 JAX process 的 kernel 提交和执行间隔

### 5. 待验证：JAX/XLA 双进程在 MPS 下的调度粒度问题

这是当前的重要假设，尚未用 Nsight 直接证明。

当前 ours 是两个独立 JAX process：

```text
VLM process -> JAX/XLA client A -> XLA executable -> CUDA/MPS
AE process  -> JAX/XLA client B -> XLA executable -> CUDA/MPS
```

JAX 调用一次 compiled function 并不等价于一个很小的 CUDA kernel。一次 VLM prefix 或 AE denoise 调用通常会由 XLA runtime 提交一串 attention、matmul、softmax、fusion、layout/copy kernel。MPS 可以让多个 CUDA client 并发提交 work，但它不保证：

- 短 AE kernel 能抢占或优先插入长 VLM kernel 之间。
- 两个 XLA executable 的 kernel 能按理想比例公平交错。
- 不同进程的 XLA work 自动形成 compute-bound 与 memory-bound 的互补 overlap。
- AE 的 `block_until_ready()` 只统计自身 kernel 执行时间；它可能把等待前面 VLM GPU work 的排队时间也算进去。

这可以解释一个关键现象：

```text
stage/standalone AE:
  sm100-bs8 action_head ~= 44.1 ms / 10 steps ~= 4.4 ms/step
  standalone B=8 AE    ~= 5.0 ms/step

full split r32:
  AE step              ~= 13 ms/step
```

如果 AE 只是少拿一点 SM，stage sm60/sm80 的结果不应恶化到 `13 ms/step`。更可能是 full split 下 AE 在 GPU queue 中等待，或与 VLM kernel 并发时只能拿到很差的执行机会。

需要用 Nsight Systems 或等价工具验证：

- AE kernel launch 到真正开始执行之间是否有长 gap。
- VLM kernel 和 AE kernel 是否真的 concurrent。
- concurrent 时 AE kernel duration 是否变长，还是主要时间花在等待。
- VLM 大 kernel 是否长时间占据 SM，使 AE 小 kernel 无法及时插入。
- MPS 下两个 JAX client 的 stream/queue 是否呈现粗粒度串行化。

## 对同卡 V-A 分离可行性的当前评估

### 什么时候可能有收益

同卡 V-A 分离仍可能在以下条件下有效：

- AE/action head 占比显著更高，例如 denoise steps 更多。
- VLM 和 AE 的资源互补更强，overlap 时 slowdown 很小。
- 有更强调度策略，让短 AE burst 快速完成，而不是被长 VLM kernel 压住。
- 实现方式避免双 JAX process + MPS + IPC，例如单进程 JAX 内部 pipeline、stream/priority 控制，或更轻的共享 buffer。
- baseline 本身没有很好 compile，或 baseline batching/serving 形态较差。

### 当前为什么不赚

当前 JAX OpenPI serving 的实际情况是：

- 5-step AE 很短，理论 overlap 收益只有约 10% 或更低。
- baseline compile 已经非常强，VLM/AE service time 接近 stage 下限。
- split 后 AE 从约 `21 ms/5 steps` 被拖到 `65 ms/5 steps`。
- split queue wait 在 r32 下仍有 `150-170 ms` 量级。
- VLM 也比 baseline 慢十几 ms。

因此当前实现不是“已经接近可用，只差一点小优化”，而是架构收益不足以覆盖同卡并发代价。

## 后续建议

优先级从高到低：

1. **做 timeline/Nsight 证明并发瓶颈。**
   - 用 `VA_SPLIT_TIMELINE_PATH` 生成 ndjson。
   - 跑 `scripts/analyze_va_split_timeline.py`。
   - 如果条件允许，用 Nsight Systems 看两个 JAX process 的 kernel queue，重点验证“短 AE work 被长 VLM/XLA work 压住”的假设。

2. **验证调度策略，而不是继续自由并发。**
   - 尝试 VLM 完成一个 batch 后，让 AE 连续 burst 跑完或跑固定几步，再放下一批 VLM。
   - 对比 free overlap、AE burst、完全串行三种策略。
   - 目标是把 full serving AE step 拉回 `4-5 ms/step` 附近。

3. **做单进程 JAX 方案的最小验证。**
   - 不通过 multiprocessing/MPS/IPC。
   - 在一个 XLA client 内显式执行 VLM/AE pipeline 或串行分阶段。
   - 先验证是否能减少 AE 被拖慢的问题。

4. **如果目标是稳定加速，优先考虑 VLM/AE 分卡。**
   - 分卡能避免同卡 SM/L2/HBM/queue 竞争。
   - 代价是跨卡传 prefix KV，需要评估 KV 大小和传输代价。

5. **如果继续做同卡 MPS quota，避免 AE 20%。**
   - 80/20 已证明不好。
   - 80/80 比较好但仍不够。
   - 可尝试 90/60、80/60、100/80 之类配置，但预期收益有限；重点仍应看 AE 是否回到 stage 下限。

6. **保留当前 standalone bench。**
   - AE/VLM standalone bench 是判断“算子本身慢”还是“服务并发慢”的关键基线。
   - 后续任何优化都应同时看 standalone 和 full profile。

## 重要文件

实现相关：

```text
src/openpi/serving/va_split_jax/ae_process.py
src/openpi/serving/va_split_jax/vlm_process.py
src/openpi/serving/va_split_jax/runtime.py
src/openpi/serving/va_split_jax/types.py
src/openpi/serving/va_split_jax/compile.py
src/openpi/serving/va_split_jax/device_slab.py
src/openpi/serving/va_split_jax/prefix_cache_pool.py
src/openpi/policies/jax_va_split_policy.py
src/openpi/policies/policy.py
scripts/profile_va_split.py
scripts/openpi_jax_stage_mps_profile.py
scripts/analyze_va_split_timeline.py
```

测试和 bench：

```text
tests/serving/va_split_jax/bench_ae_worker.py
tests/serving/va_split_jax/bench_vlm_worker.py
tests/serving/va_split_jax/bench_ae_direct_vs_slab.py
tests/serving/va_split_jax/test_ae_process.py
tests/serving/va_split_jax/test_vlm_process.py
tests/serving/va_split_jax/test_runtime.py
tests/serving/va_split_jax/test_profile_script.py
tests/serving/va_split/profile_test.py
```

主要日志：

```text
logs/JAX/baseline-compile/
logs/JAX/ours-compile/
logs/JAX/ours-compile-80-20/
logs/JAX/ours-compile-80-80/
logs/JAX/openpi_jax_stage_mps_profile/
logs/tests/AE-test/
logs/tests/VLM-test/
```

Python 环境：

```text
/data1/miliang/RLinf/openpi_libero/bin/python
PYTHONPATH=src
```

## 已运行过的验证

最近一次代码验证曾运行：

```text
/data1/miliang/RLinf/openpi_libero/bin/python -m py_compile \
  src/openpi/serving/va_split_jax/types.py \
  src/openpi/serving/va_split_jax/ae_process.py \
  src/openpi/serving/va_split_jax/runtime.py \
  scripts/profile_va_split.py \
  tests/serving/va_split_jax/bench_ae_direct_vs_slab.py \
  tests/serving/va_split_jax/bench_ae_worker.py

PYTHONPATH=src /data1/miliang/RLinf/openpi_libero/bin/python -m pytest \
  tests/serving/va_split_jax/test_ae_process.py \
  tests/serving/va_split_jax/test_runtime.py \
  tests/serving/va_split_jax/test_profile_script.py \
  tests/serving/va_split/profile_test.py -q
```

结果：

```text
54 passed
git diff --check passed
```

本次文档重构没有重新跑测试。

## 最短接手摘要

如果后续继续做：

1. 先不要再假设 batch 是问题；r32 已经能满 batch。
2. 不要再优先怀疑 slab view；direct vs slab 已证明差异很小。
3. 当前最大异常是 full serving AE step 被同卡 split 并发拖慢到 `~13 ms/step`。
4. stage profile 证明 baseline AE/VLM 接近单阶段热执行下限，split 的理论收益本来就小。
5. 下一步最值得做的是 timeline/Nsight 或 AE burst 调度实验，用证据判断同卡是否还有工程可行性。
