# JAX/XLA 双进程在 MPS 下的调度粒度验证

日期：2026-08-05

本文解释一次小型验证实验，用来判断 `docs/jax_va_split_optimization_handoff.md` 中的待验证问题：

> JAX/XLA 双进程在 CUDA MPS 下，短 AE work 是否会被长 VLM work 压住，导致 AE step 明显变慢。

## 背景问题

当前 JAX V-A split serving 中，VLM 和 AE 分别运行在两个独立进程里：

```text
VLM process -> JAX/XLA client A -> XLA executable -> CUDA/MPS
AE process  -> JAX/XLA client B -> XLA executable -> CUDA/MPS
```

handoff 文档里的异常现象是：

```text
baseline r32 AE:
  baseline_ae_step_latency_mean_ms ~= 4.27 ms/step

standalone AE B=8:
  ae_step ~= 5.0 ms/step

full split r32:
  ae_step_mean_ms ~= 13 ms/step
```

也就是说，AE 图本身单独跑并不慢，但放进双进程同卡 MPS split 后，AE step 变慢到约 3 倍。

待验证的问题是：这是否可能由 JAX/XLA 双进程在 MPS 下的 GPU 调度/排队造成，而不是 AE 代码本身变慢。

## 实验目标

这个实验不加载 OpenPI 模型，也不走 IPC、prefix slab、batching 或 serving 逻辑。

它只验证一个更小的问题：

> 一个 JAX 进程持续提交长 XLA 计算时，另一个 JAX 进程里的短 XLA 计算，在同一张 GPU + MPS 下是否会明显变慢。

如果这个小实验也能复现“短图单独很快，并发后明显变慢”，就说明 handoff 里的调度假设是合理的。

## 实验脚本

新增脚本：

```text
scripts/validate_jax_mps_two_process.py
```

主要输出：

```text
logs/tests/jax_mps_two_process_20260805_short4096_long8192/summary.json
```

运行命令：

```bash
.venv/bin/python scripts/validate_jax_mps_two_process.py \
  --gpu-id 0 \
  --output-dir logs/tests/jax_mps_two_process_20260805_short4096_long8192 \
  --short-size 4096 \
  --short-iters 2 \
  --long-size 8192 \
  --long-iters 2 \
  --measure 20 \
  --long-duration-s 15 \
  --warmup 4
```

## 实验设计

实验里有两类 JAX worker：

```text
short worker:
  运行一个较短的 JAX jit matmul 图
  用来模拟 AE 这种短 burst work

long worker:
  运行一个较长的 JAX jit matmul 图
  用来模拟 VLM prefix forward 这种长 work
```

每个 worker 都是独立 Python 进程，因此它们会创建不同的 JAX/XLA client。

父进程启动 CUDA MPS，然后分别跑以下场景：

```text
solo100:
  只跑 short worker
  short worker 使用 100% MPS active thread percentage

concurrent100:
  short worker 使用 100%
  long worker 使用 100%
  两个 JAX 进程同时跑

concurrent80:
  short worker 使用 80%
  long worker 使用 80%
  两个 JAX 进程同时跑

short20_long80:
  short worker 使用 20%
  long worker 使用 80%
  两个 JAX 进程同时跑
```

## 指标含义

### short graph latency

short worker 每次执行短 JAX 图的同步耗时。

脚本在 JAX function 返回后调用 `block_until_ready()`，所以这个时间包含：

```text
Python 发起 JAX 调用
XLA/CUDA work 提交
GPU queue 中等待
实际 GPU kernel 执行
同步等待完成
```

它不能单独区分“排队等待”和“kernel 执行变慢”，但它能代表用户侧看到的同步 latency。

### slowdown vs solo mean

并发场景下的 short mean latency 除以 solo100 的 short mean latency。

例如：

```text
solo short mean = 3.605 ms
concurrent short mean = 21.499 ms
slowdown = 21.499 / 3.605 ~= 5.96x
```

这个指标表示：短 JAX 图在双进程 MPS 并发下，相比单独执行慢了多少倍。

### long graph latency

long worker 每次执行长 JAX 图的同步耗时。

它不是主要观察对象，只用于确认 long worker 确实在持续给 GPU 施加负载。

## 实验结果

核心结果如下：

```text
case              short_sm  long_sm  short mean  short p50  short p95  slowdown
solo100              100     none       3.605      3.576      3.869      1.00x
concurrent100        100      100      21.499     17.721     28.914      5.96x
concurrent80          80       80      18.236     18.265     18.535      5.06x
short20_long80        20       80      18.459     18.419     18.835      5.12x
```

### 结果 1：短图单独跑很快

`solo100` 中，short graph 的平均耗时是：

```text
3.605 ms
```

这说明短 JAX 图本身不是慢图。它的量级和 handoff 里 AE 单独运行的 `4-5 ms/step` 接近。

### 结果 2：双进程并发后，短图明显变慢

`concurrent100` 中，short graph 平均耗时变成：

```text
21.499 ms
```

相比 solo：

```text
21.499 / 3.605 ~= 5.96x
```

这说明只要另一个 JAX 进程持续提交长 XLA work，短 JAX 图的同步耗时就会显著膨胀。

### 结果 3：80/80 配额不能消除问题

`concurrent80` 中，short graph 平均耗时仍然是：

```text
18.236 ms
```

相比 solo：

```text
18.236 / 3.605 ~= 5.06x
```

这和 OpenPI full split 里的现象一致：80/80 比 80/20 好一些，但不能把 AE 拉回 standalone/stage 下限。

### 结果 4：20/80 会让短图保持高延迟

`short20_long80` 中，short graph 平均耗时是：

```text
18.459 ms
```

相比 solo：

```text
18.459 / 3.605 ~= 5.12x
```

这也符合 handoff 里的判断：AE 给 20% SM 不是好策略，短 work 仍然会被明显拖慢。

## 和 OpenPI full split 的对应关系

OpenPI 已有结果：

```text
baseline r32:
  baseline_ae_step_latency_mean_ms = 4.27 ms/step

ours r32 sm0/0:
  ae_step_mean_ms = 13.16 ms/step

ours r32 80/80:
  ae_step_mean_ms = 12.93 ms/step

ours r32 80/20:
  ae_step_mean_ms = 16.81 ms/step
```

本次 microbench：

```text
short graph solo:
  3.605 ms

short graph concurrent:
  18-21 ms
```

两者模式一致：

```text
短 JAX work 单独跑：正常
短 JAX work 和长 JAX work 双进程同卡 MPS 并发：明显变慢
MPS quota 调整：只能缓解或改变形态，不能消除问题
```

因此，OpenPI full split 中 AE step 从 `4-5 ms` 变成 `~13 ms`，很可能不是 AE Python 逻辑、prefix slab 或 result copy 导致，而是双 JAX process 在同一张 GPU/MPS 下的调度和排队造成的。

## 结论

本次实验支持以下结论：

1. JAX/XLA 双进程在 CUDA MPS 下，短 XLA graph 会被另一个进程持续提交的长 XLA graph 明显拖慢。
2. 这种 slowdown 在合成实验里可以达到 `5x` 左右，足以解释 OpenPI full split 中 AE step 从 `4-5 ms` 膨胀到 `~13 ms` 的现象。
3. MPS active thread percentage 不是可靠解法；`80/80` 和 `20/80` 都没有把短图恢复到 solo latency。
4. 当前 JAX V-A split 的主要问题不应继续归因于 AE 小开销，而应优先看同卡双进程 JAX/XLA 调度、GPU queueing 和架构设计。

更准确的表述是：

```text
“JAX/XLA 双进程 + CUDA MPS 会让短 AE 类 work 在并发下显著变慢”
已经被 synthetic microbench 复现。

“具体慢在 kernel launch queue 还是 kernel concurrent execution”
仍需要 Nsight Systems 才能直接区分。
```


## 对 6 个假设的逐条核验

下面按真实程度逐条核验。

### 1. JAX 调用一次 compiled function，不等于一个小 kernel

结论：**真实。**

JAX 的 `jit` 调用会先 lowering/compile 成 XLA executable。对 Python 侧来说，通常表现为一次函数调用；对 GPU 侧来说，它可能对应一段由 XLA runtime 发起的 GPU work。

XLA:GPU 的官方架构说明里，HLO 会经过 lowering，之后可能走几类路径：

```text
HLO op
  -> external library call，例如 cuBLAS/cuDNN
  -> Triton emitter
  -> LLVM emitter
```

所以“一次 compiled function 调用可能包含 matmul、softmax、fusion、copy/layout 等多类 GPU work”这个方向是对的。

需要注意的是：**不能只从 Python call 数量推出 CUDA kernel 数量**。一个 XLA executable 里到底发了多少 kernel、kernel 多长、是否使用 cuBLAS/cuDNN，需要 Nsight Systems 或 XLA dump 才能确认。

本次 microbench 没有证明 kernel 数量，但证明了“一个短 JAX executable 在另一个长 JAX executable 并发存在时会显著变慢”。

### 2. 两个 JAX 进程就是两个 XLA client

结论：**对当前实现是真实的。**

当前 V-A split 是两个 Python process：

```text
VLM process: JAX client A -> XLA executable A -> CUDA/MPS
AE process : JAX client B -> XLA executable B -> CUDA/MPS
```

因为它们是两个独立进程，所以不是同一个 Python/JAX runtime 内部的两个任务。它们各自初始化 JAX、各自创建 CUDA context / XLA client，并通过 MPS 汇入同一张物理 GPU。

这点由当前 serving 架构和本次 synthetic microbench 的 worker 设计共同确认：microbench 也是两个独立 Python process，各自 import JAX、compile 自己的 function，然后同时通过 MPS 跑在 GPU 0 上。

### 3. MPS 并发不是理解 VLM/AE 优先级的抢占式公平调度

结论：**基本真实，但“完全不抢占”这句话要谨慎。**

更准确的说法是：

```text
MPS 能让多个 CUDA client 共享同一张 GPU，并改善多进程 CUDA workload 的并发。
但 MPS 不知道上层任务语义，不知道哪个 kernel 属于 VLM、哪个属于 AE，
也不会自动保证“短 AE work 优先”或“VLM/AE 按理想比例 overlap”。
```

NVIDIA 文档说明 Volta MPS 的 active thread percentage 是一种执行资源限制，不应理解成“AE 永远保留一块独占 GPU 资源，可以随时插队执行”。

本次 microbench 支持这一点：

```text
solo100 short mean:
  3.605 ms

concurrent80 short mean:
  18.236 ms

short20_long80 short mean:
  18.459 ms
```

也就是说，即使给 short/long 分别设置 MPS percentage，short graph 仍然没有恢复到 solo latency。

但本次实验还不能证明具体机制是哪一种：

```text
可能 A：short kernel 在 GPU queue 里等 long kernel
可能 B：short kernel 和 long kernel 并发，但资源竞争导致 short kernel duration 变长
可能 C：某些 kernel 因 resource requirement 不兼容，导致实际串行化
```

这三种需要 Nsight Systems 的 kernel timeline 才能区分。

### 4. XLA 的 kernel 形态可能比 PyTorch 更大、更连续

结论：**可能真实，但本次没有直接验证，不能当成已证实结论。**

合理之处在于：

```text
JAX/XLA 偏向把静态 shape 计算编译成 executable，
通过 fusion、layout/buffer 安排、library calls 等方式减少 Python overhead。
这通常有利于单进程 baseline。
```

这和当前已有结果相符：

```text
JAX baseline compile 后已经很强；
stage profile 里 VLM/AE 单独执行接近热执行下限；
split 后反而因为双进程并发和额外开销变慢。
```

但“JAX/XLA 比 PyTorch eager 或 Inductor 更大、更连续、更难交错”是一个跨框架比较。这个说法需要同时采集 JAX 和 PyTorch 的 Nsight timeline，对比 kernel 数量、kernel duration、gap、stream 和 overlap fraction。

所以现在应写成：

```text
XLA 可能生成更粗或更连续的 GPU work，从而不利于同卡细粒度 overlap。
当前 OpenPI 结果与这个解释相容，但还没有直接证明它一定比 PyTorch 更难交错。
```

### 5. block_until_ready() 会把异步队列等待算进 stage 时间

结论：**真实。**

JAX 默认是异步 dispatch。Python 调用返回时，设备上的计算可能还没完成。`block_until_ready()` 的作用就是等待设备端计算完成。

因此如果 AE 里这样计时：

```text
start timer
v_t = compiled_denoise(...)
x_t_next = x_t + dt * v_t
x_t_next.block_until_ready()
stop timer
```

那么这个时间包含：

```text
host enqueue 时间
GPU queue 等待时间
AE 自己的 GPU 执行时间
host 同步等待返回
```

所以 full profile 里的：

```text
ae_step_mean_ms ~= 13 ms
```

不能直接解释成“AE kernel 本身执行了 13 ms”。它可能包含 GPU queue wait。

本次 microbench 也使用 `block_until_ready()` 计时，因此它验证的是用户侧同步 latency 膨胀，而不是单个 kernel 的纯执行时间膨胀。

### 6. stage profile 和 full split 差很多，更像是并发排队/调度问题

结论：**被现有数据强支持，但还不是 Nsight 级别的直接证明。**

已有 OpenPI 数据：

```text
stage sm100-bs8 action head:
  44.10 ms / 10 steps ~= 4.41 ms/step

baseline r32 AE:
  4.27 ms/step

full split r32 sm0/0:
  13.16 ms/step

full split r32 80/80:
  12.93 ms/step
```

如果问题只是“AE 少拿了一点 SM”，那么 stage profile 里 `sm60/sm80` 不应该仍接近 `4-5 ms/step`，而 full split 里突然到 `~13 ms/step`。

本次 microbench 提供了一个不依赖 OpenPI 的复现：

```text
short graph solo:
  3.605 ms

short graph concurrent with long graph:
  18-21 ms
```

这强烈支持：

```text
full split AE 变慢主要来自同卡双 JAX process 并发时的 GPU 调度/排队/资源竞争，
而不是 AE computation 本身突然变重。
```

但严格来说，它还没有证明具体时间线一定是：

```text
AE enqueue -> wait wait -> run
```

也可能是：

```text
AE enqueue -> concurrent run, but every kernel runs slower
```

或者两者混合。最终需要 Nsight Systems 看 kernel launch、start、end、stream 和 overlap。

## 当前可以确认和不能确认的边界

### 可以确认

```text
1. JAX 调用是异步 dispatch，block_until_ready 会等待设备 work 完成。
2. 一次 JAX compiled function 调用可能对应一段 XLA GPU work，而不是一个 Python-visible 小 kernel。
3. 当前 split 确实是两个独立 JAX process / XLA client 通过 MPS 共享 GPU。
4. 在本机 synthetic JAX 双进程实验中，短 graph solo 约 3.6 ms，并发后变成 18-21 ms。
5. 这个现象足以解释 OpenPI AE step 从 4-5 ms 膨胀到约 13 ms。
```

### 目前不能直接确认

```text
1. AE 变慢中有多少比例是 GPU queue wait，多少比例是 kernel concurrent execution slowdown。
2. VLM kernel 和 AE kernel 是否真的 concurrent，还是大部分时间粗粒度串行。
3. 哪些具体 XLA kernel 阻塞了 AE。
4. JAX/XLA 的 kernel 形态是否一定比 PyTorch 更不利于 MPS overlap。
5. MPS scheduler 的具体内部决策。
```

## 最终判断

你给出的解释大方向是对的，但应该把结论分层表述：

```text
已验证：
  双 JAX process + MPS 下，短 XLA work 会被长 XLA work 显著拖慢。

强支持：
  OpenPI full split 的 AE step 变慢，很可能来自 GPU 调度/排队/资源竞争。

未直接证明：
  具体是 queue wait、kernel concurrent slowdown、resource incompatibility，
  还是这些因素混合。

需要谨慎：
  “JAX/XLA 一定比 PyTorch 更大、更连续、更难 overlap”。
  这需要 JAX/PyTorch Nsight timeline 对比。
```

## 参考资料

- JAX asynchronous dispatch: <https://docs.jax.dev/en/latest/async_dispatch.html>
- JAX benchmarking and `block_until_ready()`: <https://docs.jax.dev/en/latest/benchmarking.html>
- OpenXLA GPU architecture: <https://openxla.org/xla/gpu_architecture>
- NVIDIA Multi-Process Service documentation: <https://docs.nvidia.com/deploy/mps/>

## 新增控制组：区分 quota、资源争用和不良 overlap

为了进一步区分“单纯资源配额/资源争用”和“overlap 形态不好”，又补跑了两组控制实验。

输出路径：

```text
logs/tests/jax_mps_two_process_20260805_quota_controls/summary.json
logs/tests/jax_mps_two_process_20260805_long_controls/summary.json
```

### 控制组 1：short-only 不同 MPS quota

这组用来回答：short 变慢是不是单纯因为 MPS active thread percentage 变小。

```text
case       short_sm  long_sm  short mean  short p50  short p95
solo100       100     none       3.546      3.546      3.570
solo80         80     none       4.937      4.901      5.163
solo20         20     none      17.808     17.786     17.992
concurrent80   80       80      18.535     18.664     18.828
short20_long80 20       80      18.667     18.635     18.946
```

解释：

```text
solo80 只有 4.9 ms，但 concurrent80 到 18.5 ms。
所以 concurrent80 的短图变慢不能用“80% quota 本身就这么慢”解释。

solo20 本身就是 17.8 ms，short20_long80 是 18.7 ms。
所以 short20_long80 主要可以由 20% quota 解释；这也说明 AE=20% 确实会把短 work 饿慢。
```

### 控制组 2：long-only vs concurrent long

这组用来回答：如果只是两个进程并行得很好、平等争用资源，那么 long worker 也应该明显变慢。实际结果不是这样。

```text
case           short_sm  long_sm  short p50  long p50  long p95
longsolo100      none      100       n/a      26.809    26.965
concurrent100     100      100     17.796     27.000    31.655

longsolo80       none       80       n/a      34.103    34.275
concurrent80      80        80     18.186     34.294    36.430
```

解释：

```text
longsolo100 p50 = 26.809 ms
concurrent100 long p50 = 27.000 ms
long 几乎没变，但 short 从 solo100 p50 3.616 ms 变成 concurrent100 p50 17.796 ms。

longsolo80 p50 = 34.103 ms
concurrent80 long p50 = 34.294 ms
long 也几乎没变，但 short 从 solo80 p50 4.889 ms 变成 concurrent80 p50 18.186 ms。
```

这更支持下面的解释：

```text
long/VLM 类 workload 主导了 GPU 执行窗口。
short/AE 类 workload 没有按理想方式与 long/VLM 细粒度交错。
short 的同步 latency 主要被等待、不良插入或不兼容资源形态放大。
```

它不太支持下面这个解释：

```text
两个 JAX 进程其实 overlap 得很好，只是公平地共享资源，所以双方都按比例变慢。
```

原因是 long worker 在并发时 p50 基本没有变，变慢主要落在 short worker 身上。

### 当前能确认到什么程度

在没有 Nsight Systems 的情况下，还不能直接看到 CUDA kernel 的 start/end，所以仍不能精确拆分：

```text
short 到底是在 GPU queue 里等待 long kernel，
还是和 long kernel concurrent 运行但执行效率很差，
还是两者混合。
```

但新增控制组已经可以确认：

```text
这不是单纯 MPS quota 导致的慢。
也不像两个进程“并行得很好、平等争用资源”。
更像 long/VLM work 对调度窗口有主导权，short/AE work 的 overlap 形态不好。
```

## 为什么 AE 小于 VLM 仍没有形成有效收益

一个容易混淆的点是：

```text
AE_concurrent < VLM_concurrent
```

这个条件是必要的，但还不充分。真正需要的是稳态流水线周期接近：

```text
cycle_time ~= max(VLM_concurrent, AE_concurrent)
```

并且请求队列不能因为 stage 服务能力不足而持续累积。

当前 JAX r32 的关键数据是：

```text
baseline r32:
  action_latency_mean_ms        204.0
  baseline_vlm_latency_mean_ms  168.6
  baseline_ae_latency_mean_ms    21.4

ours r32 80/80:
  action_latency_mean_ms        446.7
  vlm_prefix_forward_mean_ms    186.2
  ae_step_mean_ms                12.9
  AE total, 5 steps             ~64.6
  vlm_request_queue_wait_mean   147.6
  prefix_queue_wait_mean          3.9
  ae_result_queue_wait_mean       0.9
  va_split_queue_wait_mean      152.4
```

从这些数看：

```text
AE total ~65 ms，确实小于 VLM ~186 ms。
但 action latency 不是 ~186 ms，而是 ~447 ms。
最大的额外项是 VLM request queue wait ~148 ms。
```

也就是说，问题不只是“AE 是否小于 VLM”，而是：

```text
VLM stage 作为瓶颈接近饱和；
VLM/AE 并发让 VLM 和 AE 都比理想单独阶段更慢；
请求在 VLM 入口前排队，导致流水线没有按 max(VLM, AE) 收敛；
AE 虽然可能被下一批 VLM cover，但这不能抵消前面已经形成的 VLM 队列等待。
```

还要注意依赖关系：同一条请求的 AE 不能被自己的 VLM cover，因为 AE 必须等 prefix 出来。能 overlap 的是：

```text
batch N 的 AE
和 batch N+1 的 VLM
```

所以 overlap 主要改善高负载稳态吞吐和排队，不会自然改善低负载单请求 latency。r8/r16 的数据也符合这一点：AE step 基本正常，但 split 仍比 baseline 慢，说明跨进程/队列/调度开销已经超过能 hide 的 AE 时间。

## 当前无法有效 overlap 的可能原因

### 1. VLM 入口队列已经是主要等待项

r32 下：

```text
vlm_request_queue_wait_mean_ms ~= 148-165 ms
prefix_queue_wait_mean_ms      ~= 3-4 ms
ae_result_queue_wait_mean_ms   ~= <1 ms
```

这说明 AE 不是主要卡在 prefix queue 或 result queue；主要排队发生在请求进入 VLM 前。换句话说，流水线节拍主要由 VLM stage 决定。

### 2. VLM 在 split 并发下也变慢了

```text
baseline VLM mean ~= 168.6 ms
split VLM mean    ~= 183-186 ms
```

这部分 slowdown 会直接吃掉原本可 hide 的 AE 时间。当前 baseline AE 只有 ~21 ms/5 steps，理论收益本来就小；VLM 多慢十几 ms 后，剩余收益空间很薄。

### 3. AE 变慢可以被 cover，但不能修复 VLM 排队

即使 AE 从 ~21 ms 变到 ~65 ms，仍小于 VLM ~186 ms。理论上它可以被下一批 VLM cover。

但如果 VLM stage 的有效服务能力不足，或者 VLM 连续提交形成 GPU work train，AE 的完成并不会降低 VLM 前面的 request queue wait。最终 E2E 仍然被 VLM 排队和 VLM service time 主导。

### 4. 当前是自由并发，不是受控 one-batch overlap

代码上 VLM 只受 prefix lane credits 限制，默认 prefix capacity 是 `max_vlm_batch_size * 3`。这意味着 VLM 最多可以领先 AE 约 3 个 VLM batch。

这种自由并发可能造成：

```text
VLM 连续提交多批大 XLA work；
AE 在中间尝试插入 5-step denoise；
MPS 不保证短 AE work 被及时、公平地插进去；
最终 short/AE 变慢，而且 VLM 入口队列继续增长。
```

## 缓解方向

### 方向 1：限制 VLM lookahead，而不是完全串行

目标是保留：

```text
batch N AE 和 batch N+1 VLM overlap
```

但避免 VLM 一口气领先 3 个 batch，形成连续 VLM work train。

建议实验：

```text
max_prefix_slots = max_vlm_batch_size * 2
```

也就是只允许 VLM 领先 AE 一个 batch。这样理论上仍有 overlap，但如果 AE 跟不上，VLM 会被 credits 反压，不会持续灌 GPU。

当前代码里 `max_prefix_slots` 没有从 profile CLI 暴露，默认在 `JaxProcessVASplitRuntime` 里是 `max_vlm_batch_size * 3`。可以加一个 profile 参数做 sweep：

```text
prefix_slots = 8, 16, 24
```

预期判断：

```text
如果 prefix_slots=16 降低 vlm_request_queue_wait 和 E2E，说明 VLM lookahead 过深导致 overlap 形态差。
如果 prefix_slots=8 变差，说明完全 one-batch capacity 太像串行，overlap 不够。
```

### 方向 2：减小 VLM batch，让 VLM work 边界更频繁

当前 bs8 VLM executable 比较长。可以试：

```text
MAX_VLM_BATCH_SIZE=4
MAX_AE_BATCH_SIZE=8
MAX_VLM_WAIT_MS=1.0 或 2.0
```

动机：

```text
VLM 单次 executable 更短；
AE 更容易在 VLM batch 边界之间插入；
request queue wait 可能下降。
```

代价：

```text
VLM batch efficiency 下降；
如果 VLM capacity 掉太多，吞吐会变差。
```

所以需要看：

```text
vlm_request_queue_wait 是否下降；
ae_step 是否不再出现大 p95；
action/e2e latency 是否下降；
throughput/goodput 是否提升。
```

### 方向 3：给 VLM 加轻量 pacing/yield

不是让 AE 完全独占跑完，而是在 VLM 每个 batch 后，如果 AE 有 active backlog，就给 AE 一个很短的调度窗口：

```text
VLM batch done
if AE active/backlog:
  VLM wait 5-20 ms 或等 AE 至少推进 1 step
then VLM next batch
```

这比完全 AE burst 更保守，仍然允许 VLM cover AE，但避免 VLM 连续提交把 AE 压住。

判断标准：

```text
如果少量 pacing 让 ae_step_p95 和 vlm_request_queue_wait 同时下降，说明自由并发太激进。
如果 pacing 只让 VLM 更慢，说明瓶颈已经纯 VLM，不能靠让路解决。
```

### 方向 4：MPS quota 做 sweep，但不要只试 80/20

已有结果说明 20% AE 会饿慢 AE。更值得试的是限制 VLM 峰值，而不是限制 AE：

```text
VLM/AE = 70/100
VLM/AE = 60/100
VLM/AE = 70/80
VLM/AE = 80/100
```

目的不是“公平分 SM”，而是减少 VLM 大 kernel 对 GPU 的连续占用，让 AE 更容易完成。代价是 VLM 可能明显变慢，因此要用最终 E2E/goodput 判断。

### 方向 5：单进程 JAX 受控 pipeline

如果双进程 MPS 无法稳定调度，最干净的验证是单进程里显式组织：

```text
VLM batch N
AE batch N while preparing/dispatching VLM batch N+1 的可控阶段
```

哪怕第一版不是完美多 stream，只要能避免两个 XLA client 通过 MPS 抢同卡，就能判断问题是不是主要来自双进程 MPS 调度。

## 优先实验顺序

建议按这个顺序做：

```text
1. 打开 VA_SPLIT_TIMELINE_PATH，跑 r32 80/80，量化 VLM/AE Python-level overlap fraction。
2. 暴露 max_prefix_slots，做 8/16/24 sweep。
3. 做 MAX_VLM_BATCH_SIZE=4 vs 8 sweep。
4. 做 VLM/AE quota sweep：80/100, 70/100, 60/100。
5. 如果仍不行，再做 pacing/yield 或单进程 JAX pipeline。
```

每个实验重点看：

```text
vlm_request_queue_wait_mean/p50/p95
vlm_prefix_forward_mean/p50
AE total = ae_step_mean * 5
ae_step_p95
throughput/goodput
action_latency/e2e_latency
```

## 后续建议

如果继续推进，建议优先做下面几件事：

1. 用 Nsight Systems 跑一次 full split，直接看 VLM/AE 两个进程的 CUDA kernel timeline。
2. 对比三种策略：完全自由并发、VLM 后 AE burst、完全串行。
3. 做单进程 JAX pipeline 最小验证，避免两个独立 XLA client 通过 MPS 抢同一张卡。
4. 如果目标是稳定加速，优先评估 VLM/AE 分卡，而不是继续调同卡 MPS quota。
