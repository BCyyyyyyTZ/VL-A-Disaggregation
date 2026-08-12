# JAX 多卡 V-A Split 优化说明

日期：2026-08-12

本文记录本次针对 JAX 多卡 VLA 推理拆分的优化思路。目标场景是将 VLM 和 AE 分别放到不同 GPU 上：多个 VLM worker 负责 prefix forward 和 prefix slab 写入，单个 AE worker 负责 continuous batching 的 action denoise。跨卡传输默认使用 `host-staged`，以适配当前 4090 机器上更稳定的传输路径。

## 背景

原始多卡 split 路径在高压下容易出现一个正反馈：

1. VLM worker 将 prefix 写入 AE-owned KV slab。
2. AE worker 从 prefix queue 中持续接收 ready prefix。
3. 如果 AE 一次 drain 太多 prefix，active request set 会快速变大。
4. 每个 denoise step 都需要按当前 active lanes 从 owned slab 中导出 prefix batch view。
5. active set 越大、变化越频繁，`export_batch_view` 越容易 cache miss，并在每个 denoise step 上反复物化大块 prefix 数据。
6. AE step 变慢后又会堆积更多 active 请求，进一步放大 prefix view 开销和排队延迟。

因此本次优化首先不是继续扩大 AE batch，而是控制 AE admission 的节奏，让 AE 更早进入 denoise 循环，避免 prefix view 从毫秒级退化到百毫秒级。

## 优化一：限制 AE 每轮 prefix admission

新增参数：

```text
max_prefix_admits_per_drain
```

默认值为 `1`。含义是：AE process 每次调用 `drain_prefix_ready()` 时，最多接收 1 个 ready prefix，然后回到 `step_once()` 消化当前 active 请求。

这个策略的核心作用是降低 active prefix set 的突增速度：

- 空闲时，AE 仍然可以阻塞等待第一个 prefix，不会空转。
- 有 active 请求时，AE 每轮只补少量新请求，避免一次性把 prefix queue 中的积压全部纳入 active set。
- active set 更稳定，prefix batch view 的导出成本保持在较小范围。
- 即使高压下 VLM 侧持续产出 prefix，AE 也不会因为 admission 太激进而把每个 denoise step 变成大规模 slab gather。

配置入口：

```bash
MAX_PREFIX_ADMITS_PER_DRAIN=1
```

在 profile CLI 中，`--max-prefix-admits-per-drain <= 0` 会被解释为禁用该限制，便于对照实验。

## 优化二：multi-GPU profile 默认 VLM FCFS wait 为 0

多卡 wrapper `scripts/run_profile_va_split_multigpu.sh` 的默认值调整为：

```bash
MAX_VLM_WAIT_MS=0.0
```

VLM worker 的 FCFS batching 仍然保留，`MAX_VLM_BATCH_SIZE` 也仍固定为 `8`。区别是 worker 不再为了攒 batch 额外等待时间窗口，而是立即消费当前已经可见的 request backlog。

这样做的原因是：在当前 4090 多进程 profile 中，高压下的主要队列来自 VLM worker 的真实计算吞吐，而不是缺少 1 ms 的 batching window。继续等待会增加 head-of-line latency，并且在 2 VLM + 1 AE 的 3 卡配置下，不能弥补 VLM worker 数少于 3 副本 baseline 的吞吐差距。

因此 `MAX_VLM_WAIT_MS=0.0` 的定位是 latency-first 默认值：

- 低压下减少不必要等待。
- 高压下避免 FCFS wait 放大 request queue。
- batch size 上限仍为 8，已有 backlog 足够时仍可形成 batch。

## 优化三：保留 dense prefix shadow，但默认关闭

曾验证过一个更激进的思路：AE admission 时将 sparse physical lane 的 prefix 复制到 AE-local dense active prefix pool，后续 denoise step 直接从 dense active pool 导出 batch view。

这个思路可以从机制上减少 sparse lane gather，但在 4090 真机上启用后，JAX compile warmup 风险较高，且收益不如 admission 节流稳定。因此当前实现中只保留代码和单测覆盖，不作为正式默认策略，也没有暴露到 profile wrapper 作为推荐开关。

默认行为仍是：

```text
use_prefix_shadow = False
```

## 跨卡传输策略

正式 profile 默认继续使用：

```bash
CROSS_CARD_TRANSFER_STRATEGY=host-staged
```

该路径让 VLM 侧通过 host-staged 方式写入 AE-owned slab，避免依赖 4090 上不稳定或收益不足的 device-direct 行为。相关 timing 中保留了 prefix transfer、prefix pool write、prefix view 等指标，便于继续判断跨卡写入是否成为瓶颈。

## 当前边界

本次优化解决的是 AE 侧 prefix view 坍塌和 VLM 侧不必要 FCFS 等待，不改变 3 卡拆分方案的资源结构。

在 `2 VLM + 1 AE` 的 3 卡配置下，ours 只有 2 张卡执行 VLM prefix forward；而 3 副本 baseline 有 3 张卡执行完整 VLM+AE。当前模型里 VLM prefix forward 是主导阶段，AE 相对较轻，所以高压下 ours 的 VLM 容量天然少一份。这意味着：

- AE view 可以稳定在毫秒级。
- r8/r16 这类较低压力档更容易受益。
- r32/r64 是否能超过 3 副本 baseline，取决于 VLM 容量、路由和更深的流水线重叠，而不仅是 AE batching。

后续如果继续优化高压档，应优先看：

- 更精确的 VLM worker 负载反馈和路由策略。
- 是否允许 3 VLM + 1 AE 的 4 卡配置作为高压 profile。
- VLM prefix forward 是否能在 split 路径上获得单 worker 速度优势。
- 是否有更低风险的 AE-local prefix view 缓存方案，避免 dense shadow 带来的 compile warmup 风险。
