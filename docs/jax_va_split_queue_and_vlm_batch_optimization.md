# JAX V-A Split Queue 与 VLM Batch 优化说明

本文记录当前 `jax-split-ipc` serving 路径中已落地的两项优化：请求入队回压，以及 VLM late FCFS batch 调度。重点说明设计原理，不展开性能数据和源码细节。

## 背景

JAX V-A split 将一次策略推理拆成 VLM prefix prefill 和 AE denoise 两段，并通过 IPC slab、lane credit 和进程间队列串联。高负载下，系统的主要排队通常发生在 VLM 请求队列侧；AE 本身较轻，但如果上游持续放入过多请求，VLM 会积累很长的队列等待，进而放大端到端延迟和尾部延迟。

此前的高压现象可以概括为两类：

- 在途请求数过大时，请求已经进入 runtime / VLM 队列，但尚未获得 prefix lane 或 VLM 计算机会，排队时间被堆到 `va_split_queue_wait` 和 `vlm_request_queue_wait` 中。
- FCFS 聚批在队头请求已经等待较久时仍倾向于凑大 batch。大 batch 有利于吞吐，但单次 VLM prefill 时间更长，会进一步增加已排队请求的等待。

优化目标不是简单增加 prefix slot，也不是把 VLM batch 一律调小，而是在延迟和吞吐之间做更稳健的 admission 与 batch 调度。

## 请求入队回压

runtime 侧新增了 request admission token。每个进入 `infer` / `infer_batch` 的请求在写入 VLM request queue 前，必须先拿到一个 admission token。token 总数默认跟 prefix slot 容量对齐，即 `max_prefix_slots`，当前默认仍是 `max_vlm_batch_size * 3`。

这个设计把系统的主要排队点从无界的 VLM request queue 前移到 runtime admission 层：

- 已进入 VLM 队列的请求数量被限制在 prefix slot 容量附近，避免队列无限增长。
- request queue 中的等待更接近真实可服务 backlog，而不是把过载请求提前塞进子进程。
- 当请求正常完成、worker 返回错误、worker shutdown，或父进程等待结果超时时，都会释放对应 admission token，避免回压自身漏票。

这条优化的核心作用是控制过载下的排队水位。它不会提高单次 VLM prefill 的计算速度，但能显著减少由过量 inflight 引入的尾部排队。

## VLM Late FCFS Batch 调度

VLM 进程仍保留 FCFS 聚批，但对「队头请求已经超过 FCFS 等待窗口」的情况加入 batch 上限收缩。当前目标 batch size 为 6。

调度规则如下：

- 正常情况下，VLM 仍按 `max_vlm_batch_size` 和可用 lane credit 形成 batch。
- 如果队头请求已经等待超过 `MAX_VLM_WAIT_MS`，且当前候选请求不足以立即填满原始 batch 上限，则将本次 batch limit 收缩到 6。
- 如果 backlog 已经足够深，可以立即形成满 batch，则不收缩，继续使用原始 batch 上限，避免高负载下吞吐损失过大。

这个规则针对的是「队头已经晚了，但 backlog 又没有深到必须追求满 batch」的区间。此时继续等待或凑到 B=8，会把 VLM prefill 单段时延和队头延迟一起拉高；适度降低 batch 上限，可以更快释放队头请求，同时保留一定的 batch 效率。

选择 6 而不是 4 或 7，是因为它在当前测试条件下更平衡：

- 4 能进一步降低单次 VLM prefill，但更容易牺牲高负载吞吐。
- 7 更接近原始满 batch，但对 VLM prefill 和 AE tail 的改善不稳定。
- 6 在降低 VLM prefill、稳定 AE tail 和维持吞吐之间更折中。

## 与 AE 稳定性的关系

AE 侧不是当前主瓶颈，但 AE 会受到 VLM 大 batch 和同卡 GPU 争用影响。VLM late batch 收缩后，AE 接收到 prefix 的节奏更均匀，单次 AE batch 规模也更可控，能降低 AE step 的 p95 抖动风险。

这不是通过限制 AE 并发实现的，而是通过减少上游 VLM 的长时间阻塞和批量突发，降低 AE 被动背压的概率。

## 设计边界

这两项优化都属于 admission / scheduling 层改动，不改变 IPC slab、lane credit 协议和模型计算图：

- 不增加 prefix slot 数量。
- 不改变 VLM / AE 模型函数。
- 不引入多 GPU。
- 不依赖硬 SM 配额。

因此它们适合先作为同卡 split serving 的稳定性和延迟控制基础。后续如果要继续提升高负载吞吐，重点仍应放在 VLM prefill 计算路径、输入 staging、编译形态和更细粒度的自适应调度上。

## 验证范围

当前代码配套了单元测试覆盖以下行为：

- admission token 会按 request id 释放，避免父进程侧回压漏票。
- late FCFS 且候选不足时，VLM batch 会收缩到目标上限。
- late FCFS 但候选已经足够填满 batch 时，VLM 仍保留满 batch。

实际 profile 验证使用同卡 `jax-split-ipc`、`SM=0/0`、`MAX_INFLIGHT=64`、`MAX_VLM_BATCH_SIZE=8`、`MAX_VLM_WAIT_MS=1.0`、`REQUEST_RATE_HZ_LIST=16,32`。测试过程中尽量选择当时最干净的单卡，但机器上仍存在其他常驻进程，因此 profile 数据只作为同机趋势参考。
