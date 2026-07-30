# V–A Disaggregation Report

---

## 1. Introduction

Embody AI 正使通用机器人在工厂部署日益可行。一个典型场景是：人形机器人、机械臂等部署在高度自动化的机器人工厂中（如 BMW、Tesla、Amazon），持续依赖模型推理生成动作并监控任务执行。

其中，**VLA**（Vision-Language-Action，如 Pi0.5）是关键策略模型：根据视觉观测与语言指令，经一次推理输出可执行的动作序列。一次推理可粗分为两段：

1. **VLM prefix**：视觉–语言前缀前向，构造中间表示（如 KV cache、masks、state）
2. **Action Expert (AE) denoising**：多步去噪，输出 `actions`

---

## 2. Motivation

> 传统openpi中串行执行VLM+AE的问题
> 为了AE开大batch，VLM会有什么问题；为了VLM部分不冗余，减小batch，AE会有什么问题
> 描述论证 + 画图

在传统 OpenPI 一类部署里，VLM 与 AE 通常放在**同一进程、同一条串行执行链路**中完成：先做完 VLM prefix，再在同一条链路上多步去噪出动作。问题不在单算子快慢，而在 **两段异构计算被同一套调度绑死**。

首先，两阶段 **无法时间重叠**：AE 去噪时 VLM 空等，VLM 做前缀时 AE 空等，GPU 在任一时刻只能服务其中一段。

其次，两段的资源形态本就不一样，却只能共用 **同一 batch 边界**：

- **VLM 更偏 compute-bound**：前缀前向以大算力、高算术强度为主，延迟对 batch 增大更敏感；盲目凑大 batch 会显著拉长队头等待与单次前缀耗时。
- **AE 更偏 memory-bound**：多步 denoising 反复读写 KV / 中间状态，更吃带宽与步级并行；若 batch 长期偏小，每步有效并行不足，多步迭代的吞吐上不去。

把「算力型prefix」和「带宽型多步denoise」捆成一次统一 launch，就会在调度上互相迁就，也难以各自吃满 GPU。

> 画个图？
> | 若偏向… | 做法 | 代价 |
> |---|---|---|
> | **AE 开大 batch** | 等更多请求凑齐再整段推理 | compute-bound 的 VLM 被拖进大 batch：延迟与队头阻塞上升，两阶段仍抢同一条串行执行链路 |
> | **VLM 少冗余 / 小 batch** | 尽快做完 prefix，batch 保持较小 | memory-bound 的 AE 每步并行度不足，多步 denoising 吞吐上不去 |

---

## 3. Design

> 说明设计点：画图+描述
> 1. VLM FCFS batching
> 2. AE continuous batching
> 3. lane pool transmit

### 3.1 VLM：FCFS batching

VLM 侧按 **先到先服务** 收集兼容请求：队列里已有的先尽量拼进当前 batch；**能凑满上限就开满 batch 做一次前缀前向，凑不满也不死等**，有多少就对多少做 prefix，再交给 AE。

这样 VLM 保持类似 LLM prefill 的语义：不为了喂饱 AE 而去盲目放大自己的 batch，从而控制 compute-bound 前缀的延迟。

### 3.2 AE：continuous batching

AE 侧维护一批「进行中」的请求：每个请求有自己的去噪步进度，**不必对齐同一 VLM batch**。

每一步从当前活跃请求里取出最多若干条，组成一个 GPU batch 做一次去噪；不同请求可以处于不同 timestep。谁先走完谁先返回动作，并通知 VLM 释放对应前缀。

因此即使上游 VLM batch 不大，AE 仍能靠持续到达的 prefix feature，把多步 denoising 的并行度维持在较高水平。

### 3.3 Lane pool：prefix feature 复用

VLM→AE 跨进程传前缀时，CUDA 张量走 **隐式 CUDA IPC**：实质是传 GPU 共享句柄/元数据，而不是把整份 KV 再经 CPU 全量拷贝。VLM 侧保活底层 storage，直到该请求 AE 去噪结束再释放。

AE 再把前缀写入本地预分配的 **lane pool**；之后 continuous batch 都在 pool 上读 —— 跨进程一次共享，本地槽位规整复用。

---

## 4. Report Data

> 画图、对比数据：画一个表格即可，突出ours的数据
> 分析ours带来提升的原因
> 1. disaggregation V-A
> 2. AE continuous batching
> 3. torch compile

### 提升原因（对应设计点）

1. **V–A disaggregation**  
   两阶段可重叠：VLM 继续做后续请求的 prefix，同时 AE 对已到达的 prefix 做多步去噪；打破 colocated 串行执行链路，高负载下 E2E / 吞吐收益更明显。

2. **AE continuous batching**  
   步级、混合 timestep 拼批，使 AE 不再受 VLM FCFS batch 边界束缚；小 VLM batch 也能喂出持续的 AE GPU batch，改善 memory-bound 多步阶段的利用率。

3. **Torch compile**  
   Baseline 仍串行，compile 只会让两段各自快一点，E2E 的提速近似相加，再叠加统一 batch 的排队，所以相对非 Torch baseline 只有小幅下降。  
   Ours 已解耦并可重叠，AE 还要反复执行同构去噪步——compile 落在更干净的热路径上，单步收益会被步数与 continuous batch 放大，因而相对同开 Torch 的 Baseline，E2E / Goodput 拉开更明显。

