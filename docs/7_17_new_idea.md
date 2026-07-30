# High Efficiency Serving System for Multi-Robot Factories


---

## 1. Introduction

Embody AI are making general-purpose robots increasingly practical for factory deployments.

A promising use case is the deployment of general-purpose robots, such as humanoids and manipulators, in highly-automated robot factories (e.g., BMW, Tesla, Amazon).

Robot serving systems are central to this vision, since such deployments rely on continuous streams of model inference to generate robot actions and monitor task execution.

---

## 2. SOTA Serving System Overview

Heterogeneous robots share an on-premises **edge GPU pool** to run:

- VLM planners
- VLA action models
- safety and monitoring services

under task-specific latency SLOs.

**ROSA** jointly optimizes:

- model placement
- request routing
- batch sizes
- per-task request rates

to maximize weighted SLO-qualified action throughput.

---

## 3. Limitations of SOTA Systems

### Unpredictable Physical Execution Latency & Inefficient Colocated Scheduling

Robots cannot execute actions at a fixed frequency because joint dynamics, payloads, contacts, and hardware conditions cause highly variable physical execution latency.

### Cross-Robot Similarity

Multiple robots may have similar tasks, observations, and execution states, but ROSA treats their requests independently and overlooks opportunities for computation reuse.

### Underutilized Action Chunks

VLA models often generate 32–50 actions per inference, while only 4–8 are executed.

---

## 4. Overview

（本页以系统总览图为主，文字较少。）

系统大致分为三层：

1. **Upper Layer**：Inference Serving & Scheduling（含 VLM / Action Expert 解耦与调度）
2. **Middle Layer**：Multi-Level Cache & Reuse
3. **Lower Layer**：Robot Execution & Action Management

---

## 5. Upper Layer: Disaggregated VLA Inference and Scheduling

VLA inference consists of:

- a **compute-bound VLM backbone**
- a **memory-bound Action Expert**

making a unified batching strategy inefficient.

There are strong interferences between the two stages.

Open question: like PD-disaggregated / semi-PD, should we do **VLM–AE disaggregation**?

---

## 6. Design

We separate the **Pi0.5** policy inference pipeline into two major stages:

1. **VLM prefix-cache construction**
2. **Action Expert denoising**

Key design points:

- Use **NVIDIA MPS** to constrain the effective SM allocation of a single physical GPU  
  （例如 VLM:AE = 80%:20%，或仅开 MPS、不设 SM quota）
- **VLM** uses **FCFS batching**（类似 LLM 的 chunk prefill）
- **AE** uses **mixed-timestep continuous batching**

---

## 7. Trajectory Reuse

（本页以 Trajectory Reuse 示意图为主，正文标题为 *Trajectory Reuse*。）

---

## 8. Adaptive Action Chunking

标题：Adaptive Action Chunking

实验 / 场景标注：

- Robocasa
- Libero
- Robotwin

（本页以图表为主。）

