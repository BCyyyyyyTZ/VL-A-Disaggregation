# Multi-GPU Profile Support 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 支持分卡 V-A split 及公平多卡 baseline 的 profile 入口和脚本。

**架构：** 在现有 JAX V-A split runtime 上新增 multi-GPU split runtime，并新增同 GPU 数量的 multi-replica baseline runtime。`scripts/profile_va_split.py` 暴露 `jax-multigpu-split-ipc` 与 `jax-multigpu-baseline` 两种 mode，新 bash 脚本只封装这两类分卡相关 profile。

**技术栈：** Python multiprocessing、现有 JAX split policy/runtime、现有 synthetic Poisson profile harness、bash runner。

---

### 任务 1：Profile 参数与脚本红测

**文件：**
- 修改：`tests/serving/va_split_jax/test_profile_script.py`
- 修改：`scripts/profile_va_split.py`
- 创建：`scripts/run_profile_va_split_multigpu.sh`

- [ ] **步骤 1：编写失败测试**

覆盖 `jax-multigpu-split-ipc` / `jax-multigpu-baseline` mode、`--vlm-devices`、`--ae-device`、`--baseline-devices`、以及 bash 脚本的 `PROFILE_TARGET=split|baseline` 参数映射。

- [ ] **步骤 2：运行测试验证失败**

运行：
`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_profile_script.py -q`

预期：新增测试因 mode/参数/脚本缺失失败。

### 任务 2：JAX multi-GPU runtime 接入

**文件：**
- 修改：`src/openpi/serving/va_split_jax/runtime.py`
- 修改：`src/openpi/policies/jax_va_split_policy.py`
- 测试：`tests/serving/va_split_jax/test_multigpu_runtime_unit.py`

- [ ] **步骤 1：实现 split runtime**

新增 `JaxMultiGpuProcessVASplitRuntime`，启动 1 个 AE 进程与 N 个 VLM 进程，父进程 least-backlog 路由请求到 VLM worker queue，AE release 根据 `source_worker_id` 回到对应 VLM queue。

- [ ] **步骤 2：实现 baseline runtime**

新增 `JaxMultiGpuReplicaRuntime`，每张 baseline device 一个完整 JAX policy replica 子进程，父进程 least-backlog 路由请求。baseline 不做 V/A 分离。

### 任务 3：Profile/batch runner 集成

**文件：**
- 修改：`scripts/profile_va_split.py`
- 修改：`scripts/run_profile_va_split_multigpu.sh`
- 测试：`tests/serving/va_split_jax/test_profile_script.py`

- [ ] **步骤 1：接入 mode 到 policy factory**

`jax-multigpu-split-ipc` 调用 multi-GPU split factory；`jax-multigpu-baseline` 调用 multi-replica baseline factory。PyTorch 分卡模式 fail-fast，不伪装支持。

- [ ] **步骤 2：脚本支持一次 warmup 多 rate sweep**

脚本将 `REQUEST_RATE_HZ_VALUES` 传给 `profile_va_split.py`，使用同一个 policy 对象完成现有 single warmup + sweep 逻辑。

### 任务 4：验证与提交

**文件：**
- 日志目录：`/mnt/tianze/VL-A-Disaggregation/logs/tests/multigpu`

- [ ] **步骤 1：运行 focused lint**

`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m ruff check --select I,F401,E9 ...`

- [ ] **步骤 2：运行 focused tests**

`/mnt/tianze/VL-A-Disaggregation/.venv/bin/python -m pytest tests/serving/va_split_jax/test_profile_script.py tests/serving/va_split_jax/test_multigpu_config.py tests/serving/va_split_jax/test_multigpu_router.py tests/serving/va_split_jax/test_prefix_transfer.py tests/serving/va_split_jax/test_multigpu_runtime_unit.py -q`

- [ ] **步骤 3：commit/push**

提交到 `feature/multigpu-va-disaggregation-impl` 并推送远端。
