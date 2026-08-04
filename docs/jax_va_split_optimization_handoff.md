# JAX VA-Split Optimization Handoff

Date: 2026-08-03

This note records the current JAX VA-split optimization state for continuing in a fresh conversation.

## Current Target

Default profiling target:

- System: JAX VA-split, `jax-split-ipc`
- Model: `pi05_libero`
- `num_steps=5`
- `MAX_VLM_BATCH_SIZE=8`
- `MAX_AE_BATCH_SIZE=999`
- prefix capacity: about `8 * 3 = 24`
- `REQUEST_RATE_HZ=8` for steady-state comparison
- `AE_SM_PERCENT=0`, `VLM_SM_PERCENT=0`
- entry script: `scripts/run_profile_va_split_jax.sh`
- Python: `.venv/bin/python`
- logs under `logs/tests/` or `logs/JAX/`

Main performance goal:

- JAX VA-split should match or beat PyTorch VA-split.
- JAX VA-split should beat JAX monolithic baseline under overlappable load.
- For r1, VA-split has inherent IPC/process overhead, so matching monolithic exactly is not required, but the overhead should be explainable and as small as possible.

## Latest r1 Result

Latest inspected file:

```text
/data/miliang/VL-A-Disaggregation/logs/JAX/ours-compile/r1_ours/profile.json
mtime: 2026-08-03 21:08:43
```

Summary:

```text
JAX ours r1:
action_latency_mean_ms      94.96
end_to_end_latency_p50_ms   95.90
throughput_requests_per_s   10.53

vlm_request_queue_wait_ms    0.00
vlm_request_transfer_ms      0.92
vlm_queue_wait_ms            1.25
vlm_batch_wait_ms            1.25
vlm_input_stage_ms          17.42
vlm_prefix_forward_ms       40.07

prefix_queue_wait_ms         0.00
prefix_transfer_ms           0.44
prefix_admit_wait_ms         0.56
prefix_pool_write_ms         0.006

ae_step_ms                   5.10
AE total                     25.48  # 5.10 * 5
ae_result_queue_wait_ms      0.00
ae_result_transfer_ms        0.25

va_split_queue_wait_ms       0.00
va_split_transfer_ms         1.61
```

Reference baseline r1:

```text
/data/miliang/VL-A-Disaggregation/logs/JAX/baseline-compile/r1_baseline/profile.json
mtime: 2026-08-03 19:46:19

JAX monolithic baseline r1:
action_latency_mean_ms      66.33
end_to_end_latency_p50_ms   68.50
baseline_vlm_ms             39.57
baseline_ae_ms              15.13
baseline_ae_step_ms          3.03
```

Current r1 gap:

```text
ours action - baseline action ~= 28.63 ms

Main explainable contributors:
vlm_input_stage              17.42 ms
AE extra vs baseline         10.35 ms  # 25.48 - 15.13
```

This accounts for almost the whole gap. Queue/IPC transfer is no longer the main problem.

## Fixed So Far

### 1. Statistics split was wrong

Old JAX `vlm_queue_wait_ms` included VLM child input staging. It was not true queue wait.

Current timing split:

- `vlm_request_queue_wait_ms`: time already enqueued but VLM did not start `Queue.get`.
- `vlm_request_transfer_ms`: actual queue `get` transfer time, excluding pre-enqueue blocking wait.
- `vlm_queue_wait_ms` / `vlm_batch_wait_ms`: wait after dequeue before input staging/batch handling.
- `vlm_input_stage_ms`: VLM child input staging, including observation stack and host-to-JAX/device staging.

Observed effect in r1:

```text
old vlm_request_transfer ~10.0 ms -> new ~0.9 ms
old vlm_queue_wait       ~27.0 ms -> new ~1.3 ms
```

Conclusion: the earlier 20-30 ms `vlm_queue_wait` was mostly a measurement bug / mixed timing bucket, not a scheduling queue bug.

### 2. Single-request VLM stack fast path

`_stack_request_observations()` now returns the original observation for `len==1`, avoiding unnecessary tree concat in r1.

### 3. VLM input staging optimization + sub-instrumentation

The VLM process no longer manually normalizes `uint8` images in NumPy inside `_asarray_model_input()` / `_concat_model_input()`. It keeps arrays as JAX arrays and relies on `_model.Observation.from_dict()` to do the standard `uint8 -> float32[-1,1]` conversion.

Rationale:

- Avoid CPU-side `astype(float32) / 255 * 2 - 1`.
- Avoid expanding image payload 4x before device staging.
- Preserve model input semantics.

Current code also emits sub-timing fields for `vlm_input_stage_ms`:

```text
vlm_sample_kwargs_stage_ms
vlm_observation_stack_ms
vlm_to_jax_tree_ms
vlm_observation_from_dict_ms
```

Important status:

- Latest inspected r1 profile still predates this sub-instrumented version and shows `vlm_input_stage_ms ~= 17.4 ms`.
- Rerun r1 before deciding whether the `uint8` staging change reduced `vlm_input_stage_ms`, or simply moved normalization into `Observation.from_dict` / VLM forward.
- Compare `vlm_input_stage_ms + vlm_prefix_forward_ms` before and after the change; latest old r1 sum was `17.42 + 40.07 = 57.49 ms`.

### 4. AE-side low-risk cleanup

AE `add_prefix()` no longer pulls `step_idx` back to host via `np.asarray(denoise_state.step_idx)`. Initial `step_idx` is known to be `0`.

AE `dt` is now constructed directly from `num_steps` instead of slicing/reshaping `denoise_state.dt`.

AE `step_once()` now has a single-request fast path:

- avoids `jnp.concatenate([x])`
- avoids `jnp.stack([dt])`
- reuses single request `x_t` directly

Observed current r1:

- `ae_step_ms ~= 5.10 ms`
- previous inspected r1 was about `5.40 ms`
- this is a small improvement, but split AE is still much slower than baseline AE `3.03 ms/step`.

### 5. New AE metric added after latest profile

Current code now emits:

```text
ae_init_denoise_ms
```

This was added to separate AE init/noise staging from AE denoise step time. The latest inspected r1 profile above does not show it, so rerun profile once before relying on this field.

## Current Diagnosis

### Queueing looks healthy in r1

r1 queue-related metrics:

```text
vlm_request_queue_wait_ms   0.00
vlm_queue_wait_ms           1.25
prefix_queue_wait_ms        0.00
ae_result_queue_wait_ms     0.00
va_split_queue_wait_ms      0.00
```

This does not look like a scheduler backlog bug.

### IPC/control transfer is now small

r1 transfer metrics:

```text
vlm_request_transfer_ms     0.92
prefix_transfer_ms          0.44
ae_result_transfer_ms       0.25
va_split_transfer_ms        1.61
```

This is plausible for multiprocessing control/data movement. It is not the dominant r1 gap anymore.

### Main remaining r1 costs

1. `vlm_input_stage_ms ~= 17.4 ms`

   This is now the largest non-kernel VA-split-only cost.

   It likely includes:

   - recursive tree conversion
   - `jnp.asarray` for each observation leaf
   - image dtype conversion/normalization triggered by `Observation.from_dict`
   - host-to-device staging and possible synchronization effects
   - Python overhead from building `Observation`

2. AE step is still slower than baseline:

   ```text
   baseline AE step     3.03 ms
   ours AE step         5.10 ms
   delta                2.07 ms/step
   total delta          10.35 ms over 5 steps
   ```

   Likely causes:

   - split AE calls a jitted `denoise_one_batch` once per Python step, while baseline compile can optimize a tighter monolithic/action loop boundary.
   - split AE reads prefix through AE-owned slab views, not direct local prefix arrays.
   - every step still constructs `step_idx`, `dt`, and `JaxDenoiseState` batch objects, though the per-request `step_idx` host readback and `dt` host slice have been removed.
   - `view_prefix_batch()` is outside the component baseline timing and may interact with JAX cache/sharding/layout differently.

## Recommended Next Steps

### Step 1: Rerun r1 with current sub-timing

Required new fields:

```text
vlm_sample_kwargs_stage_ms
vlm_observation_stack_ms
vlm_to_jax_tree_ms
vlm_observation_from_dict_ms
ae_init_denoise_ms
```

Goal: determine whether the 17 ms is image H2D, prompt/token arrays, Python tree work, or `Observation.from_dict` normalization.

### Step 2: Verify whether image normalization moved into VLM forward

Compare:

```text
vlm_input_stage_ms + vlm_prefix_forward_ms
```

Before and after the `uint8` staging change. Latest r1:

```text
17.42 + 40.07 = 57.49 ms
```

If this sum did not improve, the attempted optimization is not effective and should be revisited.

### Step 3: Consider caching fixed inputs

For the synthetic/profile workload and many real deployments, some inputs are fixed or deterministic:

- tokenized prompt
- tokenized prompt mask
- fixed zero/padding image
- fixed image masks

Promising direction:

- do not send generated padding image every request
- reconstruct/cache zero image in VLM worker based on shape
- cache prompt tokenization by prompt string before enqueue

Avoid preconverting real images to float32 in the parent process unless measured. It increases IPC payload size by about 4x.

### Step 4: Diagnose AE split vs baseline

Use or update existing diagnostic scripts:

```text
scripts/diag_ae_warmup_ipc.py
scripts/diag_ae_ipc_crossproc.py
```

Questions to answer:

- Does direct-prefix `denoise_one_batch` in split process match baseline `~3 ms/step`?
- Does AE-owned slab view alone raise it to `~5 ms/step`?
- Does `step_once` Python assembly account for the difference?
- Are there JAX compile cache misses caused by different `step_idx/dt` shapes or slab sharding?

### Step 5: Rerun r8 after current changes

The currently inspected r8 file is older:

```text
/data/miliang/VL-A-Disaggregation/logs/JAX/ours-compile/r8_ours/profile.json
mtime: 2026-08-03 19:37:47
```

It predates the newest r1 measurements and may not include current timing fields. Rerun r8 before drawing conclusions about p95.

## Files Changed In This Optimization Pass

Key files touched recently:

```text
src/openpi/serving/va_split_jax/vlm_process.py
src/openpi/serving/va_split_jax/ae_process.py
src/openpi/serving/va_split_jax/runtime.py
src/openpi/serving/va_split/vlm_process.py
src/openpi/serving/va_split_jax/timing.py
scripts/profile_va_split.py
tests/serving/va_split_jax/test_vlm_process.py
tests/serving/va_split_jax/test_ae_process.py
tests/serving/va_split/profile_test.py
```

The worktree contains many broader JAX VA-split changes from previous iterations. Do not assume every dirty file is part of the last micro-optimization.

## Tests Run

Latest relevant test command:

```bash
.venv/bin/python -m pytest \
  tests/serving/va_split_jax/test_vlm_process.py \
  tests/serving/va_split_jax/test_ae_process.py \
  tests/serving/va_split_jax/test_runtime.py \
  tests/serving/va_split_jax/test_compile_warmup.py \
  tests/policies/jax_va_split_policy_test.py \
  tests/serving/va_split/profile_test.py
```

Result:

```text
73 passed
```

## Short Conclusion

The large apparent `vlm_queue_wait` problem is fixed. The current r1 performance gap is now well localized:

```text
JAX ours r1 is about 28.6 ms slower than JAX monolithic baseline.
About 17.4 ms is VLM input staging.
About 10.4 ms is slower AE denoise across 5 steps.
Queue and IPC transfer are small.
```

Next work should focus on sub-instrumenting and reducing `vlm_input_stage`, then isolating why split AE slab-view denoise is around `5.1 ms/step` while baseline compile reports `3.0 ms/step`.
