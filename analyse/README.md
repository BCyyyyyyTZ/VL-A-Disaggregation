# OpenPI0.5 JAX VLM/AE Overlap Experiments

This directory contains the portable experiment package for testing whether
OpenPI0.5 JAX VLM and AE can effectively overlap on one RTX 4090 under MPS.

The target machine is assumed to have:

- repo: `/home/miliang/VL-A-Disaggregation`
- python: `/home/miliang/VL-A-Disaggregation/.venv/bin/python`
- checkpoint: `/home/miliang/model/openpi-assets/checkpoints/pi05_libero`
- main GPU: `7`

## What To Run

Run the full controlled experiment:

```bash
cd /home/miliang/VL-A-Disaggregation
bash analyse/scripts/run_all_gpu7.sh
```

This runs batch sizes:

```text
1,2,4,8,16,32,64
```

For each batch size it runs:

- `solo-vlm`
- `solo-ae`
- two-process `concurrent` VLM + AE
- `single-serial` control in one process
- `single-threads` control in one process
- optional synthetic JAX/PyTorch MPS scheduling control, see below

By default AE batch equals VLM batch. This gives the cleanest interpretation.
If you want to remove the AE-side batch cap or test a different AE batch, pass
`--ae-batch-size` directly to `analyse/src/jax_va_overlap_bench.py`.

## Memory Snapshot Requirement

Do not use `nvidia-smi` memory numbers for analysis. The benchmark writes JAX
device memory profile snapshots through:

```python
jax.profiler.save_device_memory_profile(...)
```

Snapshots and sidecar metadata are written under:

```text
analyse/summary/memory_snapshots/
```

These are the memory artifacts to archive with the run.

## Outputs

Main outputs:

```text
analyse/summary/jax_va_overlap/run_records.json
analyse/summary/jax_va_overlap/overlap_summary.json
analyse/summary/jax_va_overlap/overlap_summary.csv
analyse/summary/jax_va_overlap/overlap_summary.png
analyse/summary/jax_va_overlap/overlap_summary.pdf
analyse/summary/jax_va_overlap/cases/*/result.json
analyse/summary/memory_snapshots/*.prof
analyse/logs/*.log
```

Please preserve the whole `analyse/summary/` directory and `analyse/logs/`
after running.

## Nsight Timeline

Run one representative Nsight Systems capture, usually B=8 first:

```bash
cd /home/miliang/VL-A-Disaggregation
BATCH_SIZE=8 bash analyse/scripts/run_nsys_case_gpu7.sh
```

Optional additional captures:

```bash
BATCH_SIZE=4 bash analyse/scripts/run_nsys_case_gpu7.sh
BATCH_SIZE=16 bash analyse/scripts/run_nsys_case_gpu7.sh
```

The benchmark emits JAX/NVTX ranges:

```text
VLM_ITER
VLM_COMPILED_CALL
AE_ITER
AE_5_STEPS
AE_STEP_0 ... AE_STEP_4
SINGLE_PROCESS_THREADS
```

Use the Nsight timeline to answer:

- Does AE start kernels promptly after its host call, or wait behind VLM work?
- Do AE kernels overlap VLM kernels?
- Does AE kernel duration increase under concurrent VLM?
- Does VLM duration increase under concurrent AE?

## Quantities To Report

The summarizer computes:

```text
vlm_solo_ms
ae_solo_ms
vlm_concurrent_ms
ae_concurrent_ms
vlm_slowdown = vlm_concurrent_ms / vlm_solo_ms
ae_slowdown = ae_concurrent_ms / ae_solo_ms
ideal_speedup
actual_speedup_est
overlap_efficiency
single_serial_combined_ms
single_threads_vlm_ms
single_threads_ae_ms
```

Key interpretation:

- `AE slowdown >> 1`: AE is being delayed or starved under concurrent VLM.
- `VLM slowdown > 1.05`: AE overlap is stealing enough resources to hurt VLM.
- `overlap_efficiency <= 0`: no useful overlap after slowdown.
- `single-threads` better than two-process concurrent: dual XLA client + MPS is likely the problem.
- `single-threads` also poor: XLA graph shape/resource contention is likely the problem.

## MPS Quota

The main experiment intentionally uses free sharing (`100/100`, equivalent to
the current code's `0/0` behavior). MPS quota sweep is not the core question.
Only run quota sweeps later if the 100/100 timeline suggests AE is delayed by
VLM work and you want to test whether artificial pacing helps.

## After Running

Return these files/directories for analysis:

```text
analyse/summary/jax_va_overlap/
analyse/summary/memory_snapshots/
analyse/summary/nsys_va_overlap_b*.nsys-rep or .qdrep/.sqlite if exported
analyse/logs/
```

No interpretation is required on the remote machine; just collect the data and
generated plots.

## Synthetic JAX vs PyTorch MPS Control

To test whether dual XLA client + MPS behaves worse than PyTorch-style CUDA
workloads under the same MPS setup, run:

```bash
cd /home/miliang/VL-A-Disaggregation
bash analyse/scripts/run_synthetic_mps_gpu7.sh
```

Outputs:

```text
analyse/summary/synthetic_mps/synthetic_summary.json
analyse/summary/synthetic_mps/*/result.json
analyse/logs/synthetic_mps_gpu7.log
```

Interpretation:

- If JAX `short_slowdown` is much larger than PyTorch `short_slowdown`, this
  supports the hypothesis that dual XLA client + MPS scheduling is less friendly
  to short work than PyTorch.
- If both frameworks show similar short-work slowdown, the issue is more likely
  generic same-GPU resource contention rather than specifically XLA/MPS.
