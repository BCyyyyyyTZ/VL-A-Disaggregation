"""Multi-GPU / multi-node VLM–AE split.

Same-machine path follows the Triton bench: N VLM processes + one AE, host
shared-memory prefix lanes, fair credit prefetch, overlapped D2H, and packed
noise. Cross-machine path uses the same schedule with TCP prefix payloads,
because CUDA IPC and POSIX shared memory do not cross hosts.
"""

from openpi.serving.va_split_multinode.runtime import run_benchmark

__all__ = ["run_benchmark"]
