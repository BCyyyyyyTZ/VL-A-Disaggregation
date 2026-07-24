from __future__ import annotations

import os


def build_mps_process_envs(
    *,
    cuda_visible_devices: str,
    mps_pipe_dir: str,
    mps_log_dir: str,
    ae_sm_percent: int,
    vlm_sm_percent: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build subprocess environments for AE and VLM processes under CUDA MPS."""

    base_env = os.environ.copy()
    common_env = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CUDA_MPS_PIPE_DIRECTORY": mps_pipe_dir,
        "CUDA_MPS_LOG_DIRECTORY": mps_log_dir,
    }

    # sm_percent=0 means "no extra MPS SM quota limit" (omit/clear the env var),
    # not "0% active threads". Clear inherited values too.
    ae_env = base_env | common_env
    if ae_sm_percent != 0:
        ae_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(ae_sm_percent)
    else:
        ae_env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

    vlm_env = base_env | common_env
    if vlm_sm_percent != 0:
        vlm_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(vlm_sm_percent)
    else:
        vlm_env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

    return ae_env, vlm_env
