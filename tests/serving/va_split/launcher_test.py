from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest import mock

_LAUNCHER_PATH = Path(__file__).parents[3] / "src" / "openpi" / "serving" / "va_split" / "launcher.py"


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location("va_split_launcher_under_test", _LAUNCHER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_mps_process_envs_sets_cuda_and_mps_directories():
    launcher = _load_launcher_module()

    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        ae_env, vlm_env = launcher.build_mps_process_envs(
            cuda_visible_devices="0",
            mps_pipe_dir="/tmp/mps-pipe",
            mps_log_dir="/tmp/mps-log",
            ae_sm_percent=70,
            vlm_sm_percent=30,
        )

    assert ae_env["PATH"] == "/usr/bin"
    assert vlm_env["PATH"] == "/usr/bin"
    assert ae_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert vlm_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert ae_env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps-pipe"
    assert vlm_env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps-pipe"
    assert ae_env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/mps-log"
    assert vlm_env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/mps-log"
    assert ae_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "70"
    assert vlm_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "30"


def test_build_mps_process_envs_omits_vlm_active_thread_percentage_when_zero():
    launcher = _load_launcher_module()

    with mock.patch.dict(os.environ, {}, clear=True):
        ae_env, vlm_env = launcher.build_mps_process_envs(
            cuda_visible_devices="2,3",
            mps_pipe_dir="/tmp/pipe",
            mps_log_dir="/tmp/log",
            ae_sm_percent=100,
            vlm_sm_percent=0,
        )

    assert ae_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert vlm_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert ae_env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "100"
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in vlm_env
