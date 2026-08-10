from __future__ import annotations

import os
import pathlib
import subprocess


def _write_executable(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_run_profile_va_split_jax_defaults_and_invocation(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/run_profile_va_split_jax.sh"
    script = script_path.read_text(encoding="utf-8")
    assert os.access(script_path, os.X_OK)
    assert 'MODE="${MODE:-jax-split-ipc}"' in script
    assert 'POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"' in script
    assert 'PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"' in script
    assert 'POLICY_DIR="${POLICY_DIR:-/mnt/tianze/models/pi05_libero}"' in script
    assert 'LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/tests}"' in script
    assert 'JAX_COMPILE="${JAX_COMPILE:-1}"' in script
    assert 'JAX_COMPILE_WARMUP="${JAX_COMPILE_WARMUP:-1}"' in script
    assert 'JAX_COMPILE_WARMUP_MAX_BATCH_SIZE="${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE:-$((MAX_VLM_BATCH_SIZE * 3))}"' in script
    assert 'REQUEST_RATE_HZ_LIST="${REQUEST_RATE_HZ_LIST:-}"' in script
    assert "export XLA_PYTHON_CLIENT_PREALLOCATE=false" in script

    python_arg_log = tmp_path / "python_args.txt"
    mps_arg_log = tmp_path / "mps_args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
printf '%s\n' "$@" >"${PYTHON_ARG_LOG}"
""",
    )
    _write_executable(
        bin_dir / "nvidia-cuda-mps-control",
        """#!/usr/bin/env bash
set -euo pipefail
: "${MPS_ARG_LOG:?}"
if [[ $# -gt 0 ]]; then
  printf '%s\n' "$@" >>"${MPS_ARG_LOG}"
  touch "${CUDA_MPS_PIPE_DIRECTORY}/control_lock"
else
  cat >/dev/null || true
  printf 'stdin\n' >>"${MPS_ARG_LOG}"
fi
""",
    )
    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--query-gpu=uuid"* ]]; then
  printf 'GPU-test-uuid\n'
elif [[ "$*" == *"--query-gpu=name"* ]]; then
  printf 'Test GPU\n'
else
  printf '0, Test GPU\n'
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MPS_ARG_LOG": str(mps_arg_log),
            "RUN_TS": "testrun",
            "LOG_ROOT": str(tmp_path / "logs"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_profile_va_split_jax.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert args[:2] == [str(repo_root / "scripts/profile_va_split.py"), "--policy.config"]
    assert _flag_value(args, "--policy.config") == "pi05_libero"
    assert _flag_value(args, "--mode") == "jax-split-ipc"
    assert _flag_value(args, "--policy.dir") == "/mnt/tianze/models/pi05_libero"
    assert _flag_value(args, "--ae-sm-percent") == "20"
    assert _flag_value(args, "--vlm-sm-percent") == "0"
    assert _flag_value(args, "--jax-compile-warmup-max-batch-size") == "24"
    assert _flag_value(args, "--gpu-device-index") == "0"
    assert "--no-jax-compile" not in args
    assert "--no-jax-compile-warmup" not in args
    assert "-d" in mps_arg_log.read_text(encoding="utf-8").splitlines()


def test_run_profile_va_split_jax_passes_rate_list_for_single_warmup_sweep(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    python_arg_log = tmp_path / "python_args.txt"
    mps_arg_log = tmp_path / "mps_args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
printf '%s\n' "$@" >"${PYTHON_ARG_LOG}"
""",
    )
    _write_executable(
        bin_dir / "nvidia-cuda-mps-control",
        """#!/usr/bin/env bash
set -euo pipefail
: "${MPS_ARG_LOG:?}"
if [[ $# -gt 0 ]]; then
  printf '%s\n' "$@" >>"${MPS_ARG_LOG}"
  touch "${CUDA_MPS_PIPE_DIRECTORY}/control_lock"
else
  cat >/dev/null || true
  printf 'stdin\n' >>"${MPS_ARG_LOG}"
fi
""",
    )
    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--query-gpu=uuid"* ]]; then
  printf 'GPU-test-uuid\n'
elif [[ "$*" == *"--query-gpu=name"* ]]; then
  printf 'Test GPU\n'
else
  printf '0, Test GPU\n'
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MPS_ARG_LOG": str(mps_arg_log),
            "REQUEST_RATE_HZ_LIST": "8,16,32",
            "RUN_TS": "rates",
            "LOG_ROOT": str(tmp_path / "logs"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_profile_va_split_jax.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert _flag_value(args, "--request-rate-hz-values") == "8,16,32"


def test_run_profile_va_split_jax_can_disable_compile_flags(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    python_arg_log = tmp_path / "python_args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
printf '%s\n' "$@" >"${PYTHON_ARG_LOG}"
""",
    )
    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--query-gpu=uuid"* ]]; then
  printf 'GPU-test-uuid\n'
elif [[ "$*" == *"--query-gpu=name"* ]]; then
  printf 'Test GPU\n'
else
  printf '0, Test GPU\n'
fi
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MODE": "jax-monolithic",
            "JAX_COMPILE": "0",
            "JAX_COMPILE_WARMUP": "false",
            "LOG_ROOT": str(tmp_path / "logs"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_profile_va_split_jax.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert _flag_value(args, "--mode") == "jax-monolithic"
    assert "--no-jax-compile" in args
    assert "--no-jax-compile-warmup" in args


def test_profile_args_accept_multigpu_modes_and_devices():
    from scripts import profile_va_split

    args = profile_va_split.Args(
        mode="jax-multigpu-split-ipc",
        vlm_devices="0,1",
        ae_device="2",
        baseline_devices="0,1,2",
    )

    assert args.mode == "jax-multigpu-split-ipc"
    assert args.vlm_devices == "0,1"
    assert args.ae_device == "2"
    assert args.baseline_devices == "0,1,2"
    assert args.cross_card_transfer_strategy == "host-staged"


def test_run_profile_va_split_multigpu_split_invocation(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/run_profile_va_split_multigpu.sh"
    python_arg_log = tmp_path / "python_args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
printf '%s\n' "$@" >"${PYTHON_ARG_LOG}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "PROFILE_TARGET": "split",
            "BACKEND": "jax",
            "VLM_DEVICES": "0,1",
            "AE_DEVICE": "2",
            "REQUEST_RATE_HZ_VALUES": "8,16",
            "JAX_COMPILE": "0",
            "JAX_COMPILE_WARMUP": "false",
            "CROSS_CARD_TRANSFER_STRATEGY": "host-staged",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RUN_TS": "split",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert _flag_value(args, "--mode") == "jax-multigpu-split-ipc"
    assert _flag_value(args, "--vlm-devices") == "0,1"
    assert _flag_value(args, "--ae-device") == "2"
    assert _flag_value(args, "--cross-card-transfer-strategy") == "host-staged"
    assert _flag_value(args, "--request-rate-hz-values") == "8,16"
    assert "--no-jax-compile" in args
    assert "--no-jax-compile-warmup" in args


def test_run_profile_va_split_multigpu_baseline_invocation(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/run_profile_va_split_multigpu.sh"
    python_arg_log = tmp_path / "python_args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
printf '%s\n' "$@" >"${PYTHON_ARG_LOG}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "PROFILE_TARGET": "baseline",
            "BACKEND": "jax",
            "BASELINE_DEVICES": "0,1,2",
            "REQUEST_RATE_HZ_VALUES": "8,16",
            "LOG_ROOT": str(tmp_path / "logs"),
            "RUN_TS": "baseline",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert _flag_value(args, "--mode") == "jax-multigpu-baseline"
    assert _flag_value(args, "--baseline-devices") == "0,1,2"
    assert _flag_value(args, "--request-rate-hz-values") == "8,16"


def test_run_profile_va_split_multigpu_rejects_pytorch_backend(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts/run_profile_va_split_multigpu.sh"

    env = os.environ.copy()
    env.update({"BACKEND": "pytorch", "LOG_ROOT": str(tmp_path / "logs")})

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "PyTorch multi-GPU split profile is not implemented" in result.stderr


def _flag_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]
