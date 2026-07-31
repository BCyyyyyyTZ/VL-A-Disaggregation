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
    assert 'PYTHON_BIN="${PYTHON_BIN:-/data1/miliang/RLinf/openpi_libero/bin/python}"' in script
    assert 'JAX_COMPILE="${JAX_COMPILE:-1}"' in script
    assert 'JAX_COMPILE_WARMUP="${JAX_COMPILE_WARMUP:-1}"' in script
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
  /usr/bin/python3 - <<'PY'
import os
import socket

path = os.path.join(os.environ["CUDA_MPS_PIPE_DIRECTORY"], "control")
sock = socket.socket(socket.AF_UNIX)
sock.bind(path)
PY
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
    assert _flag_value(args, "--policy.dir") == "/data1/miliang/models/RLinf-Pi05-LIBERO-SFT"
    assert _flag_value(args, "--ae-sm-percent") == "20"
    assert _flag_value(args, "--vlm-sm-percent") == "0"
    assert _flag_value(args, "--jax-compile-warmup-max-batch-size") == "32"
    assert _flag_value(args, "--gpu-device-index") == "0"
    assert "--no-jax-compile" not in args
    assert "--no-jax-compile-warmup" not in args
    assert "-d" in mps_arg_log.read_text(encoding="utf-8").splitlines()


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


def _flag_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]
