from __future__ import annotations

import os
import pathlib
import subprocess


def _write_executable(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_run_va_split_mps_defaults_to_libero_env_python():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    script = (repo_root / "scripts/run_va_split_mps.sh").read_text(encoding="utf-8")

    assert 'PYTHON_BIN="${PYTHON_BIN:-/data1/tianze/RLinf-tianze/openpi05_libero_env/bin/python}"' in script
    assert 'WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"' in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-${MAX_VLM_BATCH_SIZE}}"' in script
    assert 'ENABLE_POLICY_BATCH="${ENABLE_POLICY_BATCH:-true}"' in script
    assert 'VA_SPLIT_MAX_VLM_BATCH_SIZE="${VA_SPLIT_MAX_VLM_BATCH_SIZE:-8}"' in script
    assert 'VA_SPLIT_MAX_VLM_WAIT_MS="${VA_SPLIT_MAX_VLM_WAIT_MS:-1.0}"' in script
    assert 'PYTORCH_COMPILE_MODE="${PYTORCH_COMPILE_MODE:-}"' in script


def test_run_va_split_mps_profile_mode_invokes_profile_workload(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    python_arg_log = tmp_path / "python_args.txt"
    mps_arg_log = tmp_path / "mps_args.txt"
    bin_dir = _prepare_stub_tools(tmp_path, python_arg_log=python_arg_log, mps_arg_log=mps_arg_log)

    log_root = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MPS_ARG_LOG": str(mps_arg_log),
            "RUN_MODE": "profile",
            "GPU_ID": "5",
            "RUN_TS": "testrun",
            "LOG_ROOT": str(log_root),
            "NUM_REQUESTS": "3",
            "REQUEST_RATE_HZ": "7",
            "MAX_INFLIGHT": "2",
            "NUM_STEPS": "4",
            "TIMEOUT_S": "9",
            "WARMUP_REQUESTS": "2",
            "PYTORCH_COMPILE_MODE": "default",
            "AE_SM_PERCENT": "60",
            "VLM_SM_PERCENT": "40",
            "VA_SPLIT_MAX_AE_BATCH_SIZE": "11",
            "MAX_VLM_BATCH_SIZE": "13",
            "MAX_VLM_WAIT_MS": "1.5",
            "BATCH_SIZE": "3",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_va_split_mps.sh", "--pytorch-device", "cuda:0"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["scripts/profile_va_split.py", "--policy.config"]
    assert _flag_value(args, "--policy.dir") == "/data2/gaobowen/model/RLinf-Pi05-LIBERO-SFT"
    assert _flag_value(args, "--mode") == "split-mps"
    assert _flag_value(args, "--num-requests") == "3"
    assert _flag_value(args, "--request-rate-hz") == "7"
    assert _flag_value(args, "--max-inflight") == "2"
    assert _flag_value(args, "--batch-size") == "3"
    assert _flag_value(args, "--num-steps") == "4"
    assert _flag_value(args, "--timeout-s") == "9"
    assert _flag_value(args, "--warmup-requests") == "2"
    assert _flag_value(args, "--pytorch-compile-mode") == "default"
    assert _flag_value(args, "--max-vlm-batch-size") == "13"
    assert _flag_value(args, "--max-vlm-wait-ms") == "1.5"
    assert _flag_value(args, "--gpu-device-index") == "5"
    assert _flag_value(args, "--json-output") == str(log_root / "testrun" / "profile.json")
    assert args[-2:] == ["--pytorch-device", "cuda:0"]
    assert "-d" in mps_arg_log.read_text(encoding="utf-8").splitlines()

    gpu_binding = (log_root / "testrun" / "gpu_binding.log").read_text(encoding="utf-8")
    assert "NUM_REQUESTS=3" in gpu_binding
    assert "REQUEST_RATE_HZ=7" in gpu_binding
    assert "MAX_INFLIGHT=2" in gpu_binding
    assert "NUM_STEPS=4" in gpu_binding
    assert "TIMEOUT_S=9" in gpu_binding
    assert "WARMUP_REQUESTS=2" in gpu_binding
    assert "PYTORCH_COMPILE_MODE=default" in gpu_binding
    assert "AE_SM_PERCENT=60" in gpu_binding
    assert "VLM_SM_PERCENT=40" in gpu_binding
    assert "VA_SPLIT_MAX_AE_BATCH_SIZE=11" in gpu_binding
    assert "MAX_VLM_BATCH_SIZE=13" in gpu_binding
    assert "MAX_VLM_WAIT_MS=1.5" in gpu_binding
    assert "BATCH_SIZE=3" in gpu_binding
    assert "ENABLE_POLICY_BATCH=true" in gpu_binding


def test_run_va_split_mps_profile_can_run_monolithic_baseline_without_mps(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    python_arg_log = tmp_path / "python_args.txt"
    mps_arg_log = tmp_path / "mps_args.txt"
    bin_dir = _prepare_stub_tools(tmp_path, python_arg_log=python_arg_log, mps_arg_log=mps_arg_log)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MPS_ARG_LOG": str(mps_arg_log),
            "RUN_MODE": "profile",
            "PROFILE_MODE": "monolithic",
            "GPU_ID": "5",
            "RUN_TS": "baseline",
            "LOG_ROOT": str(tmp_path / "logs"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_va_split_mps.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert _flag_value(args, "--mode") == "monolithic"
    assert _flag_value(args, "--batch-size") == "8"
    assert "--pytorch-compile-mode" not in args
    assert "--no-require-mps-env" in args
    assert not mps_arg_log.exists()


def test_run_va_split_mps_server_forwards_safe_compile_mode(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    python_arg_log = tmp_path / "python_args.txt"
    mps_arg_log = tmp_path / "mps_args.txt"
    bin_dir = _prepare_stub_tools(tmp_path, python_arg_log=python_arg_log, mps_arg_log=mps_arg_log)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "MPS_ARG_LOG": str(mps_arg_log),
            "RUN_MODE": "server",
            "GPU_ID": "5",
            "RUN_TS": "server",
            "LOG_ROOT": str(tmp_path / "logs"),
            "PYTORCH_COMPILE_MODE": "default",
            "ENABLE_POLICY_BATCH": "false",
            "VA_SPLIT_MAX_VLM_BATCH_SIZE": "5",
            "VA_SPLIT_MAX_VLM_WAIT_MS": "0.5",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_va_split_mps.sh"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = python_arg_log.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["scripts/serve_policy.py", "--port"]
    assert _flag_value(args, "--pytorch-compile-mode") == "default"
    assert _flag_value(args, "--va-split-max-vlm-batch-size") == "5"
    assert _flag_value(args, "--va-split-max-vlm-wait-ms") == "0.5"
    assert "--no-enable-policy-batch" in args


def _prepare_stub_tools(
    tmp_path: pathlib.Path,
    *,
    python_arg_log: pathlib.Path,
    mps_arg_log: pathlib.Path,
) -> pathlib.Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "nvidia-smi",
        """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *--query-gpu=uuid*) echo "GPU-test-uuid" ;;
  *--query-gpu=name*) echo "Test GPU" ;;
  *) echo "unexpected nvidia-smi args: $*" >&2; exit 2 ;;
esac
""",
    )
    _write_executable(
        bin_dir / "nvidia-cuda-mps-control",
        """#!/usr/bin/env bash
set -euo pipefail
: "${MPS_ARG_LOG:?}"
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    printf '%s\n' "$arg"
  done >>"${MPS_ARG_LOG}"
else
  cat >/dev/null || true
  printf 'stdin\n' >>"${MPS_ARG_LOG}"
fi
""",
    )
    _write_executable(
        bin_dir / "python",
        """#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_ARG_LOG:?}"
for arg in "$@"; do
  printf '%s\n' "$arg"
done >"${PYTHON_ARG_LOG}"
""",
    )
    return bin_dir


def _flag_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]
