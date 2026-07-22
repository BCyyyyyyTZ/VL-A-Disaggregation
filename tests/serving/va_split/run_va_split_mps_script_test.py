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


def test_run_va_split_mps_profile_mode_invokes_profile_workload(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_arg_log = tmp_path / "python_args.txt"

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
cat >/dev/null || true
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

    log_root = tmp_path / "logs"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": "python",
            "PYTHON_ARG_LOG": str(python_arg_log),
            "RUN_MODE": "profile",
            "GPU_ID": "5",
            "RUN_TS": "testrun",
            "LOG_ROOT": str(log_root),
            "NUM_REQUESTS": "3",
            "REQUEST_RATE_HZ": "7",
            "MAX_INFLIGHT": "2",
            "NUM_STEPS": "4",
            "TIMEOUT_S": "9",
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
    assert _flag_value(args, "--num-steps") == "4"
    assert _flag_value(args, "--timeout-s") == "9"
    assert _flag_value(args, "--json-output") == str(log_root / "testrun" / "profile.json")
    assert args[-2:] == ["--pytorch-device", "cuda:0"]


def _flag_value(args: list[str], flag: str) -> str:
    index = args.index(flag)
    return args[index + 1]
