"""Exercise replacement transactions with fake Slurm commands; no HPC access."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
if os.name == "nt":
    git = shutil.which("git")
    candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
    BASH = str(candidate) if candidate and candidate.exists() else None
else:
    BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH, reason="Bash required")


def posix(path: Path) -> str:
    value = path.resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


def executable(path: Path, content: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + content, encoding="utf-8", newline="\n")
    path.chmod(0o700)


@pytest.fixture
def replacement(tmp_path):
    scripts = tmp_path / "project" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(PROJECT / "scripts" / "replace_pending_matrix.sh", scripts / "replace_pending_matrix.sh")
    executable(scripts / "common.sh", '''
flow3d_die() { printf 'flow3d: %s\\n' "$*" >&2; exit 1; }
flow3d_settings() { mkdir -p "$FLOW3D_ROOT/logs"; }
''')
    executable(scripts / "submit.sh", '''
[[ "$1" = matrix-resume && "$2" = --plan && "$FLOW3D_SUBMIT_HOLD" = 1 ]]
printf 'submit-train:%s:held=%s\\n' "$FLOW3D_ARRAY_TASKS" "$FLOW3D_SUBMIT_HOLD" >> "$MOCK_DIR/events"
printf 'Submitted batch job 900\\n'
''')
    executable(scripts / "submit_test_after.sh", '''
[[ "$1" = --training-plan && "$3" = --dependency && "$4" = 900 && "$FLOW3D_SUBMIT_HOLD" = 1 ]]
printf 'submit-test:dependency=%s:held=%s\\n' "$4" "$FLOW3D_SUBMIT_HOLD" >> "$MOCK_DIR/events"
[[ "$MOCK_MODE" != submit_failure ]] || exit 42
printf 'Submitted batch job 901\\n'
''')
    shim = tmp_path / "bin"
    shim.mkdir()
    if os.name == "nt":
        executable(shim / "python", 'exec "$MOCK_REAL_PYTHON" "$1" "$2" "$(cygpath -w "$3")"\n')
    executable(shim / "squeue", '''
job=''
while (( $# )); do
    if [[ "$1" = -j ]]; then job="$2"; shift; fi
    shift
done
printf 'query:%s\\n' "${job:-all}" >> "$MOCK_DIR/events"
if [[ -z "$job" ]]; then
    [[ ! -f "$MOCK_DIR/cancelled" || "$MOCK_MODE" = cancel_partial ]] && printf '100_40|PENDING\\n'
    exit 0
fi
state=PENDING
if [[ "$job" = 100 && "$MOCK_MODE" = running_initial ]]; then state=RUNNING; fi
if [[ "$job" = 100 && "$MOCK_MODE" = running_after_hold && -f "$MOCK_DIR/held" ]]; then state=RUNNING; fi
if [[ "$job" = 100 ]]; then
    printf '100_40|%s\\n100_42|%s\\n100_47|%s\\n' "$state" "$state" "$state"
elif [[ "$job" = 200 ]]; then
    last=62
    [[ "$MOCK_MODE" != partial_test ]] || last=61
    for ((i=0; i<=last; i++)); do printf '200_%s|PENDING\\n' "$i"; done
else
    exit 1
fi
''')
    executable(shim / "scontrol", '''
printf '%s:%s\\n' "$1" "$2" >> "$MOCK_DIR/events"
if [[ "$1" = hold ]]; then touch "$MOCK_DIR/held"; fi
''')
    executable(shim / "scancel", '''
[[ "$1" = --state=PENDING && "$2" = 200 && "$3" = 100 ]]
printf 'cancel:pending:200,100\\n' >> "$MOCK_DIR/events"
touch "$MOCK_DIR/cancelled"
''')
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"mode": "matrix-train", "tasks": [{}] * 63}), encoding="utf-8")
    env = os.environ.copy()
    env.update(PATH=str(shim) + os.pathsep + env.get("PATH", ""), MOCK_DIR=posix(tmp_path),
               FLOW3D_ROOT=posix(tmp_path / "storage"), USER="fixture", MSYS_NO_PATHCONV="1",
               MOCK_REAL_PYTHON=posix(Path(sys.executable)))

    def run(mode="success"):
        env["MOCK_MODE"] = mode
        result = subprocess.run([BASH, posix(scripts / "replace_pending_matrix.sh"),
                                 "--training-plan", posix(plan), "--training-job", "100", "--test-job", "200"],
                                env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)
        events = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
        return result, events

    return run


def test_accept_both_held_jobs_before_cancelling_old_then_release(replacement):
    result, events = replacement()
    assert result.returncode == 0, result.stdout + result.stderr
    assert events == [
        "query:100", "query:200", "hold:200", "hold:100", "query:100", "query:200",
        "submit-train:40,42,47:held=1", "submit-test:dependency=900:held=1",
        "query:100", "query:200", "cancel:pending:200,100", "query:all", "release:901", "release:900",
    ]
    assert "训练=900" in result.stdout and "全量测试=901" in result.stdout


@pytest.mark.parametrize("mode", ["running_initial", "partial_test"])
def test_preflight_rejects_running_or_partially_finished_test_without_mutations(replacement, mode):
    result, events = replacement(mode)
    assert result.returncode != 0
    assert all(event.startswith("query:") for event in events)


def test_hold_race_does_not_submit_or_cancel_running_task(replacement):
    result, events = replacement("running_after_hold")
    assert result.returncode != 0
    assert "hold:100" in events
    assert not any(event.startswith(("submit-", "cancel:", "release:")) for event in events)


def test_second_submission_failure_keeps_old_and_new_jobs_held(replacement):
    result, events = replacement("submit_failure")
    assert result.returncode != 0
    assert "submit-train:40,42,47:held=1" in events
    assert not any(event.startswith(("cancel:", "release:")) for event in events)
    assert "新训练=900" in result.stderr


def test_old_job_remaining_after_cancel_prevents_releasing_new_jobs(replacement):
    result, events = replacement("cancel_partial")
    assert result.returncode != 0
    assert "cancel:pending:200,100" in events
    assert not any(event.startswith("release:") for event in events)
    assert "新训练=900" in result.stderr and "新测试=901" in result.stderr
