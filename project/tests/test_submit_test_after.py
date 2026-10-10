"""Offline orchestration tests; never contact Slurm, GitHub, or a GPU.

The existing submission tests cover real local-Git snapshot freezing. Here the
planner, freeze entry, Slurm, and flock are mocked to test deferred evaluation
and failure handling. OS flock concurrency itself is not exercised on Windows.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    from .test_submission import BASH, PROJECT, native_path, posix_path
except ImportError:  # unittest discovery with project/tests as its root.
    from test_submission import BASH, PROJECT, native_path, posix_path


@unittest.skipUnless(BASH, "需要 Bash；Windows 需要 Git for Windows")
class SubmitTestAfterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="flow3d-test-after-", dir=PROJECT / "tests")
        self.root = Path(self.temp.name)
        self.project = self.root / "repo" / "project"
        self.snapshot = self.root / "snapshot" / "project"
        self.storage = self.root / "storage"
        self.shim = self.root / "bin"
        for path in (self.project / "scripts", self.snapshot / "scripts", self.storage, self.shim):
            path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / "scripts" / "submit_test_after.sh", self.project / "scripts" / "submit_test_after.sh")
        self.write(self.project / "scripts" / "common.sh", '''#!/bin/bash
flow3d_die() { printf 'flow3d: %s\\n' "$*" >&2; exit 1; }
flow3d_settings() {
    export FLOW3D_CLUSTER=deltaai FLOW3D_ACCOUNT=biup-dtai-gh FLOW3D_PARTITION=ghx4
    export FLOW3D_GPUS=1 FLOW3D_CPUS=8 FLOW3D_MEM=64G
    mkdir -p "$FLOW3D_ROOT/logs" "$FLOW3D_ROOT/results"
}
''')
        self.write(self.project / "scripts" / "prepare_run.sh", '''#!/bin/bash
echo frozen >> "$MOCK_ROOT/freeze.calls"
printf '%s\\n' "$MOCK_SNAPSHOT"
''')
        self.write(self.snapshot / ".flow3d-commit", "b" * 40 + "\n")
        self.write(self.snapshot / "scripts" / "runtime.sh", "export FLOW3D_VISIBLE_GPUS=1\n")
        self.training_plan = self.root / "training.json"
        self.write(self.training_plan, "{}\n")
        for name, body in {
            "git": "printf '%040d\\n' 1",
            "uname": "echo aarch64",
            "flock": "exit 0",
            "sbatch": '''printf '%s\\0' "$@" > "$MOCK_ROOT/sbatch.args"
printf '%s\\n' "$FLOW3D_EVAL_COORD" > "$MOCK_ROOT/coord.txt"
export -p | grep ' FLOW3D_' > "$MOCK_ROOT/submitted.env"
printf '888000;deltaai\\n' ''',
            "python": '''case "$1" in
*/retry_srun.py)
    printf '%s\\0' "$@" >> "$MOCK_ROOT/retry.calls"
    printf '%s\\n' "$FLOW3D_EVALUATE_BUDGET_SECONDS" >> "$MOCK_ROOT/budgets.txt"
    exit 0 ;;
esac
case "$2" in
time-budget) echo 5100 ;;
plan)
    echo plan >> "$MOCK_ROOT/plan.calls"
    if [[ -f "$MOCK_ROOT/fail_plan" ]]; then
        echo 'training incomplete' >&2
        exit 1
    fi
    printf '%s\\n' "$FLOW3D_ROOT/unified-plan.json" ;;
inspect)
    case "${@: -1}" in
        mode) printf '%s\\n' "${MOCK_PLAN_MODE:-matrix-train}" ;;
        count) printf '%s\\n' "${MOCK_TASK_COUNT:-63}" ;;
        sha256)
            if [[ "$*" = *training.json* && -f "$MOCK_ROOT/change_training" ]]; then
                printf '%064d\\n' 8
            elif [[ "$*" = *unified-plan.json* && -f "$MOCK_ROOT/change_evaluation" ]]; then
                printf '%064d\\n' 9
            else
                printf '%064d\\n' 7
            fi ;;
        *) exit 98 ;;
    esac ;;
*) exit 99 ;;
esac''',
        }.items():
            self.write(self.shim / name, "#!/bin/bash\n" + body + "\n")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("FLOW3D_")}
        self.env.update({
            "FLOW3D_ROOT": posix_path(self.storage),
            "MOCK_ROOT": posix_path(self.root),
            "MOCK_SNAPSHOT": posix_path(self.snapshot),
            "TEST_SHIM": posix_path(self.shim),
            "MSYS_NO_PATHCONV": "1",
        })

    def tearDown(self) -> None:
        if not self.root.resolve().is_relative_to((PROJECT / "tests").resolve()):
            raise RuntimeError("拒绝清理测试目录以外的路径")
        for path in self.root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    @staticmethod
    def write(path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o700)

    def run_bash(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, "--noprofile", "--norc", "-c", 'export PATH="$TEST_SHIM:$PATH"; ' + script, "test-after", *args],
            cwd=self.project, env=self.env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )

    def submit(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.run_bash('exec bash scripts/submit_test_after.sh "$@"', "--training-plan", posix_path(self.training_plan), *args)

    def options(self) -> list[str]:
        return (self.root / "sbatch.args").read_bytes().decode().rstrip("\0").split("\0")

    def coordinator(self) -> Path:
        return native_path((self.root / "coord.txt").read_text().strip())

    def run_task(self) -> subprocess.CompletedProcess[str]:
        return self.run_bash('source "$1"; exec bash "$2"', posix_path(self.root / "submitted.env"), posix_path(self.coordinator() / "evaluate.slurm"))

    def test_dependency_freezes_once_and_defers_one_shared_plan(self) -> None:
        result = self.submit("--dependency", "3351820:1234567")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines()[-1], "Submitted batch job 888000")
        args = self.options()
        self.assertIn("--dependency=afterok:3351820:1234567", args)
        self.assertIn("--kill-on-invalid-dep=yes", args)
        self.assertIn("--array=0-62%16", args)
        self.assertIn("--time=01:30:00", args)
        self.assertIn("--gpus-per-node=1", args)
        self.assertEqual((self.root / "freeze.calls").read_text().splitlines(), ["frozen"])
        self.assertFalse((self.root / "plan.calls").exists())
        for _ in range(2):
            result = self.run_task()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "plan.calls").read_text().splitlines(), ["plan"])
        calls = (self.root / "retry.calls").read_bytes().decode()
        self.assertEqual(calls.count("retry_srun.py"), 2)
        self.assertEqual(calls.count("run_hpc_matrix.py"), 2)
        self.assertNotIn("--resume", calls)
        self.assertTrue(all(0 < int(value) <= 5100 for value in (self.root / "budgets.txt").read_text().splitlines()))
        self.assertEqual((self.coordinator() / "job_id.txt").read_text().strip(), "888000")

    def test_no_dependency_validates_training_before_submission(self) -> None:
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(arg.startswith("--dependency") for arg in self.options()))
        self.assertEqual((self.root / "plan.calls").read_text().splitlines(), ["plan"])

    def test_incomplete_training_without_dependency_does_not_submit(self) -> None:
        self.write(self.root / "fail_plan", "")
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("training incomplete", result.stderr)
        self.assertFalse((self.root / "sbatch.args").exists())
        self.assertEqual(len(list((self.storage / "results").glob("evaluate-all.*/plan.failed"))), 1)

    def test_deferred_planning_failure_is_shared_and_not_retried(self) -> None:
        result = self.submit("--dependency", "3351820")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write(self.root / "fail_plan", "")
        for _ in range(2):
            result = self.run_task()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("training incomplete", result.stderr)
        self.assertEqual((self.root / "plan.calls").read_text().splitlines(), ["plan"])
        self.assertFalse((self.root / "retry.calls").exists())

    def test_training_plan_change_after_submission_is_rejected(self) -> None:
        result = self.submit("--dependency", "3351820")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write(self.root / "change_training", "")
        result = self.run_task()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "plan.calls").exists())
        self.assertTrue((self.coordinator() / "plan.failed").exists())

    def test_evaluation_plan_change_is_rejected(self) -> None:
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write(self.root / "change_evaluation", "")
        result = self.run_task()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "retry.calls").exists())

    def test_dry_run_does_not_freeze_submit_or_create_coordinator(self) -> None:
        result = self.submit("--dependency", "3351820", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--dependency=afterok:3351820", result.stdout)
        self.assertFalse((self.root / "freeze.calls").exists())
        self.assertFalse((self.root / "sbatch.args").exists())
        self.assertFalse((self.root / "plan.calls").exists())
        self.assertFalse(list((self.storage / "results").glob("evaluate-all.*")))

    def test_explicit_settings_and_hold_are_forwarded(self) -> None:
        self.env.update({"FLOW3D_EVALUATE_TIME": "02:00:00", "FLOW3D_ARRAY_CONCURRENCY": "3", "FLOW3D_SUBMIT_HOLD": "1", "MOCK_TASK_COUNT": "12"})
        result = self.submit("--dependency", "3351820")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--time=02:00:00", self.options())
        self.assertIn("--array=0-11%3", self.options())
        self.assertIn("--hold", self.options())

    def test_invalid_dependency_or_plan_mode_is_rejected(self) -> None:
        for dependency in ("", "12,34", "afterok:12", "12:foo", "0"):
            result = self.submit("--dependency", dependency)
            self.assertNotEqual(result.returncode, 0, dependency)
            self.assertFalse((self.root / "sbatch.args").exists())
        self.env["MOCK_PLAN_MODE"] = "matrix-evaluate"
        result = self.submit("--dependency", "3351820")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "sbatch.args").exists())


if __name__ == "__main__":
    unittest.main()
