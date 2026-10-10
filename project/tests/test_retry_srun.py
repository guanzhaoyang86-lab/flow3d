"""Exercise Slurm launch retry without a cluster or GPU.

Only the srun executable is replaced. Its subprocess runs the real marker
wrapper and a real Python payload, including on Windows development machines.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
HELPER = PROJECT / "scripts" / "retry_srun.py"
SPEC = importlib.util.spec_from_file_location("flow3d_retry_srun_test", HELPER)
assert SPEC is not None and SPEC.loader is not None
retry_srun = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry_srun)


FAKE_SRUN = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ["MOCK_LAUNCH_ROOT"])
count_file = root / "count.txt"
count = int(count_file.read_text()) + 1 if count_file.exists() else 1
count_file.write_text(str(count))
arguments = sys.argv[1:]
with (root / "srun-args.jsonl").open("a") as stream:
    stream.write(json.dumps(arguments) + "\n")
assert arguments[:2] == ["--ntasks=1", "--gpu-bind=none"], arguments
mode = os.environ.get("MOCK_LAUNCH_MODE", "success")
failures = int(os.environ.get("MOCK_LAUNCH_FAILURES", "0"))
if mode == "cancelled":
    print("srun: error: task 0 launch failed: Error configuring interconnect", file=sys.stderr)
    print("slurmstepd: error: JOB CANCELLED DUE TO TIME LIMIT", file=sys.stderr)
    sys.exit(1)
if mode == "signal-exit":
    print("srun: error: task 0 launch failed: Error configuring interconnect", file=sys.stderr)
    sys.exit(143)
if mode == "generic-error":
    print("srun: error: Unable to create step: Invalid job id specified", file=sys.stderr)
    sys.exit(1)
if count <= failures:
    print("srun: error: task 0 launch failed: Error configuring interconnect", file=sys.stderr)
    sys.exit(1)
if os.name == "nt":
    # Windows CRT execvp is not POSIX exec: it can lose argument quoting and
    # return before the replacement process. Emulate only that OS operation.
    sys.exit(subprocess.call([sys.executable, str(root / "posix_exec.py"), *arguments[3:]]))
sys.exit(subprocess.call(arguments[2:]))
'''


POSIX_EXEC_ADAPTER = r'''
import importlib.util
import subprocess
import sys

spec = importlib.util.spec_from_file_location("retry_helper_child", sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
def execvp(executable, arguments):
    raise SystemExit(subprocess.call(arguments))
helper.os.execvp = execvp
sys.exit(helper.main(sys.argv[2:]))
'''


PAYLOAD = r'''
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["MOCK_LAUNCH_ROOT"])
(root / "payload.json").write_text(json.dumps({
    "args": sys.argv[1:],
    "budget": os.environ.get("FLOW3D_EVALUATE_BUDGET_SECONDS"),
}))
message = os.environ.get("MOCK_PAYLOAD_STDERR", "")
if message:
    print(message, file=sys.stderr, flush=True)
sys.exit(int(os.environ.get("MOCK_PAYLOAD_EXIT", "0")))
'''


class RetrySrunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="flow3d-retry-")
        self.root = Path(self.temp.name)
        self.storage = self.root / "shared storage"
        self.fake = self.root / "fake_srun.py"
        self.payload = self.root / "payload.py"
        self.fake.write_text(FAKE_SRUN, encoding="utf-8")
        self.payload.write_text(PAYLOAD, encoding="utf-8")
        (self.root / "posix_exec.py").write_text(POSIX_EXEC_ADAPTER, encoding="utf-8")
        self.env = {
            "SLURM_JOB_ID": "3351820", "SLURM_ARRAY_TASK_ID": "40",
            "FLOW3D_ROOT": str(self.storage),
            "FLOW3D_LAUNCH_RETRIES": "3", "FLOW3D_LAUNCH_RETRY_DELAY": "0",
            "MOCK_LAUNCH_ROOT": str(self.root),
        }
        self.real_popen = subprocess.Popen
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def popen(self, command, *args, **kwargs):
        self.assertEqual(command[0], "srun")
        return self.real_popen([sys.executable, str(self.fake), *command[1:]], *args, **kwargs)

    def run_helper(self, options=(), arguments=("argument with spaces", "雪"), unset=()) -> int:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("FLOW3D_", "SLURM_", "MOCK_"))}
        env.update(self.env)
        for key in unset:
            env.pop(key, None)
        with patch.dict(os.environ, env, clear=True), \
                patch.object(retry_srun.subprocess, "Popen", side_effect=self.popen), \
                contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            return retry_srun.main([*options, "--", sys.executable, str(self.payload), *arguments])

    def count(self) -> int:
        path = self.root / "count.txt"
        return int(path.read_text()) if path.exists() else 0

    def records(self) -> tuple[Path, list[dict]]:
        directories = list((self.storage / "logs").glob("launch-*"))
        self.assertEqual(len(directories), 1)
        records = [json.loads(line) for line in (directories[0] / "attempts.jsonl").read_text().splitlines()]
        return directories[0], records

    def test_success_once_preserves_arguments_and_records_start(self) -> None:
        self.assertEqual(self.run_helper(), 0, self.stderr.getvalue())
        self.assertEqual(self.count(), 1)
        self.assertEqual(json.loads((self.root / "payload.json").read_text())["args"],
                         ["argument with spaces", "雪"])
        directory, records = self.records()
        self.assertTrue(records[0]["payload_started"])
        self.assertFalse(records[0]["will_retry"])
        launcher = json.loads((directory / "launcher.json").read_text())
        self.assertEqual(launcher["array_task_id"], "40")
        self.assertEqual(launcher["max_extra_retries"], 3)

    def test_three_launch_failures_then_success(self) -> None:
        self.env["MOCK_LAUNCH_FAILURES"] = "3"
        self.assertEqual(self.run_helper(), 0, self.stderr.getvalue())
        self.assertEqual(self.count(), 4)
        directory, records = self.records()
        self.assertEqual([record["payload_started"] for record in records], [False, False, False, True])
        self.assertEqual([record["will_retry"] for record in records], [True, True, True, False])
        self.assertIn("Error configuring interconnect", (directory / "attempt-1.stderr").read_text())
        self.assertIn("retrying", self.stderr.getvalue())

    def test_launch_failures_exhaust_three_extra_attempts(self) -> None:
        self.env["MOCK_LAUNCH_FAILURES"] = "9"
        self.assertEqual(self.run_helper(), 1)
        self.assertEqual(self.count(), 4)
        self.assertFalse((self.root / "payload.json").exists())
        self.assertFalse(self.records()[1][-1]["will_retry"])

    def test_retries_can_be_disabled(self) -> None:
        self.env["MOCK_LAUNCH_FAILURES"] = "9"
        self.assertEqual(self.run_helper(options=("--retries", "0")), 1)
        self.assertEqual(self.count(), 1)

    def test_generic_launch_failure_is_not_retried(self) -> None:
        self.env["MOCK_LAUNCH_MODE"] = "generic-error"
        self.assertEqual(self.run_helper(), 1)
        self.assertEqual(self.count(), 1)

    def test_started_python_error_is_not_retried(self) -> None:
        self.env.update(MOCK_PAYLOAD_EXIT="1", MOCK_PAYLOAD_STDERR="RuntimeError: CUDA out of memory")
        self.assertEqual(self.run_helper(), 1)
        self.assertEqual(self.count(), 1)
        self.assertIn("CUDA out of memory", self.stderr.getvalue())
        self.assertTrue(self.records()[1][0]["payload_started"])

    def test_payload_cannot_trigger_retry_by_printing_matching_error(self) -> None:
        self.env.update(MOCK_PAYLOAD_EXIT="1", MOCK_PAYLOAD_STDERR=
                        "srun: error: task 0 launch failed: Error configuring interconnect")
        self.assertEqual(self.run_helper(), 1)
        self.assertEqual(self.count(), 1)
        self.assertTrue(self.records()[1][0]["payload_started"])

    def test_cancellation_message_prevents_retry(self) -> None:
        self.env["MOCK_LAUNCH_MODE"] = "cancelled"
        self.assertEqual(self.run_helper(), 1)
        self.assertEqual(self.count(), 1)

    def test_signal_exit_prevents_retry(self) -> None:
        self.env["MOCK_LAUNCH_MODE"] = "signal-exit"
        self.assertEqual(self.run_helper(), 143)
        self.assertEqual(self.count(), 1)

    def test_partial_evaluation_exit_75_is_preserved(self) -> None:
        self.env["MOCK_PAYLOAD_EXIT"] = "75"
        self.assertEqual(self.run_helper(), 75)
        self.assertEqual(self.count(), 1)

    def test_requires_allocation_and_absolute_shared_storage(self) -> None:
        for variable in ("SLURM_JOB_ID", "FLOW3D_ROOT"):
            with self.subTest(variable=variable), self.assertRaises(SystemExit) as error:
                self.run_helper(unset=(variable,))
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.count(), 0)

    def test_invalid_limits_fail_before_launch(self) -> None:
        for options in (("--retries", "-1"), ("--retries", "4"), ("--delay", "-1"),
                        ("--delay", "nan"), ("--delay", "inf"), ("--delay", "301")):
            with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                self.run_helper(options=options)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.count(), 0)

    def test_evaluation_budget_is_reduced_by_launch_overhead(self) -> None:
        self.env["FLOW3D_EVALUATE_BUDGET_SECONDS"] = "60"
        self.assertEqual(self.run_helper(), 0, self.stderr.getvalue())
        remaining = int(json.loads((self.root / "payload.json").read_text())["budget"])
        self.assertGreater(remaining, 0)
        self.assertLess(remaining, 60)

    def test_exhausted_budget_does_not_run_payload_or_retry(self) -> None:
        self.env["FLOW3D_EVALUATE_BUDGET_SECONDS"] = "0"
        self.assertEqual(self.run_helper(), 75)
        self.assertEqual(self.count(), 1)
        self.assertFalse((self.root / "payload.json").exists())
        self.assertTrue(self.records()[1][0]["payload_started"])


if __name__ == "__main__":
    unittest.main()
