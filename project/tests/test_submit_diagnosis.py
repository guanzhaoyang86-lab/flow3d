"""Offline GH200 diagnostic submission checks with local Git and mocked Slurm."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    from .test_submission import BASH, GIT, PROJECT, native_path, posix_path
except ImportError:
    from test_submission import BASH, GIT, PROJECT, native_path, posix_path


@unittest.skipUnless(BASH and GIT, "Requires Bash and Git; Git for Windows is supported")
class SubmitDiagnosisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="flow3d-diagnosis-", dir=PROJECT / "tests")
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.project = self.repo / "project"
        self.storage = self.root / "storage"
        self.shim = self.root / "bin"
        (self.project / "scripts").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        self.shim.mkdir()
        for name in ("submit_diagnosis.sh", "diagnose.slurm", "prepare_run.sh", "common.sh", "retry_srun.py"):
            shutil.copyfile(PROJECT / "scripts" / name, self.project / "scripts" / name)
        # Only the fixture permits a developer-local storage path.
        common = self.project / "scripts" / "common.sh"
        with common.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write('\nflow3d_settings() { mkdir -p "$FLOW3D_ROOT/logs"; }\n')
        self.write(self.project / "scripts" / "runtime.sh", "export FLOW3D_VISIBLE_GPUS=1\n")
        self.write(self.repo / "scripts" / "diagnose_diffusion_sampling.py", "# frozen runner fixture\n")
        self.manifest = self.root / "manifest.json"
        self.checkpoint = self.root / "checkpoint with spaces.pt"
        self.write(self.manifest, "{}\n")
        self.write(self.checkpoint, "unchanged checkpoint\n")
        self.git("init", "-b", "main", cwd=self.repo)
        self.git("config", "user.name", "Diagnosis Test", cwd=self.repo)
        self.git("config", "user.email", "diagnosis@example.invalid", cwd=self.repo)
        self.git("config", "core.autocrlf", "false", cwd=self.repo)
        self.git("add", ".", cwd=self.repo)
        self.git("commit", "-m", "diagnostic fixture", cwd=self.repo)
        remote = self.root / "remote.git"
        self.git("init", "--bare", str(remote), cwd=self.root)
        self.git("remote", "add", "origin", str(remote), cwd=self.repo)
        self.git("push", "-u", "origin", "main", cwd=self.repo)
        for name, body in {
            "git": '''if [[ "${1:-}" = remote && "${2:-}" = get-url ]]; then
    echo https://github.com/fixture/flow3d.git
    exit 0
fi
exec "$REAL_GIT" "$@"''',
            "uname": "echo aarch64",
            "sbatch": '''printf '%s\\0' "$@" > "$MOCK_ROOT/sbatch.args"
printf '%s\\n' "$FLOW3D_DIAGNOSIS_RUN_DIR" > "$MOCK_ROOT/run_dir.txt"
export -p | grep ' FLOW3D_' > "$MOCK_ROOT/submitted.env"
printf '%s\\n' "${!SBATCH_@}" > "$MOCK_ROOT/ambient_sbatch.txt"
echo '888001;deltaai' ''',
            "python": '''printf '%s\\0' "$@" > "$MOCK_ROOT/python.args"
exit "${MOCK_JOB_EXIT:-0}"''',
        }.items():
            self.write(self.shim / name, "#!/bin/bash\n" + body + "\n")
        self.env = {key: value for key, value in os.environ.items() if not key.startswith(("FLOW3D_", "SBATCH_"))}
        self.env.update({
            "FLOW3D_ROOT": posix_path(self.storage),
            "FLOW3D_SNAPSHOT_ROOT": posix_path(self.root / "snapshots"),
            "REAL_GIT": posix_path(GIT),
            "MOCK_ROOT": posix_path(self.root),
            "TEST_SHIM": posix_path(self.shim),
            "MSYS_NO_PATHCONV": "1",
        })

    @staticmethod
    def write(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", newline="\n")
        path.chmod(0o700)

    @staticmethod
    def git(*args: str, cwd: Path) -> str:
        result = subprocess.run([GIT, *args], cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def tearDown(self) -> None:
        if not self.root.resolve().is_relative_to((PROJECT / "tests").resolve()):
            raise RuntimeError("Refusing to clean paths outside the test workspace")
        for path in self.root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    def run_bash(self, command: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, "--noprofile", "--norc", "-c", 'export PATH="$TEST_SHIM:$PATH"; ' + command, "diagnosis-test", *args],
            cwd=self.repo, env=self.env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )

    def submit(self, *args: str, inputs: bool = True) -> subprocess.CompletedProcess[str]:
        defaults = ("--manifest", posix_path(self.manifest), "--checkpoint", posix_path(self.checkpoint)) if inputs else ()
        return self.run_bash('exec bash project/scripts/submit_diagnosis.sh "$@"', *defaults, *args)

    def options(self) -> list[str]:
        return (self.root / "sbatch.args").read_bytes().decode().rstrip("\0").split("\0")

    def run_dir(self) -> Path:
        return native_path((self.root / "run_dir.txt").read_text().strip())

    def test_dry_run_needs_no_inputs_or_hpc_and_creates_nothing(self) -> None:
        self.write(self.shim / "uname", "#!/bin/bash\necho x86_64\n")
        result = self.submit("--dry-run", inputs=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--time=00:30:00", result.stdout)
        self.assertIn("dit3d_N002_s31/best.pt", result.stdout)
        self.assertIn("tensor-dit_N002_s31/best.pt", result.stdout)
        self.assertFalse(self.storage.exists())
        self.assertFalse((self.root / "snapshots").exists())
        self.assertFalse((self.root / "sbatch.args").exists())

    def test_frozen_single_job_clears_old_array_dependency_and_settings(self) -> None:
        self.env.update({
            "SBATCH_DEPENDENCY": "afterok:100",
            "SBATCH_ARRAY_INX": "0-62",
            "SBATCH_HOLD": "1",
            "FLOW3D_CLUSTER": "delta",
            "FLOW3D_TIME": "48:00:00",
            "FLOW3D_GPUS": "4",
            "FLOW3D_EVALUATE_BUDGET_SECONDS": "1",
        })
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.options()
        for option in ("--account=biup-dtai-gh", "--partition=ghx4", "--gpus-per-node=1", "--cpus-per-task=8", "--mem=64G", "--time=00:30:00", "--no-requeue"):
            self.assertIn(option, args)
        self.assertFalse(any(arg.startswith(("--array", "--dependency", "--hold")) for arg in args))
        self.assertEqual((self.root / "ambient_sbatch.txt").read_text().strip(), "")
        snapshot = native_path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir=")))
        commit = self.git("rev-parse", "HEAD", cwd=self.repo)
        self.assertEqual((snapshot / ".flow3d-commit").read_text().strip(), commit)
        self.assertEqual((self.run_dir() / "commit.txt").read_text().strip(), commit)
        self.assertTrue((snapshot.parent / "scripts" / "diagnose_diffusion_sampling.py").is_file())
        self.assertFalse((self.run_dir() / "diagnosis").exists())
        self.assertEqual((self.run_dir() / "job_id.txt").read_text().strip(), "888001")
        self.assertIn(posix_path(self.checkpoint), args)
        environment = (self.root / "submitted.env").read_text()
        self.assertIn("python/miniforge3_pytorch/2.10.0", environment)
        self.assertNotIn("FLOW3D_EVALUATE_BUDGET_SECONDS", environment)
        self.assertEqual(self.checkpoint.read_text(), "unchanged checkpoint\n")
        self.write(self.repo / "scripts" / "diagnose_diffusion_sampling.py", "changed after submission\n")
        self.assertEqual((snapshot.parent / "scripts" / "diagnose_diffusion_sampling.py").read_text(), "# frozen runner fixture\n")

    def test_repeated_checkpoints_and_custom_indices_reach_retry_wrapper(self) -> None:
        result = self.submit("--checkpoint", posix_path(self.manifest), "--indices", "1,2", "--time", "01:00:00")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.options()
        script_index = next(i for i, arg in enumerate(args) if arg.endswith("/diagnose.slurm"))
        self.env.update({"SLURM_JOB_ID": "888001", "MOCK_JOB_EXIT": "9"})
        result = self.run_bash('source "$1"; shift; exec bash "$@"', posix_path(self.root / "submitted.env"), *args[script_index:])
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertEqual((self.run_dir() / "exit_code.txt").read_text().strip(), "9")
        payload = (self.root / "python.args").read_bytes().decode().rstrip("\0").split("\0")
        self.assertTrue(payload[0].endswith("/retry_srun.py"))
        self.assertTrue(payload[3].endswith("/diagnose_diffusion_sampling.py"))
        self.assertEqual(payload.count("--checkpoint"), 2)
        self.assertEqual(payload[payload.index("--indices") + 1], "1,2")
        self.assertEqual(payload[payload.index("--device") + 1], "cuda")

    def test_each_submission_uses_a_new_result_directory(self) -> None:
        self.assertEqual(self.submit().returncode, 0)
        first = self.run_dir()
        self.assertEqual(self.submit().returncode, 0)
        self.assertNotEqual(first, self.run_dir())
        self.assertTrue((first / "job_id.txt").exists())

    def test_missing_input_and_dirty_source_do_not_submit(self) -> None:
        result = self.submit("--manifest", posix_path(self.root / "missing.json"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "sbatch.args").exists())
        self.write(self.repo / "scripts" / "diagnose_diffusion_sampling.py", "dirty\n")
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "sbatch.args").exists())

    def test_invalid_arguments_never_submit(self) -> None:
        for args in (("--time", "01:00:01"), ("--time", "00:00:00"), ("--time", "48:00:00"), ("--indices", "0,abc"), ("--sampling-steps", "0"), ("--num-probe-particles", "-1"), ("--seed", "-2"), ("--manifest",)):
            with self.subTest(args=args):
                result = self.submit(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "sbatch.args").exists())


if __name__ == "__main__":
    unittest.main()
