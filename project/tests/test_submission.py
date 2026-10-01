"""离线验证 GitHub 同步和 Slurm 提交；不连接 HPC 或消耗 GPU 配额。"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def find_bash() -> str | None:
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
            if candidate.exists():
                return str(candidate)
        return None  # WindowsApps/bash.exe may launch an unconfigured WSL.
    return shutil.which("bash")


BASH = find_bash()
GIT = shutil.which("git")


def posix_path(path: Path | str) -> str:
    path = Path(path).resolve()
    if os.name == "nt":
        return f"/{path.drive[0].lower()}{path.as_posix()[2:]}"
    return str(path)


def native_path(path: str) -> Path:
    if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == "/":
        return Path(f"{path[1]}:{path[2:]}")
    return Path(path)


@unittest.skipUnless(BASH and GIT, "需要 Git 和 Bash；Windows 需要 Git for Windows")
class SubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="flow3d-submit-", dir=PROJECT / "tests")
        self.root = Path(self.temp.name)
        self.publisher = self.root / "publisher"
        self.repo = self.root / "hpc"
        self.remote = self.root / "remote.git"
        self.capture = self.root / "sbatch.args"
        self.publisher.mkdir()
        self.git("init", "--bare", str(self.remote), cwd=self.root)
        self.git("init", "-b", "main", cwd=self.publisher)
        self.identity(self.publisher)
        shutil.copytree(PROJECT / "scripts", self.publisher / "scripts")
        # Fixture-only storage adapter: production /scratch and /work paths
        # are unavailable on a developer PC. Keep all writes in this temp dir.
        common = self.publisher / "scripts" / "common.sh"
        with common.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write('\nflow3d_settings() {\n'
                         '  export FLOW3D_CLUSTER=delta FLOW3D_ACCOUNT=PHY260443\n'
                         '  export FLOW3D_PARTITION=gpuA100x4 FLOW3D_GPUS=${FLOW3D_GPUS:-2}\n'
                         '  export FLOW3D_TIME=00:05:00 FLOW3D_CPUS=8 FLOW3D_MEM=32G\n'
                         '  mkdir -p "$FLOW3D_ROOT/logs"\n'
                         '}\n')
        (self.publisher / "payload.txt").write_text("version one\n", encoding="utf-8")
        self.commit(self.publisher, "initial")
        self.git("remote", "add", "origin", str(self.remote), cwd=self.publisher)
        self.git("push", "-u", "origin", "main", cwd=self.publisher)
        self.git("-c", "core.autocrlf=false", "clone", "--branch", "main", str(self.remote), str(self.repo), cwd=self.root)
        self.identity(self.repo)
        shim = self.root / "bin"
        shim.mkdir()
        # Report a GitHub code remote to the production URL guard while all
        # Git transport remains on a local bare repository.
        self.write_script(shim / "git", '#!/bin/bash\n'
                          'if [[ "${1:-}" = remote && "${2:-}" = get-url ]]; then\n'
                          '  printf "https://github.com/fixture/flow3d.git\\n"\n'
                          '  exit 0\n'
                          'fi\n'
                          'if [[ "${1:-}" = pull && "${MOCK_PULL_FAIL:-0}" = 1 ]]; then\n'
                          '  printf "simulated network failure\\n" >&2; exit 42\n'
                          'fi\n'
                          'exec "$REAL_GIT" "$@"\n')
        self.write_script(shim / "sbatch", '#!/bin/bash\n'
                          'printf "%s\\0" "$@" > "$MOCK_SBATCH_CAPTURE"\n'
                          'printf "Submitted batch job 12345\\n"\n')
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(shim) + os.pathsep + self.env.get("PATH", ""),
            "REAL_GIT": posix_path(GIT),
            "FLOW3D_ROOT": posix_path(self.root / "storage"),
            "FLOW3D_GPUS": "2",
            "FLOW3D_SNAPSHOT_ROOT": posix_path(self.root / "snapshots"),
            "MOCK_SBATCH_CAPTURE": posix_path(self.capture),
            "FLOW3D_TEST_SHIM": posix_path(shim),
            "MSYS_NO_PATHCONV": "1",
        })
        for variable in ("FLOW3D_CODE_DIR", "FLOW3D_COMMIT", "FLOW3D_GPU_TYPE", "FLOW3D_CONSTRAINT"):
            self.env.pop(variable, None)

    def tearDown(self) -> None:
        # prepare_run intentionally makes snapshots read-only. Restore only
        # files inside this test's temporary directory before its cleanup.
        if not self.root.resolve().is_relative_to((PROJECT / "tests").resolve()):
            raise RuntimeError("拒绝清理测试目录以外的路径")
        for path in self.root.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        for attempt in range(5):
            try:
                self.temp.cleanup()
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))

    @staticmethod
    def write_script(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o700)

    @staticmethod
    def git(*args: str, cwd: Path) -> str:
        result = subprocess.run([GIT, *args], cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def identity(self, path: Path) -> None:
        self.git("config", "user.name", "Flow3D Test", cwd=path)
        self.git("config", "user.email", "flow3d-test@example.invalid", cwd=path)
        self.git("config", "core.autocrlf", "false", cwd=path)
        self.git("config", "gc.auto", "0", cwd=path)

    def commit(self, path: Path, message: str) -> None:
        self.git("add", ".", cwd=path)
        self.git("commit", "-m", message, cwd=path)

    def submit(self, script="scripts/submit.sh", mode="train", arguments=None) -> subprocess.CompletedProcess[str]:
        if arguments is None:
            arguments = ["--name", "two words"]
        return subprocess.run([BASH, "-c", 'export PATH="$FLOW3D_TEST_SHIM:$PATH"; exec bash "$@"',
                               "flow3d-test", script, mode, *arguments],
                              cwd=self.repo, env=self.env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    def test_latest_commit_snapshot_and_submission_arguments(self) -> None:
        (self.publisher / "payload.txt").write_text("version two\n", encoding="utf-8")
        self.commit(self.publisher, "published update")
        self.git("push", cwd=self.publisher)
        expected_commit = self.git("rev-parse", "HEAD", cwd=self.publisher)
        result = self.submit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.capture.read_bytes().decode("utf-8").rstrip("\0").split("\0")
        self.assertIn("--account=PHY260443", args)
        self.assertIn("--gpus-per-node=2", args)
        self.assertIn("--time=00:05:00", args)
        self.assertEqual(args[-2:], ["--name", "two words"])
        snapshot = native_path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir=")))
        self.assertEqual((snapshot / ".flow3d-commit").read_text().strip(), expected_commit)
        self.assertEqual((snapshot / "payload.txt").read_text(), "version two\n")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.repo), expected_commit)
        (self.repo / "payload.txt").write_text("changed after queueing\n", encoding="utf-8")
        self.assertEqual((snapshot / "payload.txt").read_text(), "version two\n")

    def test_dirty_working_tree_is_rejected(self) -> None:
        (self.repo / "payload.txt").write_text("uncommitted\n", encoding="utf-8")
        self.assertNotEqual(self.submit().returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_workflow_subdirectory_keeps_full_repository_snapshot(self) -> None:
        shutil.copytree(self.publisher / "scripts", self.publisher / "project" / "scripts")
        self.commit(self.publisher, "add workflow subdirectory")
        self.git("push", cwd=self.publisher)
        self.git("pull", "--ff-only", cwd=self.repo)
        result = self.submit("project/scripts/submit.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.capture.read_bytes().decode("utf-8").rstrip("\0").split("\0")
        snapshot = native_path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir=")))
        self.assertEqual(snapshot.name, "project")
        self.assertEqual((snapshot.parent / "payload.txt").read_text(), "version one\n")
        self.assertEqual((snapshot / ".flow3d-commit").read_text().strip(),
                         self.git("rev-parse", "HEAD", cwd=self.publisher))

    def test_unpublished_commit_is_rejected(self) -> None:
        (self.repo / "payload.txt").write_text("not pushed\n", encoding="utf-8")
        self.commit(self.repo, "HPC-only change")
        self.assertNotEqual(self.submit().returncode, 0)
        self.assertFalse(self.capture.exists())

    def test_diffusion_uses_real_entry_in_complete_snapshot(self) -> None:
        shutil.copytree(self.publisher / "scripts", self.publisher / "project" / "scripts")
        (self.publisher / "scripts" / "run_hpc_diffusion.py").write_text("# research entry fixture\n")
        self.commit(self.publisher, "add research entry")
        self.git("push", cwd=self.publisher)
        self.git("pull", "--ff-only", cwd=self.repo)
        self.env["FLOW3D_GPUS"] = "1"
        result = self.submit("project/scripts/submit.sh", "diffusion-smoke", [])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.capture.read_bytes().decode("utf-8").rstrip("\0").split("\0")
        self.assertIn("--gpus-per-node=1", args)
        self.assertTrue(args[-2].endswith("/scripts/diffusion.slurm"))
        self.assertEqual(args[-1], "smoke")
        snapshot = native_path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir=")))
        self.assertTrue((snapshot.parent / "scripts" / "run_hpc_diffusion.py").is_file())

    def test_diffusion_rejects_multiple_gpus_before_submission(self) -> None:
        result = self.submit(mode="diffusion-train", arguments=[])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FLOW3D_GPUS=1", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_generation_uses_frozen_entry_and_forwards_mode(self) -> None:
        shutil.copytree(self.publisher / "scripts", self.publisher / "project" / "scripts")
        (self.publisher / "scripts" / "run_hpc_generation.py").write_text("# generation fixture\n")
        self.commit(self.publisher, "add generation entry")
        self.git("push", cwd=self.publisher)
        self.git("pull", "--ff-only", cwd=self.repo)
        upstream = self.root / "upstream" / "Single_phase"
        upstream.mkdir(parents=True)
        (upstream / "LBM_3D_SinglePhase_Solver.py").write_text("# upstream fixture\n")
        self.env["FLOW3D_UPSTREAM_REPO"] = posix_path(upstream.parent)
        self.env["FLOW3D_GPUS"] = "1"
        result = self.submit("project/scripts/submit.sh", "generate-pilot", [])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.capture.read_bytes().decode("utf-8").rstrip("\0").split("\0")
        self.assertIn("--gpus-per-node=1", args)
        self.assertTrue(args[-2].endswith("/scripts/generate.slurm"))
        self.assertEqual(args[-1], "pilot")
        snapshot = native_path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--chdir=")))
        self.assertTrue((snapshot.parent / "scripts" / "run_hpc_generation.py").is_file())

    def test_generation_rejects_multiple_gpus_before_submission(self) -> None:
        result = self.submit(mode="generate-full", arguments=[])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FLOW3D_GPUS=1", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_generation_rejects_deltaai_before_solver_lookup(self) -> None:
        self.env["FLOW3D_GPUS"] = "1"
        self.write_script(self.root / "bin" / "uname", '#!/bin/bash\nprintf "aarch64\\n"\n')
        result = self.submit(mode="generate-full", arguments=[])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dt-login", result.stderr)
        self.assertNotIn("setup_delta_generation.sh", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_network_failure_prevents_submission(self) -> None:
        self.env["MOCK_PULL_FAIL"] = "1"
        result = self.submit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated network failure", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
