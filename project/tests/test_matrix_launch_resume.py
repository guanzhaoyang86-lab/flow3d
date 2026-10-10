"""A new operational launcher must not silently migrate an old experiment."""
import importlib.util
from pathlib import Path
import shutil

import pytest


def test_resume_freezes_new_launcher_but_keeps_original_science(tmp_path):
    spec = importlib.util.spec_from_file_location("submission_fixture", Path(__file__).with_name("test_submission.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if not fixture.BASH or not fixture.GIT:
        pytest.skip("Git Bash required")
    case = fixture.SubmissionTests()
    case.setUp()
    try:
        shutil.copytree(case.publisher / "scripts", case.publisher / "project/scripts")
        (case.publisher / "scripts/run_hpc_matrix.py").write_text("# planner fixture\n")
        with (case.publisher / "project/scripts/common.sh").open("a", newline="\n") as stream:
            stream.write('\nflow3d_settings() {\n'
                         'export FLOW3D_CLUSTER=deltaai FLOW3D_ACCOUNT=biup-dtai-gh\n'
                         'export FLOW3D_PARTITION=ghx4 FLOW3D_GPUS=1 FLOW3D_CPUS=8 FLOW3D_MEM=64G\n'
                         'mkdir -p "$FLOW3D_ROOT/logs"\n}\n')
        old = case.root / "old-science/project"
        old.mkdir(parents=True)
        (old / ".flow3d-commit").write_text("1" * 40)
        capture_env = case.root / "captured.env"
        case.env.update(OLD_SCIENCE=fixture.posix_path(old), CAPTURE_ENV=fixture.posix_path(capture_env),
                        FLOW3D_ARRAY_TASKS="40,42,47", FLOW3D_TRAIN_TIME="02:00:00",
                        FLOW3D_ARRAY_CONCURRENCY="3", FLOW3D_SUBMIT_HOLD="1")
        case.write_script(case.root / "bin/uname", '#!/bin/bash\necho aarch64\n')
        case.write_script(case.root / "bin/python", '#!/bin/bash\n'
                          'case "$2" in\n'
                          'resume) printf "%s/old-plan.json\\n" "$FLOW3D_ROOT" ;;\n'
                          'inspect) case "$6" in\n'
                          'code_dir) echo "$OLD_SCIENCE" ;;\n'
                          'commit) echo 1111111111111111111111111111111111111111 ;;\n'
                          'count) echo 63 ;; mode) echo matrix-train ;; sha256) echo oldhash ;;\n'
                          'esac ;;\n'
                          'array) echo "$FLOW3D_ARRAY_TASKS" ;; check-time) exit 0 ;;\n'
                          '*) exit 3 ;; esac\n')
        case.write_script(case.root / "bin/sbatch", '#!/bin/bash\n'
                          'printf "%s\\0" "$@" > "$MOCK_SBATCH_CAPTURE"\n'
                          'printf "%s\\n" "$FLOW3D_CODE_DIR" "$FLOW3D_COMMIT" '
                          '"$FLOW3D_LAUNCHER_DIR" "$FLOW3D_LAUNCHER_COMMIT" > "$CAPTURE_ENV"\n'
                          'echo "Submitted batch job 12345"\n')
        case.commit(case.publisher, "updated launcher")
        case.git("push", cwd=case.publisher)
        case.git("pull", "--ff-only", cwd=case.repo)
        result = case.submit("project/scripts/submit.sh", "matrix-resume", ["--plan", "old-plan.json", "--dry-run"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert not case.capture.exists()
        assert not (case.root / "snapshots").exists()
        result = case.submit("project/scripts/submit.sh", "matrix-resume", ["--plan", "old-plan.json"])
        assert result.returncode == 0, result.stdout + result.stderr
        args = case.capture.read_bytes().decode().rstrip("\0").split("\0")
        science, commit, launcher, launcher_commit = capture_env.read_text().splitlines()
        assert science == fixture.posix_path(old)
        assert commit == "1" * 40
        assert launcher != science
        assert launcher_commit == case.git("rev-parse", "HEAD", cwd=case.repo)
        assert fixture.native_path(launcher).joinpath("scripts/retry_srun.py").is_file()
        assert f"--chdir={science}" in args
        assert "--array=40,42,47%3" in args
        assert "--time=02:00:00" in args
        assert "--hold" in args
        assert args[-4] == launcher + "/scripts/matrix.slurm"
        assert args[-1] == "--resume"
        assert (old / ".flow3d-commit").read_text() == "1" * 40
    finally:
        case.tearDown()
