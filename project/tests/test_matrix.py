"""Offline matrix/provenance tests; never allocates Slurm or a GPU."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hpc_matrix as matrix


def plan_args(tmp_path, **overrides):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mode": "scientific", "splits": {
        "train": [{"case_id": str(i)} for i in range(798)],
        "validation": [{"case_id": str(i)} for i in range(101)],
        "test": [{"case_id": str(i)} for i in range(101)]}}))
    artifact = tmp_path / "rank_8.pt"
    artifact.write_bytes(b"fixture artifact identity")
    values = dict(mode="matrix-train", storage_root=tmp_path / "storage",
                  code_dir=ROOT / "project", commit="a" * 40, dry_run=True,
                  manifest=manifest, tensor_artifact=artifact, training_plan=None,
                  phase="full", epochs=None, ranks=[4, 8, 12, 16])
    values.update(overrides)
    return argparse.Namespace(**values)


def test_matrix_exact_cross_product():
    full = matrix.training_rows("full")
    pilot = matrix.training_rows("pilot")
    assert len(full) == len({row["key"] for row in full}) == 63
    assert len(pilot) == 9
    assert {row["num_particles"] for row in full} == {2, 4, 6, 12, 24, 48, 96}
    assert {row["seed"] for row in full} == {31, 32, 33}
    assert {row["architecture"] for row in full} == set(matrix.ARCHITECTURES)
    assert {row["num_particles"] for row in pilot} == {2, 24, 96}
    assert {row["seed"] for row in pilot} == {31}


def test_plan_requires_chosen_artifact(tmp_path):
    args = plan_args(tmp_path, tensor_artifact=None)
    with pytest.raises(ValueError, match="choose a validated rank"):
        matrix.make_plan(args)


def test_plan_preserves_inputs_and_isolates_checkpoints(tmp_path):
    args = plan_args(tmp_path)
    path = matrix.make_plan(args)
    document = matrix.read_plan(path)
    assert document["manifest"]["sha256"] == matrix.digest(args.manifest)
    assert document["tensor_artifact"]["sha256"] == matrix.digest(args.tensor_artifact)
    assert document["evaluation"]["num_test_cases"] == 101
    assert document["evaluation"]["num_probe_particles"] == 64
    assert document["epochs"] == 100
    assert len({row["checkpoint_dir"] for row in document["tasks"]}) == 63
    assert matrix.selected_array(document, "all") == "0-62"
    assert matrix.selected_array(document, "3,7") == "3,7"
    for selection in ("63", "-1", "1,1", "0-3", "1%4"):
        with pytest.raises(ValueError):
            matrix.selected_array(document, selection)
    args.tensor_artifact.write_bytes(b"replaced artifact")
    with pytest.raises(ValueError, match="changed"):
        matrix.verify_identity(document["tensor_artifact"])


def test_pilot_and_commands_use_expected_architecture(tmp_path):
    document = matrix.read_plan(matrix.make_plan(plan_args(tmp_path, phase="pilot")))
    assert document["epochs"] == 10
    for row in document["tasks"]:
        command = matrix.task_command(document, row)
        assert command[command.index("--architecture") + 1] == row["architecture"]
        assert command[command.index("--num-particles") + 1] == str(row["num_particles"])
        assert command[command.index("--epochs") + 1] == "10"
        assert ("--tensor-artifact" in command) == (row["architecture"] == "tensor-dit")
    task = document["tasks"][0]
    checkpoints = Path(task["checkpoint_dir"])
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "latest.pt"
    latest.write_bytes(b"checkpoint")
    assert matrix.task_command(document, task, resume=True)[-2:] == ["--resume", str(latest)]


def test_evaluation_requires_all_training_complete_and_pins_checkpoints(tmp_path):
    args = plan_args(tmp_path, phase="pilot", dry_run=False)
    training_path = matrix.make_plan(args)
    training = matrix.read_plan(training_path)
    evaluation_args = argparse.Namespace(**{**vars(args), "mode": "matrix-evaluate",
                                            "training_plan": training_path})
    with pytest.raises(FileNotFoundError):
        matrix.make_plan(evaluation_args)
    for task in training["tasks"]:
        checkpoint = Path(task["checkpoint_dir"]) / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"trained checkpoint")
        record = Path(task["record_dir"])
        record.mkdir(parents=True)
        (record / "results.json").write_text(json.dumps({"status": "completed",
            "plan_sha256": matrix.digest(training_path), "task": task, "manifest": training["manifest"],
            "training": {"status": "completed", "epochs": training["epochs"], "scientific_result": True,
                         "architecture": task["architecture"], "num_particles_per_condition": task["num_particles"]},
            "best_checkpoint": matrix.identity(checkpoint)}))
    evaluation = matrix.read_plan(matrix.make_plan(evaluation_args))
    command = matrix.task_command(evaluation, evaluation["tasks"][0])
    assert command[command.index("--num-test-cases") + 1] == "101"
    assert command[command.index("--num-probe-particles") + 1] == "64"
    assert command[command.index("--num-samples") + 1] == "16"
    assert command[command.index("--sampling-steps") + 1] == "50"
    assert command[command.index("--boundary-projection") + 1] == "final"
    assert matrix.task_command(evaluation, evaluation["tasks"][0], resume=True)[-1] == "--resume"
    checkpoint.write_bytes(b"modified after completion")
    with pytest.raises(ValueError, match="changed"):
        matrix.make_plan(evaluation_args)


def test_prepare_command_and_scientific_manifest_guard(tmp_path):
    args = plan_args(tmp_path, mode="tensor-prepare", tensor_artifact=None)
    document = matrix.read_plan(matrix.make_plan(args))
    command = matrix.task_command(document, document["tasks"][0])
    assert command[-2:] == ["--ranks", "4,8,12,16"]
    assert "prepare_tensor_space.py" in command[1]
    with pytest.raises(ValueError, match="cannot resume"):
        matrix.task_command(document, document["tasks"][0], resume=True)
    args.manifest.write_text(json.dumps({"mode": "smoke_test"}))
    with pytest.raises(ValueError, match="scientific"):
        matrix.make_plan(args)


@pytest.mark.parametrize("value", ["00:00:00", "48:00:01", "2-01:00:00", "08:60:00", "infinite"])
def test_time_rejects_invalid_or_over_site_limit(value):
    with pytest.raises(ValueError):
        matrix.validate_time(value)


def test_login_node_execution_rejected_before_writes(tmp_path):
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_hpc_matrix.py"),
                             "run", "--plan", str(tmp_path / "does-not-exist.json")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires Slurm" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_generated_training_arguments_match_real_parser(tmp_path):
    import train_sparse_track_diffusion as training
    document = matrix.read_plan(matrix.make_plan(plan_args(tmp_path, phase="pilot")))
    for task in document["tasks"]:
        parsed = training._build_parser().parse_args(matrix.task_command(document, task)[2:])
        assert parsed.architecture == task["architecture"]
        assert parsed.num_particles == task["num_particles"]
        assert parsed.dit_hidden_dim == 192


def test_evaluation_guard_rejects_nonscientific_or_partial():
    document = {"evaluation": {"num_test_cases": 101}}
    task = {"num_particles": 96}
    summary = {"complete": True, "scientific_result": True, "dry_run": False,
               "failed_runs": 0, "completed_runs": 101, "boundary_projection": "final",
               "per_particle_count": {"96": {"completed_runs": 101, "trained_particles": 96}}}
    matrix.validate_evaluation(summary, document, task)
    for field, value in (("scientific_result", False), ("complete", False),
                         ("completed_runs", 100), ("boundary_projection", "each-step")):
        with pytest.raises(ValueError, match="incomplete"):
            matrix.validate_evaluation({**summary, field: value}, document, task)


def test_aggregate_seed_statistics_keeps_missing_and_failed_tasks(tmp_path):
    tasks = [{"index": index, "key": f"dit3d_N002_s{seed}", "architecture": "dit3d",
              "num_particles": 2, "seed": seed, "record_dir": str(tmp_path / f"task{index}")}
             for index, seed in enumerate((31, 32, 33))]
    tasks.append({"index": 3, "key": "tensor-dit_N004_s31", "architecture": "tensor-dit",
                  "num_particles": 4, "seed": 31, "record_dir": str(tmp_path / "missing")})
    tasks.append({"index": 4, "key": "unet3d_N004_s31", "architecture": "unet3d",
                  "num_particles": 4, "seed": 31, "record_dir": str(tmp_path / "failed")})
    document = {"format_version": 1, "mode": "matrix-evaluate", "tasks": tasks,
                "evaluation": {"num_test_cases": 101}}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(document))
    for task, mean in zip(tasks[:3], (1.0, 2.0, 3.0)):
        record = Path(task["record_dir"])
        evaluation = record / "evaluation"
        evaluation.mkdir(parents=True)
        summary = {"complete": True, "scientific_result": True, "dry_run": False,
                   "failed_runs": 0, "completed_runs": 101, "boundary_projection": "final",
                   "per_particle_count": {"2": {"completed_runs": 101, "trained_particles": 2,
                                                    "metrics": {"l2": {"mean": mean, "std": 0.5}}}}}
        summary_path = evaluation / "summary.json"
        summary_path.write_text(json.dumps(summary))
        (record / "results.json").write_text(json.dumps({"status": "completed", "summary": matrix.identity(summary_path)}))
        (evaluation / "runs.jsonl").write_text(json.dumps({"status": "completed", "metrics": {"l2": mean}}) + "\n")
    failed = Path(tasks[-1]["record_dir"])
    failed.mkdir()
    (failed / "results.json").write_text(json.dumps({"status": "failed", "error": "OOM"}))
    result = matrix.summarize_plan(plan)
    assert result["complete"] is False
    assert result["expected_tasks"] == 5 and result["completed_tasks"] == 3
    assert [task["status"] for task in result["tasks"]] == ["completed"] * 3 + ["missing", "failed"]
    aggregate = next(row for row in result["aggregates"] if row["architecture"] == "dit3d")
    assert aggregate["seed_mean"] == 2.0 and aggregate["seed_std"] == 1.0
    assert aggregate["valid_seeds"] == 3 and aggregate["complete"] is True
    assert len((tmp_path / "summary/per_case.jsonl").read_text().splitlines()) == 3
    assert "OOM" in (tmp_path / "summary/per_seed.csv").read_text()


def _load_submission_fixture():
    spec = importlib.util.spec_from_file_location("flow3d_submission_fixture", ROOT / "project/tests/test_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode,dependency,phase", [("matrix-train", "", None),
                                                  ("matrix-evaluate", "", None),
                                                  ("matrix-followup", "12345", None),
                                                  ("matrix-followup", "", None),
                                                  ("matrix-followup", "", "full")])
@pytest.mark.parametrize("cluster", ["deltaai", "delta"])
def test_slurm_array_reuses_one_snapshot_and_preserves_dry_run(tmp_path, mode, dependency, phase, cluster):
    """Real local Git snapshots + mocked planner/sbatch; no network/GPU."""
    fixture = _load_submission_fixture()
    if not fixture.BASH or not fixture.GIT:
        pytest.skip("Git for Windows Bash required")
    case = fixture.SubmissionTests()
    case.setUp()
    try:
        account, partition, arch, time_limit, concurrency = (
            ("biup-dtai-gh", "ghx4", "aarch64", "08:00:00", 4) if cluster == "deltaai" else
            ("biup-delta-gpu", "gpuA100x4", "x86_64", "02:00:00", 16))
        case.env.pop("FLOW3D_TRAIN_TIME", None)
        if mode == "matrix-evaluate":
            time_limit = "01:30:00"
            case.env["FLOW3D_EVALUATE_TIME"] = time_limit
        case.env["FLOW3D_ARRAY_CONCURRENCY"] = str(concurrency)
        shutil.copytree(case.publisher / "scripts", case.publisher / "project/scripts")
        (case.publisher / "scripts/run_hpc_matrix.py").write_text("# mocked by python shim\n")
        common = case.publisher / "project/scripts/common.sh"
        with common.open("a", newline="\n") as stream:
            stream.write('\nflow3d_settings() {\n'
                         f'export FLOW3D_CLUSTER={cluster} FLOW3D_ACCOUNT={account}\n'
                         f'export FLOW3D_PARTITION={partition} FLOW3D_GPUS=1 FLOW3D_CPUS=8 FLOW3D_MEM=32G\n'
                         'mkdir -p "$FLOW3D_ROOT/logs"\n}\n')
        case.write_script(case.root / "bin/python", '#!/bin/bash\n'
                          'case "$2" in\n'
                          'plan) printf "%s/plan.json\\n" "$FLOW3D_ROOT" ;;\n'
                          'inspect) case "${6}" in\n'
                          'count) echo "$MOCK_MATRIX_COUNT" ;;\n'
                          'mode) echo "$MOCK_MATRIX_MODE" ;;\n'
                          'sha256) echo 0123456789abcdef ;;\n'
                          'esac ;;\n'
                          'array) echo "0-$((MOCK_MATRIX_COUNT - 1))" ;;\n'
                          'dependency) printf "%s\\n" "$MOCK_MATRIX_DEPENDENCY" ;;\n'
                          'check-time) exit 0 ;;\n'
                          'time-budget) echo 5100 ;;\n'
                          '*) exit 3 ;;\n'
                          'esac\n')
        count = 12 if mode == "matrix-evaluate" else 9 if mode == "matrix-followup" and phase != "full" else 63
        arguments = ["--phase", phase] if phase else []
        case.env.update(MOCK_MATRIX_COUNT=str(count), MOCK_MATRIX_DEPENDENCY=dependency)
        case.env["MOCK_MATRIX_MODE"] = "matrix-evaluate" if mode == "matrix-evaluate" else "matrix-train"
        wrong_arch = "x86_64" if arch == "aarch64" else "aarch64"
        case.write_script(case.root / "bin/uname", f'#!/bin/bash\necho {wrong_arch}\n')
        case.commit(case.publisher, "matrix fixture")
        case.git("push", cwd=case.publisher)
        case.git("pull", "--ff-only", cwd=case.repo)
        result = case.submit("project/scripts/submit.sh", mode, [*arguments, "--dry-run"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert not case.capture.exists()
        assert not (case.root / "snapshots").exists()
        result = case.submit("project/scripts/submit.sh", mode, arguments)
        assert result.returncode != 0
        assert ("DeltaAI" if cluster == "deltaai" else "dt-login") in result.stderr
        assert not case.capture.exists()
        assert not (case.root / "snapshots").exists()
        case.write_script(case.root / "bin/uname", f'#!/bin/bash\necho {arch}\n')
        result = case.submit("project/scripts/submit.sh", mode, arguments)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"总任务：{count}" in result.stdout
        options = case.capture.read_bytes().decode().rstrip("\0").split("\0")
        assert f"--array=0-{count - 1}%{concurrency}" in options
        if dependency:
            assert f"--dependency=afterok:{dependency}" in options
            assert "--kill-on-invalid-dep=yes" in options
        else:
            assert not any(option.startswith("--dependency") for option in options)
        assert "--gpus-per-node=1" in options
        assert f"--account={account}" in options
        assert f"--partition={partition}" in options
        assert f"--time={time_limit}" in options
        assert any("%A_%a.out" in option for option in options)
        assert len(list((case.root / "snapshots").iterdir())) == 1
    finally:
        case.tearDown()


def test_delta_plan_reuses_cache_but_keeps_results_independent(tmp_path):
    args = plan_args(tmp_path, cluster="deltaai")
    gh = matrix.read_plan(matrix.make_plan(args))
    args.cluster = "delta"
    a100 = matrix.read_plan(matrix.make_plan(args))
    assert gh["cluster"] == "deltaai" and a100["cluster"] == "delta"
    assert gh["manifest"] == a100["manifest"]
    assert gh["tensor_artifact"] == a100["tensor_artifact"]
    assert gh["train_options"] == a100["train_options"]
    assert a100["epochs"] == 100 and len(a100["tasks"]) == 63
    for left, right in zip(gh["tasks"], a100["tasks"]):
        assert left["key"] == right["key"]
        assert left["record_dir"] != right["record_dir"]
        assert left["checkpoint_dir"] != right["checkpoint_dir"]


@pytest.mark.parametrize("document,cluster,architecture,gpu", [
    ({}, "deltaai", "aarch64", "NVIDIA GH200 120GB"),
    ({"cluster": "deltaai"}, "deltaai", "aarch64", "NVIDIA GH200 120GB"),
    ({"cluster": "delta"}, "delta", "x86_64", "NVIDIA A100-SXM4-40GB"),
])
def test_cluster_hardware_accepts_matching_gpu(document, cluster, architecture, gpu):
    matrix.validate_hardware(document, cluster=cluster, architecture=architecture, gpu=gpu)


@pytest.mark.parametrize("document,cluster,architecture,gpu", [
    ({}, "delta", "x86_64", "NVIDIA A100-SXM4-40GB"),  # old plans remain GH200
    ({"cluster": "delta"}, "deltaai", "aarch64", "NVIDIA GH200 120GB"),
    ({"cluster": "delta"}, "delta", "aarch64", "NVIDIA A100-SXM4-40GB"),
    ({"cluster": "delta"}, "delta", "x86_64", "NVIDIA A40"),
    ({"cluster": "deltaai"}, "deltaai", "x86_64", "NVIDIA GH200 120GB"),
    ({"cluster": "unknown"}, "unknown", "x86_64", "NVIDIA A100-SXM4-40GB"),
])
def test_cluster_hardware_rejects_wrong_profile_or_gpu(document, cluster, architecture, gpu):
    with pytest.raises(ValueError):
        matrix.validate_hardware(document, cluster=cluster, architecture=architecture, gpu=gpu)


def test_resume_cannot_silently_migrate_a_plan_to_another_cluster(tmp_path):
    path = matrix.make_plan(plan_args(tmp_path, cluster="deltaai", dry_run=False))
    env = {**os.environ, "FLOW3D_CLUSTER": "delta"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_hpc_matrix.py"),
                             "resume", "--plan", str(path)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires deltaai profile" in result.stderr


def test_inspect_reuses_exact_selected_artifact_and_rejects_tampering(tmp_path):
    args = plan_args(tmp_path)
    path = matrix.make_plan(args)
    command = [sys.executable, str(ROOT / "scripts/run_hpc_matrix.py"),
               "inspect", "--plan", str(path), "--field", "tensor_artifact"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == args.tensor_artifact.resolve()
    args.tensor_artifact.write_bytes(b"modified cache")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "changed" in result.stderr
