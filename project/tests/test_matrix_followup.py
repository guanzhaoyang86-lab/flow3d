"""Offline deferred-rank plans and Slurm dependency provenance checks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hpc_matrix as matrix


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    """A real pinned preparation plan; no artifacts are present initially."""
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"mode": "scientific", "splits": {
        "train": [{"case_id": "train"}], "validation": [{"case_id": "validation"}],
        "test": [{"case_id": "test"}]}})
    code = tmp_path / "snapshot" / "project"
    code.mkdir(parents=True)
    (code / ".flow3d-commit").write_text("a" * 40)
    args = argparse.Namespace(
        mode="tensor-prepare", manifest=manifest, storage_root=tmp_path / "storage",
        code_dir=code, commit="a" * 40, dry_run=False, phase=None, epochs=None,
        ranks=[4, 8, 12, 16], tensor_artifact=None, training_plan=None,
        prepare_plan=None, after_job=None, rank_mean_limit=.05, rank_max_limit=.20,
    )
    prepare_path = matrix.make_plan(args)
    prepare = matrix.read_plan(prepare_path)
    followup_args = argparse.Namespace(**{**vars(args), "mode": "matrix-followup",
                                         "prepare_plan": prepare_path, "after_job": 12345})
    return {"args": followup_args, "prepare_path": prepare_path, "prepare": prepare}


def followup(prepared):
    path = matrix.make_plan(prepared["args"])
    return path, matrix.read_plan(path)


def finish_preparation(prepared):
    prepare = prepared["prepare"]
    output = Path(prepare["prepare_output"])
    output.mkdir(parents=True)
    report = {"manifest": prepare["manifest"]["path"], "ranks": {},
              "rank_selection_split": "validation", "test_metrics_computed": False}
    artifacts = []
    for rank, mean, maximum in ((4, .10, .3), (8, .04, .12), (12, .02, .08), (16, .01, .04)):
        artifact = output / f"rank_{rank}.pt"
        artifact.write_bytes(f"dummy prepared tensor rank {rank}".encode())
        artifact_id = matrix.identity(artifact)
        artifacts.append(artifact_id)
        report["ranks"][str(rank)] = {
            "rank": rank, "artifact": str(artifact), "artifact_sha256": artifact_id["sha256"],
            "rank_selection_split": "validation", "test_metrics_computed": False,
            "validation_mean_relative_l2_fluid": mean, "validation_max_relative_l2_fluid": maximum,
            "anchor_overlap_min_singular_value": {"train": [.2] * 3, "validation": [.1] * 3},
            "low_anchor_overlap_threshold": 1e-5,
            "low_anchor_overlap_case_count": {"train": 0, "validation": 0},
        }
    report_path = output / "report.json"
    write_json(report_path, report)
    record = Path(prepare["tasks"][0]["record_dir"])
    attempt = record / "attempts" / "12346_fixture"
    environment_path = attempt / "environment.json"
    write_json(environment_path, {"slurm_job_id": "12346", "slurm_array_job_id": "12345"})
    result = {"status": "completed", "plan_sha256": matrix.digest(prepared["prepare_path"]),
              "attempt": str(attempt), "report": matrix.identity(report_path), "artifacts": artifacts}
    result_path = record / "results.json"
    write_json(result_path, result)
    return {"result_path": result_path, "result": result, "report_path": report_path,
            "report": report, "environment_path": environment_path, "artifacts": artifacts}


def update_report(completed):
    write_json(completed["report_path"], completed["report"])
    completed["result"]["report"] = matrix.identity(completed["report_path"])
    write_json(completed["result_path"], completed["result"])


def test_followup_pins_source_before_artifact_exists_and_builds_nine_pilot_tasks(prepared):
    path, document = followup(prepared)
    assert document["mode"] == "matrix-train"
    assert document["phase"] == "pilot" and document["epochs"] == 10
    assert len(document["tasks"]) == 9
    assert {task["num_particles"] for task in document["tasks"]} == {2, 24, 96}
    assert {task["seed"] for task in document["tasks"]} == {31}
    assert document["tensor_artifact"] is None
    assert not Path(prepared["prepare"]["prepare_output"]).exists()
    assert document["followup"]["prepare_plan"] == matrix.identity(prepared["prepare_path"])
    assert document["followup"]["rank_policy"] == {"mean_limit": .05, "max_limit": .2}
    assert document["manifest"] == prepared["prepare"]["manifest"]
    assert matrix.followup_dependency(document, path) == "12345"


@pytest.mark.parametrize("change,expected", [
    ({"dry_run": True}, "submitted single-task"),
    ({"mode": "matrix-train"}, "submitted single-task"),
])
def test_followup_rejects_unsubmitted_or_wrong_source_plan(prepared, change, expected):
    write_json(prepared["prepare_path"], {**prepared["prepare"], **change})
    with pytest.raises(ValueError, match=expected):
        followup(prepared)


@pytest.mark.parametrize("field,value,expected", [
    ("storage_root", "other-root", "storage root"),
    ("phase", "full", "pilot only"),
    ("tensor_artifact", "rank_8.pt", "automatically"),
    ("rank_mean_limit", float("nan"), "finite"),
    ("rank_max_limit", .01, "must not exceed"),
    ("after_job", None, "requires"),
])
def test_followup_rejects_invalid_requested_configuration(prepared, tmp_path, field, value, expected):
    if field in {"storage_root", "tensor_artifact"}:
        value = tmp_path / value
    setattr(prepared["args"], field, value)
    with pytest.raises(ValueError, match=expected):
        followup(prepared)


def test_followup_dependency_waits_for_pending_or_running_and_refuses_failed(prepared):
    path, document = followup(prepared)
    result_path = Path(prepared["prepare"]["tasks"][0]["record_dir"]) / "results.json"
    assert matrix.followup_dependency(document, path) == "12345"
    write_json(result_path, {"status": "running"})
    assert matrix.followup_dependency(document, path) == "12345"
    write_json(result_path, {"status": "failed"})
    with pytest.raises(ValueError, match="preparation failed"):
        matrix.followup_dependency(document, path)


def test_completed_followup_selects_once_and_drops_obsolete_slurm_dependency(prepared):
    path, document = followup(prepared)
    completed = finish_preparation(prepared)
    resolved = matrix.resolve_followup(document, path)
    assert resolved["tensor_artifact"] == completed["artifacts"][1]
    selection_path = path.parent / "rank_selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["selected_rank"] == 8
    assert selection["followup_plan_sha256"] == matrix.digest(path)
    before = selection_path.read_bytes()
    assert matrix.resolve_followup(document, path) == resolved
    assert matrix.followup_dependency(document, path) == ""
    assert selection_path.read_bytes() == before
    assert matrix.read_plan(path)["tensor_artifact"] is None


@pytest.mark.parametrize("kind", ["source-plan", "result-plan", "job", "report", "artifact",
                                  "artifact-not-recorded", "report-manifest", "report-ranks"])
def test_resolution_rejects_mutated_or_mismatched_preparation(prepared, kind):
    path, document = followup(prepared)
    completed = finish_preparation(prepared)
    if kind == "source-plan":
        write_json(prepared["prepare_path"], {**prepared["prepare"], "commit": "b" * 40})
    elif kind == "result-plan":
        completed["result"]["plan_sha256"] = "0" * 64
        write_json(completed["result_path"], completed["result"])
    elif kind == "job":
        write_json(completed["environment_path"], {"slurm_array_job_id": "54321", "slurm_job_id": "12345"})
    elif kind == "report":
        completed["report_path"].write_text("changed report")
    elif kind == "artifact":
        Path(completed["artifacts"][1]["path"]).write_bytes(b"replaced tensor cache")
    elif kind == "artifact-not-recorded":
        completed["result"]["artifacts"] = [completed["artifacts"][0]]
        write_json(completed["result_path"], completed["result"])
    elif kind == "report-manifest":
        completed["report"]["manifest"] += ".other"
        update_report(completed)
    else:
        completed["report"]["ranks"].pop("16")
        update_report(completed)
    with pytest.raises(ValueError):
        matrix.resolve_followup(document, path)
    assert not (path.parent / "rank_selection.json").exists()


def test_resolved_selection_cannot_be_replaced_by_new_report(prepared):
    path, document = followup(prepared)
    completed = finish_preparation(prepared)
    matrix.resolve_followup(document, path)
    before = (path.parent / "rank_selection.json").read_bytes()
    completed["report"]["ranks"]["8"]["validation_mean_relative_l2_fluid"] = .08
    update_report(completed)
    with pytest.raises(ValueError, match="changed after first resolution"):
        matrix.resolve_followup(document, path)
    assert (path.parent / "rank_selection.json").read_bytes() == before


def test_evaluation_resolves_deferred_training_artifact_and_rejects_inconsistent_tasks(prepared):
    path, document = followup(prepared)
    completed = finish_preparation(prepared)
    for task in document["tasks"]:
        checkpoint = Path(task["checkpoint_dir"]) / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"dummy trained weights")
        write_json(Path(task["record_dir"]) / "results.json", {
            "status": "completed", "best_checkpoint": matrix.identity(checkpoint),
            "tensor_artifact": completed["artifacts"][1],
        })
    args = argparse.Namespace(**{**vars(prepared["args"]), "mode": "matrix-evaluate", "training_plan": path})
    evaluation = matrix.read_plan(matrix.make_plan(args))
    assert len(evaluation["tasks"]) == 9
    assert evaluation["tensor_artifact"] == completed["artifacts"][1]
    assert evaluation["training_rank_selection"] == matrix.identity(path.parent / "rank_selection.json")
    bad_result_path = Path(document["tasks"][0]["record_dir"]) / "results.json"
    bad_result = json.loads(bad_result_path.read_text())
    bad_result["tensor_artifact"] = completed["artifacts"][2]
    write_json(bad_result_path, bad_result)
    with pytest.raises(ValueError, match="inconsistent automatic rank"):
        matrix.make_plan(args)


def test_resume_command_resolves_original_deferred_artifact(prepared):
    path, document = followup(prepared)
    finish_preparation(prepared)
    environment = {**os.environ, "FLOW3D_ROOT": document["storage_root"]}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_hpc_matrix.py"),
                             "resume", "--plan", str(path)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == path.resolve()
    assert (path.parent / "rank_selection.json").exists()


def test_cli_rejects_nonpositive_dependency_id(prepared):
    args = prepared["args"]
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_hpc_matrix.py"),
                             "plan", "matrix-followup", "--prepare-plan", str(args.prepare_plan),
                             "--after-job", "0", "--storage-root", str(args.storage_root),
                             "--code-dir", str(args.code_dir), "--commit", args.commit],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "must be positive" in result.stderr


def test_real_cpu_preparation_report_resolves_followup_without_schema_adaptation(prepared, tmp_path):
    """Run the actual cache writer so report/selector contracts cannot drift."""
    import torch
    import prepare_tensor_space

    spec = importlib.util.spec_from_file_location("tensor_manifest_fixture", ROOT / "tests/test_tensor_space.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    data_dir = tmp_path / "real_cases"
    data_dir.mkdir()
    manifest = fixture._manifest(data_dir)
    args = argparse.Namespace(**{**vars(prepared["args"]), "mode": "tensor-prepare",
                                 "manifest": manifest, "ranks": [2, 4]})
    prepared["prepare_path"] = matrix.make_plan(args)
    prepared["prepare"] = matrix.read_plan(prepared["prepare_path"])
    prepared["args"] = argparse.Namespace(**{**vars(args), "mode": "matrix-followup",
                                             "prepare_plan": prepared["prepare_path"]})
    path, document = followup(prepared)
    output = Path(prepared["prepare"]["prepare_output"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        prepare_tensor_space.main(["--manifest", str(manifest), "--output-dir", str(output),
                                   "--ranks", "2,4", "--device", "cpu"])
    finally:
        torch.set_num_threads(old_threads)
    record = Path(prepared["prepare"]["tasks"][0]["record_dir"])
    attempt = record / "attempts" / "12346_real"
    write_json(attempt / "environment.json", {"slurm_array_job_id": "12345", "slurm_job_id": "12346"})
    write_json(record / "results.json", {
        "status": "completed", "plan_sha256": matrix.digest(prepared["prepare_path"]),
        "attempt": str(attempt), "report": matrix.identity(output / "report.json"),
        "artifacts": [matrix.identity(output / f"rank_{rank}.pt") for rank in [2, 4]],
    })
    resolved = matrix.resolve_followup(document, path)
    selection = json.loads((path.parent / "rank_selection.json").read_text())
    assert selection["selected_rank"] in [2, 4]
    assert resolved["tensor_artifact"] == matrix.identity(output / f"rank_{selection['selected_rank']}.pt")
    assert matrix.followup_dependency(document, path) == ""
