"""Evaluation of completed subsets keeps source identity and complete test coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_hpc_matrix as matrix


@pytest.fixture
def training(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"mode": "scientific", "splits": {
        "train": [{}] * 798, "validation": [{}] * 101, "test": [{}] * 101}}))
    artifact = tmp_path / "rank.pt"
    artifact.write_bytes(b"fixture")
    args = argparse.Namespace(mode="matrix-train", storage_root=tmp_path, code_dir=tmp_path / "project",
        commit="b" * 40, manifest=manifest, tensor_artifact=artifact, training_plan=None,
        phase="full", epochs=100, ranks=[4, 8], dry_run=False, cluster="deltaai")
    path = matrix.make_plan(args)
    plan = matrix.read_plan(path)
    for index in (3, 6, 9):
        task = plan["tasks"][index]
        checkpoint = Path(task["checkpoint_dir"]) / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"trained model {index}".encode())
        record = Path(task["record_dir"]) / "results.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"status": "completed", "plan_sha256": matrix.digest(path),
            "task": task, "manifest": plan["manifest"], "best_checkpoint": matrix.identity(checkpoint),
            "training": {"status": "completed", "epochs": 100, "scientific_result": True,
                         "architecture": task["architecture"], "num_particles_per_condition": task["num_particles"]}}))
    args.mode = "matrix-evaluate"
    args.training_plan = path
    return args, plan


def test_completed_selection_is_frozen_reindexed_and_keeps_all_test_cases(training):
    args, source = training
    args.training_tasks = "completed"
    plan = matrix.read_plan(matrix.make_plan(args))
    assert [task["index"] for task in plan["tasks"]] == [0, 1, 2]
    assert [task["training_index"] for task in plan["tasks"]] == [3, 6, 9]
    assert [task["key"] for task in plan["tasks"]] == [source["tasks"][i]["key"] for i in (3, 6, 9)]
    assert plan["training_selection"]["total_training_tasks"] == 63
    assert len(plan["training_selection"]["excluded"]) == 60
    assert plan["evaluation"]["num_test_cases"] == 101
    assert matrix.selected_array(plan, "all") == "0-2"
    assert plan["evaluation_figures"] is True
    for task in plan["tasks"]:
        matrix.verify_identity(task["training_result"])
        assert "--figures" in matrix.task_command(plan, task)


def test_explicit_selection_uses_original_training_ids(training):
    args, _ = training
    args.training_tasks = "9,3"
    plan = matrix.read_plan(matrix.make_plan(args))
    assert [task["training_index"] for task in plan["tasks"]] == [3, 9]
    assert [task["index"] for task in plan["tasks"]] == [0, 1]
    args.training_tasks = "0,3"
    with pytest.raises(FileNotFoundError):
        matrix.make_plan(args)


@pytest.mark.parametrize("selection", ["63", "3,3", "-1", "3-9"])
def test_invalid_evaluation_training_indices(training, selection):
    args, _ = training
    args.training_tasks = selection
    with pytest.raises(ValueError):
        matrix.make_plan(args)


def test_completed_selection_skips_running_but_does_not_claim_it_completed(training):
    args, source = training
    args.training_tasks = "completed"
    path = Path(source["tasks"][3]["record_dir"]) / "results.json"
    path.write_text(json.dumps({"status": "running"}))
    plan = matrix.read_plan(matrix.make_plan(args))
    assert [task["training_index"] for task in plan["tasks"]] == [6, 9]
    assert next(item for item in plan["training_selection"]["excluded"] if item["index"] == 3)["reason"] == "running"


@pytest.mark.parametrize("mutation", ["plan", "manifest", "task", "epochs", "scientific", "checkpoint"])
def test_completed_selection_rejects_invalid_completed_record(training, tmp_path, mutation):
    args, source = training
    args.training_tasks = "completed"
    path = Path(source["tasks"][3]["record_dir"]) / "results.json"
    record = json.loads(path.read_text())
    if mutation == "plan":
        record["plan_sha256"] = "wrong"
    elif mutation == "manifest":
        record["manifest"] = {}
    elif mutation == "task":
        record["task"]["seed"] = 99
    elif mutation == "epochs":
        record["training"]["epochs"] = 1
    elif mutation == "scientific":
        record["training"]["scientific_result"] = False
    else:
        other = tmp_path / "other.pt"
        other.write_bytes(b"different weights")
        record["best_checkpoint"] = matrix.identity(other)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        matrix.make_plan(args)


def test_no_completed_training_tasks_is_an_error(training):
    args, source = training
    args.training_tasks = "completed"
    for task in source["tasks"]:
        path = Path(task["record_dir"]) / "results.json"
        if path.exists():
            path.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="no completed"):
        matrix.make_plan(args)


def test_runtime_budget_is_operational_not_frozen_scientific_protocol(training, monkeypatch):
    args, _ = training
    args.training_tasks = "completed"
    plan = matrix.read_plan(matrix.make_plan(args))
    monkeypatch.setenv("FLOW3D_EVALUATE_BUDGET_SECONDS", "5100")
    command = matrix.task_command(plan, plan["tasks"][0], resume=True)
    assert command[command.index("--max-runtime-seconds") + 1] == "5100"
    assert command[-1] == "--resume"
    assert matrix.validate_time("01:30:00") == 5400
    assert "max_runtime_seconds" not in plan["evaluation"]


def test_budget_exit_code_survives_matrix_wrapper(monkeypatch):
    def stopped(args):
        raise subprocess.CalledProcessError(75, ["sampler"])
    monkeypatch.setattr(matrix, "run_task", stopped)
    monkeypatch.setattr(sys, "argv", ["matrix", "run", "--plan", "unused.json"])
    with pytest.raises(SystemExit) as caught:
        matrix.main()
    assert caught.value.code == 75


def test_new_evaluation_summary_must_match_full_sampling_protocol(training):
    args, _ = training
    args.training_tasks = "completed"
    plan = matrix.read_plan(matrix.make_plan(args))
    task = plan["tasks"][0]
    summary = {"complete": True, "scientific_result": True, "dry_run": False,
        "failed_runs": 0, "completed_runs": 101, "boundary_projection": "final",
        "requested_test_cases_per_count": 101, "expected_runs": 101,
        "particle_counts": [task["num_particles"]], "base_seed": 47, "num_posterior_samples": 16,
        "num_probe_particles": 64, "sampling_steps": 50, "eta": 1.0, "cfg_scale": 1.5,
        "manifest": plan["manifest"]["path"], "per_particle_count": {str(task["num_particles"]): {
            "completed_runs": 101, "trained_particles": task["num_particles"],
            "checkpoint": task["checkpoint"]["path"]}}}
    matrix.validate_evaluation(summary, plan, task)
    for field, value in (("num_posterior_samples", 4), ("sampling_steps", 10), ("base_seed", 999),
                         ("manifest", "different.json"), ("cfg_scale", 1.0)):
        with pytest.raises(ValueError):
            matrix.validate_evaluation({**summary, field: value}, plan, task)
