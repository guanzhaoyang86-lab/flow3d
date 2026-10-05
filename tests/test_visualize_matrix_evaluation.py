"""Synthetic fixtures only: test figures and statistics, not scientific results."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "visualize_matrix_evaluation.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("matrix_evaluation_figures", SCRIPT)
viz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viz)


def synthetic_archive(path: Path, count: int = 2) -> Path:
    """Deliberately labelled NON-SCIENTIFIC field with distinct u/v/w components."""
    z, y, x = np.meshgrid(np.arange(4), np.arange(6), np.arange(8), indexing="ij")
    field = np.stack([1 + x * .1, 2 + y * .2, 3 + z * .3]).astype(np.float32)[None]
    bounds = np.asarray([[[0, 7], [0, 5], [0, 3]]], dtype=np.float32)
    projection = np.asarray([[[[1, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]]], dtype=np.float32)
    def tracks(n):
        t = np.linspace(0, 1, 4)
        result = np.empty((1, 3, n, 4, 2), dtype=np.float32)
        for particle in range(n):
            result[0, :, particle, :, 0] = t + particle * .02
            result[0, :, particle, :, 1] = t * .5 + particle * .02
        return result
    observed, probes = tracks(count), tracks(3)
    metrics = {"num_observed_particles": count, "num_posterior_samples": 2,
               "num_probe_particles": 3, "synthetic_fixture": True}
    metadata = {"architecture": "unet3d", "sampled_particles": count, "trained_particles": count,
                "training_seed": 31, "split": "test", "case_id": "synthetic-fixture-only",
                "scientific_result": False, "scientific_result_reasons": ["synthetic test fixture"]}
    np.savez(path, reference_field=field, posterior_mean=field + .1,
             posterior_variance=np.ones_like(field) * .01, domain_bounds=bounds,
             observed_tracks=observed, replay_tracks=observed + .03,
             observation_mask=np.ones(observed.shape[:-1], dtype=bool),
             replay_validity=np.ones((1, count, 4), dtype=bool), projection_matrix=projection,
             probe_observed_tracks=probes, probe_replay_tracks=probes + .05,
             probe_observation_mask=np.ones(probes.shape[:-1], dtype=bool),
             probe_replay_validity=np.ones((1, 3, 4), dtype=bool),
             metrics=np.asarray(json.dumps(metrics)), metadata=np.asarray(json.dumps(metadata)))
    return path


@pytest.mark.parametrize("name,shape,components,labels", [
    ("xy", (6, 8), (1, 2), ("x", "y")),
    ("xz", (4, 8), (1, 3), ("x", "z")),
    ("yz", (4, 6), (2, 3), ("y", "z")),
])
def test_velocity_plane_components_and_axes(name, shape, components, labels):
    field = np.ones((3, 4, 6, 8)) * np.asarray([1, 2, 3])[:, None, None, None]
    bounds = np.asarray([[0, 7], [0, 5], [0, 3]])
    result = viz.center_plane(field, bounds, name)
    assert result["scalar"].shape == shape
    np.testing.assert_allclose(result["u"], components[0])
    np.testing.assert_allclose(result["v"], components[1])
    np.testing.assert_allclose(result["scalar"], math.sqrt(14))
    assert (result["xlabel"], result["ylabel"]) == labels


def test_even_center_interpolates_and_projection_labels_are_dynamic():
    field = np.zeros((3, 4, 6, 8))
    field[0] = np.arange(4)[:, None, None]
    bounds = np.asarray([[0, 7], [0, 5], [0, 3]])
    result = viz.center_plane(field, bounds, "xy")
    np.testing.assert_allclose(result["u"], 1.5)
    assert result["fixed"] == "z=1.5"
    assert viz._view_labels(np.asarray([[0, 1, 0], [0, 0, 1]])) == ("YZ", "y", "z")
    assert viz._view_labels(np.asarray([[.5, .5, 0], [0, 0, 1]]))[1] == "projection 1"


@pytest.mark.parametrize("count", [2, 6])
def test_case_png_pdf_and_manifest_outputs(tmp_path, count):
    archive = synthetic_archive(tmp_path / "synthetic.npz", count)
    out = tmp_path / "figures"
    result = viz.write_task_figures(archive, out, architecture="unet3d", num_particles=count,
                                   training_seed=31, dpi=40)
    assert len(result["files"]) == 6
    assert result["scientific_result"] is False
    assert result["test_case_index"] == 0
    assert "not median" in result["selection"]
    assert result["track_display"]["Observed particles"]["available"] == count
    for file in result["files"]:
        assert Path(file).stat().st_size > 500
    assert (out / "figure_manifest.json").is_file()
    assert not viz.plt.get_fignums()


def test_case_rejects_wrong_particle_label_and_cherrypicked_index(tmp_path):
    archive = synthetic_archive(tmp_path / "synthetic.npz", 2)
    with pytest.raises(ValueError, match="particle count"):
        viz.write_task_figures(archive, tmp_path / "bad", architecture="unet3d", num_particles=4, training_seed=31)
    with pytest.raises(ValueError, match="preselected test index"):
        viz.write_task_figures(archive, tmp_path / "bad", architecture="unet3d", num_particles=2, training_seed=31, test_case_index=1)
    with pytest.raises(ValueError, match="training seed"):
        viz.write_task_figures(archive, tmp_path / "bad", architecture="unet3d", num_particles=2, training_seed=33)


def synthetic_record(task, index, l2, **overrides):
    metrics = {key: l2 for key in viz.METRICS}
    metrics.update(num_observed_particles=task["num_particles"], num_posterior_samples=8,
                   num_probe_particles=64, all_probe_particles_valid=True, all_replay_particles_valid=True)
    result = {"fixture_only": "synthetic unit-test record, not a research output", "status": "completed", "returncode": 0,
              "scientific_result": True, "source_scientific_result": True, "scientific_design": True,
              "scientific_result_reasons": [], "test_case_index": index, "case_id": f"synthetic-{index}",
              "num_particles": task["num_particles"], "trained_particles": task["num_particles"],
              "checkpoint": task["checkpoint"]["path"], "seed": 47 + index, "metrics": metrics}
    result.update(overrides)
    return result


def synthetic_plan(tmp_path):
    original = [{"index": i, "architecture": "unet3d", "num_particles": count, "seed": seed}
                for i, (count, seed) in enumerate((c, s) for c in (2, 4, 6) for s in (31, 32, 33))]
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"fixture_only": True, "tasks": original}), encoding="utf-8")
    tasks = [{**original[index], "record_dir": str(tmp_path / f"task_{index}"),
              "checkpoint": {"path": f"synthetic_checkpoint_{index}.pt"}} for index in (0, 1, 3, 6)]
    for task in tasks:
        (Path(task["record_dir"]) / "evaluation").mkdir(parents=True)
    document = {"fixture_only": True, "mode": "matrix-evaluate", "dry_run": False, "tasks": tasks,
                "training_plan": {"path": str(training)},
                "evaluation": {"num_test_cases": 2, "num_samples": 8, "num_probe_particles": 64,
                               "sampling_steps": 50, "seed": 47, "boundary_projection": "final"}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, tasks


def write_runs(task, records, complete=False):
    folder = Path(task["record_dir"]) / "evaluation"
    (folder / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    if complete:
        (folder / "summary.json").write_text(json.dumps({"complete": True, "scientific_result": True,
            "dry_run": False, "failed_runs": 0, "completed_runs": 2, "boundary_projection": "final",
            "per_particle_count": {str(task["num_particles"]): {"completed_runs": 2, "trained_particles": task["num_particles"]}}}), encoding="utf-8")


def test_report_separates_test_std_seed_std_partial_and_missing(tmp_path):
    plan, tasks = synthetic_plan(tmp_path)
    write_runs(tasks[0], [synthetic_record(tasks[0], 0, 0), synthetic_record(tasks[0], 1, 4)], complete=True)
    write_runs(tasks[1], [synthetic_record(tasks[1], 0, 2), synthetic_record(tasks[1], 1, 6)], complete=True)
    write_runs(tasks[2], [synthetic_record(tasks[2], 0, 9), synthetic_record(tasks[2], 1, 99, status="failed"),
                          synthetic_record(tasks[2], 1, 99, scientific_result=False)])
    # A stale outer runner status must not disqualify a validated completed sweep.
    (Path(tasks[0]["record_dir"]) / "results.json").write_text('{"status":"running"}')
    result = viz.write_report(plan, dpi=40)
    assert result["status"] == "partial"
    assert result["complete_tasks"] == 2
    assert result["actual_case_runs"] == 5
    assert result["expected_case_runs"] == 8
    assert result["original_training_matrix_tasks"] == 9
    assert [t["status"] for t in result["tasks"]] == ["complete", "complete", "partial", "missing"]
    metric = "field_relative_l2_unknown_interior"
    assert result["tasks"][0]["metrics"][metric]["mean"] == 2
    assert result["tasks"][0]["metrics"][metric]["sample_std"] == pytest.approx(math.sqrt(8))
    group = result["groups"][0]
    assert group["expected_training_seeds"] == 3
    assert group["actual_training_seeds"] == 2
    assert group["metrics_across_training_seed_means"][metric]["mean"] == 3
    assert group["metrics_across_training_seed_means"][metric]["sample_std"] == pytest.approx(math.sqrt(2))
    # Partial N=4 case contributes to the per-seed table, never the full-test curve.
    assert result["tasks"][2]["metrics"][metric]["mean"] == 9
    assert result["curves"][1]["metrics_across_training_seed_means"][metric]["mean"] is None
    assert result["tasks"][3]["metrics"][metric]["mean"] is None
    assert result["warnings"]
    for name in ("summary.json", "per_seed.csv", "particle_error_curves.png", "particle_error_curves.pdf", "meeting_report.md"):
        assert (plan.parent / "report" / name).is_file()
    assert "部分结果" in (plan.parent / "report" / "meeting_report.md").read_text(encoding="utf-8")


def test_report_rejects_mismatch_nonfinite_duplicate_and_incomplete_summary(tmp_path):
    plan, tasks = synthetic_plan(tmp_path)
    good = synthetic_record(tasks[0], 0, .2)
    bad_value = synthetic_record(tasks[0], 1, float("nan"))
    write_runs(tasks[0], [good, good, bad_value, synthetic_record(tasks[0], 1, 100, checkpoint="wrong.pt")], complete=True)
    result = viz.write_report(plan, dpi=40)
    assert result["actual_case_runs"] == 1
    assert result["complete_tasks"] == 0
    assert len(result["warnings"]) == 3
    assert result["tasks"][0]["metrics"]["field_relative_l2_unknown_interior"]["sample_std"] is None


def test_report_copies_small_case_figures_for_portable_package(tmp_path):
    plan, tasks = synthetic_plan(tmp_path)
    # Match run_sparse_particle_sweep._write_case_figures exactly.
    folder = Path(tasks[0]["record_dir"]) / "evaluation" / "figures" / "N002_case0000"
    folder.mkdir(parents=True)
    (folder / "figure_manifest.json").write_text('{"fixture_only":true}')
    (folder / "flow_slices.png").write_bytes(b"synthetic image fixture")
    viz.write_report(plan, dpi=40)
    copied = tmp_path / "report" / "cases" / "000_unet3d_N2_seed31"
    assert (copied / "flow_slices.png").read_bytes() == b"synthetic image fixture"
    text = (tmp_path / "report" / "meeting_report.md").read_text(encoding="utf-8")
    assert "cases/000_unet3d_N2_seed31/flow_slices.png" in text


def test_report_refuses_dry_run_plan(tmp_path):
    plan, _ = synthetic_plan(tmp_path)
    value = json.loads(plan.read_text())
    value["dry_run"] = True
    plan.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="submitted"):
        viz.write_report(plan)
