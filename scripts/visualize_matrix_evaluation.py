#!/usr/bin/env python
"""Create traceable CPU-only figures and a Chinese report from evaluation outputs.

No inference or checkpoint loading occurs here. Per-case illustrations use the
preselected test index 0, not a retrospectively chosen best or median case.
Partial runs remain visibly partial; absent measurements are never zero-filled.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_hpc_matrix import validate_evaluation


METRICS = {
    "field_relative_l2_unknown_interior": "Field relative L2 (unknown interior)",
    "field_cosine_unknown_interior": "Velocity cosine similarity",
    "observed_track_rmse_cells": "Observed-track RMSE (cells)",
    "probe_track_rmse_cells": "Held-out probe RMSE (cells)",
    "posterior_mean_divergence_mse": "Posterior-mean divergence MSE",
}
PLANES = ("xy", "xz", "yz")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": statistics.fmean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def center_plane(field: np.ndarray, bounds: np.ndarray, name: str,
                 scalar: np.ndarray | None = None) -> dict:
    """Slice [u,v,w; z,y,x] through the physical centre, averaging even grids."""
    if field.ndim != 4 or field.shape[0] != 3 or min(field.shape[1:]) < 2:
        raise ValueError("field must have shape [3, D, H, W] with each grid size >= 2")
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must contain ascending [x,y,z] min/max pairs")
    scalar = np.linalg.norm(field, axis=0) if scalar is None else scalar
    if scalar.shape != field.shape[1:]:
        raise ValueError("scalar volume does not match the field grid")
    x, y, z = [np.linspace(*bounds[i], field.shape[3 - i]) for i in range(3)]
    choices = {"xy": (0, x, y, 0, 1, "x", "y", "z", bounds[2].mean()),
               "xz": (1, x, z, 0, 2, "x", "z", "y", bounds[1].mean()),
               "yz": (2, y, z, 1, 2, "y", "z", "x", bounds[0].mean())}
    if name not in choices:
        raise ValueError(f"unknown plane: {name}")
    axis, horizontal, vertical, ci, cj, hl, vl, fixed, value = choices[name]

    def take_center(array: np.ndarray, dim: int) -> np.ndarray:
        upper = array.shape[dim] // 2
        lower = (array.shape[dim] - 1) // 2
        return (np.take(array, lower, axis=dim) + np.take(array, upper, axis=dim)) / 2

    vector = take_center(field, axis + 1)
    return {"scalar": take_center(scalar, axis), "horizontal": horizontal, "vertical": vertical,
            "u": vector[ci], "v": vector[cj], "xlabel": hl, "ylabel": vl,
            "fixed": f"{fixed}={value:g}"}


def _draw(ax: plt.Axes, plane: dict, *, vmax: float, cmap: str, vectors: bool = False,
          vector_max: float = 1.0):
    h, v = plane["horizontal"], plane["vertical"]
    image = ax.imshow(plane["scalar"], origin="lower", extent=(h[0], h[-1], v[0], v[-1]),
                      vmin=0, vmax=max(vmax, np.finfo(float).eps), cmap=cmap, aspect="equal")
    if vectors:
        stride = max(1, max(len(h), len(v)) // 12)
        xx, yy = np.meshgrid(h[::stride], v[::stride])
        # Identical scale for all truth/prediction panels, in axis-width units.
        ax.quiver(xx, yy, plane["u"][::stride, ::stride], plane["v"][::stride, ::stride],
                  color="white", scale=max(vector_max, 1e-12) * 14, scale_units="width", width=.004)
    ax.set(xlabel=plane["xlabel"], ylabel=plane["ylabel"])
    return image


def _save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    paths = []
    try:
        for extension in ("png", "pdf"):
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(str(path.resolve()))
    finally:
        plt.close(fig)
    return paths


def _view_labels(matrix: np.ndarray) -> tuple[str, str, str]:
    labels = []
    for row in matrix:
        hits = [i for i in range(3) if np.allclose(row, np.eye(3)[i], atol=1e-6)]
        labels.append("xyz"[hits[0]] if hits else f"projection {len(labels) + 1}")
    title = "".join(labels).upper() if all(label in "xyz" for label in labels) else "Projected view"
    return title, labels[0], labels[1]


def write_task_figures(posterior: Path, output_dir: Path, *, architecture: str,
                       num_particles: int, training_seed: int, test_case_index: int = 0,
                       dpi: int = 170) -> dict:
    """Write three PNG/PDF figures from one real posterior archive, CPU only."""
    if dpi < 30 or test_case_index != 0:
        raise ValueError("figures require dpi >= 30 and the preselected test index 0")
    posterior, output_dir = Path(posterior), Path(output_dir)
    with np.load(posterior, allow_pickle=False) as archive:
        # Never decompress the potentially large posterior_samples array.
        fields = ("reference_field", "posterior_mean", "posterior_variance", "domain_bounds",
                  "observed_tracks", "observation_mask", "replay_tracks", "replay_validity",
                  "projection_matrix", "probe_observed_tracks", "probe_observation_mask",
                  "probe_replay_tracks", "probe_replay_validity")
        data = {key: archive[key] for key in fields if key in archive}
        metadata = json.loads(str(archive["metadata"].item()))
        metrics = json.loads(str(archive["metrics"].item()))
    if metadata.get("sampled_particles") != num_particles or metrics.get("num_observed_particles") != num_particles:
        raise ValueError("figure particle count differs from the posterior archive")
    if metadata.get("architecture") != architecture or metadata.get("split") != "test":
        raise ValueError("posterior architecture or test split does not match figure request")
    if metadata.get("training_seed", training_seed) != training_seed:
        raise ValueError("posterior training seed differs from the requested figure label")
    if metadata.get("test_case_index", test_case_index) != test_case_index:
        raise ValueError("posterior test case index differs from the preselected example")
    required = ("reference_field", "posterior_mean", "posterior_variance", "domain_bounds")
    for key in required:
        if key not in data or data[key].shape[0] != 1 or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid posterior data: {key}")
    reference, prediction, variance = (data[key][0] for key in required[:3])
    if reference.shape != prediction.shape or reference.shape != variance.shape or np.any(variance < 0):
        raise ValueError("posterior fields must have matching shapes and nonnegative variance")
    bounds = data["domain_bounds"][0]
    sample_count = metrics.get("num_posterior_samples")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("posterior sample count is missing or invalid")
    scientific = (metadata.get("scientific_result") is True
                  and metadata.get("scientific_result_reasons") == []
                  and metadata.get("trained_particles") == num_particles
                  and metadata.get("checkpoint_has_training_provenance") is True
                  and metadata.get("evaluation_manifest_provenance", {}).get("manifest_mode") == "scientific")
    case_id = str(metadata.get("case_id", "unknown"))
    label = (f"{architecture} | N={num_particles} | training seed={training_seed} | "
             f"test index=0 ({case_id}) | posterior samples={sample_count}")
    if not scientific:
        label = "NON-SCIENTIFIC / diagnostic illustration\n" + label
    output_dir.mkdir(parents=True, exist_ok=True)
    error = np.linalg.norm(prediction - reference, axis=0)
    uncertainty = np.sqrt(variance.sum(axis=0))
    planes = {name: {"truth": center_plane(reference, bounds, name),
                     "prediction": center_plane(prediction, bounds, name),
                     "error": center_plane(reference, bounds, name, error),
                     "uncertainty": center_plane(reference, bounds, name, uncertainty)} for name in PLANES}
    speed_max = max(float(planes[name][key]["scalar"].max()) for name in PLANES for key in ("truth", "prediction"))
    vector_max = max(float(np.linalg.norm(field, axis=0).max()) for field in (reference, prediction))
    error_max = max(float(planes[name]["error"]["scalar"].max()) for name in PLANES)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row, name in enumerate(PLANES):
        for col, (key, title) in enumerate((("truth", "Ground truth |velocity|"),
                                          ("prediction", "Posterior mean |velocity|"),
                                          ("error", "Vector error ||prediction - truth||"))):
            plane = planes[name][key]
            im = _draw(axes[row, col], plane, vmax=error_max if key == "error" else speed_max,
                       cmap="magma" if key == "error" else "viridis", vectors=key != "error", vector_max=vector_max)
            axes[row, col].set_title(f"{name.upper()}, {plane['fixed']}\n{title}", fontsize=10)
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    fig.suptitle(label + "\nFixed first test case; centre slices; velocity in dataset units", fontsize=11)
    files = _save(fig, output_dir, "flow_slices", dpi)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), constrained_layout=True)
    uncertainty_max = max(float(planes[name]["uncertainty"]["scalar"].max()) for name in PLANES)
    for col, name in enumerate(PLANES):
        for row, key in enumerate(("error", "uncertainty")):
            im = _draw(axes[row, col], planes[name][key], vmax=error_max if row == 0 else uncertainty_max,
                       cmap="magma" if row == 0 else "cividis")
            axes[row, col].set_title(f"{name.upper()} | " + ("Vector error" if row == 0 else "sqrt(sum component variance)"), fontsize=10)
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    note = "Uncertainty is posterior spread, not a calibrated error bound"
    if sample_count == 1:
        note += "; one sample cannot estimate uncertainty"
    fig.suptitle(label + "\n" + note, fontsize=11)
    files += _save(fig, output_dir, "uncertainty", dpi)

    projection = data["projection_matrix"][0]
    if projection.ndim != 3 or projection.shape[1:] != (2, 3):
        raise ValueError("projection_matrix must have shape [1,V,2,3]")
    views = len(projection)
    fig, axes = plt.subplots(2, views, figsize=(4.3 * views, 7.8), constrained_layout=True, squeeze=False)
    track_details = {}
    corners = np.asarray(list(itertools.product(*bounds)))
    for row, (prefix, group_name) in enumerate((("", "Observed particles"), ("probe_", "Held-out probes"))):
        if prefix + "observed_tracks" not in data:
            for ax in axes[row]:
                ax.text(.5, .5, "No held-out probes recorded", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            track_details[group_name] = {"available": 0, "displayed": 0}
            continue
        observed = data[prefix + "observed_tracks"][0]
        replay = data[prefix + "replay_tracks"][0]
        mask = data[prefix + "observation_mask"][0].astype(bool)
        valid = data[prefix + "replay_validity"][0].astype(bool)
        if (observed.ndim != 4 or observed.shape[0] != views or observed.shape[-1] != 2
                or replay.shape != observed.shape or mask.shape != observed.shape[:-1]
                or valid.shape != observed.shape[1:3]):
            raise ValueError("trajectory shapes or validity masks are inconsistent")
        total, shown = observed.shape[1], min(observed.shape[1], 12)
        invalid = int(np.count_nonzero(~valid))
        track_details[group_name] = {"available": total, "displayed": shown, "selection": "first 12 or fewer",
                                    "invalid_replay_points_all_particles": invalid}
        for view, ax in enumerate(axes[row]):
            for particle in range(shown):
                obs = observed[view, particle].copy()
                pred = replay[view, particle].copy()
                obs[~mask[view, particle]] = np.nan
                pred[~(mask[view, particle] & valid[particle])] = np.nan
                ax.plot(*obs.T, color="#0072B2", lw=1, alpha=.8)
                ax.plot(*pred.T, color="#D55E00", lw=1, ls="--", alpha=.8)
            title, xlabel, ylabel = _view_labels(projection[view])
            projected = corners @ projection[view].T
            low, high = projected.min(axis=0), projected.max(axis=0)
            pad = max(float((high - low).max()) * .03, 1e-6)
            ax.set(title=f"{group_name} | {title}\nfirst {shown}/{total}; invalid replay points={invalid}",
                   xlabel=xlabel, ylabel=ylabel, xlim=(low[0] - pad, high[0] + pad),
                   ylim=(low[1] - pad, high[1] + pad), aspect="equal")
            ax.grid(alpha=.2)
    fig.suptitle(label + "\nTracks advected in the posterior-mean field; invalid prediction segments omitted", fontsize=11)
    axes[0, 0].legend(handles=[Line2D([], [], color="#0072B2", label="Reference track"),
                               Line2D([], [], color="#D55E00", ls="--", label="Reconstructed track")], fontsize=8)
    files += _save(fig, output_dir, "trajectory_projections", dpi)
    result = {"source": str(posterior.resolve()), "architecture": architecture, "num_particles": num_particles,
              "training_seed": training_seed, "test_case_index": 0, "case_id": case_id,
              "selection": "preselected test index 0, not median/best", "scientific_result": scientific,
              "num_posterior_samples": sample_count, "metrics": metrics, "track_display": track_details,
              "field_axes": "component(u,v,w), z, y, x", "error_definition": "Euclidean norm of vector difference",
              "uncertainty_definition": "sqrt(var(u)+var(v)+var(w)), empirical population variance",
              "shared_speed_color_max": speed_max, "files": files}
    _write_json(output_dir / "figure_manifest.json", result)
    return result


def _collect_records(task: dict, expected: int, evaluation: dict) -> tuple[dict[int, dict], list[str]]:
    path = Path(task["record_dir"]) / "evaluation" / "runs.jsonl"
    if not path.is_file():
        return {}, []
    accepted, warnings = {}, []
    # A trailing partial line can occur while another process appends progress.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            if not (record.get("status") == "completed" and record.get("returncode") == 0
                    and record.get("scientific_result") is True
                    and record.get("source_scientific_result") is True
                    and record.get("scientific_design") is True
                    and record.get("scientific_result_reasons") == []):
                warnings.append(f"line {line_number}: excluded incomplete/failed/non-scientific record")
                continue
            index = record["test_case_index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < expected:
                raise ValueError("test index outside requested test split")
            if (record.get("num_particles") != task["num_particles"]
                    or record.get("trained_particles") != task["num_particles"]):
                raise ValueError("particle count differs from the task")
            if task.get("checkpoint") and record.get("checkpoint") != task["checkpoint"]["path"]:
                raise ValueError("checkpoint differs from the frozen plan")
            if "seed" in evaluation and record.get("seed") != evaluation["seed"] + index:
                raise ValueError("sampling seed differs from the frozen plan")
            if not isinstance(record.get("case_id"), str) or not record["case_id"]:
                raise ValueError("missing case identity")
            metrics = record.get("metrics", {})
            if (metrics.get("num_observed_particles") != task["num_particles"]
                    or any(not _numeric(metrics.get(name)) for name in METRICS
                           if name != "probe_track_rmse_cells" or evaluation.get("num_probe_particles", 0) > 0)):
                raise ValueError("missing/nonfinite required metric or inconsistent particle count")
            for option, metric in (("num_samples", "num_posterior_samples"), ("num_probe_particles", "num_probe_particles")):
                if option in evaluation and metrics.get(metric) != evaluation[option]:
                    raise ValueError(f"{metric} differs from the frozen plan")
            if index in accepted:
                raise ValueError("duplicate successful test index; first valid measurement retained")
            accepted[index] = record
        except (ValueError, TypeError, KeyError) as exc:
            warnings.append(f"line {line_number}: {exc}")
    return accepted, warnings


def write_report(plan: Path, output_dir: Path | None = None, *, dpi: int = 170) -> dict:
    """Summarize valid case measurements, with test and training-seed variance separate."""
    plan = Path(plan).resolve()
    document = json.loads(plan.read_text(encoding="utf-8"))
    if document.get("mode") != "matrix-evaluate" or document.get("dry_run") is not False:
        raise ValueError("report requires a submitted matrix-evaluate plan")
    expected = document["evaluation"]["num_test_cases"]
    if not isinstance(expected, int) or expected <= 0 or not document["tasks"]:
        raise ValueError("evaluation must request positive test cases and at least one task")
    output_dir = Path(output_dir) if output_dir else plan.parent / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_source = document.get("training_plan")
    if not isinstance(training_source, dict) or not Path(training_source.get("path", "")).is_file():
        raise ValueError("original training plan is required to report intended model/seed coverage")
    training_bytes = Path(training_source["path"]).read_bytes()
    if training_source.get("sha256") and hashlib.sha256(training_bytes).hexdigest() != training_source["sha256"]:
        raise ValueError("original training plan changed after evaluation submission")
    training_document = json.loads(training_bytes)
    original_tasks = training_document["tasks"]
    original_groups = {}
    for original in original_tasks:
        original_groups.setdefault((original["architecture"], original["num_particles"]), set()).add(original["seed"])
    tasks, records, warnings = [], {}, []
    identities = {}
    test_entries = document.get("manifest_content", {}).get("splits", {}).get("test", [])
    if len(test_entries) >= expected:
        identities = {index: entry["case_id"] for index, entry in enumerate(test_entries[:expected])
                      if isinstance(entry, dict) and "case_id" in entry}
    for task in document["tasks"]:
        task_records, rejected = _collect_records(task, expected, document["evaluation"])
        warnings.extend(f"task {task['index']}: {item}" for item in rejected)
        for index in list(task_records):
            case_id = task_records[index]["case_id"]
            if index in identities and identities[index] != case_id:
                warnings.append(f"task {task['index']}: inconsistent case identity at test index {index}; excluded")
                del task_records[index]
            else:
                identities[index] = case_id
        records[task["index"]] = task_records
        summary = {key: task[key] for key in ("index", "architecture", "num_particles", "seed")}
        summary_path = Path(task["record_dir"]) / "evaluation" / "summary.json"
        validated_complete = False
        if summary_path.is_file():
            try:
                sweep_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                validate_evaluation(sweep_summary, document, task)
                validated_complete = True
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                warnings.append(f"task {task['index']}: invalid sweep summary: {exc}")
        status = "complete" if len(task_records) == expected and validated_complete else "partial" if task_records else "missing"
        summary.update(status=status, expected_cases=expected, actual_cases=len(task_records),
                       test_case_indices=sorted(task_records), metrics={})
        for name in METRICS:
            values = [record["metrics"][name] for record in task_records.values() if _numeric(record["metrics"].get(name))]
            summary["metrics"][name] = _stats(values)
        summary["all_probe_particles_valid"] = all(r["metrics"].get("all_probe_particles_valid") is True for r in task_records.values()) if task_records else None
        summary["all_replay_particles_valid"] = all(r["metrics"].get("all_replay_particles_valid") is True for r in task_records.values()) if task_records else None
        tasks.append(summary)

    groups, curves = [], []
    for architecture in sorted({task["architecture"] for task in tasks}):
        for count in sorted({task["num_particles"] for task in tasks if task["architecture"] == architecture}):
            requested = [task for task in tasks if task["architecture"] == architecture and task["num_particles"] == count]
            observed = [task for task in requested if task["status"] == "complete"]
            intended_seeds = sorted(original_groups.get((architecture, count), set()))
            if not intended_seeds or any(task["seed"] not in intended_seeds for task in requested):
                raise ValueError("evaluation task model/count/seed is absent from its original training plan")
            indices = list(range(expected)) if observed else []
            group = {"architecture": architecture, "num_particles": count,
                     "expected_training_seeds": len(intended_seeds), "selected_training_seeds": len(requested),
                     "actual_training_seeds": len(observed), "expected_seed_values": intended_seeds,
                     "training_seeds": [task["seed"] for task in observed], "common_test_cases": len(indices),
                     "status": "complete" if len(observed) == len(intended_seeds) else "partial",
                     "metrics_across_training_seed_means": {}}
            curve = {"architecture": architecture, "num_particles": count, "training_seeds": group["training_seeds"],
                     "expected_training_seeds": len(intended_seeds),
                     "paired_test_case_indices": indices, "metrics_across_training_seed_means": {}}
            for name in METRICS:
                for result in (group, curve):
                    values = []
                    for task in observed:
                        selected = [records[task["index"]][i]["metrics"].get(name) for i in indices]
                        if selected and all(_numeric(value) for value in selected):
                            values.append(statistics.fmean(selected))
                    result["metrics_across_training_seed_means"][name] = _stats(values)
            groups.append(group)
            curves.append(curve)
    complete = all(task["status"] == "complete" for task in tasks)
    result = {"plan": str(plan), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "complete" if complete else "partial", "expected_tasks": len(tasks),
              "complete_tasks": sum(task["status"] == "complete" for task in tasks),
              "original_training_matrix_tasks": len(original_tasks), "selected_evaluation_tasks": len(tasks),
              "original_matrix_complete_evaluations": sum(task["status"] == "complete" for task in tasks),
              "training_selection": document.get("training_selection"),
              "expected_case_runs": len(tasks) * expected, "actual_case_runs": sum(task["actual_cases"] for task in tasks),
              "evaluation": document["evaluation"], "tasks": tasks, "groups": groups, "curves": curves,
              "warnings": warnings, "std_definition": "sample standard deviation, ddof=1; null for fewer than two observations",
              "curve_protocol": "only seeds covering every requested test case with validated complete scientific sweep summaries; partial test means excluded; missing seeds are labelled"}
    _write_json(output_dir / "summary.json", result)
    with (output_dir / "per_seed.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "architecture", "N", "training_seed", "status", "actual_cases", "expected_cases",
                         *[f"{name}_{suffix}" for name in METRICS for suffix in ("mean", "test_sample_std")]])
        for task in tasks:
            writer.writerow([task[key] for key in ("index", "architecture", "num_particles", "seed", "status", "actual_cases", "expected_cases")]
                            + [task["metrics"][name][key] for name in METRICS for key in ("mean", "sample_std")])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    count_ticks = sorted({task["num_particles"] for task in tasks})
    for ax, name in zip(axes, ("field_relative_l2_unknown_interior", "probe_track_rmse_cells")):
        any_data = False
        for architecture in sorted({task["architecture"] for task in tasks}):
            selected = [group for group in curves if group["architecture"] == architecture
                        and group["metrics_across_training_seed_means"][name]["mean"] is not None]
            if not selected:
                continue
            any_data = True
            x = [group["num_particles"] for group in selected]
            y = [group["metrics_across_training_seed_means"][name]["mean"] for group in selected]
            line, = ax.plot(x, y, marker="o", label=f"{architecture}; paired test cases={len(selected[0]['paired_test_case_indices'])}")
            for group, xpos, ypos in zip(selected, x, y):
                stats = group["metrics_across_training_seed_means"][name]
                if stats["sample_std"] is not None:
                    ax.errorbar(xpos, ypos, yerr=stats["sample_std"], color=line.get_color(), capsize=3)
                ax.annotate(f"{stats['count']}/{group['expected_training_seeds']} seeds", (xpos, ypos), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
        if any_data:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "No valid comparable results yet", ha="center", va="center", transform=ax.transAxes)
        ax.set(xlabel="Number of observed particles", ylabel=METRICS[name])
        ax.set_xscale("log", base=2)
        ax.set_xticks(count_ticks, labels=[str(value) for value in count_ticks])
        ax.set_xlim(min(count_ticks) * .8, max(count_ticks) * 1.25)
        ax.grid(alpha=.2)
    fig.suptitle(f"Selected evaluation {'complete' if complete else 'PARTIAL'}: {result['complete_tasks']}/{len(tasks)} models; "
                 f"original matrix={len(original_tasks)} models\n"
                 "Error bars: sample SD across training-seed test means (not a confidence interval)", fontsize=11)
    result["curve_files"] = _save(fig, output_dir, "particle_error_curves", dpi)
    _write_json(output_dir / "summary.json", result)

    def number(value):
        return "—" if value is None else f"{value:.5g}"

    lines = ["# 流场重建测试：组会结果包", "", f"状态：**{'全量完成' if complete else '部分结果，请勿当作完整实验结论'}**。",
             f"本次计划 {len(tasks)} 个模型任务，每个 {expected} 个测试案例；已完整完成 {result['complete_tasks']} 个任务，",
             f"已有 {result['actual_case_runs']}/{result['expected_case_runs']} 条通过科学性检查的案例记录。", "",
             f"原训练矩阵共有 **{len(original_tasks)}** 个模型设置；本次只选入 **{len(tasks)}** 个，完整评估覆盖为 **{result['complete_tasks']}/{len(original_tasks)}**。",
             "已选模型的全部测试案例完成，不代表原训练矩阵全部模型已评估。选择在提交时冻结，之后完成的训练不会自动加入。", "",
             "## 测试协议", "", f"后验样本数：{document['evaluation'].get('num_samples', '未记录')}；采样步数：{document['evaluation'].get('sampling_steps', '未记录')}；"
             f"独立探针粒子数：{document['evaluation'].get('num_probe_particles', '未记录')}。",
             "只统计成功、科学性标识通过、粒子数和冻结计划相符的记录。失败、缺失、非有限指标不会填成零。",
             "统计单位是一个测试流场；后验样本不是独立测试案例。训练验证损失不等于以下流场重建误差。", "",
             "## 各训练种子的测试结果", "",
             "| 模型 | 粒子数 | 训练种子 | 已测/应测 | 流场相对 L2（案例均值 ± 案例样本标准差） | 独立探针 RMSE（cells） | 状态 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for task in tasks:
        l2, probe = task["metrics"]["field_relative_l2_unknown_interior"], task["metrics"]["probe_track_rmse_cells"]
        lines.append(f"| {task['architecture']} | {task['num_particles']} | {task['seed']} | {task['actual_cases']}/{expected} | "
                     f"{number(l2['mean'])} ± {number(l2['sample_std'])} | {number(probe['mean'])} ± {number(probe['sample_std'])} | {task['status']} |")
    lines += ["", "## 粒子数量对比", "", "![粒子数与测试误差](particle_error_curves.png)", "",
              "曲线只使用已覆盖全部指定测试案例、且完整科学性汇总通过检查的训练种子，再对各训练种子的测试集均值取平均。",
              "部分案例的均值只出现在上述逐种子表格，不混入完整测试集曲线。每点标明完成种子数/原训练计划预期种子数。",
              "误差棒是训练种子间的样本标准差，不是案例间标准差，也不是置信区间；只有一个种子时不画误差棒。",
              "不同粒子数若完成的训练种子不同，应将结论视为初步结果；JSON 中保留具体种子和完整配对案例。", "",
              "## 可用于汇报的单案例图片", "", "固定展示测试索引 0；这是事先选定的示例，不是中位、最好或最差样例。",
              "流场图展示真实场、后验均值和三分量向量误差的 XY/XZ/YZ 中心切片，真实场和预测共享速度色标。",
              "轨迹图区分观测粒子与未参与条件输入的独立探针；为保持清晰，每组最多展示前 12 条轨迹，指标仍按全部粒子计算。",
              "不确定性图展示后验分散程度，不能直接解释为已校准的误差区间。", ""]
    for task in document["tasks"]:
        directory = (Path(task["record_dir"]) / "evaluation" / "figures"
                     / f"N{task['num_particles']:03d}_case0000")
        if (directory / "figure_manifest.json").is_file():
            relative = Path("cases") / f"{task['index']:03d}_{task['architecture']}_N{task['num_particles']}_seed{task['seed']}"
            destination = output_dir / relative
            destination.mkdir(parents=True, exist_ok=True)
            allowed = ["figure_manifest.json"] + [f"{stem}.{extension}" for stem in ("flow_slices", "uncertainty", "trajectory_projections") for extension in ("png", "pdf")]
            for name in allowed:
                source, target = directory / name, destination / name
                if source.is_file() and source.resolve() != target.resolve():
                    shutil.copy2(source, target)
            label = f"{task['architecture']}，N={task['num_particles']}，训练种子 {task['seed']}"
            lines.append(f"- {label}：[流场]({relative.as_posix()}/flow_slices.png)、[轨迹]({relative.as_posix()}/trajectory_projections.png)、[不确定性]({relative.as_posix()}/uncertainty.png)、[来源记录]({relative.as_posix()}/figure_manifest.json)")
    lines += ["", "## 汇报边界", "", "目前报告只覆盖上述已完成或部分完成的模型；其余训练任务不能据此推断结果。",
              "先核对所有探针/重放有效性，再解释轨迹误差。图包为实际推理结果，不把损失下降等同于流场重建成功。",
              "全量数据保存在 `summary.json` 和 `per_seed.csv`；跨训练种子统计与每个种子的跨测试案例统计分开保存。"]
    invalid = [task for task in tasks if task["all_probe_particles_valid"] is False or task["all_replay_particles_valid"] is False]
    if invalid:
        lines += ["", "**注意：以下任务存在无效或未报告有效性的轨迹点，不能仅凭 RMSE 判断轨迹重建成功：** "
                  + ", ".join(str(task["index"]) for task in invalid)]
    if warnings:
        lines += ["", f"排除/解析提示共 {len(warnings)} 条，详见 `summary.json` 的 warnings。"]
    (output_dir / "meeting_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    case = commands.add_parser("case", help="Plot the fixed first test case from a posterior NPZ")
    case.add_argument("--posterior", type=Path, required=True)
    case.add_argument("--output-dir", type=Path, required=True)
    case.add_argument("--architecture", required=True)
    case.add_argument("--num-particles", type=int, required=True)
    case.add_argument("--training-seed", type=int, required=True)
    case.add_argument("--test-case-index", type=int, default=0)
    case.add_argument("--dpi", type=int, default=170)
    report = commands.add_parser("report", help="Summarize complete or partial evaluation tasks")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()
    if args.command == "case":
        result = write_task_figures(args.posterior, args.output_dir, architecture=args.architecture,
                                   num_particles=args.num_particles, training_seed=args.training_seed,
                                   test_case_index=args.test_case_index, dpi=args.dpi)
        print(f"Wrote {len(result['files'])} figures to {args.output_dir}")
    else:
        result = write_report(args.plan, args.output_dir, dpi=args.dpi)
        print(f"Evaluation report: {result['status']}, {result['actual_case_runs']}/{result['expected_case_runs']} case runs")


if __name__ == "__main__":
    main()
