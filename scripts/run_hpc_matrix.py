#!/usr/bin/env python
"""Plan and execute reproducible single-GH200 Slurm experiment arrays.

Planning uses the Python standard library only. GPU imports and computation
occur exclusively in the `run` command, which requires a Slurm allocation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import tempfile
import time
import uuid

REPOSITORY = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("unet3d", "dit3d", "tensor-dit")
COUNTS = (2, 4, 6, 12, 24, 48, 96)
SEEDS = (31, 32, 33)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path: Path) -> dict:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return {"path": str(path), "sha256": digest(path)}


def verify_identity(item: dict) -> Path:
    path = Path(item["path"])
    if not path.is_file() or digest(path) != item["sha256"]:
        raise ValueError(f"file changed or disappeared since planning: {path}")
    return path


def read_plan(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format_version") != 1 or not document.get("tasks"):
        raise ValueError("invalid matrix plan")
    return document


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def ranks(value: str) -> list[int]:
    values = [int(part) for part in value.split(",")]
    if not values or len(set(values)) != len(values) or any(r < 1 or r >= 32 for r in values):
        raise argparse.ArgumentTypeError("ranks must be distinct integers between 1 and 31")
    return values


def training_rows(phase: str) -> list[dict]:
    counts = (2, 24, 96) if phase == "pilot" else COUNTS
    seeds = (31,) if phase == "pilot" else SEEDS
    return [{"index": index, "architecture": architecture, "num_particles": count,
             "seed": seed, "key": f"{architecture}_N{count:03d}_s{seed}"}
            for index, (architecture, count, seed) in
            enumerate(itertools.product(ARCHITECTURES, counts, seeds))]


def selected_array(document: dict, selection: str) -> str:
    count = len(document["tasks"])
    if selection == "all":
        return f"0-{count - 1}" if count > 1 else "0"
    if not re.fullmatch(r"\d+(?:,\d+)*", selection):
        raise ValueError("FLOW3D_ARRAY_TASKS must be all or explicit comma-separated task IDs")
    indices = [int(value) for value in selection.split(",")]
    if len(set(indices)) != len(indices) or any(index >= count for index in indices):
        raise ValueError("duplicate or out-of-range array task ID")
    return ",".join(str(index) for index in indices)


def validate_time(value: str) -> None:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError("time must be HH:MM:SS or D-HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if minutes >= 60 or seconds >= 60 or not 0 < total <= 48 * 3600:
        raise ValueError("ghx4 time must be positive and at most 48:00:00")


def resolve_followup(document: dict, plan_path: Path) -> dict:
    """Resolve a deferred artifact from verified successful preparation only.

    All array tasks independently reproduce the same deterministic selection.
    Atomic publication pins it for subsequent tasks, resumes and evaluation.
    No tensor loading or computation is needed here.
    """
    followup = document.get("followup")
    if not followup:
        return document
    from select_tensor_rank import select_rank

    prepare_path = verify_identity(followup["prepare_plan"])
    prepare = read_plan(prepare_path)
    verify_identity(document["manifest"])
    if prepare["mode"] != "tensor-prepare" or prepare["manifest"] != document["manifest"]:
        raise ValueError("preparation does not match the follow-up manifest")
    result_path = Path(prepare["tasks"][0]["record_dir"]) / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed" or result.get("plan_sha256") != followup["prepare_plan"]["sha256"]:
        raise ValueError("preparation has not completed successfully with the expected plan")
    environment_path = Path(result["attempt"]) / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    preparation_job = environment.get("slurm_array_job_id") or environment.get("slurm_job_id")
    if str(preparation_job) != followup["after_job"]:
        raise ValueError("--after-job does not match the completed preparation job")
    report_path = verify_identity(result["report"])
    if report_path.resolve() != (Path(prepare["prepare_output"]) / "report.json").resolve():
        raise ValueError("preparation report is outside the planned output")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if Path(report["manifest"]).resolve() != Path(document["manifest"]["path"]):
        raise ValueError("rank report uses a different manifest")
    if set(report["ranks"]) != {str(rank) for rank in prepare["ranks"]}:
        raise ValueError("rank report does not contain exactly the planned ranks")
    selected = select_rank(report, **followup["rank_policy"])
    artifact = selected["artifact"]
    expected = Path(prepare["prepare_output"]) / f"rank_{selected['selected_rank']}.pt"
    if Path(artifact["path"]).resolve() != expected.resolve() or artifact not in result["artifacts"]:
        raise ValueError("selected artifact is not recorded by the successful preparation")
    verify_identity(artifact)
    selection = {"followup_plan_sha256": digest(plan_path), "prepare_plan": followup["prepare_plan"],
                 "prepare_result": identity(result_path), "prepare_environment": identity(environment_path),
                 "report": result["report"], **selected}
    selection_path = plan_path.parent / "rank_selection.json"
    # Hard-link publication is atomic on the target filesystem and will not
    # replace another task's selection. Concurrent tasks must agree exactly.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".selection-", dir=selection_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, selection_path)
        except FileExistsError:
            if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
                raise ValueError("rank selection or preparation changed after first resolution")
    finally:
        temporary.unlink(missing_ok=True)
    return {**document, "tensor_artifact": artifact, "rank_selection": identity(selection_path)}


def followup_dependency(document: dict, plan_path: Path) -> str:
    followup = document.get("followup")
    if not followup:
        return ""
    prepare = read_plan(verify_identity(followup["prepare_plan"]))
    result_path = Path(prepare["tasks"][0]["record_dir"]) / "results.json"
    if result_path.is_file():
        status = json.loads(result_path.read_text(encoding="utf-8")).get("status")
        if status == "completed":
            resolve_followup(document, plan_path)
            # Old completed Slurm jobs may no longer accept dependencies.
            return ""
        if status != "running":
            raise ValueError("preparation failed; cannot queue a successful follow-up")
    return followup["after_job"]


def make_plan(args: argparse.Namespace) -> Path:
    root = args.storage_root.expanduser().resolve()
    if not re.fullmatch(r"[0-9a-f]{40,64}", args.commit):
        raise ValueError("a full Git commit is required")
    code_dir = args.code_dir.expanduser().resolve()
    training_plan = None
    followup = None
    effective_mode = args.mode
    phase = args.phase
    if args.mode == "matrix-evaluate":
        if args.training_plan is None:
            raise ValueError("matrix-evaluate requires --training-plan")
        training_plan = read_plan(args.training_plan)
        if training_plan["mode"] != "matrix-train" or training_plan["dry_run"]:
            raise ValueError("evaluation requires a submitted training plan")
        training_plan = resolve_followup(training_plan, args.training_plan)
        manifest = verify_identity(training_plan["manifest"])
        rows = []
        for training_task in training_plan["tasks"]:
            result_path = Path(training_task["record_dir"]) / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"training not complete: {training_task['key']}")
            if training_plan.get("followup") and result.get("tensor_artifact") != training_plan["tensor_artifact"]:
                raise ValueError("training tasks used inconsistent automatic rank selections")
            verify_identity(result["best_checkpoint"])
            rows.append({key: training_task[key] for key in
                         ("index", "key", "architecture", "num_particles", "seed")})
            rows[-1]["checkpoint"] = result["best_checkpoint"]
    elif args.mode == "matrix-followup":
        if args.prepare_plan is None or args.after_job is None:
            raise ValueError("matrix-followup requires --prepare-plan and --after-job")
        if args.tensor_artifact is not None:
            raise ValueError("matrix-followup selects its artifact automatically")
        if args.phase not in (None, "pilot"):
            raise ValueError("matrix-followup starts the 9-task pilot only")
        for limit in (args.rank_mean_limit, args.rank_max_limit):
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError("automatic rank limits must be finite and positive")
        if args.rank_mean_limit > args.rank_max_limit:
            raise ValueError("rank mean limit must not exceed the maximum limit")
        prepare_path = args.prepare_plan.expanduser().resolve(strict=True)
        prepare = read_plan(prepare_path)
        if prepare["mode"] != "tensor-prepare" or prepare["dry_run"] or len(prepare["tasks"]) != 1:
            raise ValueError("follow-up requires a submitted single-task preparation plan")
        if Path(prepare["storage_root"]).resolve() != root:
            raise ValueError("follow-up must use the preparation storage root")
        manifest = verify_identity(prepare["manifest"])
        phase, effective_mode = "pilot", "matrix-train"
        rows = training_rows(phase)
        followup = {"prepare_plan": identity(prepare_path), "prepare_commit": prepare["commit"],
                    "after_job": str(args.after_job),
                    "rank_policy": {"mean_limit": args.rank_mean_limit, "max_limit": args.rank_max_limit}}
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required")
        manifest = args.manifest.expanduser().resolve(strict=True)
        phase = phase or "full"
        rows = training_rows(phase) if args.mode == "matrix-train" else [{"index": 0, "key": "tensor-prepare"}]
    manifest_id = identity(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("mode", "scientific") != "scientific" or data.get("allow_overlap_for_smoke_test"):
        raise ValueError("matrix experiments require a scientific manifest")
    split_sizes = {split: len(data.get("splits", {}).get(split, []))
                   for split in ("train", "validation", "test")}
    if not all(split_sizes.values()):
        raise ValueError("manifest requires nonempty train, validation and test splits")
    artifact = training_plan.get("tensor_artifact") if training_plan else None
    if artifact:
        verify_identity(artifact)
    if args.mode == "matrix-train":
        if args.tensor_artifact is None:
            raise ValueError("choose a validated rank explicitly with --tensor-artifact rank_R.pt")
        artifact = identity(args.tensor_artifact)
    plan_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S") + f"_{args.mode}_{uuid.uuid4().hex[:8]}"
    directory = root / "results" / "matrices" / plan_id
    for row in rows:
        row["record_dir"] = str(directory / "tasks" / f"{row['index']:03d}_{row['key']}")
        if effective_mode == "matrix-train":
            row["checkpoint_dir"] = str(root / "checkpoints" / plan_id / row["key"])
    document = {
        "format_version": 1, "plan_id": plan_id, "mode": effective_mode,
        "dry_run": args.dry_run, "commit": args.commit, "code_dir": str(code_dir),
        "storage_root": str(root), "manifest": manifest_id, "split_sizes": split_sizes,
        "manifest_content": data, "tensor_artifact": artifact, "tasks": rows,
        "phase": phase or "full", "epochs": args.epochs or (10 if phase == "pilot" else 100),
        "ranks": prepare["ranks"] if followup else args.ranks,
        "prepare_output": prepare["prepare_output"] if followup else str(root / "datasets" / "tensor_space" / plan_id),
        "train_options": {"batch_size": 1, "num_workers": 0, "learning_rate": 0.0002,
                          "weight_decay": 0.0001, "diffusion_steps": 1000,
                          "dit_hidden_dim": 192, "dit_depth": 4, "dit_heads": 6, "patch_size": 4},
        "evaluation": {"num_test_cases": split_sizes["test"], "num_probe_particles": 64,
                       "num_samples": 16, "sampling_steps": 50, "seed": 47,
                       "eta": 1.0, "cfg_scale": 1.5, "boundary_projection": "final"},
        "training_plan": identity(args.training_plan) if training_plan else None,
        "training_commit": training_plan["commit"] if training_plan else None,
        "followup": followup,
        "training_rank_selection": training_plan.get("rank_selection") if training_plan else None,
    }
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "plan.json"
    write_json(path, document)
    path.chmod(0o444)
    return path


def task_command(document: dict, task: dict, *, resume: bool = False) -> list[str]:
    repo = Path(document["code_dir"]).parent
    common = ["--manifest", document["manifest"]["path"], "--device", "cuda"]
    if document["mode"] == "tensor-prepare":
        if resume:
            raise ValueError("tensor preparation cannot resume; submit a new preparation plan")
        return [sys.executable, str(repo / "scripts" / "prepare_tensor_space.py"), *common,
                "--output-dir", document["prepare_output"], "--ranks", ",".join(map(str, document["ranks"]))]
    if document["mode"] == "matrix-evaluate":
        command = [sys.executable, str(repo / "scripts" / "run_sparse_particle_sweep.py"), *common,
                   "--particle-counts", str(task["num_particles"]), "--checkpoint",
                   f"{task['num_particles']}={task['checkpoint']['path']}",
                   "--output-dir", str(Path(task["record_dir"]) / "evaluation")]
        for key, value in document["evaluation"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
        if resume:
            command += ["--resume"]
        return command
    command = [sys.executable, str(repo / "scripts" / "train_sparse_track_diffusion.py"), *common,
               "--architecture", task["architecture"], "--num-particles", str(task["num_particles"]),
               "--seed", str(task["seed"]), "--epochs", str(document["epochs"]),
               "--output-dir", task["checkpoint_dir"], "--save-every", "10"]
    for key, value in document["train_options"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if task["architecture"] == "tensor-dit":
        command += ["--tensor-artifact", document["tensor_artifact"]["path"]]
    if resume:
        latest = Path(task["checkpoint_dir"]) / "latest.pt"
        if latest.is_file():
            command += ["--resume", str(latest)]
        elif Path(task["checkpoint_dir"]).exists() and any(Path(task["checkpoint_dir"]).iterdir()):
            raise ValueError("checkpoint directory is occupied but latest.pt is missing")
    return command


def validate_evaluation(summary: dict, document: dict, task: dict) -> None:
    expected = document["evaluation"]["num_test_cases"]
    group = summary.get("per_particle_count", {}).get(str(task["num_particles"]), {})
    if (summary.get("complete") is not True or summary.get("scientific_result") is not True
            or summary.get("dry_run") is not False or summary.get("failed_runs") != 0
            or summary.get("completed_runs") != expected or group.get("completed_runs") != expected
            or group.get("trained_particles") != task["num_particles"]
            or summary.get("boundary_projection") != "final"):
        raise ValueError("evaluation incomplete or scientific provenance/boundary protocol failed")


def summarize_plan(plan_path: Path, output_dir: Path | None = None) -> dict:
    """Aggregate per-seed test means; retain missing/failed tasks and case records."""
    document = read_plan(plan_path)
    if document["mode"] != "matrix-evaluate":
        raise ValueError("summarize requires an evaluation plan")
    output_dir = output_dir or plan_path.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, per_seed, per_case = [], [], []
    by_group = {}
    for task in document["tasks"]:
        item = {key: task[key] for key in ("index", "key", "architecture", "num_particles", "seed")}
        item.update(status="missing", metrics={})
        record = Path(task["record_dir"])
        result_path = record / "results.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                item["status"] = result.get("status", "unknown")
                item["error"] = result.get("error")
                if item["status"] == "completed":
                    summary = json.loads(verify_identity(result["summary"]).read_text(encoding="utf-8"))
                    validate_evaluation(summary, document, task)
                    item["metrics"] = summary["per_particle_count"][str(task["num_particles"])]["metrics"]
            except (ValueError, OSError, KeyError) as error:
                item.update(status="invalid", error=str(error), metrics={})
        cases_path = record / "evaluation" / "runs.jsonl"
        if cases_path.exists():
            for line in cases_path.read_text(encoding="utf-8").splitlines():
                try:
                    case_record = json.loads(line)
                except ValueError:
                    case_record = {"status": "invalid_json", "raw": line}
                per_case.append({"matrix_task": task["key"], "architecture": task["architecture"],
                                 "training_seed": task["seed"], "record": case_record})
        tasks.append(item)
        by_group.setdefault((task["architecture"], task["num_particles"]), []).append(item)
        base = {key: item[key] for key in ("architecture", "num_particles", "seed", "status")}
        if not item["metrics"]:
            per_seed.append({**base, "error": item.get("error"), "metric": "", "test_mean": "", "test_std": ""})
        for name, values in item["metrics"].items():
            per_seed.append({**base, "error": "", "metric": name,
                             "test_mean": values["mean"], "test_std": values["std"]})
    aggregates = []
    for (architecture, count), members in sorted(by_group.items()):
        valid = [member for member in members if member["status"] == "completed"]
        metric_names = sorted({name for member in valid for name in member["metrics"]}) or [""]
        for name in metric_names:
            values = [member["metrics"][name]["mean"] for member in valid
                      if name in member["metrics"] and math.isfinite(member["metrics"][name]["mean"])]
            aggregates.append({"architecture": architecture, "num_particles": count,
                               "expected_seeds": len(members), "valid_seeds": len(values),
                               "complete": len(values) == len(members), "metric": name,
                               "seed_mean": statistics.fmean(values) if values else None,
                               "seed_std": statistics.stdev(values) if len(values) > 1 else None})
    result = {"plan": str(plan_path.resolve()), "plan_sha256": digest(plan_path),
              "complete": all(item["status"] == "completed" for item in tasks),
              "expected_tasks": len(tasks), "completed_tasks": sum(item["status"] == "completed" for item in tasks),
              "seed_std_ddof": 1, "aggregation": "mean and sample std of per-training-seed test means",
              "tasks": tasks, "aggregates": aggregates}
    write_json(output_dir / "summary.json", result)
    for name, rows, fields in (
        ("per_seed.csv", per_seed, ["architecture", "num_particles", "seed", "status", "error", "metric", "test_mean", "test_std"]),
        ("aggregate.csv", aggregates, ["architecture", "num_particles", "metric", "expected_seeds", "valid_seeds", "complete", "seed_mean", "seed_std"]),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "per_case.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_case), encoding="utf-8")
    return result


def run_task(args: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU execution requires Slurm; submit via project/scripts/submit.sh")
    if digest(args.plan) != os.environ.get("FLOW3D_PLAN_SHA256"):
        raise ValueError("plan hash missing or changed since submission")
    document = read_plan(args.plan)
    if document["dry_run"]:
        raise ValueError("dry-run plans cannot execute")
    if os.environ.get("FLOW3D_COMMIT") != document["commit"]:
        raise ValueError("code commit does not match frozen plan")
    if Path(document["code_dir"]).parent.resolve() != REPOSITORY:
        raise ValueError("execution must use the original frozen repository")
    root = Path(document["storage_root"]).resolve()
    if not root.is_relative_to(Path("/work")):
        raise ValueError("GPU outputs must be under /work")
    if os.environ.get("FLOW3D_CLUSTER") != "deltaai":
        raise ValueError("matrix jobs require the DeltaAI profile")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index < 0 or index >= len(document["tasks"]):
        raise ValueError("invalid Slurm array index")
    task = document["tasks"][index]
    verify_identity(document["manifest"])
    if document.get("training_rank_selection"):
        verify_identity(document["training_rank_selection"])
    if document["tensor_artifact"]:
        verify_identity(document["tensor_artifact"])
    if task.get("checkpoint"):
        verify_identity(task["checkpoint"])
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if platform.machine() != "aarch64" or "GH200" not in gpu:
        raise ValueError(f"this matrix requires ARM/GH200, got {platform.machine()}/{gpu}")
    record = Path(task["record_dir"])
    record.mkdir(parents=True, exist_ok=True)
    import fcntl
    from run_hpc_diffusion import run_logged
    with (record / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("this matrix task is already running") from error
        result_path = record / "results.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("plan_sha256") != digest(args.plan):
                raise ValueError("previous attempt used a different plan; resume must preserve configuration")
            if prior.get("status") == "completed":
                raise ValueError("completed task cannot be overwritten")
            if not args.resume:
                raise ValueError("task has previous output; use explicit matrix-resume")
        attempt = record / "attempts" / (os.environ["SLURM_JOB_ID"] + "_" + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True)
        write_json(attempt / "environment.json", {
            "commit": document["commit"], "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu,
            "architecture": platform.machine(), "host": platform.node(),
            "modules": os.environ.get("LOADEDMODULES"), "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "array_task_id": index,
            "account": os.environ.get("SLURM_JOB_ACCOUNT"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "submit_command": os.environ.get("FLOW3D_SUBMIT_COMMAND"),
        })
        started = time.monotonic()
        result = {"status": "running", "task": task, "attempt": str(attempt),
                  "manifest": document["manifest"], "tensor_artifact": document["tensor_artifact"],
                  "plan_sha256": digest(args.plan)}
        write_json(result_path, result)
        try:
            document = resolve_followup(document, args.plan)
            result["tensor_artifact"] = document["tensor_artifact"]
            result["rank_selection"] = document.get("rank_selection")
            if result["rank_selection"]:
                print(f"Automatic rank selection: {result['rank_selection']['path']}", flush=True)
            if document["tensor_artifact"]:
                # Every architecture verifies the same prepared dataset even
                # when it does not consume the low-dimensional representation.
                sys.path.insert(0, str(REPOSITORY / "src"))
                from flow_observation.tensor_space import load_tensor_space_artifact
                load_tensor_space_artifact(document["tensor_artifact"]["path"],
                                           manifest_path=document["manifest"]["path"], verify_files=True)
            command = task_command(document, task, resume=args.resume)
            write_json(attempt / "config.yaml", {"plan": str(args.plan), "plan_sha256": digest(args.plan),
                                                "task": task, "command": command,
                                                "rank_selection": document.get("rank_selection")})
            run_logged(command, attempt)
            if document["mode"] == "matrix-train":
                result["training"] = json.loads((Path(task["checkpoint_dir"]) / "training_summary.json").read_text())
                result["best_checkpoint"] = identity(Path(task["checkpoint_dir"]) / "best.pt")
            elif document["mode"] == "matrix-evaluate":
                summary = Path(task["record_dir"]) / "evaluation" / "summary.json"
                validate_evaluation(json.loads(summary.read_text(encoding="utf-8")), document, task)
                result["summary"] = identity(summary)
            else:
                report = Path(document["prepare_output"]) / "report.json"
                result["report"] = identity(report)
                result["artifacts"] = [identity(Path(document["prepare_output"]) / f"rank_{rank}.pt")
                                       for rank in document["ranks"]]
            result["status"] = "completed"
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(result_path, result)
            write_json(attempt / "results.json", result)
        print(f"Completed. Results: {result_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    plan = subparsers.add_parser("plan", allow_abbrev=False)
    plan.add_argument("mode", choices=("tensor-prepare", "matrix-train", "matrix-evaluate", "matrix-followup"))
    plan.add_argument("--manifest", type=Path, default=os.environ.get("FLOW3D_MANIFEST"))
    plan.add_argument("--tensor-artifact", type=Path)
    plan.add_argument("--training-plan", type=Path)
    plan.add_argument("--prepare-plan", type=Path)
    plan.add_argument("--after-job", type=positive)
    plan.add_argument("--rank-mean-limit", type=float, default=0.05)
    plan.add_argument("--rank-max-limit", type=float, default=0.20)
    plan.add_argument("--storage-root", type=Path, default=os.environ.get("FLOW3D_ROOT"), required=not os.environ.get("FLOW3D_ROOT"))
    plan.add_argument("--code-dir", type=Path, required=True)
    plan.add_argument("--commit", required=True)
    plan.add_argument("--phase", choices=("pilot", "full"), default=None)
    plan.add_argument("--epochs", type=positive)
    plan.add_argument("--ranks", type=ranks, default=[4, 8, 12, 16])
    plan.add_argument("--dry-run", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--plan", type=Path, required=True)
    inspect.add_argument("--field", choices=("mode", "code_dir", "commit", "count", "sha256"), required=True)
    array = subparsers.add_parser("array")
    array.add_argument("--plan", type=Path, required=True)
    array.add_argument("--tasks", default="all")
    dependency = subparsers.add_parser("dependency")
    dependency.add_argument("--plan", type=Path, required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--plan", type=Path, required=True)
    resume.add_argument("--dry-run", action="store_true")
    timing = subparsers.add_parser("check-time")
    timing.add_argument("value")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            print(make_plan(args))
        elif args.action == "run":
            run_task(args)
        elif args.action == "inspect":
            document = read_plan(args.plan)
            print(digest(args.plan) if args.field == "sha256" else
                  len(document["tasks"]) if args.field == "count" else document[args.field])
        elif args.action == "array":
            print(selected_array(read_plan(args.plan), args.tasks))
        elif args.action == "dependency":
            print(followup_dependency(read_plan(args.plan), args.plan))
        elif args.action == "resume":
            document = read_plan(args.plan)
            if document["dry_run"] or document["mode"] == "tensor-prepare":
                raise ValueError("only submitted training/evaluation plans support resume")
            marker = Path(document["code_dir"]) / ".flow3d-commit"
            if not marker.is_file() or marker.read_text().strip() != document["commit"]:
                raise ValueError("original code snapshot unavailable")
            if Path(os.environ.get("FLOW3D_ROOT", "")).resolve() != Path(document["storage_root"]):
                raise ValueError("FLOW3D_ROOT differs from original plan")
            verify_identity(document["manifest"])
            document = resolve_followup(document, args.plan)
            if document["tensor_artifact"]:
                verify_identity(document["tensor_artifact"])
            print(args.plan.resolve())
        elif args.action == "summarize":
            summary = summarize_plan(args.plan, args.output_dir)
            print(json.dumps({key: summary[key] for key in ("complete", "expected_tasks", "completed_tasks")}))
        else:
            validate_time(args.value)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"flow3d matrix: {error}\n")


if __name__ == "__main__":
    main()
