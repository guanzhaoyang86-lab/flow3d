#!/usr/bin/env python
"""Run and summarize the sparse-particle posterior reconstruction sweep.

The scientific default requires a checkpoint trained for every requested
particle count.  Reusing one fixed-count checkpoint at other counts is allowed
only behind the explicit ``--allow-untrained-count-extrapolation`` flag, and
all resulting records are marked non-scientific.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SAMPLER = _REPOSITORY_ROOT / "scripts" / "sample_sparse_track_diffusion.py"
_SUPPORTED_PARTICLE_COUNTS = (2, 4, 8, 32, 64, 128)
_EXTRAPOLATION_WARNING = (
    "NON-SCIENTIFIC: one fixed-particle-count checkpoint was evaluated at "
    "untrained particle counts. These runs are exploratory extrapolations, "
    "not a valid particle-count ablation."
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _particle_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "particle counts must be comma-separated integers"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one particle count is required")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("particle counts contain duplicates")
    unsupported = [count for count in parsed if count not in _SUPPORTED_PARTICLE_COUNTS]
    if unsupported:
        allowed = ", ".join(str(item) for item in _SUPPORTED_PARTICLE_COUNTS)
        raise argparse.ArgumentTypeError(
            f"particle counts must be selected from {allowed}"
        )
    return parsed


def _checkpoint_spec(value: str) -> tuple[int, Path]:
    count_text, separator, path_text = value.partition("=")
    if not separator or not count_text.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("checkpoint must use N=PATH syntax")
    try:
        count = int(count_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint N must be an integer") from error
    if count not in _SUPPORTED_PARTICLE_COUNTS:
        allowed = ", ".join(str(item) for item in _SUPPORTED_PARTICLE_COUNTS)
        raise argparse.ArgumentTypeError(f"checkpoint N must be one of {allowed}")
    return count, Path(path_text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a requested subset of sparse-track diffusion models and "
            "write per-run plus mean/std summaries."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--particle-counts",
        type=_particle_counts,
        default=_SUPPORTED_PARTICLE_COUNTS,
        help=(
            "Comma-separated subset selected from 2,4,8,32,64,128 "
            "(default: all six)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_spec,
        default=[],
        metavar="N=PATH",
        help=(
            "Checkpoint trained for exactly N particles. Repeat once for every "
            "N in 2,4,8,32,64,128 (scientific default)."
        ),
    )
    parser.add_argument(
        "--allow-untrained-count-extrapolation",
        action="store_true",
        help=(
            "Explicitly allow one fixed-N checkpoint at all particle counts. "
            "The summary will be marked NON-SCIENTIFIC."
        ),
    )
    parser.add_argument(
        "--shared-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used only with --allow-untrained-count-extrapolation.",
    )
    parser.add_argument("--num-test-cases", type=_positive_integer, default=5)
    parser.add_argument("--num-probe-particles", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
        help=(
            "Base seed. Test case i uses seed+i for every N, keeping particle "
            "subsets and posterior noise comparable across counts."
        ),
    )
    parser.add_argument("--num-samples", type=_positive_integer, default=16)
    parser.add_argument("--sampling-steps", type=_positive_integer, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--eta", type=_nonnegative_float, default=1.0)
    parser.add_argument("--cfg-scale", type=_nonnegative_float, default=1.5)
    parser.add_argument(
        "--trajectory-guidance-strength", type=_nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--divergence-guidance-weight", type=_nonnegative_float, default=0.0
    )
    parser.add_argument("--trajectory-substeps", type=_positive_integer, default=1)
    parser.add_argument("--use-raw-weights", action="store_true")

    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sampler", type=Path, default=_DEFAULT_SAMPLER)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print the complete command plan without loading models.",
    )
    existing_output = parser.add_mutually_exclusive_group()
    existing_output.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted sweep. Completed run IDs are retained only "
            "when their expected posterior NPZ still exists; all other runs "
            "are executed again."
        ),
    )
    existing_output.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sweep summary/run log in the output directory.",
    )
    return parser


def _resolve_checkpoint_policy(
    args: argparse.Namespace,
) -> tuple[dict[int, Path], bool]:
    assignments: dict[int, Path] = {}
    for count, path in args.checkpoint:
        if count in assignments:
            raise ValueError(f"--checkpoint was provided more than once for N={count}")
        assignments[count] = path.expanduser().resolve()

    if args.allow_untrained_count_extrapolation:
        if args.shared_checkpoint is None:
            raise ValueError(
                "--allow-untrained-count-extrapolation requires --shared-checkpoint"
            )
        if assignments:
            raise ValueError(
                "do not combine per-count --checkpoint values with the explicit "
                "shared-checkpoint extrapolation mode"
            )
        shared = args.shared_checkpoint.expanduser().resolve()
        return {count: shared for count in args.particle_counts}, False

    if args.shared_checkpoint is not None:
        raise ValueError(
            "--shared-checkpoint is forbidden unless "
            "--allow-untrained-count-extrapolation is explicit"
        )
    requested = tuple(args.particle_counts)
    unexpected = [count for count in assignments if count not in requested]
    if unexpected:
        rendered = ", ".join(str(count) for count in unexpected)
        raise ValueError(
            "checkpoints were provided for unrequested particle counts: " + rendered
        )
    missing = [count for count in requested if count not in assignments]
    if missing:
        rendered = ", ".join(f"--checkpoint {count}=PATH" for count in missing)
        raise ValueError(
            "scientific sweep requires a matching checkpoint for every N; missing "
            + rendered
        )
    return {count: assignments[count] for count in requested}, True


def _checkpoint_particle_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    data_config = checkpoint.get("data_config")
    if not isinstance(data_config, dict) or "num_particles" not in data_config:
        raise ValueError(f"checkpoint has no data_config.num_particles: {path}")
    trained_particles = int(data_config["num_particles"])
    if trained_particles < 1:
        raise ValueError(f"checkpoint has invalid trained particle count: {path}")
    return trained_particles


def _validate_checkpoint_training_counts(
    checkpoints: dict[int, Path], scientific_design: bool
) -> dict[int, int]:
    cached: dict[Path, int] = {}
    result: dict[int, int] = {}
    for requested, path in checkpoints.items():
        if path not in cached:
            cached[path] = _checkpoint_particle_count(path)
        trained = cached[path]
        if scientific_design and trained != requested:
            raise ValueError(
                f"checkpoint assigned to N={requested} was trained with N={trained}: {path}"
            )
        result[requested] = trained
    return result


def _build_sampler_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    particle_count: int,
    case_index: int,
    run_seed: int,
    output_path: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(args.sampler),
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(args.manifest.expanduser().resolve()),
        "--split",
        "test",
        "--index",
        str(case_index),
        "--num-particles",
        str(particle_count),
        "--num-samples",
        str(args.num_samples),
        "--num-probe-particles",
        str(args.num_probe_particles),
        "--sampling-steps",
        str(args.sampling_steps),
        "--eta",
        format(args.eta, ".12g"),
        "--cfg-scale",
        format(args.cfg_scale, ".12g"),
        "--trajectory-guidance-strength",
        format(args.trajectory_guidance_strength, ".12g"),
        "--divergence-guidance-weight",
        format(args.divergence_guidance_weight, ".12g"),
        "--trajectory-substeps",
        str(args.trajectory_substeps),
        "--seed",
        str(run_seed),
        "--device",
        args.device,
        "--output",
        str(output_path),
    ]
    if args.use_raw_weights:
        command.append("--use-raw-weights")
    return command


def _decode_json_scalar(value: np.ndarray, *, name: str, path: Path) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError(f"{name} in {path} must be a scalar JSON string")
    decoded = json.loads(str(value.item()))
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} in {path} must decode to a JSON object")
    return decoded


def _load_run_result(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"sampler did not write its expected output: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "metrics" not in archive.files or "metadata" not in archive.files:
            raise ValueError(f"sampler output lacks metrics/metadata: {path}")
        metrics = _decode_json_scalar(
            np.array(archive["metrics"], copy=True), name="metrics", path=path
        )
        metadata = _decode_json_scalar(
            np.array(archive["metadata"], copy=True), name="metadata", path=path
        )
    return metrics, metadata


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_resumable_records(
    path: Path, expected_outputs: dict[str, Path]
) -> list[dict[str, Any]]:
    """Return valid completed records in deterministic sweep order.

    A record is resumable only when it belongs to the current run plan, reports
    completion, names the expected output path, and that posterior archive is
    still present. Failed, planned, malformed, duplicate, and stale records are
    discarded so their run IDs are executed again.
    """

    if not path.is_file():
        return []

    retained: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            decoded = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in resume log {path} at line {line_number}: {error}"
            ) from error
        if not isinstance(decoded, dict):
            continue
        run_id = decoded.get("run_id")
        if (
            not isinstance(run_id, str)
            or run_id not in expected_outputs
            or decoded.get("status") != "completed"
        ):
            continue
        output_value = decoded.get("output")
        if not isinstance(output_value, str) or not output_value:
            continue
        expected_output = expected_outputs[run_id]
        if Path(output_value).expanduser().resolve() != expected_output:
            continue
        if not expected_output.is_file():
            continue
        retained[run_id] = decoded

    return [
        retained[run_id]
        for run_id in expected_outputs
        if run_id in retained
    ]


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    _atomic_write_text(path, content)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool):
            result[name] = float(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            converted = float(value)
            if math.isfinite(converted):
                result[name] = converted
    return result


def _build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    checkpoints: dict[int, Path],
    trained_counts: dict[int, int | None],
    *,
    scientific_design: bool,
    complete: bool,
    dry_run: bool,
    partial_reason: str | None = None,
) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    failed = [record for record in records if record["status"] == "failed"]
    planned = [record for record in records if record["status"] == "planned"]
    per_count: dict[str, Any] = {}
    for count in args.particle_counts:
        selected = [record for record in completed if record["num_particles"] == count]
        metric_names = sorted(
            {
                name
                for record in selected
                for name in _numeric_metrics(record["metrics"])
            }
        )
        aggregate: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            values = [
                _numeric_metrics(record["metrics"])[metric_name]
                for record in selected
                if metric_name in _numeric_metrics(record["metrics"])
            ]
            aggregate[metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        per_count[str(count)] = {
            "checkpoint": str(checkpoints[count]),
            "trained_particles": trained_counts.get(count),
            "completed_runs": len(selected),
            "scientific_result": bool(selected) and all(
                bool(record.get("scientific_result", False)) for record in selected
            ),
            "non_scientific_reasons": sorted(
                {
                    reason
                    for record in selected
                    for reason in record.get("scientific_result_reasons", [])
                }
            ),
            "metrics": aggregate,
        }

    if not scientific_design:
        scientific_result: bool | None = False
    elif dry_run:
        scientific_result = None
    elif not complete:
        scientific_result = False
    else:
        scientific_result = bool(completed) and all(
            bool(record.get("source_scientific_result", False)) for record in completed
        )

    if not scientific_design:
        warning = _EXTRAPOLATION_WARNING
    elif complete and scientific_result is False:
        warning = (
            "NON-SCIENTIFIC: at least one sampler run failed checkpoint, "
            "manifest, split, or particle-count provenance checks."
        )
    else:
        warning = None

    return {
        "format_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "dry_run": dry_run,
        "partial_reason": partial_reason,
        "particle_counts": list(args.particle_counts),
        "checkpoint_policy": (
            "matching_checkpoint_per_particle_count"
            if scientific_design
            else "shared_fixed_count_checkpoint_extrapolation"
        ),
        "scientific_design": scientific_design,
        "scientific_result": scientific_result,
        "warning": warning,
        "population_std_ddof": 0,
        "requested_test_cases_per_count": args.num_test_cases,
        "expected_runs": len(args.particle_counts) * args.num_test_cases,
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "planned_runs": len(planned),
        "base_seed": args.seed,
        "num_posterior_samples": args.num_samples,
        "num_probe_particles": args.num_probe_particles,
        "sampling_steps": args.sampling_steps,
        "device": args.device,
        "manifest": str(args.manifest.expanduser().resolve()),
        "per_particle_count": per_count,
    }


def _summary_csv(summary: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fieldnames = [
        "num_particles",
        "checkpoint",
        "trained_particles",
        "scientific_design",
        "scientific_result",
        "completed_runs",
        "metric",
        "mean",
        "std",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for count in summary["particle_counts"]:
        group = summary["per_particle_count"][str(count)]
        for metric_name, statistics_value in group["metrics"].items():
            writer.writerow(
                {
                    "num_particles": count,
                    "checkpoint": group["checkpoint"],
                    "trained_particles": group["trained_particles"],
                    "scientific_design": summary["scientific_design"],
                    "scientific_result": summary["scientific_result"],
                    "completed_runs": group["completed_runs"],
                    "metric": metric_name,
                    "mean": format(statistics_value["mean"], ".12g"),
                    "std": format(statistics_value["std"], ".12g"),
                }
            )
    return stream.getvalue()


def _write_summaries(output_dir: Path, summary: dict[str, Any]) -> None:
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(output_dir / "summary.csv", _summary_csv(summary))


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_probe_particles < 0:
        raise ValueError("--num-probe-particles must be non-negative")
    if (
        args.divergence_guidance_weight > 0.0
        and args.trajectory_guidance_strength == 0.0
    ):
        raise ValueError(
            "--divergence-guidance-weight is non-zero, but total guidance is "
            "disabled; set --trajectory-guidance-strength greater than zero"
        )
    checkpoints, scientific_design = _resolve_checkpoint_policy(args)
    if not args.dry_run:
        if not args.manifest.expanduser().is_file():
            raise FileNotFoundError(f"manifest does not exist: {args.manifest}")
        if not args.sampler.expanduser().is_file():
            raise FileNotFoundError(f"sampler does not exist: {args.sampler}")
        trained_counts: dict[int, int | None] = _validate_checkpoint_training_counts(
            checkpoints, scientific_design
        )
    else:
        trained_counts = {count: None for count in args.particle_counts}

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.jsonl"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    expected_outputs: dict[str, Path] = {}
    for case_index in range(args.num_test_cases):
        run_seed = args.seed + case_index
        for particle_count in args.particle_counts:
            run_id = f"N{particle_count:03d}_case_{case_index:04d}"
            expected_outputs[run_id] = (
                output_dir
                / f"N{particle_count:03d}"
                / f"case_{case_index:04d}_seed_{run_seed}.npz"
            ).resolve()
    occupied = [path for path in (runs_path, summary_path, csv_path) if path.exists()]
    if occupied and not (args.overwrite or args.resume):
        names = ", ".join(path.name for path in occupied)
        raise FileExistsError(
            f"output directory already contains {names}; use --resume to continue "
            "or --overwrite to restart"
        )
    if args.resume:
        records = _load_resumable_records(runs_path, expected_outputs)
        _rewrite_jsonl(runs_path, records)
        print(
            f"Resuming with {len(records)}/{len(expected_outputs)} completed runs",
            flush=True,
        )
    else:
        records = []
        runs_path.write_text("", encoding="utf-8")
    completed_run_ids = {record["run_id"] for record in records}

    if not scientific_design:
        print(_EXTRAPOLATION_WARNING, file=sys.stderr, flush=True)

    for case_index in range(args.num_test_cases):
        run_seed = args.seed + case_index
        for particle_count in args.particle_counts:
            run_directory = output_dir / f"N{particle_count:03d}"
            run_directory.mkdir(parents=True, exist_ok=True)
            output_path = run_directory / f"case_{case_index:04d}_seed_{run_seed}.npz"
            log_path = output_path.with_suffix(".log")
            run_id = f"N{particle_count:03d}_case_{case_index:04d}"
            if run_id in completed_run_ids:
                print(f"Skipping completed {run_id}", flush=True)
                continue
            command = _build_sampler_command(
                args,
                checkpoint=checkpoints[particle_count],
                particle_count=particle_count,
                case_index=case_index,
                run_seed=run_seed,
                output_path=output_path,
            )
            base_record: dict[str, Any] = {
                "run_id": run_id,
                "num_particles": particle_count,
                "test_case_index": case_index,
                "seed": run_seed,
                "checkpoint": str(checkpoints[particle_count]),
                "trained_particles": trained_counts.get(particle_count),
                "scientific_design": scientific_design,
                "output": str(output_path),
                "log": str(log_path),
                "command": command,
                "command_line": subprocess.list2cmdline(command),
            }

            if args.dry_run:
                record = {**base_record, "status": "planned"}
                records.append(record)
                _append_jsonl(runs_path, record)
                print(record["command_line"])
                continue

            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                log.write(base_record["command_line"] + "\n\n")
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=_REPOSITORY_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            elapsed = time.perf_counter() - started
            if result.returncode != 0:
                record = {
                    **base_record,
                    "status": "failed",
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                }
                records.append(record)
                _append_jsonl(runs_path, record)
                summary = _build_summary(
                    args,
                    records,
                    checkpoints,
                    trained_counts,
                    scientific_design=scientific_design,
                    complete=False,
                    dry_run=False,
                    partial_reason=f"subprocess failed for {record['run_id']}",
                )
                _write_summaries(output_dir, summary)
                raise RuntimeError(
                    f"sampler failed for {record['run_id']} with exit code "
                    f"{result.returncode}; progress is preserved in {runs_path} "
                    f"and details are in {log_path}"
                )

            try:
                metrics, metadata = _load_run_result(output_path)
                observed_particles = int(metrics["num_observed_particles"])
                if observed_particles != particle_count:
                    raise ValueError(
                        f"output reports N={observed_particles}, expected N={particle_count}"
                    )
                output_trained = int(metadata["trained_particles"])
                expected_trained = trained_counts[particle_count]
                if expected_trained is not None and output_trained != expected_trained:
                    raise ValueError(
                        f"output reports trained N={output_trained}, expected "
                        f"N={expected_trained}"
                    )
                if int(metadata["sampled_particles"]) != particle_count:
                    raise ValueError(
                        "sampler metadata sampled_particles does not match "
                        f"requested N={particle_count}"
                    )
                if metadata.get("split") != "test":
                    raise ValueError("sampler metadata must report split='test'")
                expected_method = (
                    f"{particle_count}-particle conditional 3D flow diffusion"
                )
                if metadata.get("method") != expected_method:
                    raise ValueError(
                        f"sampler metadata method must be {expected_method!r}"
                    )
                source_flag = metadata.get("scientific_result")
                if not isinstance(source_flag, bool):
                    raise TypeError("sampler scientific_result must be boolean")
                source_reasons = metadata.get("scientific_result_reasons")
                if not isinstance(source_reasons, list) or any(
                    not isinstance(reason, str) for reason in source_reasons
                ):
                    raise TypeError(
                        "sampler scientific_result_reasons must be a list of strings"
                    )
                if source_flag:
                    if source_reasons:
                        raise ValueError(
                            "scientific sampler output cannot contain rejection reasons"
                        )
                    if metadata.get("checkpoint_has_training_provenance") is not True:
                        raise ValueError(
                            "scientific sampler output lacks checkpoint provenance"
                        )
                    evaluation = metadata.get("evaluation_manifest_provenance")
                    if not isinstance(evaluation, dict) or evaluation.get(
                        "manifest_mode"
                    ) != "scientific":
                        raise ValueError(
                            "scientific sampler output lacks a scientific "
                            "evaluation-manifest provenance record"
                        )
                elif not source_reasons:
                    raise ValueError(
                        "non-scientific sampler output must explain why it is "
                        "non-scientific"
                    )
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                record = {
                    **base_record,
                    "status": "failed",
                    "returncode": result.returncode,
                    "elapsed_seconds": elapsed,
                    "error": str(error),
                }
                records.append(record)
                _append_jsonl(runs_path, record)
                summary = _build_summary(
                    args,
                    records,
                    checkpoints,
                    trained_counts,
                    scientific_design=scientific_design,
                    complete=False,
                    dry_run=False,
                    partial_reason=f"invalid output for {record['run_id']}: {error}",
                )
                _write_summaries(output_dir, summary)
                raise RuntimeError(
                    f"invalid sampler output for {record['run_id']}; progress is "
                    f"preserved in {runs_path}: {error}"
                ) from error

            source_scientific = metadata["scientific_result"]
            record = {
                **base_record,
                "status": "completed",
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "case_id": metadata.get("case_id"),
                "source_scientific_result": source_scientific,
                "scientific_result": scientific_design and source_scientific,
                "scientific_result_reasons": metadata[
                    "scientific_result_reasons"
                ],
                "metrics": metrics,
            }
            records.append(record)
            _append_jsonl(runs_path, record)
            partial_summary = _build_summary(
                args,
                records,
                checkpoints,
                trained_counts,
                scientific_design=scientific_design,
                complete=False,
                dry_run=False,
                partial_reason="sweep in progress",
            )
            _write_summaries(output_dir, partial_summary)
            print(
                f"Completed N={particle_count}, case={case_index + 1}/"
                f"{args.num_test_cases}",
                flush=True,
            )

    final_summary = _build_summary(
        args,
        records,
        checkpoints,
        trained_counts,
        scientific_design=scientific_design,
        complete=not args.dry_run,
        dry_run=bool(args.dry_run),
        partial_reason="dry-run command plan" if args.dry_run else None,
    )
    _write_summaries(output_dir, final_summary)
    return final_summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        summary = run_sweep(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"Wrote run records to {args.output_dir / 'runs.jsonl'}")
    print(f"Wrote JSON summary to {args.output_dir / 'summary.json'}")
    print(f"Wrote CSV summary to {args.output_dir / 'summary.csv'}")
    if summary["warning"]:
        print(summary["warning"], file=sys.stderr)


if __name__ == "__main__":
    main()
