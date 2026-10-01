#!/usr/bin/env python
"""Run the full sparse-particle diffusion experiment queue.

The queue waits for the shared Taichi-LBM3D collection, creates a group-disjoint
manifest, trains one model for every (training seed, particle count), performs
physics fine-tuning, evaluates every held-out test flow, and aggregates all
three training seeds.  Every stage is restartable from its latest checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPOSITORY_ROOT / "scripts" / "train_sparse_track_diffusion.py"
SWEEP_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_sparse_particle_sweep.py"
SAMPLER_SCRIPT = REPOSITORY_ROOT / "scripts" / "sample_sparse_track_diffusion.py"
MANIFEST_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_sparse_flow_manifest.py"
COLLECTION_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "generate_taichi_lbm3d_flow_collection.py"
)
TRAINING_CODE_PATHS = (
    TRAIN_SCRIPT,
    REPOSITORY_ROOT / "src" / "flow_observation" / "diffusion.py",
    REPOSITORY_ROOT / "src" / "flow_observation" / "sparse_dataset.py",
    REPOSITORY_ROOT / "src" / "flow_observation" / "sparse_physics.py",
    REPOSITORY_ROOT / "src" / "flow_observation" / "models" / "trajectory_encoder.py",
    REPOSITORY_ROOT / "src" / "flow_observation" / "models" / "unet3d.py",
)
SAMPLING_CODE_PATHS = (
    SWEEP_SCRIPT,
    SAMPLER_SCRIPT,
    *TRAINING_CODE_PATHS[1:],
)
REQUIRED_METRICS = (
    "field_relative_l2_unknown_interior",
    "field_cosine_unknown_interior",
    "observed_track_rmse_cells",
    "probe_track_rmse_cells",
    "posterior_mean_divergence_mse",
    "posterior_mean_variance",
)


def _csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer list contains duplicates")
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "lbm3d_multiflow",
    )
    parser.add_argument("--expected-cases", type=_positive_int, default=1000)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "lbm3d_multiflow" / "manifest.json",
    )
    parser.add_argument(
        "--data-python",
        type=Path,
        default=REPOSITORY_ROOT / ".venv-lbm3d" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--torch-python",
        type=Path,
        default=Path(r"D:\py\anaocnda\envs\torch5090\python.exe"),
    )
    parser.add_argument(
        "--torch-env",
        type=Path,
        default=Path(r"D:\py\anaocnda\envs\torch5090"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "particle_count_3seed",
    )
    parser.add_argument("--particle-counts", type=_csv_ints, default=(2, 4, 8, 32, 64, 128))
    parser.add_argument("--training-seeds", type=_csv_ints, default=(31, 32, 33))
    parser.add_argument("--observations-per-flow", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrain-epochs", type=_positive_int, default=100)
    parser.add_argument("--final-epochs", type=_positive_int, default=150)
    parser.add_argument("--num-test-cases", type=_positive_int, default=100)
    parser.add_argument("--num-probe-particles", type=_positive_int, default=32)
    parser.add_argument("--num-samples", type=_positive_int, default=16)
    parser.add_argument("--sampling-steps", type=_positive_int, default=50)
    parser.add_argument("--poll-seconds", type=_positive_int, default=30)
    return parser


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return decoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())): _sha256(path)
        for path in paths
    }


def _torch_environment(torch_environment: Path) -> dict[str, str]:
    environment = dict(os.environ)
    torch_library = torch_environment / "Lib" / "site-packages" / "torch" / "lib"
    conda_library = torch_environment / "Library" / "bin"
    # torch ships the matching uv.dll.  It must precede the base Conda uv.dll
    # on Windows, otherwise importing shm.dll fails with WinError 127.
    prefix = os.pathsep.join(
        [str(torch_library.resolve()), str(conda_library.resolve()), str(torch_environment.resolve())]
    )
    environment["PATH"] = prefix + os.pathsep + environment.get("PATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _update_state(state_path: Path, **updates: Any) -> None:
    state = _read_json(state_path) if state_path.exists() else {}
    state.update(updates)
    state["updated_at_utc"] = _timestamp()
    _atomic_json(state_path, state)


def _collection_is_complete(collection_dir: Path, expected_cases: int) -> bool:
    collection_path = collection_dir / "collection.json"
    if not collection_path.is_file():
        return False
    collection = _read_json(collection_path)
    if not bool(collection.get("complete")):
        return False
    if bool(collection.get("dry_run")):
        raise ValueError("completed collection is a dry-run plan")
    cases = collection.get("cases", [])
    recorded_cases = len(cases) if isinstance(cases, list) else 0
    if recorded_cases != expected_cases:
        raise ValueError(
            f"completed collection contains {recorded_cases} cases, "
            f"expected {expected_cases}"
        )
    archives = sorted(collection_dir.glob("case_*.npz"))
    if len(archives) != expected_cases:
        raise ValueError(
            f"completed collection contains {len(archives)} archives, "
            f"expected {expected_cases}"
        )
    _validate_collection_archives(collection_dir, archives, expected_cases)
    return True


def _ensure_collection(args: argparse.Namespace, state_path: Path) -> None:
    if _collection_is_complete(args.collection_dir, args.expected_cases):
        return
    command = [
        str(args.data_python.resolve()),
        "-u",
        str(COLLECTION_SCRIPT),
        "--output-dir",
        str(args.collection_dir.resolve()),
        "--sampling",
        "random",
        "--num-cases",
        str(args.expected_cases),
        "--num-particles",
        "160",
        "--particle-selection",
        "random",
        "--backend",
        "cpu",
        "--grid-size",
        "32",
        "--num-observation-times",
        "20",
        "--duration",
        "120",
        "--projections",
        "xz,xy,yz",
        "--integration-substeps",
        "4",
        "--planner-seed",
        "20260918",
    ]
    _run_logged(
        command,
        args.output_root / "collection_generation.log",
        None,
        state_path,
        "generating_collection",
    )
    if not _collection_is_complete(args.collection_dir, args.expected_cases):
        raise RuntimeError("collection generator returned without a complete collection")


def _validate_collection_archives(
    collection_dir: Path, archives: list[Path], expected_cases: int
) -> None:
    collection = _read_json(collection_dir / "collection.json")
    cases = collection.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError("collection case records do not match the expected case count")
    group_ids = [record.get("flow_group_id") for record in cases]
    if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
        raise ValueError("collection contains a missing flow_group_id")
    if len(set(group_ids)) != expected_cases:
        raise ValueError(
            "formal collection must contain one independent physical flow group per case"
        )
    for archive_path in archives:
        try:
            with np.load(archive_path, allow_pickle=False) as archive:
                trajectories = np.asarray(archive["trajectories_2d"])
                if trajectories.ndim != 5 or trajectories.shape[2] < 160:
                    raise ValueError(
                        f"{archive_path} stores only shape {trajectories.shape}; "
                        "at least 160 particles are required"
                    )
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"invalid collection archive {archive_path}: {error}") from error


def _run_logged(
    command: list[str],
    log_path: Path,
    environment: dict[str, str] | None,
    state_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _update_state(
        state_path,
        stage=label,
        active_command=subprocess.list2cmdline(command),
        active_log=str(log_path),
    )
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\n[{_timestamp()}] {subprocess.list2cmdline(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        _update_state(
            state_path,
            stage="failed",
            failed_label=label,
            returncode=result.returncode,
            failed_log=str(log_path),
        )
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {log_path}")


def _stage_signature(
    args: argparse.Namespace,
    *,
    count: int,
    seed: int,
    epochs: int,
    physics: bool,
    initial_resume: Path | None,
) -> dict[str, Any]:
    return {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "num_particles": count,
        "seed": seed,
        "observations_per_flow": args.observations_per_flow,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": epochs,
        "physics": physics,
        "track_loss_weight": 0.1 if physics else 0.0,
        "divergence_loss_weight": 0.01 if physics else 0.0,
        "boundary_loss_weight": 0.01 if physics else 0.0,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "diffusion_steps": 1000,
        "schedule": "cosine",
        "base_channels": 16,
        "condition_dim": 128,
        "time_embedding_dim": 64,
        "condition_dropout": 0.1,
        "ema_decay": 0.999,
        "amp": True,
        "initial_resume": (
            None
            if initial_resume is None
            else {
                "path": str(initial_resume.resolve()),
                "sha256": _sha256(initial_resume.resolve()),
            }
        ),
        "training_code_sha256": _code_hashes(TRAINING_CODE_PATHS),
    }


def _prepare_stage_directory(output_dir: Path, signature: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = output_dir / "queue_stage_config.json"
    if signature_path.is_file():
        if _read_json(signature_path) != signature:
            raise ValueError(
                f"existing stage configuration differs from the request: {output_dir}"
            )
        return
    protected = (
        output_dir / "best.pt",
        output_dir / "latest.pt",
        output_dir / "training_summary.json",
    )
    if any(path.exists() for path in protected):
        raise ValueError(
            f"untracked existing training artifacts found in {output_dir}; "
            "refusing to mix configurations"
        )
    _atomic_json(signature_path, signature)


def _summary_complete(
    output_dir: Path, epochs: int, signature: dict[str, Any]
) -> bool:
    summary_path = output_dir / "training_summary.json"
    checkpoint_path = output_dir / "best.pt"
    signature_path = output_dir / "queue_stage_config.json"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        return False
    if not signature_path.is_file() or _read_json(signature_path) != signature:
        return False
    summary = _read_json(summary_path)
    return (
        summary.get("status") == "completed"
        and int(summary.get("epochs", -1)) == epochs
        and int(summary.get("num_particles_per_condition", -1))
        == int(signature["num_particles"])
        and summary.get("training_provenance", {}).get("manifest_sha256")
        == signature["manifest_sha256"]
    )


def _train_stage(
    *,
    args: argparse.Namespace,
    count: int,
    seed: int,
    output_dir: Path,
    epochs: int,
    environment: dict[str, str],
    state_path: Path,
    label: str,
    initial_resume: Path | None,
    physics: bool,
) -> Path:
    signature = _stage_signature(
        args,
        count=count,
        seed=seed,
        epochs=epochs,
        physics=physics,
        initial_resume=initial_resume,
    )
    _prepare_stage_directory(output_dir, signature)
    if _summary_complete(output_dir, epochs, signature):
        return output_dir / "best.pt"

    own_latest = output_dir / "latest.pt"
    resume = own_latest if own_latest.is_file() else initial_resume
    command = [
        str(args.torch_python.resolve()),
        "-u",
        str(TRAIN_SCRIPT),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--num-particles",
        str(count),
        "--observations-per-flow",
        str(args.observations_per_flow),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume.resolve())])
    if physics:
        command.extend(
            [
                "--track-loss-weight",
                "0.1",
                "--divergence-loss-weight",
                "0.01",
                "--boundary-loss-weight",
                "0.01",
            ]
        )
    _run_logged(command, output_dir / "run.log", environment, state_path, label)
    if not _summary_complete(output_dir, epochs, signature):
        raise RuntimeError(f"{label} returned success without a complete summary")
    return output_dir / "best.pt"


def _build_manifest(args: argparse.Namespace, state_path: Path) -> None:
    if args.manifest.is_file():
        return
    command = [
        str(args.data_python.resolve()),
        "-u",
        str(MANIFEST_SCRIPT),
        "--input-dir",
        str(args.collection_dir.resolve()),
        "--output",
        str(args.manifest.resolve()),
        "--seed",
        "20260918",
        "--train-fraction",
        "0.8",
        "--validation-fraction",
        "0.1",
    ]
    _run_logged(
        command,
        args.output_root / "manifest_build.log",
        None,
        state_path,
        "building_manifest",
    )


def _run_sweep(
    args: argparse.Namespace,
    seed: int,
    checkpoints: dict[int, Path],
    environment: dict[str, str],
    state_path: Path,
) -> None:
    output_dir = args.output_root / "sweeps" / f"training_seed_{seed}"
    signature = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest.resolve()),
        "training_seed": seed,
        "checkpoints": {
            str(count): {
                "path": str(checkpoints[count].resolve()),
                "sha256": _sha256(checkpoints[count].resolve()),
            }
            for count in args.particle_counts
        },
        "particle_counts": list(args.particle_counts),
        "num_test_cases": args.num_test_cases,
        "base_sampling_seed": 47000,
        "num_probe_particles": args.num_probe_particles,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "eta": 1.0,
        "cfg_scale": 1.5,
        "trajectory_guidance_strength": 0.0,
        "divergence_guidance_weight": 0.0,
        "sampling_code_sha256": _code_hashes(SAMPLING_CODE_PATHS),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    signature_path = output_dir / "queue_sweep_config.json"
    if signature_path.is_file():
        if _read_json(signature_path) != signature:
            raise ValueError(
                f"existing sweep configuration differs from the request: {output_dir}"
            )
    else:
        existing_markers = (
            output_dir / "runs.jsonl",
            output_dir / "summary.json",
            output_dir / "summary.csv",
        )
        if any(path.exists() for path in existing_markers):
            raise ValueError(
                f"untracked existing sweep artifacts found in {output_dir}"
            )
        _atomic_json(signature_path, signature)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        existing_summary = _read_json(summary_path)
        if bool(existing_summary.get("complete")):
            if existing_summary.get("scientific_result") is not True:
                raise ValueError(f"completed sweep is not scientific: {summary_path}")
            return
    command = [
        str(args.torch_python.resolve()),
        "-u",
        str(SWEEP_SCRIPT),
        "--manifest",
        str(args.manifest.resolve()),
        "--particle-counts",
        ",".join(str(count) for count in args.particle_counts),
    ]
    for count in args.particle_counts:
        command.extend(["--checkpoint", f"{count}={checkpoints[count].resolve()}"])
    command.extend(
        [
            "--num-test-cases",
            str(args.num_test_cases),
            "--num-probe-particles",
            str(args.num_probe_particles),
            "--num-samples",
            str(args.num_samples),
            "--sampling-steps",
            str(args.sampling_steps),
            "--seed",
            "47000",
            "--device",
            "cuda",
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )
    sweep_markers = (
        output_dir / "runs.jsonl",
        output_dir / "summary.json",
        output_dir / "summary.csv",
    )
    if any(path.exists() for path in sweep_markers):
        command.append("--resume")
    _run_logged(
        command,
        output_dir / "sweep.log",
        environment,
        state_path,
        f"sweep_seed_{seed}",
    )


def _aggregate(args: argparse.Namespace) -> None:
    pooled_values: dict[int, dict[str, list[float]]] = {
        count: {} for count in args.particle_counts
    }
    per_seed: dict[str, Any] = {}
    for training_seed in args.training_seeds:
        sweep_root = (
            args.output_root / "sweeps" / f"training_seed_{training_seed}"
        )
        sweep_summary = _read_json(sweep_root / "summary.json")
        expected_runs = len(args.particle_counts) * args.num_test_cases
        if (
            sweep_summary.get("complete") is not True
            or sweep_summary.get("scientific_result") is not True
            or sweep_summary.get("particle_counts") != list(args.particle_counts)
            or int(sweep_summary.get("completed_runs", -1)) != expected_runs
            or int(sweep_summary.get("expected_runs", -1)) != expected_runs
        ):
            raise ValueError(
                f"incomplete or non-scientific sweep for training seed {training_seed}"
            )
        per_count_summary = sweep_summary.get("per_particle_count", {})
        for count in args.particle_counts:
            count_summary = per_count_summary.get(str(count), {})
            if (
                int(count_summary.get("completed_runs", -1)) != args.num_test_cases
                or count_summary.get("scientific_result") is not True
            ):
                raise ValueError(
                    f"invalid N={count} sweep summary for training seed {training_seed}"
                )
            missing_required = [
                name
                for name in REQUIRED_METRICS
                if name not in count_summary.get("metrics", {})
            ]
            if missing_required:
                raise ValueError(
                    f"training seed {training_seed}, N={count} lacks metrics: "
                    + ", ".join(missing_required)
                )
        runs_path = (
            sweep_root / "runs.jsonl"
        )
        seed_counts: dict[str, int] = {}
        seed_values: dict[int, dict[str, list[float]]] = {
            count: {} for count in args.particle_counts
        }
        seen_run_ids: set[str] = set()
        for raw_line in runs_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if record.get("status") != "completed":
                continue
            run_id = str(record.get("run_id"))
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate completed sweep run_id: {run_id}")
            seen_run_ids.add(run_id)
            if record.get("scientific_result") is not True:
                raise ValueError(f"non-scientific completed sweep run: {run_id}")
            count = int(record["num_particles"])
            seed_counts[str(count)] = seed_counts.get(str(count), 0) + 1
            for name, raw_value in record.get("metrics", {}).items():
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    pooled_values[count].setdefault(name, []).append(value)
                    seed_values[count].setdefault(name, []).append(value)
        for count in args.particle_counts:
            if seed_counts.get(str(count), 0) != args.num_test_cases:
                raise ValueError(
                    f"training seed {training_seed}, N={count} has "
                    f"{seed_counts.get(str(count), 0)} completed runs; "
                    f"expected {args.num_test_cases}"
                )
            for name in REQUIRED_METRICS:
                if len(seed_values[count].get(name, [])) != args.num_test_cases:
                    raise ValueError(
                        f"training seed {training_seed}, N={count}, metric {name} "
                        "is incomplete"
                    )
        metric_means = {
            str(count): {
                name: statistics.fmean(samples)
                for name, samples in metrics.items()
                if samples
            }
            for count, metrics in seed_values.items()
        }
        per_seed[str(training_seed)] = {
            "completed_per_particle_count": seed_counts,
            "metric_means": metric_means,
        }

    aggregate: dict[str, Any] = {
        "created_at_utc": _timestamp(),
        "particle_counts": list(args.particle_counts),
        "training_seeds": list(args.training_seeds),
        "num_test_cases_per_training_seed": args.num_test_cases,
        "per_training_seed": per_seed,
        "per_particle_count": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for count, metrics in pooled_values.items():
        count_payload: dict[str, Any] = {}
        for name, samples in sorted(metrics.items()):
            seed_means = [
                float(per_seed[str(seed)]["metric_means"][str(count)][name])
                for seed in args.training_seeds
                if name in per_seed[str(seed)]["metric_means"][str(count)]
            ]
            if len(seed_means) != len(args.training_seeds):
                if name in REQUIRED_METRICS:
                    raise ValueError(
                        f"required metric {name} for N={count} does not cover all seeds"
                    )
                continue
            stats = {
                "n_training_seeds": len(seed_means),
                "mean_over_training_seeds": statistics.fmean(seed_means),
                "std_over_training_seeds": (
                    statistics.pstdev(seed_means) if len(seed_means) > 1 else 0.0
                ),
                "n_pooled_cases": len(samples),
                "pooled_case_mean": statistics.fmean(samples),
                "pooled_case_std": (
                    statistics.pstdev(samples) if len(samples) > 1 else 0.0
                ),
            }
            count_payload[name] = stats
            csv_rows.append({"num_particles": count, "metric": name, **stats})
        aggregate["per_particle_count"][str(count)] = count_payload

    _atomic_json(args.output_root / "aggregate_summary.json", aggregate)
    with (args.output_root / "aggregate_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "num_particles",
                "metric",
                "n_training_seeds",
                "mean_over_training_seeds",
                "std_over_training_seeds",
                "n_pooled_cases",
                "pooled_case_mean",
                "pooled_case_std",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)


def run(args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.final_epochs <= args.pretrain_epochs:
        raise ValueError("--final-epochs must exceed --pretrain-epochs")
    if max(args.particle_counts) + args.num_probe_particles > 160:
        raise ValueError("observed particles plus probes exceed the stored pool of 160")
    if not args.data_python.is_file() or not args.torch_python.is_file():
        raise FileNotFoundError("a configured Python interpreter does not exist")

    args.collection_dir = args.collection_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "experiment_state.json"
    _update_state(
        state_path,
        status="running",
        started_at_utc=_read_json(state_path).get("started_at_utc", _timestamp())
        if state_path.exists()
        else _timestamp(),
        particle_counts=list(args.particle_counts),
        training_seeds=list(args.training_seeds),
    )

    _ensure_collection(args, state_path)
    _build_manifest(args, state_path)
    environment = _torch_environment(args.torch_env)

    checkpoints_by_seed: dict[int, dict[int, Path]] = {}
    for seed in args.training_seeds:
        checkpoints_by_seed[seed] = {}
        for count in args.particle_counts:
            model_root = (
                args.output_root / "models" / f"training_seed_{seed}" / f"N{count:03d}"
            )
            pretrain = _train_stage(
                args=args,
                count=count,
                seed=seed,
                output_dir=model_root / "pretrain",
                epochs=args.pretrain_epochs,
                environment=environment,
                state_path=state_path,
                label=f"pretrain_seed_{seed}_N{count}",
                initial_resume=None,
                physics=False,
            )
            physics = _train_stage(
                args=args,
                count=count,
                seed=seed,
                output_dir=model_root / "physics",
                epochs=args.final_epochs,
                environment=environment,
                state_path=state_path,
                label=f"physics_seed_{seed}_N{count}",
                initial_resume=pretrain,
                physics=True,
            )
            checkpoints_by_seed[seed][count] = physics

    for seed in args.training_seeds:
        _run_sweep(args, seed, checkpoints_by_seed[seed], environment, state_path)
    _aggregate(args)
    _update_state(
        state_path,
        status="completed",
        stage="completed",
        completed_at_utc=_timestamp(),
        aggregate_summary=str(args.output_root / "aggregate_summary.json"),
    )


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output_root = args.output_root.expanduser().resolve()
        state_path = output_root / "experiment_state.json"
        output_root.mkdir(parents=True, exist_ok=True)
        _update_state(state_path, status="failed", stage="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
