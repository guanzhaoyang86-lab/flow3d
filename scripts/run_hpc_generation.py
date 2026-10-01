#!/usr/bin/env python
"""Generate real Taichi-LBM3D data on Delta and validate a shared manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from run_hpc_diffusion import REPOSITORY, run_logged, write_json

UPSTREAM_COMMIT = "fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c"


def generation_parameters(mode: str) -> list[str]:
    # Match the existing local collection's physical and observation settings.
    # A pilot has its own plan; its three cases are not a prefix of the full plan.
    return [
        "--sampling", "random", "--num-cases", "3" if mode == "pilot" else "1000",
        "--planner-seed", "20260918", "--backend", "cuda", "--grid-size", "32",
        "--lid-speed-range", "0.03", "0.06", "--upstream-niu-range", "0.12", "0.21",
        "--warmup-step-range", "600", "1200", "--seed-range", "0", "2147483647",
        "--num-particles", "160", "--num-observation-times", "20", "--duration", "120",
        "--projections", "xz,xy,yz", "--integration-substeps", "4",
        "--particle-margin", "2", "--candidate-multiplier", "32",
        "--minimum-path-length", "0", "--particle-selection", "random",
        "--progress-interval", "200",
    ]


def validate_archives(directory: Path, expected_count: int) -> list[Path]:
    paths = sorted(directory.glob("*.npz"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} archives, found {len(paths)}")
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("backend") != "cuda" or metadata.get("upstream_commit") != UPSTREAM_COMMIT:
                raise ValueError(f"Incompatible solver provenance: {path}")
            for name in archive.files:
                values = archive[name]
                if values.dtype.kind in "fc" and not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {name}: {path}")
    return paths


def check_resume(directory: Path, configuration: dict) -> None:
    previous = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
    if previous != configuration:
        raise ValueError("Resume requires the same code, solver, packages, GPU and parameters")
    if (directory / "COMPLETE.json").exists():
        raise ValueError("Dataset is already complete; use its manifest instead of regenerating")


def train_after_generation(output: Path, storage_root: Path, epochs: int, record: Path) -> None:
    """Reuse the allocated A100 only after generation and validation succeed."""
    complete = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
    manifest = output / "manifest.json"
    if complete.get("status") != "completed" or not manifest.is_file():
        raise ValueError("Follow-up training requires a completed dataset and manifest")
    command = [
        sys.executable, str(REPOSITORY / "scripts/run_hpc_diffusion.py"), "train",
        "--manifest", str(manifest), "--storage-root", str(storage_root), "--device", "cuda",
        "--", "--epochs", str(epochs), "--batch-size", "1", "--num-workers", "0",
        "--num-particles", "2", "--seed", "31", "--save-every", "1",
    ]
    state = {"status": "running", "dataset": str(output), "training_epochs": epochs,
             "command": command, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    started = time.monotonic()
    write_json(record / "pipeline.json", state)
    print(f"Data generation validated. Starting follow-up training: {epochs} epoch(s)", flush=True)
    try:
        run_logged(command, record)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["training_elapsed_seconds"] = time.monotonic() - started
        write_json(record / "pipeline.json", state)
    print(f"Pipeline completed: generation and training. Record: {record / 'pipeline.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--storage-root", type=Path, default=os.environ.get("FLOW3D_ROOT"))
    parser.add_argument("--upstream-repo", type=Path, default=os.environ.get("FLOW3D_UPSTREAM_REPO"))
    parser.add_argument("--resume", type=Path, help="An incomplete dataset directory from this mode")
    parser.add_argument("--train-epochs", type=int, default=0,
                        help="After successful generation, train in this allocation; 0 disables training")
    args = parser.parse_args()
    if args.train_epochs < 0:
        parser.error("--train-epochs must be nonnegative")
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        parser.error("Data generation requires a Slurm job; use scripts/submit.sh generate-pilot")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("This Taichi environment requires Linux x86_64 on Delta")
    if args.storage_root is None or not str(args.storage_root.resolve()).startswith("/work/"):
        parser.error("FLOW3D_ROOT must be shared /work storage")
    if args.upstream_repo is None:
        parser.error("Set FLOW3D_UPSTREAM_REPO; run setup_delta_generation.sh first")
    commit = os.environ.get("FLOW3D_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        parser.error("A frozen GitHub code commit is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error("Exactly one working CUDA GPU is required")
    upstream = args.upstream_repo.expanduser().resolve()
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if actual != UPSTREAM_COMMIT or dirty:
        parser.error("Upstream solver must be clean and at the pinned commit; run setup first")

    configuration = {
        "format_version": 1, "mode": args.mode, "code_commit": commit,
        "upstream_commit": actual, "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(), "cuda_runtime": torch.version.cuda,
        "packages": {name: version(name) for name in ("torch", "numpy", "taichi", "sympy", "pyevtk")},
        "generation_parameters": generation_parameters(args.mode), "split_seed": 20260918,
    }
    dataset_root = args.storage_root.resolve() / "datasets"
    dataset_root.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{job}_{uuid.uuid4().hex[:8]}"
    if args.resume:
        output = args.resume.expanduser().resolve()
        if output.parent != dataset_root or not output.is_dir():
            parser.error("--resume must be an existing dataset directly under FLOW3D_ROOT/datasets")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        output = dataset_root / f"{stamp}_lbm3d-{args.mode}_{attempt_id}"
        output.mkdir()  # Unique directory; never overwrite another collection.

    # Linux advisory lock is released by the OS even if Slurm kills the job.
    import fcntl
    with (output / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another job is already generating this dataset")
        if args.resume:
            check_resume(output, configuration)
        else:
            write_json(output / "generation.json", configuration)
        record = output / "attempts" / attempt_id
        record.mkdir(parents=True)
        os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = str(record / "taichi-cache")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        write_json(record / "environment.json", {
            **configuration, "slurm_job_id": job, "host": platform.node(),
            "architecture": platform.machine(), "modules": os.environ.get("LOADEDMODULES"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "submit_command": os.environ.get("FLOW3D_SUBMIT_COMMAND"),
        })
        (record / "pip-freeze.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        started = time.monotonic()
        result = {"status": "running", "mode": args.mode, "dataset": str(output)}
        write_json(record / "results.json", result)
        print(f"Dataset: {output}", flush=True)
        print(f"Attempt record: {record}", flush=True)
        try:
            run_logged([
                sys.executable, str(REPOSITORY / "scripts/generate_taichi_lbm3d_flow_collection.py"),
                "--output-dir", str(output), "--upstream-repo", str(upstream),
                *configuration["generation_parameters"],
            ], record)
            paths = validate_archives(output, 3 if args.mode == "pilot" else 1000)
            run_logged([
                sys.executable, str(REPOSITORY / "scripts/build_sparse_flow_manifest.py"),
                "--input-dir", str(output), "--output", str(output / "manifest.json"),
                "--seed", str(configuration["split_seed"]),
            ], record)
            # Hash the actual regenerated files: CPU and CUDA results need not be bitwise equal.
            checksums = output / "SHA256SUMS"
            with checksums.open("w", encoding="utf-8", newline="\n") as stream:
                for path in [*paths, output / "manifest.json", output / "collection.json", output / "generation.json"]:
                    stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            result.update(
                status="completed", num_cases=len(paths), manifest=str(output / "manifest.json"),
                split_counts={key: len(value) for key, value in manifest["splits"].items()},
                dataset_version=hashlib.sha256(checksums.read_bytes()).hexdigest(),
                usage="pipeline validation only" if args.mode == "pilot" else "candidate training dataset; assess physical validity separately",
            )
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(record / "results.json", result)
        write_json(output / "COMPLETE.json", result)
        print(f"Completed. Manifest: {output / 'manifest.json'}", flush=True)
    if args.train_epochs:
        train_after_generation(output, args.storage_root.resolve(), args.train_epochs, record)


if __name__ == "__main__":
    main()
