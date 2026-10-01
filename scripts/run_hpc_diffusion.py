#!/usr/bin/env python
"""Run the real diffusion CLI with Slurm provenance and isolated work storage.

The smoke mode generates a tiny synthetic fixture without Taichi. It is an
environment/integration check, never a scientific reconstruction result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def smoke_manifest(directory: Path) -> Path:
    """Create the same small data contract used by the real CLI CPU test."""
    directory.mkdir(parents=True)
    z, y, x = np.meshgrid(*(np.arange(8, dtype=np.float32),) * 3, indexing="ij")
    flow = np.stack((0.010 + 0.001*x + 0.0002*y,
                     -0.006 + 0.0008*y + 0.0001*z,
                     0.004 + 0.0007*z - 0.0001*x))[None]
    initial = np.asarray(((2, 2, 2), (3, 4, 2.5), (4, 2.5, 5), (5, 5, 4)), dtype=np.float32)
    times = np.asarray((0, 0.5, 1), dtype=np.float32)
    trajectories = initial[:, None] + times[None, :, None] * np.asarray((0.04, -0.02, 0.03), dtype=np.float32)
    projection = np.asarray((((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1))), dtype=np.float32)
    tracks = np.einsum("ntc,voc->vnto", trajectories, projection)[None]
    solid = np.zeros((8, 8, 8), dtype=bool)
    solid[:, :, 0] = True
    np.savez_compressed(
        directory / "case.npz", flow_field=flow, trajectories_2d=tracks,
        projection_matrix=projection, observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
        observation_times=times, domain_bounds=np.asarray(((0, 7),) * 3, dtype=np.float32),
        solid_mask=solid, particle_counts=np.asarray([4], dtype=np.int64),
        metadata=np.asarray(json.dumps({"case_id": "hpc-smoke", "lid_face": "x=max",
                                       "lid_velocity_xyz": [0, 0, 0.05], "scope": "smoke only"})),
    )
    entry = {"case_id": "hpc-smoke", "path": "case.npz"}
    path = directory / "manifest.json"
    write_json(path, {"format_version": 1, "mode": "smoke_test", "scientific_result": False,
                     "allow_overlap_for_smoke_test": True,
                     "splits": {name: [entry] for name in ("train", "validation", "test")}})
    return path


def run_logged(command: list[str], directory: Path) -> None:
    rendered = shlex.join(command)
    with (directory / "command.txt").open("a", encoding="utf-8") as commands:
        commands.write(rendered + "\n")
    with (directory / "log.txt").open("a", encoding="utf-8") as log:
        print(rendered, flush=True)
        log.write(rendered + "\n")
        with subprocess.Popen(command, cwd=REPOSITORY, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("smoke", "train", "inference"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--storage-root", type=Path, default=os.environ.get("FLOW3D_ROOT"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    # Options after -- are parsed by the original research training/sampling CLI.
    raw = sys.argv[1:]
    separator = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    extra = raw[separator + 1:]
    if args.storage_root is None:
        parser.error("set FLOW3D_ROOT or --storage-root")
    if args.mode == "smoke" and (extra or args.manifest or args.checkpoint):
        parser.error("smoke uses fixed tiny data and model settings")
    if args.mode != "smoke" and (args.manifest is None or not args.manifest.is_file()):
        parser.error("train/inference requires an existing --manifest on this machine")
    if args.mode == "inference" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("inference requires an existing --checkpoint")
    if args.mode == "train" and args.checkpoint is not None:
        parser.error("use -- --resume /absolute/path/latest.pt to resume training")
    managed = {"--manifest", "--checkpoint", "--output", "--output-dir", "--device", "--"}
    if any(arg.split("=", 1)[0] in managed for arg in extra):
        parser.error("manifest/checkpoint/device/output are managed by this wrapper")
    job = os.environ.get("SLURM_JOB_ID")
    if args.device == "cuda":
        if not job:
            parser.error("GPU execution requires a Slurm job; use scripts/submit.sh")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            parser.error("exactly one working CUDA GPU is required")
        if not str(args.storage_root.resolve()).startswith("/work/"):
            parser.error("HPC output storage must be under /work")
    elif job or args.mode != "smoke":
        parser.error("CPU execution is only for a local smoke test outside Slurm")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_id += f"_diffusion-{args.mode}_{job or 'local'}_{uuid.uuid4().hex[:8]}"
    root = args.storage_root.expanduser().resolve()
    record = root / "results" / "experiments" / run_id
    checkpoints = root / "checkpoints" / run_id
    record.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(record / ".matplotlib"))
    print(f"Experiment record: {record}", flush=True)
    print(f"Checkpoints: {checkpoints}", flush=True)
    environment = {
        "commit": os.environ.get("FLOW3D_COMMIT", "unrecorded-local-run"),
        "slurm_job_id": job, "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "modules": os.environ.get("LOADEDMODULES", ""), "host": platform.node(),
        "architecture": platform.machine(), "python": platform.python_version(),
        "executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "gpu_capability": torch.cuda.get_device_capability(0) if args.device == "cuda" else None,
        "torch_cuda_architectures": torch.cuda.get_arch_list() if args.device == "cuda" else [],
        "submit_command": os.environ.get("FLOW3D_SUBMIT_COMMAND"),
    }
    write_json(record / "environment.json", environment)
    result = {"status": "running", "mode": args.mode, "checkpoint_dir": str(checkpoints)}
    write_json(record / "results.json", result)
    try:
        manifest = smoke_manifest(record / "smoke_data") if args.mode == "smoke" else args.manifest.resolve()
        # Validate all case files and split provenance before starting training.
        sys.path.insert(0, str(REPOSITORY / "src"))
        from flow_observation.sparse_dataset import load_sparse_flow_manifest
        document, cases = load_sparse_flow_manifest(manifest)
        write_json(record / "dataset.json", {
            "manifest": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "mode": document.get("mode", "scientific"), "manifest_content": document,
            "resolved_files": {split: [str(case.path) for case in entries] for split, entries in cases.items()},
        })
        commands = []
        if args.mode in ("smoke", "train"):
            train_args = extra if args.mode == "train" else [
                "--epochs", "1", "--batch-size", "1", "--num-workers", "0",
                "--num-particles", "2", "--diffusion-steps", "4", "--base-channels", "8",
                "--condition-dim", "32", "--time-embedding-dim", "16",
                "--condition-dropout", "0", "--seed", "31", "--save-every", "1",
            ]
            if args.device == "cpu":
                train_args += ["--no-amp"]
            commands.append([sys.executable, str(REPOSITORY / "scripts" / "train_sparse_track_diffusion.py"),
                             *train_args, "--manifest", str(manifest), "--output-dir", str(checkpoints),
                             "--device", args.device])
        if args.mode in ("smoke", "inference"):
            sample_args = extra if args.mode == "inference" else [
                "--num-particles", "2", "--num-probe-particles", "2", "--num-samples", "1",
                "--sampling-steps", "2", "--eta", "0", "--cfg-scale", "1", "--seed", "47",
            ]
            checkpoint = checkpoints / "best.pt" if args.mode == "smoke" else args.checkpoint.resolve()
            commands.append([sys.executable, str(REPOSITORY / "scripts" / "sample_sparse_track_diffusion.py"),
                             *sample_args, "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                             "--output", str(record / "posterior.npz"), "--device", args.device])
        # The original parsers capture defaults too, including seeds and model sizes.
        configurations = []
        for command in commands:
            module = importlib.import_module(Path(command[1]).stem)
            configurations.append(vars(module._build_parser().parse_args(command[2:])))
        # JSON is also valid YAML; avoid adding a runtime dependency on PyYAML.
        write_json(record / "config.yaml", {"mode": args.mode, "commands": commands, "parameters": configurations})
        for command in commands:
            run_logged(command, record)
        if args.mode in ("smoke", "train"):
            result["training"] = json.loads((checkpoints / "training_summary.json").read_text())
        if args.mode in ("smoke", "inference"):
            with np.load(record / "posterior.npz", allow_pickle=False) as posterior:
                if not np.isfinite(posterior["posterior_samples"]).all():
                    raise RuntimeError("posterior samples contain nonfinite values")
                result["sampling_metadata"] = json.loads(str(posterior["metadata"].item()))
                result["metrics"] = json.loads(str(posterior["metrics"].item()))
        if args.mode == "smoke":
            result["scientific_result"] = False
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(record / "results.json", result)
    print(f"Completed. Results: {record / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
