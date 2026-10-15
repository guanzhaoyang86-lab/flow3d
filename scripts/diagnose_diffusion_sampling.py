#!/usr/bin/env python
"""Trace frozen checkpoint sampling on validation cases without changing training.

This is a small, paired diagnostic, not a final test-set evaluation. Ground truth
is used only for metrics and explicitly labelled teacher-forced checks. No field,
latent, cached target, model weight, or checkpoint is written by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sample_sparse_track_diffusion import (
    _as_batch, _field_metrics, _load_models, _positive_integer,
    _nonnegative_integer, _resolve_device,
)
from flow_observation.checkpoint_io import load_trusted_checkpoint
from flow_observation.sparse_dataset import (
    SparseFlowDataset, denormalize_flow, normalize_flow,
)
from flow_observation.sparse_physics import (
    apply_hard_boundary_conditions, batched_trajectory_consistency,
)
from flow_observation.tensor_space import TensorSpaceCodec


VARIANTS = (
    {"name": "correct_cfg1p5_eta1", "condition": "correct", "cfg_scale": 1.5, "eta": 1.0},
    {"name": "zero_condition_cfg1p5_eta1", "condition": "zero", "cfg_scale": 1.5, "eta": 1.0},
    {"name": "other_case_condition_cfg1p5_eta1", "condition": "other_case", "cfg_scale": 1.5, "eta": 1.0},
    {"name": "correct_cfg1_eta1", "condition": "correct", "cfg_scale": 1.0, "eta": 1.0},
    {"name": "correct_cfg1p5_eta0", "condition": "correct", "cfg_scale": 1.5, "eta": 0.0},
)
_TRACE_FIELDS = (
    "sample", "predicted_noise", "predicted_x0_raw", "predicted_x0",
    "clean_for_update", "updated_sample",
)


def _parse_indices(value: str) -> list[int]:
    """Parse a nonempty, unique, nonnegative comma-separated validation index list."""
    try:
        indices = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("indices must be unique nonnegative integers")
    return indices


def _build_parser() -> argparse.ArgumentParser:
    """Build the validation-only command line; no test/train split is accepted."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=_parse_indices, default=_parse_indices("0,1,2"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=_positive_integer, default=1)
    parser.add_argument("--sampling-steps", type=_positive_integer, default=50)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--num-probe-particles", type=_nonnegative_integer, default=16)
    parser.add_argument("--trajectory-substeps", type=_positive_integer, default=1)
    return parser


def _json_safe(value: Any) -> Any:
    """Convert metadata to strict JSON, representing nonfinite scalars as null.

    Large tensors are summarized, never dumped; checkpoint tensors themselves
    must not be supplied as provenance. Path-valued training config is supported.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return _json_safe(value.detach().cpu().tolist()) if value.numel() <= 64 else _tensor_summary(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "tracked_worktree_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": os.environ.get("FLOW3D_COMMIT"),
                "commit_source": "FLOW3D_COMMIT" if os.environ.get("FLOW3D_COMMIT") else None,
                "error": str(error)}


def _tensor_summary(value: Tensor) -> dict[str, Any]:
    """Return finite counts and float64 RMS/extrema without retaining a tensor.

    RMS/extrema are null when the tensor contains nonfinite values; finite_count
    remains available so a failure is never silently hidden by finite filtering.
    """
    value = value.detach()
    finite_count = int(torch.isfinite(value).sum().cpu())
    result: dict[str, Any] = {"count": value.numel(), "finite_count": finite_count,
                              "all_finite": finite_count == value.numel()}
    if not result["all_finite"] or not value.numel():
        return {**result, "rms": None, "max_abs": None, "min": None, "max": None}
    values = value.double()
    result.update(rms=float(values.square().mean().sqrt().cpu()),
                  max_abs=float(values.abs().max().cpu()),
                  min=float(values.min().cpu()), max=float(values.max().cpu()))
    return _json_safe(result)


def _tensor_factor_metrics(codec: TensorSpaceCodec, latent: Tensor) -> dict[str, Any]:
    """Inspect unstandardized Tucker core/factors without modifying their gauge."""
    raw = latent.detach().float() * codec.latent_std + codec.latent_mean
    offset = 3 * codec.rank ** 3
    core = raw[:, :offset]
    result: dict[str, Any] = {"core": _tensor_summary(core), "factors": []}
    result["core"]["frobenius_norm"] = _json_safe(float(core.double().norm().cpu()))
    for axis, size in enumerate(codec.spatial_shape):
        end = offset + size * codec.rank
        factor = raw[:, offset:end].reshape(-1, size, codec.rank).double()
        info = {"axis": axis, **_tensor_summary(factor)}
        info["frobenius_norm"] = _json_safe(float(factor.norm().cpu()))
        if info["all_finite"]:
            info["spectral_norm"] = float(torch.linalg.matrix_norm(factor, ord=2).max().cpu())
            identity = torch.eye(codec.rank, device=factor.device, dtype=factor.dtype)
            info["orthogonality_residual_frobenius"] = float((factor.transpose(-1, -2) @ factor - identity).norm().cpu())
        else:
            info.update(spectral_norm=None, orthogonality_residual_frobenius=None)
        result["factors"].append(info)
        offset = end
    return _json_safe(result)


def _physical_summary(physical: Tensor, reference: Tensor, interior: Tensor) -> dict[str, Any]:
    weights = interior[:, None].expand_as(reference)
    values = physical[weights].double()
    truth = reference[weights].double()
    reference_norm = float(truth.norm().cpu())
    result = _tensor_summary(physical)
    result.update(reference_interior_norm=reference_norm,
                  physical_interior_norm=_json_safe(float(values.norm().cpu())),
                  physical_norm_ratio_unknown_interior=(
                      _json_safe(float((values.norm() / truth.norm()).cpu()))
                      if reference_norm > 0 else None),
                  physical_norm_ratio_defined=reference_norm > 0)
    return result


class _TraceObserver:
    """Read-only sampler observer that flushes one aggregate JSON record per step."""

    def __init__(self, path: Path, decode: Callable[[Tensor], Tensor], reference: Tensor,
                 interior: Tensor, codec: TensorSpaceCodec | None) -> None:
        self.path, self.decode, self.reference, self.interior, self.codec = path, decode, reference, interior, codec
        self.records: list[dict[str, Any]] = []
        self.first_anomaly: dict[str, Any] | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        record: dict[str, Any] = {key: event[key] for key in ("step", "timestep", "previous_timestep")}
        record["tensors"] = {key: _tensor_summary(event[key]) for key in _TRACE_FIELDS}
        record["physical"] = {}
        reasons = [f"{key} contains nonfinite values" for key in _TRACE_FIELDS
                   if not record["tensors"][key]["all_finite"]]
        for key in ("sample", "predicted_x0_raw", "predicted_x0", "updated_sample"):
            if not record["tensors"][key]["all_finite"]:
                continue
            try:
                physical = self.decode(event[key])
                info = _physical_summary(physical, self.reference, self.interior)
                record["physical"][key] = info
                ratio = info["physical_norm_ratio_unknown_interior"]
                if not info["all_finite"]:
                    reasons.append(f"decoded {key} contains nonfinite values")
                # x_t and the updated state include intentional diffusion noise;
                # their decoded magnitudes are recorded but are not clean-field
                # failures. Only predicted clean fields trigger this threshold.
                if key in ("predicted_x0_raw", "predicted_x0") and ratio is not None and ratio > 10:
                    reasons.append(f"decoded {key} interior norm exceeds 10 times reference")
            except (ValueError, RuntimeError) as error:
                record["physical"][key] = {"decode_error": str(error)}
                reasons.append(f"{key} decoding failed: {error}")
        if self.codec is not None:
            record["tucker"] = {key: _tensor_factor_metrics(self.codec, event[key])
                                 for key in ("predicted_x0_raw", "predicted_x0", "updated_sample")}
        if self.first_anomaly is None and reasons:
            self.first_anomaly = {"step": event["step"], "timestep": event["timestep"], "reasons": reasons}
        record["anomalies"] = reasons
        record = _json_safe(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
        self.records.append(record)


@torch.no_grad()
def _replay_metrics(physical: Tensor, batch: dict[str, Any], substeps: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, track_key, mask_key in (("observed", "tracks", "mask"), ("probe", "probe_tracks", "probe_mask")):
        if track_key not in batch:
            continue
        loss, _, validity = batched_trajectory_consistency(
            physical, batch[track_key], batch["projection"], batch["times"], batch["bounds"],
            observation_mask=batch[mask_key], num_substeps=substeps, boundary_mode="clamp",
        )
        result[f"{prefix}_track_rmse_cells"] = float(loss.sqrt().cpu())
        result[f"{prefix}_replay_valid_fraction"] = float(validity.float().mean().cpu())
        result["all_replay_particles_valid" if prefix == "observed" else "all_probe_particles_valid"] = bool(validity.all().cpu())
    return _json_safe(result)


@torch.no_grad()
def _teacher_forced(diffusion: Any, model: Any, clean: Tensor,
                    conditions: dict[str, Tensor | None], seed: int) -> list[dict[str, Any]]:
    """Probe epsilon prediction on fresh noised validation targets, never a cache.

    All condition variants share the exact target/noise at each timestep. The
    oracle identity uses the known noise to distinguish algebra from prediction.
    """
    requested = [999, 979, 500, 0]
    timesteps = sorted({t for t in requested if t < diffusion.num_steps} | {diffusion.num_steps - 1}, reverse=True)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    rows = []
    for timestep in timesteps:
        time = torch.tensor([timestep], device=clean.device, dtype=torch.long)
        noisy = diffusion.q_sample(clean, time, noise=noise)
        unconditional = model(noisy, time, None)
        oracle = (noisy - diffusion.sqrt_one_minus_alpha_bars[timestep] * noise) / diffusion.sqrt_alpha_bars[timestep].clamp_min(1e-12)
        row: dict[str, Any] = {"timestep": timestep, "oracle_x0_raw_mse": float((oracle.double() - clean.double()).square().mean().cpu()), "conditions": {}}
        for name, condition in conditions.items():
            if condition is None:
                continue
            prediction = model(noisy, time, condition)
            guided = unconditional + 1.5 * (prediction - unconditional)
            row["conditions"][name] = {
                "noise_mse_cfg1": float((prediction.double() - noise.double()).square().mean().cpu()),
                "noise_mse_cfg1p5": float((guided.double() - noise.double()).square().mean().cpu()),
                "predicted_noise": _tensor_summary(prediction),
            }
        rows.append(_json_safe(row))
    return rows


def _plot_traces(path: Path, variants: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; numerical JSONL traces are complete"
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for variant in variants:
        trace = variant.get("trace_file")
        if not trace or not Path(trace).is_file():
            continue
        rows = [json.loads(line) for line in Path(trace).read_text(encoding="utf-8").splitlines()]
        x = [row["step"] for row in rows]
        axes[0].plot(x, [row["tensors"]["predicted_x0_raw"]["rms"] for row in rows], label=variant["name"])
        axes[1].plot(x, [row["physical"].get("predicted_x0", {}).get("physical_norm_ratio_unknown_interior") for row in rows])
    for axis in axes:
        axis.set_xlabel("Sampling step (zero based)")
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Raw predicted x0 RMS")
    axes[1].set_ylabel("Decoded x0 / reference interior norm")
    axes[1].axhline(10, linestyle=":", color="black", linewidth=1)
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return None


def run_diagnostics(args: argparse.Namespace) -> Path:
    """Run fixed paired validation ablations, preserving partial failure evidence.

    Return summary.json on success. Any Python failure is serialized and causes
    RuntimeError after remaining independent checkpoints/cases/variants finish.
    An existing output directory is refused before any diagnostic write.
    """
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)
    if device.type == "cuda" and os.name != "nt" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA diagnostics on the Linux HPC site require a Slurm allocation")
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary: dict[str, Any] = {
        "format_version": 1, "diagnostic_only": True, "scientific_result": False,
        "warning": "Validation diagnosis only: one posterior per case/variant, not a final test-set result.",
        "split": "validation", "status": "running", "device": str(device),
        "git": _git_provenance(), "command": sys.argv, "arguments": vars(args),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "source_sha256": {"diagnose_diffusion_sampling.py": _sha256(Path(__file__)),
                          "diffusion.py": _sha256(_ROOT / "src" / "flow_observation" / "diffusion.py"),
                          "sample_sparse_track_diffusion.py": _sha256(_ROOT / "scripts" / "sample_sparse_track_diffusion.py")},
        "manifest": str(args.manifest.expanduser().resolve()),
        "boundary_projection": "final", "weights": "ema",
        "sampling_policy": "checkpoint schedule/buffers/clipping unchanged; no trajectory guidance",
        "anomaly_policy": "first nonfinite/decode failure at any stage or predicted clean x0 physical interior norm ratio >10; noisy-state magnitudes are recorded only; descriptive, never affects sampling",
        "checkpoints": [], "failures": [],
    }

    def save() -> None:
        _write_json(summary_path, summary)

    def failed(record: dict[str, Any], error: Exception, location: Path) -> None:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        failure = {"location": str(location), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        summary["failures"].append(failure)
        _write_json(location / "failure.json", failure)
        save()

    save()
    try:
        summary["manifest_sha256"] = _sha256(args.manifest.expanduser().resolve())
    except Exception as error:
        failed(summary, error, output)
        raise RuntimeError(f"diagnostics failed; see {summary_path}") from error
    for checkpoint_index, checkpoint_path_arg in enumerate(args.checkpoint):
        checkpoint_path = checkpoint_path_arg.expanduser().resolve()
        checkpoint_dir = output / f"checkpoint_{checkpoint_index:02d}"
        checkpoint_dir.mkdir()
        record: dict[str, Any] = {"checkpoint": str(checkpoint_path), "status": "running", "cases": []}
        summary["checkpoints"].append(record)
        save()
        try:
            record["checkpoint_sha256"] = _sha256(checkpoint_path)
            checkpoint = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
                raise ValueError("unsupported sparse diffusion checkpoint")
            record.update(architecture=checkpoint["model_config"].get("architecture", "unet3d"),
                          model_config=checkpoint["model_config"], train_config=checkpoint.get("train_config"),
                          data_config=checkpoint["data_config"], normalization=checkpoint["normalization"],
                          training_provenance=checkpoint.get("training_provenance"),
                          tensor_artifact_sha256=checkpoint.get("tensor_artifact_sha256"))
            record["training_progress"] = {
                key: checkpoint.get(key) for key in (
                    "epoch", "best_epoch", "best_validation_loss",
                    "best_validation_total_loss", "best_validation_diffusion_loss",
                )
            }
            history = checkpoint.get("history", [])
            if isinstance(history, list):
                record["training_progress"].update(history_rows=len(history),
                                                     history_first=history[0] if history else None,
                                                     history_last=history[-1] if history else None)
                validation_rows = [row for row in history if isinstance(row, dict)
                                   and isinstance(row.get("validation_total"), (int, float))
                                   and math.isfinite(row["validation_total"])]
                if validation_rows:
                    best_row = min(validation_rows, key=lambda row: row["validation_total"])
                    record["training_progress"]["history_best_validation_row"] = best_row
                    record["training_progress"]["history_best_epoch"] = best_row.get("epoch")
            training_provenance = checkpoint.get("training_provenance")
            trained_manifest = training_provenance.get("manifest_sha256") if isinstance(training_provenance, dict) else None
            record["manifest_matches_training"] = trained_manifest == summary["manifest_sha256"] if trained_manifest else None
            record["provenance_warnings"] = []
            if record["manifest_matches_training"] is not True:
                record["provenance_warnings"].append(
                    "evaluation manifest SHA256 differs from training manifest" if trained_manifest
                    else "checkpoint does not record a training manifest SHA256")
                print(f"Warning: {checkpoint_path.name}: {record['provenance_warnings'][-1]}", flush=True)
            encoder, model, diffusion, stats = _load_models(checkpoint, device, False)
            if args.sampling_steps > diffusion.num_steps:
                raise ValueError("sampling-steps exceeds checkpoint diffusion steps")
            record["diffusion"] = {"num_steps": diffusion.num_steps, "schedule": diffusion.schedule_name,
                                   "clip_x0": diffusion.clip_x0, "alpha_bar_first": float(diffusion.alpha_bars[0].cpu()),
                                   "alpha_bar_last": float(diffusion.alpha_bars[-1].cpu())}
            dataset = SparseFlowDataset(args.manifest, "validation", num_particles=int(checkpoint["data_config"]["num_particles"]),
                                        num_probe_particles=args.num_probe_particles, observations_per_flow=1, seed=args.seed, normalize=False)
            if any(index >= len(dataset) for index in args.indices):
                raise IndexError(f"validation indices {args.indices} exceed dataset length {len(dataset)}")
            record["smoke_test_manifest"] = dataset.is_smoke_test
            codec = None
            if record["architecture"] == "tensor-dit":
                if not isinstance(checkpoint.get("tensor_codec"), dict):
                    raise ValueError("tensor-dit checkpoint is missing its self-contained codec")
                codec = TensorSpaceCodec.from_artifact(checkpoint["tensor_codec"]).to(device).eval()
                if codec.rank != checkpoint["model_config"]["tensor_rank"]:
                    raise ValueError("checkpoint tensor rank and codec disagree")
            for index in args.indices:
                case_dir = checkpoint_dir / f"validation_{index:04d}"
                case_dir.mkdir()
                case_record: dict[str, Any] = {"index": index, "status": "running", "variants": []}
                record["cases"].append(case_record)
                try:
                    item = dataset[index]
                    batch = _as_batch(item, device)
                    reference = batch["flow"]
                    interior = ~batch["solid_mask"] & ~batch["boundary_mask"]
                    if not bool(interior.any()):
                        raise ValueError("validation field has no unknown interior")
                    if codec is not None and tuple(codec.spatial_shape) != tuple(reference.shape[-3:]):
                        raise ValueError("checkpoint spatial shape differs from validation field")
                    case_record.update(case_id=item["case_id"], flow_group_id=item["flow_group_id"],
                                       source_sample_index=item["source_sample_index"], particle_indices=item["particle_indices"],
                                       probe_particle_indices=item.get("probe_particle_indices"),
                                       reference=_tensor_summary(reference))

                    def encode_condition(value: dict[str, Any]) -> Tensor:
                        with torch.no_grad():
                            return encoder(value["tracks"], value["mask"], value["projection"], value["times"], value["bounds"])

                    correct = encode_condition(batch)
                    conditions = {"correct": correct, "zero": torch.zeros_like(correct), "other_case": None}
                    # Cycle across the complete validation split, never shuffle a
                    # set encoder's particle order as a fake conditioning ablation.
                    donor_index = next(((index + offset) % len(dataset) for offset in range(1, len(dataset))
                                        if dataset.cases[(index + offset) % len(dataset)].flow_group_id != item["flow_group_id"]), None)
                    if donor_index is not None:
                        donor_item = dataset[donor_index]
                        conditions["other_case"] = encode_condition(_as_batch(donor_item, device))
                        case_record["other_condition"] = {"index": donor_index, "case_id": donor_item["case_id"], "flow_group_id": donor_item["flow_group_id"]}
                    else:
                        case_record["other_condition"] = {"unavailable": "validation split has no different physical flow group"}

                    def decode(value: Tensor) -> Tensor:
                        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                            normalized = codec.decode(value.float()) if codec is not None else value.float()
                            return denormalize_flow(normalized, stats)

                    case_record["ground_truth_replay"] = _replay_metrics(reference, batch, args.trajectory_substeps)
                    with torch.no_grad():
                        clean = normalize_flow(reference, stats)
                        if codec is not None:
                            clean = codec.encode(clean)  # Never use the artifact's latent cache.
                            case_record["codec_validation_reconstruction"] = _field_metrics(reference, decode(clean), interior)
                            case_record["reference_tucker"] = _tensor_factor_metrics(codec, clean)
                        case_record["teacher_forced_target"] = "fresh codec.encode(validation flow)" if codec is not None else "normalized validation flow"
                        case_record["teacher_forced"] = _teacher_forced(diffusion, model, clean, conditions, args.seed + index + 100000)
                    sample_seed = args.seed + index
                    initial_generator = torch.Generator(device=device).manual_seed(sample_seed)
                    initial = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=initial_generator)
                    case_record["initial_noise_sha256"] = hashlib.sha256(initial.detach().cpu().numpy().tobytes()).hexdigest()
                    case_record["sampling_seed"] = sample_seed
                    case_record["stochastic_update_seed"] = sample_seed + 10000
                    save()
                    for variant in VARIANTS:
                        variant_dir = case_dir / variant["name"]
                        variant_dir.mkdir()
                        result = {**variant, "status": "running", "trace_file": str(variant_dir / "trace.jsonl")}
                        case_record["variants"].append(result)
                        if conditions[variant["condition"]] is None:
                            result.update(status="skipped", reason="no validation case with a different physical flow group")
                            save()
                            continue
                        observer = _TraceObserver(variant_dir / "trace.jsonl", decode, reference, interior, codec)
                        save()
                        try:
                            generator = torch.Generator(device=device).manual_seed(sample_seed + 10000)
                            normalized = diffusion.ddim_sample(
                                model, tuple(initial.shape), conditions[variant["condition"]],
                                sampling_steps=args.sampling_steps, eta=variant["eta"], guidance_scale=variant["cfg_scale"],
                                unconditional_condition=None, projection=None, initial_noise=initial,
                                generator=generator, device=device, trace_callback=observer,
                            )
                            physical = apply_hard_boundary_conditions(decode(normalized), batch["boundary_mask"].unsqueeze(1), batch["boundary_values"])
                            if not bool(torch.isfinite(physical).all()):
                                raise RuntimeError("sampling produced a nonfinite decoded velocity field")
                            metrics = _field_metrics(reference, physical, interior)
                            metrics.update(_replay_metrics(physical, batch, args.trajectory_substeps))
                            velocity = _tensor_summary(physical)
                            metrics.update(velocity_min=velocity["min"], velocity_max=velocity["max"], velocity_max_abs=velocity["max_abs"],
                                           physical_norm_ratio_unknown_interior=_physical_summary(physical, reference, interior)["physical_norm_ratio_unknown_interior"])
                            result.update(status="completed", metrics=metrics, first_anomaly=observer.first_anomaly,
                                          trace_steps=len(observer.records), boundary_projection="final")
                            _write_json(variant_dir / "result.json", result)
                            print(f"{checkpoint_path.name} validation[{index}] {variant['name']}: completed", flush=True)
                        except Exception as error:
                            result.update(first_anomaly=observer.first_anomaly, trace_steps=len(observer.records))
                            failed(result, error, variant_dir)
                        save()
                    case_record["status"] = "failed" if any(v["status"] == "failed" for v in case_record["variants"]) else "completed"
                    try:
                        warning = _plot_traces(case_dir / "sampling_trace.png", case_record["variants"])
                        if warning:
                            case_record["plot_warning"] = warning
                        else:
                            case_record["plot"] = str(case_dir / "sampling_trace.png")
                    except Exception as error:
                        failed(case_record, error, case_dir)
                    save()
                except Exception as error:
                    failed(case_record, error, case_dir)
            record["status"] = "failed" if any(case["status"] == "failed" for case in record["cases"]) else "completed"
            del encoder, model, diffusion, codec, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save()
        except Exception as error:
            failed(record, error, checkpoint_dir)
    summary["status"] = "failed" if summary["failures"] else "completed"
    save()
    if summary["failures"]:
        raise RuntimeError(f"{len(summary['failures'])} diagnostic failure(s); partial results saved to {summary_path}")
    return summary_path


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        output = run_diagnostics(args)
    except Exception as error:
        parser.error(str(error))
    print(f"Validation diagnostics saved to {output}")


if __name__ == "__main__":
    main()
