#!/usr/bin/env python
"""Sample and evaluate 3D flow posteriors from sparse particle tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import numpy as np
from torch import Tensor


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.diffusion import GaussianDiffusion
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.unet3d import ConditionalUNet3D
from flow_observation.models.factory import build_denoiser, diffusion_clip
from flow_observation.sparse_dataset import (
    FlowNormalizationStats,
    SparseFlowDataset,
    denormalize_flow,
    load_sparse_flow_manifest,
    normalize_flow,
)
from flow_observation.sparse_physics import (
    apply_hard_boundary_conditions,
    batched_divergence_mse,
    batched_trajectory_consistency,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Generate posterior 3D flow samples from sparse tracks."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-particles", type=_positive_integer, default=None)
    parser.add_argument("--num-probe-particles", type=_nonnegative_integer, default=0)
    parser.add_argument("--num-samples", type=_positive_integer, default=16)
    parser.add_argument("--sampling-steps", type=_positive_integer, default=50)
    parser.add_argument("--eta", type=_nonnegative_float, default=1.0)
    parser.add_argument("--cfg-scale", type=_nonnegative_float, default=1.5)
    parser.add_argument("--boundary-projection", choices=("final", "each-step"), default="each-step",
                        help="Use final for comparisons with tensor-dit; latent decoding enforces boundaries at the end.")
    parser.add_argument(
        "--trajectory-guidance-strength", type=_nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--divergence-guidance-weight", type=_nonnegative_float, default=0.0
    )
    parser.add_argument("--trajectory-substeps", type=_positive_integer, default=1)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/sparse_track_posterior.npz")
    )
    parser.add_argument(
        "--use-raw-weights",
        action="store_true",
        help="Use non-EMA weights; EMA is the default for posterior sampling.",
    )
    return parser


def _cuda_architecture_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in set(torch.cuda.get_arch_list())


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    supported = _cuda_architecture_supported()
    if requested == "cuda":
        if not supported:
            raise RuntimeError(
                "the current PyTorch build does not support the visible CUDA GPU"
            )
        return torch.device("cuda")
    return torch.device("cuda" if supported else "cpu")


def _as_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, Tensor):
            result[key] = value.unsqueeze(0).to(device=device)
        else:
            result[key] = value
    return result


def _field_metrics(reference: Tensor, estimate: Tensor, mask: Tensor) -> dict[str, float]:
    weights = mask[:, None].expand_as(reference)
    reference_values = reference[weights]
    estimate_values = estimate[weights]
    difference = estimate_values - reference_values
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference_values
    ).clamp_min(1e-12)
    cosine = torch.dot(reference_values, estimate_values) / (
        torch.linalg.vector_norm(reference_values)
        * torch.linalg.vector_norm(estimate_values)
    ).clamp_min(1e-12)
    return {
        "field_relative_l2_unknown_interior": float(relative_l2.detach().cpu()),
        "field_cosine_unknown_interior": float(cosine.detach().cpu()),
    }


def _uncertainty_error_correlation(
    reference: Tensor,
    estimate: Tensor,
    posterior_variance: Tensor,
    mask: Tensor,
) -> tuple[float, bool]:
    """Correlate elementwise posterior variance with squared field error.

    Pearson correlation is undefined when either vector has zero variance or
    fewer than two entries.  In that case we serialize ``0.0`` and return a
    separate ``False`` flag instead of writing NaN.
    """

    weights = mask[:, None].expand_as(reference)
    uncertainty = posterior_variance[weights].to(torch.float64)
    squared_error = (estimate - reference).square()[weights].to(torch.float64)
    if uncertainty.numel() < 2:
        return 0.0, False
    uncertainty = uncertainty - uncertainty.mean()
    squared_error = squared_error - squared_error.mean()
    denominator = torch.linalg.vector_norm(uncertainty) * torch.linalg.vector_norm(
        squared_error
    )
    if not bool(torch.isfinite(denominator).detach()) or float(denominator) == 0.0:
        return 0.0, False
    correlation = torch.dot(uncertainty, squared_error) / denominator
    if not bool(torch.isfinite(correlation).detach()):
        return 0.0, False
    return float(correlation.detach().cpu()), True


def _manifest_provenance(manifest_path: Path) -> dict[str, object]:
    """Compute the same immutable split fingerprint stored by training."""

    resolved = manifest_path.expanduser().resolve()
    manifest, splits = load_sparse_flow_manifest(resolved)
    return {
        "manifest_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "manifest_mode": str(manifest.get("mode", "scientific")),
        "train_flow_group_ids": sorted(
            {case.flow_group_id for case in splits["train"]}
        ),
        "validation_flow_group_ids": sorted(
            {case.flow_group_id for case in splits["validation"]}
        ),
        "test_flow_group_ids": sorted(
            {case.flow_group_id for case in splits["test"]}
        ),
    }


def _scientific_sampling_assessment(
    checkpoint: dict[str, Any],
    manifest_path: Path,
    *,
    split: str,
    trained_particles: int,
    sampled_particles: int,
) -> tuple[bool, list[str], dict[str, object]]:
    """Conservatively decide whether this evaluation can be scientific.

    Legacy checkpoints remain usable, but never acquire a scientific label
    merely because they are evaluated with a disjoint manifest.
    """

    evaluation = _manifest_provenance(manifest_path)
    reasons: list[str] = []
    if evaluation["manifest_mode"] != "scientific":
        reasons.append("evaluation manifest mode is not scientific")
    if split != "test":
        reasons.append("evaluation split is not test")
    if sampled_particles != trained_particles:
        reasons.append(
            f"checkpoint was trained with N={trained_particles}, not N={sampled_particles}"
        )

    training = checkpoint.get("training_provenance")
    required = (
        "manifest_sha256",
        "manifest_mode",
        "train_flow_group_ids",
        "validation_flow_group_ids",
        "test_flow_group_ids",
    )
    if not isinstance(training, dict):
        reasons.append("checkpoint lacks training_provenance")
    else:
        missing = [name for name in required if name not in training]
        if missing:
            reasons.append(
                "checkpoint training_provenance lacks " + ", ".join(missing)
            )
        else:
            if training["manifest_mode"] != "scientific":
                reasons.append("checkpoint was not trained with a scientific manifest")
            for name in required:
                expected = evaluation[name]
                actual = training[name]
                if name.endswith("_flow_group_ids"):
                    if not isinstance(actual, (list, tuple)) or any(
                        not isinstance(value, str) for value in actual
                    ):
                        reasons.append(
                            f"checkpoint training_provenance.{name} is malformed"
                        )
                        continue
                    actual = sorted(set(actual))
                if actual != expected:
                    reasons.append(
                        f"checkpoint training_provenance.{name} does not match "
                        "the evaluation manifest"
                    )
    return not reasons, reasons, evaluation


def _load_models(
    checkpoint: dict[str, Any], device: torch.device, use_raw: bool
) -> tuple[TrackSetEncoder, ConditionalUNet3D, GaussianDiffusion, FlowNormalizationStats]:
    config = checkpoint["model_config"]
    encoder = TrackSetEncoder(condition_dim=int(config["condition_dim"]))
    unet = build_denoiser(config)
    if use_raw:
        encoder.load_state_dict(checkpoint["encoder_state"])
        unet.load_state_dict(checkpoint["unet_state"])
    else:
        encoder.load_state_dict(checkpoint["ema_encoder_state"])
        unet.load_state_dict(checkpoint["ema_unet_state"])
    diffusion = GaussianDiffusion(
        int(config["diffusion_steps"]),
        schedule=str(config["schedule"]),
        clip_x0=diffusion_clip(config),
    )
    diffusion.load_state_dict(checkpoint["diffusion_state"])
    stats = FlowNormalizationStats.from_dict(checkpoint["normalization"])
    return (
        encoder.to(device).eval(),
        unet.to(device).eval(),
        diffusion.to(device).eval(),
        stats,
    )


def sample(args: argparse.Namespace) -> Path:
    if (
        args.divergence_guidance_weight > 0.0
        and args.trajectory_guidance_strength == 0.0
    ):
        raise ValueError(
            "--divergence-guidance-weight is non-zero, but total guidance is "
            "disabled; set --trajectory-guidance-strength greater than zero"
        )
    device = _resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported sparse diffusion checkpoint")
    trained_particles = int(checkpoint["data_config"]["num_particles"])
    num_particles = trained_particles if args.num_particles is None else args.num_particles
    if num_particles != trained_particles:
        print(
            f"Warning: checkpoint was trained with N={trained_particles}, but "
            f"sampling requests N={num_particles}. Treat this as exploratory "
            "unless the model was subsequently fine-tuned with variable N.",
            flush=True,
        )

    dataset = SparseFlowDataset(
        args.manifest,
        args.split,
        num_particles=num_particles,
        num_probe_particles=args.num_probe_particles,
        observations_per_flow=1,
        seed=args.seed,
        normalize=False,
    )
    item = dataset[args.index]
    batch = _as_batch(item, device)
    boundary_mask = batch["boundary_mask"].unsqueeze(1)
    scientific_result, scientific_reasons, evaluation_provenance = (
        _scientific_sampling_assessment(
            checkpoint,
            args.manifest,
            split=args.split,
            trained_particles=trained_particles,
            sampled_particles=num_particles,
        )
    )
    if not scientific_result:
        print(
            "WARNING: output is NON-SCIENTIFIC: " + "; ".join(scientific_reasons),
            flush=True,
        )
    encoder, unet, diffusion, stats = _load_models(
        checkpoint, device, args.use_raw_weights
    )
    tensor_codec = None
    architecture = checkpoint["model_config"].get("architecture", "unet3d")
    if architecture == "tensor-dit":
        from flow_observation.tensor_space import TensorSpaceCodec
        if not isinstance(checkpoint.get("tensor_codec"), dict):
            raise ValueError("tensor-dit checkpoint is missing its self-contained codec")
        tensor_codec = TensorSpaceCodec.from_artifact(checkpoint["tensor_codec"]).to(device)
        if tuple(tensor_codec.spatial_shape) != tuple(batch["flow"].shape[-3:]):
            raise ValueError("checkpoint spatial shape differs from evaluation field")
        if tensor_codec.rank != checkpoint["model_config"]["tensor_rank"]:
            raise ValueError("checkpoint tensor rank and codec disagree")
    with torch.no_grad():
        condition = encoder(
            batch["tracks"],
            batch["mask"],
            batch["projection"],
            batch["times"],
            batch["bounds"],
        )

    def hard_projection(normalized: Tensor) -> Tensor:
        physical = denormalize_flow(normalized, stats)
        physical = apply_hard_boundary_conditions(
            physical, boundary_mask, batch["boundary_values"]
        )
        return normalize_flow(physical, stats)

    def decode(value: Tensor) -> Tensor:
        with torch.autocast(device_type=device.type, enabled=False):
            return tensor_codec.decode(value.float()) if tensor_codec is not None else value.float()

    def observation_guidance(normalized: Tensor, timesteps: Tensor) -> Tensor:
        del timesteps
        physical = denormalize_flow(decode(normalized), stats)
        physical = apply_hard_boundary_conditions(
            physical, boundary_mask, batch["boundary_values"]
        )
        track_loss, _, _ = batched_trajectory_consistency(
            physical,
            batch["tracks"],
            batch["projection"],
            batch["times"],
            batch["bounds"],
            observation_mask=batch["mask"],
            num_substeps=args.trajectory_substeps,
            boundary_mode="clamp",
        )
        span = batch["bounds"][..., 1] - batch["bounds"][..., 0]
        loss = track_loss / span.square().mean().clamp_min(1e-12)
        if args.divergence_guidance_weight > 0.0:
            loss = loss + args.divergence_guidance_weight * batched_divergence_mse(
                physical, batch["bounds"], ~batch["solid_mask"]
            )
        return loss

    shape = (1, *tuple(int(value) for value in batch["flow"].shape[1:]))
    if tensor_codec is not None:
        shape = (1, tensor_codec.latent_dim)
    posterior_samples: list[Tensor] = []
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    for sample_index in range(args.num_samples):
        normalized = diffusion.ddim_sample(
            unet,
            shape,
            condition,
            sampling_steps=args.sampling_steps,
            eta=args.eta,
            guidance_scale=args.cfg_scale,
            unconditional_condition=None,
            guidance_loss=(
                observation_guidance
                if args.trajectory_guidance_strength > 0.0
                else None
            ),
            guidance_strength=args.trajectory_guidance_strength,
            projection=(hard_projection if tensor_codec is None and args.boundary_projection == "each-step" else None),
            generator=generator,
            device=device,
        )
        physical = apply_hard_boundary_conditions(
            denormalize_flow(decode(normalized), stats),
            boundary_mask,
            batch["boundary_values"],
        )
        posterior_samples.append(physical[0].detach().cpu())
        if not bool(torch.isfinite(physical).all()):
            raise RuntimeError("sampling produced a non-finite decoded velocity field")
        print(
            f"sample {sample_index + 1}/{args.num_samples} complete", flush=True
        )

    samples = torch.stack(posterior_samples, dim=0)
    posterior_mean = samples.mean(dim=0, keepdim=True)
    posterior_variance = samples.var(dim=0, unbiased=False, keepdim=True)
    reference = batch["flow"].detach().cpu()
    solid = batch["solid_mask"].detach().cpu()
    boundary = batch["boundary_mask"].detach().cpu()
    unknown_interior = (~solid) & (~boundary)
    metrics = _field_metrics(reference, posterior_mean, unknown_interior)
    uncertainty_correlation, uncertainty_correlation_defined = (
        _uncertainty_error_correlation(
            reference, posterior_mean, posterior_variance, unknown_interior
        )
    )
    with torch.no_grad():
        replay_loss, replay, validity = batched_trajectory_consistency(
            posterior_mean.to(device),
            batch["tracks"],
            batch["projection"],
            batch["times"],
            batch["bounds"],
            observation_mask=batch["mask"],
            num_substeps=args.trajectory_substeps,
            boundary_mode="clamp",
        )
        divergence = batched_divergence_mse(
            posterior_mean.to(device), batch["bounds"], ~batch["solid_mask"]
        )
        if args.num_probe_particles:
            probe_loss, probe_replay, probe_validity = (
                batched_trajectory_consistency(
                    posterior_mean.to(device),
                    batch["probe_tracks"],
                    batch["projection"],
                    batch["times"],
                    batch["bounds"],
                    observation_mask=batch["probe_mask"],
                    num_substeps=args.trajectory_substeps,
                    boundary_mode="clamp",
                )
            )
        else:
            probe_loss = None
            probe_replay = batch["tracks"][:, :, :0]
            probe_validity = validity[:, :0]
    metrics.update(
        {
            "observed_track_rmse_cells": float(replay_loss.sqrt().cpu()),
            "posterior_mean_divergence_mse": float(divergence.cpu()),
            "posterior_mean_variance": float(posterior_variance.mean()),
            "uncertainty_error_pearson_unknown_interior": uncertainty_correlation,
            "uncertainty_error_pearson_defined": uncertainty_correlation_defined,
            "all_replay_particles_valid": bool(validity.all().cpu()),
            "num_posterior_samples": args.num_samples,
            "num_observed_particles": num_particles,
            "num_probe_particles": args.num_probe_particles,
        }
    )
    if probe_loss is not None:
        metrics.update(
            {
                "probe_track_rmse_cells": float(probe_loss.sqrt().cpu()),
                "all_probe_particles_valid": bool(probe_validity.all().cpu()),
            }
        )
    metadata = {
        "method": f"{num_particles}-particle conditional 3D flow diffusion",
        "architecture": architecture,
        "model_config": checkpoint["model_config"],
        "tensor_artifact_sha256": checkpoint.get("tensor_artifact_sha256"),
        "boundary_projection": "final" if tensor_codec is not None else args.boundary_projection,
        "checkpoint": str(checkpoint_path),
        "case_id": item["case_id"],
        "split": args.split,
        "smoke_test_manifest": dataset.is_smoke_test,
        "scientific_result": scientific_result,
        "scientific_result_reasons": scientific_reasons,
        "evaluation_manifest_provenance": evaluation_provenance,
        "checkpoint_has_training_provenance": isinstance(
            checkpoint.get("training_provenance"), dict
        ),
        "trained_particles": trained_particles,
        "sampled_particles": num_particles,
        "probe_particles": args.num_probe_particles,
        "sampling_steps": args.sampling_steps,
        "eta": args.eta,
        "cfg_scale": args.cfg_scale,
        "trajectory_guidance_strength": args.trajectory_guidance_strength,
        "metrics": metrics,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "posterior_samples": samples.numpy().astype(np.float32),
        "posterior_mean": posterior_mean.numpy().astype(np.float32),
        "posterior_variance": posterior_variance.numpy().astype(np.float32),
        "reference_field": reference.numpy().astype(np.float32),
        "observed_tracks": batch["tracks"].detach().cpu().numpy().astype(np.float32),
        "observation_mask": batch["mask"].detach().cpu().numpy().astype(bool),
        "replay_tracks": replay.detach().cpu().numpy().astype(np.float32),
        "replay_validity": validity.detach().cpu().numpy().astype(bool),
        "projection_matrix": batch["projection"].detach().cpu().numpy().astype(np.float32),
        "observation_times": batch["times"].detach().cpu().numpy().astype(np.float32),
        "domain_bounds": batch["bounds"].detach().cpu().numpy().astype(np.float32),
        "particle_indices": batch["particle_indices"].detach().cpu().numpy().astype(np.int64),
        "metrics": np.asarray(json.dumps(metrics, sort_keys=True)),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if args.num_probe_particles:
        archive.update(
            {
                "probe_observed_tracks": batch["probe_tracks"].detach().cpu().numpy().astype(np.float32),
                "probe_observation_mask": batch["probe_mask"].detach().cpu().numpy().astype(bool),
                "probe_replay_tracks": probe_replay.detach().cpu().numpy().astype(np.float32),
                "probe_replay_validity": probe_validity.detach().cpu().numpy().astype(bool),
                "probe_particle_indices": batch["probe_particle_indices"].detach().cpu().numpy().astype(np.int64),
            }
        )
    np.savez_compressed(output, **archive)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        output = sample(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote posterior archive to {output}")


if __name__ == "__main__":
    main()
