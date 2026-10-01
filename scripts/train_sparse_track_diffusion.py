#!/usr/bin/env python
"""Train a two-particle conditional 3D flow diffusion model.

The split manifest is case based: complete Taichi-LBM3D fields are targets,
while exactly ``--num-particles`` projected trajectories are conditions.  The
default is the professor-requested two-particle setting.  Auxiliary replay and
physics losses are implemented but disabled by default for the initial DDPM
pretraining stage; enable them for low-noise fine-tuning after the denoiser is
stable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.diffusion import GaussianDiffusion
from flow_observation.models.trajectory_encoder import TrackSetEncoder
from flow_observation.models.unet3d import ConditionalUNet3D
from flow_observation.sparse_dataset import (
    FlowNormalizationStats,
    SparseFlowDataset,
    compute_train_flow_stats,
    denormalize_flow,
    load_sparse_flow_manifest,
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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Train conditional 3D diffusion from sparse projected particle tracks."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sparse_track_diffusion")
    )
    parser.add_argument("--num-particles", type=_positive_integer, default=2)
    parser.add_argument("--observations-per-flow", type=_positive_integer, default=1)
    parser.add_argument("--epochs", type=_positive_integer, default=100)
    parser.add_argument("--batch-size", type=_positive_integer, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=_positive_integer, default=1)
    parser.add_argument("--diffusion-steps", type=_positive_integer, default=1000)
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--base-channels", type=_positive_integer, default=16)
    parser.add_argument("--condition-dim", type=_positive_integer, default=128)
    parser.add_argument("--time-embedding-dim", type=_positive_integer, default=64)
    parser.add_argument("--dropout", type=_probability, default=0.0)
    parser.add_argument("--condition-dropout", type=_probability, default=0.1)
    parser.add_argument("--ema-decay", type=_probability, default=0.999)
    parser.add_argument(
        "--track-loss-weight",
        type=_nonnegative_float,
        default=0.0,
        help="Enable after DDPM pretraining; replay is expensive and low-noise only.",
    )
    parser.add_argument(
        "--divergence-loss-weight", type=_nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--boundary-loss-weight", type=_nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--auxiliary-max-t-fraction",
        type=_probability,
        default=0.2,
        help="Apply trajectory/physics losses only below this noise-time fraction.",
    )
    parser.add_argument("--trajectory-substeps", type=_positive_integer, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=_positive_integer, default=10)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a compatible checkpoint; --epochs is the final total epoch.",
    )
    return parser


def _cuda_architecture_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    architecture = f"sm_{major}{minor}"
    available = set(torch.cuda.get_arch_list())
    return architecture in available


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    supported = _cuda_architecture_supported()
    if requested == "cuda":
        if not supported:
            capability = (
                torch.cuda.get_device_capability() if torch.cuda.is_available() else None
            )
            raise RuntimeError(
                "CUDA was requested, but this PyTorch build does not contain a "
                f"kernel for the visible GPU capability {capability}. Install a "
                "PyTorch build supporting the GPU before full 32^3 training."
            )
        return torch.device("cuda")
    if supported:
        return torch.device("cuda")
    if torch.cuda.is_available():
        print(
            "Warning: CUDA is visible but unsupported by this PyTorch build; "
            "falling back to CPU. CPU is suitable only for smoke tests.",
            flush=True,
        )
    return torch.device("cpu")


def _seed_everything(seed: int, *, use_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def _tensor_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def _encode_condition(
    encoder: TrackSetEncoder,
    batch: dict[str, Any],
    *,
    dropout_probability: float = 0.0,
) -> Tensor:
    condition = encoder(
        batch["tracks"],
        batch["mask"],
        batch["projection"],
        batch["times"],
        batch["bounds"],
    )
    if dropout_probability > 0.0:
        dropped = torch.rand(
            (condition.shape[0], 1), device=condition.device
        ) < dropout_probability
        condition = torch.where(dropped, torch.zeros_like(condition), condition)
    return condition


def _masked_boundary_loss(
    prediction: Tensor,
    boundary_mask: Tensor,
    boundary_values: Tensor,
    stats: FlowNormalizationStats,
) -> Tensor:
    if (
        boundary_mask.ndim != 5
        or boundary_mask.shape[0] != prediction.shape[0]
        or boundary_mask.shape[1] != 1
        or boundary_mask.shape[2:] != prediction.shape[2:]
    ):
        raise ValueError("boundary_mask must have shape [B,1,D,H,W]")
    mask = boundary_mask.expand_as(prediction)
    scale = stats.std.to(device=prediction.device, dtype=prediction.dtype).reshape(
        1, 3, 1, 1, 1
    )
    squared = ((prediction - boundary_values) / scale).square()
    return squared[mask].mean()


def _auxiliary_losses(
    predicted_normalized: Tensor,
    timesteps: Tensor,
    batch: dict[str, Any],
    stats: FlowNormalizationStats,
    diffusion_steps: int,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    zero = predicted_normalized.sum() * 0.0
    result = {"track": zero, "divergence": zero, "boundary": zero}
    if not any(
        weight > 0.0
        for weight in (
            args.track_loss_weight,
            args.divergence_loss_weight,
            args.boundary_loss_weight,
        )
    ):
        return result
    maximum = max(0, int(args.auxiliary_max_t_fraction * diffusion_steps) - 1)
    selected = torch.nonzero(timesteps <= maximum, as_tuple=False).flatten()
    if selected.numel() == 0:
        return result

    normalized = predicted_normalized.index_select(0, selected)
    physical = denormalize_flow(normalized, stats)
    # Rank-4 [B,D,H,W] is ambiguous when B == 3 because the physics helper
    # also accepts a shared component-specific [3,D,H,W] mask.
    boundary_mask = batch["boundary_mask"].index_select(0, selected).unsqueeze(1)
    boundary_values = batch["boundary_values"].index_select(0, selected)
    if args.boundary_loss_weight > 0.0:
        result["boundary"] = _masked_boundary_loss(
            physical, boundary_mask, boundary_values, stats
        )
    constrained = apply_hard_boundary_conditions(
        physical, boundary_mask, boundary_values
    )
    if args.divergence_loss_weight > 0.0:
        fluid_mask = ~batch["solid_mask"].index_select(0, selected)
        result["divergence"] = batched_divergence_mse(
            constrained,
            batch["bounds"].index_select(0, selected),
            fluid_mask,
        )
    if args.track_loss_weight > 0.0:
        raw_track, _, _ = batched_trajectory_consistency(
            constrained,
            batch["tracks"].index_select(0, selected),
            batch["projection"].index_select(0, selected),
            batch["times"].index_select(0, selected),
            batch["bounds"].index_select(0, selected),
            observation_mask=batch["mask"].index_select(0, selected),
            num_substeps=args.trajectory_substeps,
            boundary_mode="clamp",
        )
        span = (
            batch["bounds"].index_select(0, selected)[..., 1]
            - batch["bounds"].index_select(0, selected)[..., 0]
        )
        result["track"] = raw_track / span.square().mean().clamp_min(1e-12)
    return result


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].lerp_(parameter, 1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer)


def _training_provenance(manifest_path: str | Path) -> dict[str, object]:
    """Return immutable split provenance for scientific/audit checks."""

    resolved = Path(manifest_path).expanduser().resolve()
    raw_manifest = resolved.read_bytes()
    manifest, splits = load_sparse_flow_manifest(resolved)

    def group_ids(split: str) -> list[str]:
        return sorted({case.flow_group_id for case in splits[split]})

    mode = str(manifest.get("mode", "scientific"))
    return {
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "manifest_mode": mode,
        "train_flow_group_ids": group_ids("train"),
        "validation_flow_group_ids": group_ids("validation"),
        "test_flow_group_ids": group_ids("test"),
        "smoke_test": mode == "smoke_test",
    }


def _capture_training_state(
    loader_generator: torch.Generator,
    scaler: torch.amp.GradScaler,
    *,
    use_cuda: bool = False,
) -> dict[str, object]:
    """Capture all stochastic state required for an exact continuation."""

    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if use_cuda else None,
        "dataloader_generator_state": loader_generator.get_state(),
        "amp_scaler_state": scaler.state_dict(),
    }


def _restore_training_state(
    state: object,
    loader_generator: torch.Generator,
    scaler: torch.amp.GradScaler,
    *,
    use_cuda: bool,
) -> bool:
    """Restore stochastic state, returning False for a legacy checkpoint."""

    if not isinstance(state, dict):
        return False
    required = {
        "python_random_state",
        "numpy_random_state",
        "torch_rng_state",
        "dataloader_generator_state",
        "amp_scaler_state",
    }
    if not required.issubset(state):
        return False
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    loader_generator.set_state(state["dataloader_generator_state"].cpu())
    scaler.load_state_dict(state["amp_scaler_state"])
    cuda_state = state.get("cuda_rng_state_all")
    if use_cuda and cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])
    return True


def _accumulation_window_size(
    batch_index: int, num_batches: int, gradient_accumulation: int
) -> int:
    """Number of micro-batches in the current optimizer-step window."""

    window_start = (batch_index // gradient_accumulation) * gradient_accumulation
    return min(gradient_accumulation, num_batches - window_start)


def _checkpoint_payload(
    *,
    epoch: int,
    best_validation_total: float,
    best_validation_diffusion: float,
    encoder: TrackSetEncoder,
    unet: ConditionalUNet3D,
    ema_encoder: TrackSetEncoder,
    ema_unet: ConditionalUNet3D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    diffusion: GaussianDiffusion,
    stats: FlowNormalizationStats,
    provenance: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "method": "two-particle conditional 3D flow diffusion",
        "epoch": epoch,
        "best_validation_total_loss": best_validation_total,
        # Retained for older format-version-1 readers.  New training runs use
        # best_validation_total_loss for checkpoint selection.
        "best_validation_diffusion_loss": best_validation_diffusion,
        "encoder_state": encoder.state_dict(),
        "unet_state": unet.state_dict(),
        "ema_encoder_state": ema_encoder.state_dict(),
        "ema_unet_state": ema_unet.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_state": _capture_training_state(
            loader_generator,
            scaler,
            use_cuda=next(unet.parameters()).is_cuda,
        ),
        "diffusion_state": diffusion.state_dict(),
        "normalization": stats.to_dict(),
        "model_config": {
            "condition_dim": args.condition_dim,
            "base_channels": args.base_channels,
            "channel_multipliers": [1, 2, 4],
            "time_embedding_dim": args.time_embedding_dim,
            "dropout": args.dropout,
            "diffusion_steps": args.diffusion_steps,
            "schedule": args.schedule,
        },
        "data_config": {
            "manifest": str(args.manifest.expanduser().resolve()),
            "num_particles": args.num_particles,
            "observations_per_flow": args.observations_per_flow,
            "split_unit": "independent flow case",
        },
        "training_provenance": provenance,
        "train_config": vars(args),
    }


def _fixed_validation_inputs(
    clean: Tensor,
    *,
    sample_offset: int,
    diffusion_steps: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create per-example validation timesteps/noise independent of global RNG."""

    timesteps: list[int] = []
    noises: list[Tensor] = []
    for local_index in range(clean.shape[0]):
        example_index = sample_offset + local_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) + 1_000_003 * example_index) % (2**63 - 1))
        timesteps.append(
            int(torch.randint(diffusion_steps, (1,), generator=generator).item())
        )
        noises.append(
            torch.randn(
                tuple(clean.shape[1:]),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        )
    timestep_tensor = torch.tensor(timesteps, dtype=torch.long, device=clean.device)
    noise_tensor = torch.stack(noises).to(device=clean.device, dtype=clean.dtype)
    return timestep_tensor, noise_tensor


def _run_validation(
    encoder: TrackSetEncoder,
    unet: ConditionalUNet3D,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    stats: FlowNormalizationStats,
    args: argparse.Namespace,
) -> dict[str, float]:
    encoder.eval()
    unet.eval()
    accumulated = {
        "total": 0.0,
        "diffusion": 0.0,
        "track": 0.0,
        "divergence": 0.0,
        "boundary": 0.0,
    }
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _tensor_batch_to_device(raw_batch, device)
            condition = _encode_condition(encoder, batch)
            batch_size = int(batch["flow"].shape[0])
            timesteps, noise = _fixed_validation_inputs(
                batch["flow"],
                sample_offset=count,
                diffusion_steps=args.diffusion_steps,
                seed=args.seed + 10_000_019,
            )
            result = diffusion.training_loss(
                unet,
                batch["flow"],
                condition,
                timesteps=timesteps,
                noise=noise,
            )
            auxiliary = _auxiliary_losses(
                result["predicted_x0"].float(),
                result["timesteps"],
                batch,
                stats,
                args.diffusion_steps,
                args,
            )
            total_loss = (
                result["loss"]
                + args.track_loss_weight * auxiliary["track"]
                + args.divergence_loss_weight * auxiliary["divergence"]
                + args.boundary_loss_weight * auxiliary["boundary"]
            )
            values = {"total": total_loss, "diffusion": result["loss"], **auxiliary}
            for name, value in values.items():
                accumulated[name] += float(value.detach().cpu()) * batch_size
            count += batch_size
    return {name: value / max(count, 1) for name, value in accumulated.items()}


def train(args: argparse.Namespace) -> Path:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0.0:
        raise ValueError("--gradient-clip must be finite and positive")
    device = _resolve_device(args.device)
    _seed_everything(args.seed, use_cuda=device.type == "cuda")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _training_provenance(args.manifest)

    stats = compute_train_flow_stats(args.manifest)
    train_dataset = SparseFlowDataset(
        args.manifest,
        "train",
        num_particles=args.num_particles,
        observations_per_flow=args.observations_per_flow,
        seed=args.seed,
        normalize=True,
        normalization_stats=stats,
    )
    validation_dataset = SparseFlowDataset(
        args.manifest,
        "validation",
        num_particles=args.num_particles,
        observations_per_flow=1,
        seed=args.seed + 1,
        normalize=True,
        normalization_stats=stats,
    )
    if train_dataset.is_smoke_test:
        print(
            "WARNING: manifest is explicitly marked smoke_test with overlapping "
            "flows. Outputs validate code only and are not scientific results.",
            flush=True,
        )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder = TrackSetEncoder(condition_dim=args.condition_dim).to(device)
    unet = ConditionalUNet3D(
        condition_dim=args.condition_dim,
        base_channels=args.base_channels,
        channel_multipliers=(1, 2, 4),
        time_embedding_dim=args.time_embedding_dim,
        dropout=args.dropout,
    ).to(device)
    diffusion = GaussianDiffusion(
        args.diffusion_steps, schedule=args.schedule, clip_x0=8.0
    ).to(device)
    ema_encoder = copy.deepcopy(encoder).eval().requires_grad_(False)
    ema_unet = copy.deepcopy(unet).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *unet.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    # Best is scoped to this invocation.  In particular, the first completed
    # resumed epoch must materialize best.pt in either a new or reused output
    # directory; it must never silently return a stale/missing artifact.
    best_validation = float("inf")
    best_validation_diffusion = float("inf")
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    resume_at_completed_target: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if not isinstance(resume, dict) or resume.get("format_version") != 1:
            raise ValueError("--resume is not a supported sparse diffusion checkpoint")
        saved_config = resume.get("model_config", {})
        expected_config = {
            "condition_dim": args.condition_dim,
            "base_channels": args.base_channels,
            "channel_multipliers": [1, 2, 4],
            "time_embedding_dim": args.time_embedding_dim,
            "dropout": args.dropout,
            "diffusion_steps": args.diffusion_steps,
            "schedule": args.schedule,
        }
        if saved_config != expected_config:
            raise ValueError(
                "--resume model configuration differs from the current arguments"
            )
        saved_particles = int(resume["data_config"]["num_particles"])
        if saved_particles != args.num_particles:
            raise ValueError(
                f"--resume was trained with {saved_particles} particles, not "
                f"the requested {args.num_particles}"
            )
        saved_provenance = resume.get("training_provenance")
        if isinstance(saved_provenance, dict):
            provenance_keys = (
                "manifest_sha256",
                "manifest_mode",
                "train_flow_group_ids",
                "validation_flow_group_ids",
                "test_flow_group_ids",
                "smoke_test",
            )
            if any(
                saved_provenance.get(key) != provenance.get(key)
                for key in provenance_keys
            ):
                raise ValueError(
                    "--resume training provenance differs from the current manifest"
                )
        saved_stats = FlowNormalizationStats.from_dict(resume["normalization"])
        if not torch.allclose(saved_stats.mean, stats.mean) or not torch.allclose(
            saved_stats.std, stats.std
        ):
            raise ValueError(
                "--resume normalization differs from the current training manifest"
            )
        encoder.load_state_dict(resume["encoder_state"])
        unet.load_state_dict(resume["unet_state"])
        ema_encoder.load_state_dict(resume["ema_encoder_state"])
        ema_unet.load_state_dict(resume["ema_unet_state"])
        diffusion.load_state_dict(resume["diffusion_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["weight_decay"] = args.weight_decay
        start_epoch = int(resume["epoch"]) + 1
        saved_history = resume.get("history", [])
        if not isinstance(saved_history, list):
            raise ValueError("--resume checkpoint history is invalid")
        history = list(saved_history)
        if start_epoch == args.epochs + 1:
            resume_at_completed_target = resume
            best_validation = float(
                resume.get("best_validation_total_loss", float("inf"))
            )
            best_validation_diffusion = float(
                resume.get("best_validation_diffusion_loss", float("inf"))
            )
        elif start_epoch > args.epochs + 1:
            raise ValueError(
                f"--resume starts at epoch {start_epoch}, beyond --epochs={args.epochs}"
            )
        restored_exact_state = _restore_training_state(
            resume.get("training_state"),
            loader_generator,
            scaler,
            use_cuda=device.type == "cuda",
        )
        if not restored_exact_state:
            print(
                "Warning: legacy checkpoint has no complete RNG/DataLoader/AMP "
                "state; weights can be resumed, but the continuation is not "
                "bitwise reproducible.",
                flush=True,
            )
        print(f"Resuming {resume_path} at epoch {start_epoch}", flush=True)
    optimizer.zero_grad(set_to_none=True)

    # A process may be interrupted after the final latest.pt was atomically
    # written but before best.pt/training_summary.json were materialized.  This
    # is already a completed target epoch, so finalize it without attempting an
    # impossible extra epoch.
    if resume_at_completed_target is not None:
        best_path = output_dir / "best.pt"
        if not best_path.is_file():
            torch.save(resume_at_completed_target, best_path)
        summary = {
            "status": "completed",
            "device": str(device),
            "best_validation_total_loss": best_validation,
            "best_validation_diffusion_loss": best_validation_diffusion,
            "epochs": args.epochs,
            "num_particles_per_condition": args.num_particles,
            "manifest_mode": train_dataset.manifest.get("mode", "scientific"),
            "scientific_result": not train_dataset.is_smoke_test,
            "training_provenance": provenance,
            "normalization": stats.to_dict(),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return best_path

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_dataset.set_epoch(epoch - 1)
        encoder.train()
        unet.train()
        accumulated = {"total": 0.0, "diffusion": 0.0, "track": 0.0, "divergence": 0.0, "boundary": 0.0}
        seen = 0
        for batch_index, raw_batch in enumerate(train_loader):
            batch = _tensor_batch_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                condition = _encode_condition(
                    encoder,
                    batch,
                    dropout_probability=args.condition_dropout,
                )
                diffusion_result = diffusion.training_loss(
                    unet, batch["flow"], condition
                )
                auxiliary = _auxiliary_losses(
                    diffusion_result["predicted_x0"].float(),
                    diffusion_result["timesteps"],
                    batch,
                    stats,
                    args.diffusion_steps,
                    args,
                )
                total_loss = (
                    diffusion_result["loss"]
                    + args.track_loss_weight * auxiliary["track"]
                    + args.divergence_loss_weight * auxiliary["divergence"]
                    + args.boundary_loss_weight * auxiliary["boundary"]
                )
                window_size = _accumulation_window_size(
                    batch_index, len(train_loader), args.gradient_accumulation
                )
                scaled_loss = total_loss / window_size
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [*encoder.parameters(), *unet.parameters()], args.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _update_ema(ema_encoder, encoder, args.ema_decay)
                _update_ema(ema_unet, unet, args.ema_decay)

            batch_size = int(batch["flow"].shape[0])
            seen += batch_size
            values = {
                "total": total_loss,
                "diffusion": diffusion_result["loss"],
                **auxiliary,
            }
            for name, value in values.items():
                accumulated[name] += float(value.detach().cpu()) * batch_size

        validation = _run_validation(
            ema_encoder,
            ema_unet,
            diffusion,
            validation_loader,
            device,
            stats,
            args,
        )
        if not math.isfinite(validation["total"]):
            raise RuntimeError("validation total loss is not finite")
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_total": accumulated["total"] / max(seen, 1),
            "train_diffusion": accumulated["diffusion"] / max(seen, 1),
            "train_track": accumulated["track"] / max(seen, 1),
            "train_divergence": accumulated["divergence"] / max(seen, 1),
            "train_boundary": accumulated["boundary"] / max(seen, 1),
            "validation_total": validation["total"],
            "validation_diffusion": validation["diffusion"],
            "validation_track": validation["track"],
            "validation_divergence": validation["divergence"],
            "validation_boundary": validation["boundary"],
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        improves_best = validation["total"] < best_validation
        payload = _checkpoint_payload(
            epoch=epoch,
            best_validation_total=min(best_validation, validation["total"]),
            best_validation_diffusion=(
                validation["diffusion"]
                if improves_best
                else best_validation_diffusion
            ),
            encoder=encoder,
            unet=unet,
            ema_encoder=ema_encoder,
            ema_unet=ema_unet,
            optimizer=optimizer,
            scaler=scaler,
            loader_generator=loader_generator,
            diffusion=diffusion,
            stats=stats,
            provenance=provenance,
            args=args,
        )
        payload["history"] = history
        torch.save(payload, output_dir / "latest.pt")
        if improves_best:
            best_validation = validation["total"]
            best_validation_diffusion = validation["diffusion"]
            payload["best_validation_total_loss"] = best_validation
            payload["best_validation_diffusion_loss"] = best_validation_diffusion
            torch.save(payload, output_dir / "best.pt")
        if epoch % args.save_every == 0:
            torch.save(payload, output_dir / f"epoch_{epoch:04d}.pt")

    summary = {
        "status": "completed",
        "device": str(device),
        "best_validation_total_loss": best_validation,
        "best_validation_diffusion_loss": best_validation_diffusion,
        "epochs": args.epochs,
        "num_particles_per_condition": args.num_particles,
        "manifest_mode": train_dataset.manifest.get("mode", "scientific"),
        "scientific_result": not train_dataset.is_smoke_test,
        "training_provenance": provenance,
        "normalization": stats.to_dict(),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir / "best.pt"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        checkpoint = train(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote best checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
