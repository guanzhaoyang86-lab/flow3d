#!/usr/bin/env python
"""Recover a steady Taichi-LBM3D velocity field with a coarse-grid baseline.

The optimizer consumes synchronized projected tracks, camera matrices, masks,
times, domain bounds, and known cavity boundary conditions. Reference 3D
trajectories and precomputed hidden depths are deliberately not loaded. The
reference velocity array is used only after optimization for post-hoc metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.advection import advect_particles
from flow_observation.multiview import triangulate_initial_positions
from flow_observation.recovery import (
    CoarseVelocityField,
    divergence_mse,
    masked_trajectory_mse,
    project_trajectory_views,
    spatial_smoothness_mse,
)


_REQUIRED_KEYS = {
    "domain_bounds",
    "flow_field",
    "metadata",
    "observation_mask",
    "observation_times",
    "projection_matrix",
    "solid_mask",
    "trajectories_2d",
    "velocity_times",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParticleSplit:
    train: Tensor
    validation: Tensor
    test: Tensor


@dataclass
class OptimizationResult:
    recovered_coarse: Tensor
    train_iterations: list[int]
    train_total: list[float]
    train_track_mse: list[float]
    train_smoothness: list[float]
    train_divergence: list[float]
    learning_rates: list[float]
    validation_iterations: list[int]
    validation_track_mse: list[float]
    best_iteration: int
    elapsed_seconds: float


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a steady 3D velocity field from synchronized projected "
            "tracks with a regularized coarse-grid baseline."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-index", type=_nonnegative_integer, default=0)
    parser.add_argument(
        "--views",
        default="all",
        help="Comma-separated projection names/indices, or 'all' (default).",
    )
    parser.add_argument("--coarse-grid-size", type=_positive_integer, default=8)
    parser.add_argument("--max-iterations", type=_positive_integer, default=150)
    parser.add_argument("--learning-rate", type=_positive_float, default=3e-3)
    parser.add_argument(
        "--max-speed",
        type=_positive_float,
        default=None,
        help="Component bound; defaults to 1.5 times the known lid speed.",
    )
    parser.add_argument("--smoothness-weight", type=_nonnegative_float, default=100.0)
    parser.add_argument("--divergence-weight", type=_nonnegative_float, default=1e4)
    parser.add_argument("--gradient-clip", type=_positive_float, default=10.0)
    parser.add_argument("--validation-fraction", type=_fraction, default=0.1875)
    parser.add_argument("--test-fraction", type=_fraction, default=0.1875)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-every", type=_positive_integer, default=5)
    parser.add_argument(
        "--patience",
        type=_positive_integer,
        default=30,
        help="Stop after this many iterations without validation improvement.",
    )
    parser.add_argument("--minimum-improvement", type=_nonnegative_float, default=1e-7)
    parser.add_argument(
        "--training-substeps",
        type=_positive_integer,
        default=1,
        help="RK4 substeps per interval during optimization (fast baseline default: 1).",
    )
    parser.add_argument(
        "--evaluation-substeps",
        type=_positive_integer,
        default=None,
        help="Final replay substeps; defaults to the data-generation setting.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--known-lid-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard-code the known moving-lid velocity (default: enabled).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lbm3d_cavity_baseline_recovery.npz"),
    )
    parser.add_argument(
        "--convergence-plot",
        type=Path,
        default=Path(
            "outputs/lbm3d_cavity_baseline_visualizations/optimization_convergence.png"
        ),
    )
    return parser


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_KEYS.difference(archive.files))
        if missing:
            raise ValueError(
                f"archive is missing required key(s): {', '.join(missing)}"
            )
        # Do not load trajectories_3d or initial_hidden_depth. They are not
        # optimization inputs for this synchronized multi-view baseline.
        arrays = {key: np.array(archive[key], copy=True) for key in _REQUIRED_KEYS}
        if "particle_counts" in archive.files:
            arrays["particle_counts"] = np.array(archive["particle_counts"], copy=True)

    metadata_array = arrays.pop("metadata")
    if metadata_array.shape != ():
        raise ValueError("metadata must be a scalar JSON string")
    try:
        metadata = json.loads(str(metadata_array.item()))
    except json.JSONDecodeError as error:
        raise ValueError("metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON must decode to an object")
    _validate_dataset(arrays)
    return arrays, metadata


def _validate_dataset(data: dict[str, np.ndarray]) -> None:
    field = data["flow_field"]
    observations = data["trajectories_2d"]
    projection = data["projection_matrix"]
    mask = data["observation_mask"]
    if field.ndim != 5 or field.shape[0] != 1 or field.shape[1] != 3:
        raise ValueError("flow_field must be one steady snapshot [1,3,D,H,W]")
    if min(field.shape[-3:]) < 3 or not np.issubdtype(field.dtype, np.floating):
        raise ValueError(
            "flow_field spatial dimensions must be at least 3 and floating"
        )
    if observations.ndim != 5 or observations.shape[-1] != 2:
        raise ValueError("trajectories_2d must have shape [S,V,N,T,2]")
    if tuple(projection.shape) != (observations.shape[1], 2, 3):
        raise ValueError("projection_matrix must have shape [V,2,3]")
    if tuple(mask.shape) != tuple(observations.shape[:-1]):
        raise ValueError(
            "observation_mask must match trajectories_2d without coordinate"
        )
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
            raise ValueError("observation_mask must be boolean or finite numeric")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("numeric observation_mask values must be exactly 0 or 1")
    if data["observation_times"].shape != (observations.shape[3],):
        raise ValueError("observation_times must match the trajectory time axis")
    if data["velocity_times"].shape != (1,):
        raise ValueError("velocity_times must contain the steady snapshot time")
    if data["domain_bounds"].shape != (3, 2):
        raise ValueError("domain_bounds must have shape [3,2]")
    if tuple(data["solid_mask"].shape) != tuple(field.shape[-3:]):
        raise ValueError("solid_mask must match flow_field spatial dimensions")
    numeric = (
        field,
        observations,
        projection,
        data["observation_times"],
        data["velocity_times"],
        data["domain_bounds"],
    )
    if any(not np.isfinite(array).all() for array in numeric):
        raise ValueError("numeric dataset arrays must contain only finite values")


def _resolve_views(
    specification: str, metadata: dict[str, Any], count: int
) -> list[int]:
    names_value = metadata.get("projection_names", [])
    names = (
        [str(value).lower() for value in names_value]
        if isinstance(names_value, list)
        else []
    )
    if specification.strip().lower() == "all":
        return list(range(count))
    indices: list[int] = []
    for token in specification.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in names:
            index = names.index(value)
        else:
            try:
                index = int(value)
            except ValueError as error:
                raise ValueError(f"unknown projection view {value!r}") from error
        if index < 0 or index >= count:
            raise ValueError(f"projection view {index} is outside [0,{count - 1}]")
        if index in indices:
            raise ValueError("projection views must not be repeated")
        indices.append(index)
    if not indices:
        raise ValueError("--views must select at least one projection")
    return indices


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"}:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            probe = torch.ones(1, device="cuda")
            _ = (probe + 1.0).sum().item()
            return torch.device("cuda")
        except (RuntimeError, AssertionError) as error:
            if requested == "cuda":
                raise RuntimeError(f"CUDA device check failed: {error}") from error
            print(f"CUDA unavailable for this PyTorch build; using CPU ({error})")
    return torch.device("cpu")


def _face_mask(shape: tuple[int, int, int], face: str, device: torch.device) -> Tensor:
    mapping = {"x": 2, "y": 1, "z": 0}
    try:
        axis_name, side = face.lower().split("=", maxsplit=1)
        axis = mapping[axis_name]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported lid face {face!r}") from error
    if side not in {"min", "max"}:
        raise ValueError(f"unsupported lid face {face!r}")
    result = torch.zeros(shape, dtype=torch.bool, device=device)
    selection = [slice(None), slice(None), slice(None)]
    selection[axis] = 0 if side == "min" else shape[axis] - 1
    result[tuple(selection)] = True
    return result


def _boundary_conditions(
    shape: tuple[int, int, int],
    metadata: dict[str, Any],
    device: torch.device,
    *,
    known_lid: bool,
    solid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    lid_face = str(metadata.get("lid_face", "x=max"))
    lid_velocity = torch.as_tensor(
        metadata.get("lid_velocity_xyz", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device=device,
    )
    if lid_velocity.shape != (3,) or not bool(torch.isfinite(lid_velocity).all()):
        raise ValueError("metadata lid_velocity_xyz must contain three finite values")

    if solid_mask is None:
        solid = torch.zeros(shape, dtype=torch.bool, device=device)
        faces = ("x=min", "x=max", "y=min", "y=max", "z=min", "z=max")
        for face in faces:
            if face != lid_face:
                solid |= _face_mask(shape, face, device)
    else:
        solid = torch.as_tensor(solid_mask, dtype=torch.bool, device=device)
        if tuple(solid.shape) != shape:
            raise ValueError("solid_mask must match the requested boundary grid")

    lid = _face_mask(shape, lid_face, device) & ~solid
    fixed = solid.clone()
    values = torch.zeros((3, *shape), dtype=torch.float32, device=device)
    if known_lid:
        fixed |= lid
        values[:, lid] = lid_velocity[:, None]
    return fixed, values, ~solid


def _projected_motion_scores(observations: Tensor, mask: Tensor) -> Tensor:
    pair_mask = mask[..., 1:] & mask[..., :-1]
    displacement = torch.linalg.vector_norm(
        observations[..., 1:, :] - observations[..., :-1, :], dim=-1
    )
    weighted = torch.where(pair_mask, displacement, torch.zeros_like(displacement))
    counts = pair_mask.sum(dim=(0, 1, 3)).clamp_min(1)
    return weighted.sum(dim=(0, 1, 3)) / counts


def _stratified_particle_split(
    motion_scores: Tensor,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> ParticleSplit:
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1")
    count = motion_scores.numel()
    if count < 12:
        raise ValueError("at least 12 particles are required for train/validation/test")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    ordered = torch.argsort(motion_scores.detach().cpu())
    groups = torch.tensor_split(ordered, min(4, count))
    train_parts: list[Tensor] = []
    validation_parts: list[Tensor] = []
    test_parts: list[Tensor] = []
    for group in groups:
        shuffled = group[torch.randperm(group.numel(), generator=generator)]
        validation_count = max(1, int(round(group.numel() * validation_fraction)))
        test_count = max(1, int(round(group.numel() * test_fraction)))
        if validation_count + test_count >= group.numel():
            raise ValueError("split fractions leave no training particle in a stratum")
        validation_parts.append(shuffled[:validation_count])
        test_parts.append(shuffled[validation_count : validation_count + test_count])
        train_parts.append(shuffled[validation_count + test_count :])
    return ParticleSplit(
        train=torch.sort(torch.cat(train_parts)).values,
        validation=torch.sort(torch.cat(validation_parts)).values,
        test=torch.sort(torch.cat(test_parts)).values,
    )


def _trajectory_mask(mask: Tensor) -> Tensor:
    result = mask.clone()
    result[..., 0] = False
    if not bool(result.any()):
        raise ValueError("observations must contain at least one point after time zero")
    return result


def _predict(
    velocity_field: Tensor,
    initial_positions: Tensor,
    observation_times: Tensor,
    projection_matrices: Tensor,
    domain_bounds: Tensor,
    velocity_times: Tensor,
    *,
    integrator: str,
    substeps: int,
    boundary_mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    trajectories, validity = advect_particles(
        velocity_field,
        initial_positions,
        observation_times,
        integrator=integrator,
        num_substeps=substeps,
        domain_bounds=domain_bounds,
        boundary_mode=boundary_mode,
        velocity_times=velocity_times,
        return_validity=True,
    )
    projected = project_trajectory_views(trajectories, projection_matrices)
    return trajectories, projected, validity


def _subset(values: Tensor, indices: Tensor) -> Tensor:
    return values.index_select(2, indices.to(device=values.device))


def _optimize(
    model: CoarseVelocityField,
    observations: Tensor,
    observation_mask: Tensor,
    initial_positions: Tensor,
    projection_matrices: Tensor,
    observation_times: Tensor,
    velocity_times: Tensor,
    domain_bounds: Tensor,
    fluid_mask: Tensor,
    split: ParticleSplit,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> OptimizationResult:
    train_indices = split.train.to(device=observations.device)
    validation_indices = split.validation.to(device=observations.device)
    train_observations = _subset(observations, train_indices)
    train_mask = _subset(observation_mask, train_indices)
    train_initial = initial_positions.index_select(1, train_indices)
    validation_observations = _subset(observations, validation_indices)
    validation_mask = _subset(observation_mask, validation_indices)
    validation_initial = initial_positions.index_select(1, validation_indices)

    integrator = str(metadata.get("integrator", "rk4"))
    boundary_mode = str(metadata.get("boundary_mode", "terminate"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iterations, eta_min=args.learning_rate * 0.01
    )

    train_iterations: list[int] = []
    train_total: list[float] = []
    train_track: list[float] = []
    train_smoothness: list[float] = []
    train_divergence: list[float] = []
    learning_rates: list[float] = []
    validation_iterations: list[int] = []
    validation_track: list[float] = []
    best_validation = math.inf
    best_iteration = 0
    best_raw = model.raw_velocity.detach().clone()
    start = time.perf_counter()

    for iteration in range(args.max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate = model()
        _, projected, validity = _predict(
            candidate,
            train_initial,
            observation_times,
            projection_matrices,
            domain_bounds,
            velocity_times,
            integrator=integrator,
            substeps=args.training_substeps,
            boundary_mode=boundary_mode,
        )
        if not bool(validity.all().detach()):
            raise RuntimeError(
                "a training particle became invalid; lower --learning-rate or "
                "--max-speed instead of masking invalid predictions"
            )
        track_loss = masked_trajectory_mse(projected, train_observations, train_mask)
        smoothness = spatial_smoothness_mse(candidate, fluid_mask)
        divergence = divergence_mse(candidate, domain_bounds, fluid_mask)
        total = (
            track_loss
            + args.smoothness_weight * smoothness
            + args.divergence_weight * divergence
        )
        if not bool(torch.isfinite(total).detach()):
            raise RuntimeError("optimization loss became non-finite")

        train_iterations.append(iteration)
        train_total.append(float(total.detach().cpu()))
        train_track.append(float(track_loss.detach().cpu()))
        train_smoothness.append(float(smoothness.detach().cpu()))
        train_divergence.append(float(divergence.detach().cpu()))
        learning_rates.append(float(optimizer.param_groups[0]["lr"]))

        should_validate = (
            iteration % args.validation_every == 0 or iteration == args.max_iterations
        )
        if should_validate:
            with torch.no_grad():
                _, validation_projected, validation_validity = _predict(
                    candidate,
                    validation_initial,
                    observation_times,
                    projection_matrices,
                    domain_bounds,
                    velocity_times,
                    integrator=integrator,
                    substeps=args.training_substeps,
                    boundary_mode=boundary_mode,
                )
                if not bool(validation_validity.all()):
                    raise RuntimeError(
                        "a validation particle became invalid; lower the speed bound"
                    )
                validation_loss = masked_trajectory_mse(
                    validation_projected,
                    validation_observations,
                    validation_mask,
                )
            validation_value = float(validation_loss.cpu())
            validation_iterations.append(iteration)
            validation_track.append(validation_value)
            if validation_value < best_validation - args.minimum_improvement:
                best_validation = validation_value
                best_iteration = iteration
                best_raw = model.raw_velocity.detach().clone()
            print(
                f"iteration {iteration:4d} | "
                f"train RMSE {math.sqrt(train_track[-1]):.6f} | "
                f"validation RMSE {math.sqrt(validation_value):.6f} | "
                f"total {train_total[-1]:.6e}"
            )
            if iteration - best_iteration >= args.patience:
                print(
                    f"Early stopping at iteration {iteration}; "
                    f"best validation was iteration {best_iteration}."
                )
                break

        if iteration == args.max_iterations:
            break
        total.backward()
        if model.raw_velocity.grad is None or not bool(
            torch.isfinite(model.raw_velocity.grad).all().detach()
        ):
            raise RuntimeError("velocity gradient is missing or non-finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        model.raw_velocity.copy_(best_raw)
        recovered = model().detach().clone()
    return OptimizationResult(
        recovered_coarse=recovered,
        train_iterations=train_iterations,
        train_total=train_total,
        train_track_mse=train_track,
        train_smoothness=train_smoothness,
        train_divergence=train_divergence,
        learning_rates=learning_rates,
        validation_iterations=validation_iterations,
        validation_track_mse=validation_track,
        best_iteration=best_iteration,
        elapsed_seconds=time.perf_counter() - start,
    )


def _upsample_with_boundaries(
    coarse_field: Tensor,
    dense_shape: tuple[int, int, int],
    fixed_mask: Tensor,
    fixed_values: Tensor,
) -> Tensor:
    dense = F.interpolate(
        coarse_field,
        size=dense_shape,
        mode="trilinear",
        align_corners=True,
    )
    return torch.where(fixed_mask[None, None], fixed_values[None], dense)


def _track_rmse(
    predicted: Tensor, observed: Tensor, mask: Tensor, indices: Tensor
) -> float:
    value = masked_trajectory_mse(
        _subset(predicted, indices),
        _subset(observed, indices),
        _subset(mask, indices),
    )
    return math.sqrt(float(value.detach().cpu()))


def _field_metrics(
    reference: Tensor, candidate: Tensor, fluid_mask: Tensor
) -> dict[str, Any]:
    mask = fluid_mask[None, None].expand_as(reference)
    reference_values = reference[mask]
    candidate_values = candidate[mask]
    difference = candidate_values - reference_values
    denominator = torch.linalg.vector_norm(reference_values).clamp_min(1e-12)
    global_relative_l2 = torch.linalg.vector_norm(difference) / denominator
    cosine_denominator = (
        torch.linalg.vector_norm(reference_values)
        * torch.linalg.vector_norm(candidate_values)
    ).clamp_min(1e-12)
    global_cosine = torch.dot(reference_values, candidate_values) / cosine_denominator

    component_relative: list[float] = []
    component_cosine: list[float] = []
    reference_rms: list[float] = []
    candidate_rms: list[float] = []
    component_mask = fluid_mask[None].expand(reference.shape[0], *fluid_mask.shape)
    for component in range(3):
        ref = reference[:, component][component_mask]
        cand = candidate[:, component][component_mask]
        component_relative.append(
            float(
                (
                    torch.linalg.vector_norm(cand - ref)
                    / torch.linalg.vector_norm(ref).clamp_min(1e-12)
                ).cpu()
            )
        )
        cosine = torch.dot(ref, cand) / (
            torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cand)
        ).clamp_min(1e-12)
        component_cosine.append(float(cosine.cpu()))
        reference_rms.append(float(torch.sqrt(ref.square().mean()).cpu()))
        candidate_rms.append(float(torch.sqrt(cand.square().mean()).cpu()))
    return {
        "field_relative_l2_fluid": float(global_relative_l2.cpu()),
        "field_cosine_fluid": float(global_cosine.cpu()),
        "component_relative_l2_xyz": component_relative,
        "component_cosine_xyz": component_cosine,
        "reference_component_rms_xyz": reference_rms,
        "recovered_component_rms_xyz": candidate_rms,
    }


def _save_convergence(
    result: OptimizationResult,
    path: Path,
    *,
    smoothness_weight: float,
    divergence_weight: float,
) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.asarray(result.train_iterations)
    train_rmse = np.sqrt(np.asarray(result.train_track_mse))
    validation_rmse = np.sqrt(np.asarray(result.validation_track_mse))
    total = np.asarray(result.train_total)
    track = np.asarray(result.train_track_mse)
    smooth = smoothness_weight * np.asarray(result.train_smoothness)
    divergence = divergence_weight * np.asarray(result.train_divergence)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(iterations, train_rmse, label="train")
    axes[0].plot(
        result.validation_iterations,
        validation_rmse,
        "o-",
        markersize=3,
        label="validation",
    )
    axes[0].axvline(
        result.best_iteration,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"selected={result.best_iteration}",
    )
    axes[0].set(
        xlabel="iteration",
        ylabel="projected-coordinate RMSE (lattice cells)",
        title="Trajectory fit",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    epsilon = np.finfo(np.float64).tiny
    axes[1].semilogy(iterations, np.maximum(total, epsilon), label="total")
    axes[1].semilogy(iterations, np.maximum(track, epsilon), label="track")
    axes[1].semilogy(
        iterations,
        np.maximum(smooth, epsilon),
        label=f"{smoothness_weight:g} x smoothness",
    )
    axes[1].semilogy(
        iterations,
        np.maximum(divergence, epsilon),
        label=f"{divergence_weight:g} x divergence",
    )
    axes[1].set(xlabel="iteration", ylabel="loss", title="Optimization terms")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Taichi-LBM3D coarse-grid recovery baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_output(
    path: Path,
    result: OptimizationResult,
    recovered_dense: Tensor,
    projected_dense: Tensor,
    trajectories_dense: Tensor,
    validity_dense: Tensor,
    initial_positions: Tensor,
    selected_view_indices: Tensor,
    split: ParticleSplit,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        recovered_field=recovered_dense.detach().cpu().numpy().astype(np.float32),
        coarse_field=result.recovered_coarse.detach().cpu().numpy().astype(np.float32),
        predicted_trajectories_3d=trajectories_dense.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        predicted_trajectories_2d=projected_dense.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        prediction_validity=validity_dense.detach().cpu().numpy().astype(bool),
        triangulated_initial_positions=initial_positions.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        selected_view_indices=selected_view_indices.detach()
        .cpu()
        .numpy()
        .astype(np.int64),
        train_indices=split.train.cpu().numpy().astype(np.int64),
        validation_indices=split.validation.cpu().numpy().astype(np.int64),
        test_indices=split.test.cpu().numpy().astype(np.int64),
        iterations=np.asarray(result.train_iterations, dtype=np.int64),
        total_loss=np.asarray(result.train_total, dtype=np.float64),
        track_mse=np.asarray(result.train_track_mse, dtype=np.float64),
        smoothness_loss=np.asarray(result.train_smoothness, dtype=np.float64),
        divergence_loss=np.asarray(result.train_divergence, dtype=np.float64),
        learning_rate=np.asarray(result.learning_rates, dtype=np.float64),
        validation_iterations=np.asarray(result.validation_iterations, dtype=np.int64),
        validation_track_mse=np.asarray(result.validation_track_mse, dtype=np.float64),
        metrics=np.asarray(json.dumps(metrics, sort_keys=True)),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def recover(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    source_path = args.input.expanduser().resolve()
    source_sha256 = _sha256(source_path)
    data, source_metadata = _load_dataset(source_path)
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    observations_all = torch.as_tensor(
        data["trajectories_2d"], dtype=torch.float32, device=device
    )
    projection_all = torch.as_tensor(
        data["projection_matrix"], dtype=torch.float32, device=device
    )
    masks_all = torch.as_tensor(
        data["observation_mask"], dtype=torch.bool, device=device
    )
    if args.sample_index >= observations_all.shape[0]:
        raise ValueError(
            f"sample index {args.sample_index} is outside "
            f"[0,{observations_all.shape[0] - 1}]"
        )
    count = observations_all.shape[2]
    if "particle_counts" in data:
        count = int(data["particle_counts"][args.sample_index])
    if count < 12 or count > observations_all.shape[2]:
        raise ValueError("particle count must be within stored bounds and at least 12")

    views = _resolve_views(args.views, source_metadata, observations_all.shape[1])
    view_indices = torch.tensor(views, dtype=torch.long, device=device)
    observations = observations_all[
        args.sample_index : args.sample_index + 1, :, :count
    ].index_select(1, view_indices)
    projection = projection_all.index_select(0, view_indices)
    mask = masks_all[args.sample_index : args.sample_index + 1, :, :count].index_select(
        1, view_indices
    )
    if not bool(mask[..., 0].all()):
        raise ValueError(
            "every selected view must observe time zero for multi-view triangulation"
        )
    initial_positions = triangulate_initial_positions(observations, projection)
    mask = _trajectory_mask(mask)

    motion_scores = _projected_motion_scores(observations, mask)
    split = _stratified_particle_split(
        motion_scores,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
    )
    observation_times = torch.as_tensor(
        data["observation_times"], dtype=torch.float32, device=device
    )
    velocity_times = torch.as_tensor(
        data["velocity_times"], dtype=torch.float32, device=device
    )
    domain_bounds = torch.as_tensor(
        data["domain_bounds"], dtype=torch.float32, device=device
    )

    coarse_size = args.coarse_grid_size
    if coarse_size < 3:
        raise ValueError("--coarse-grid-size must be at least 3")
    coarse_shape = (coarse_size, coarse_size, coarse_size)
    coarse_fixed, coarse_values, coarse_fluid = _boundary_conditions(
        coarse_shape,
        source_metadata,
        device,
        known_lid=args.known_lid_boundary,
    )
    lid_velocity = np.asarray(
        source_metadata.get("lid_velocity_xyz", [0.0, 0.0, 0.0]), dtype=float
    )
    lid_speed = float(np.linalg.norm(lid_velocity))
    max_speed = args.max_speed
    if max_speed is None:
        max_speed = 1.5 * lid_speed if lid_speed > 0.0 else 0.1
    model = CoarseVelocityField(
        coarse_shape,
        coarse_shape,
        max_speed=max_speed,
        fixed_mask=coarse_fixed,
        fixed_values=coarse_values,
        device=device,
    )

    result = _optimize(
        model,
        observations,
        mask,
        initial_positions,
        projection,
        observation_times,
        velocity_times,
        domain_bounds,
        coarse_fluid,
        split,
        source_metadata,
        args,
    )

    dense_shape = tuple(int(value) for value in data["flow_field"].shape[-3:])
    dense_solid = torch.as_tensor(data["solid_mask"], dtype=torch.bool, device=device)
    dense_fixed, dense_values, dense_fluid = _boundary_conditions(
        dense_shape,
        source_metadata,
        device,
        known_lid=args.known_lid_boundary,
        solid_mask=dense_solid,
    )
    recovered_dense = _upsample_with_boundaries(
        result.recovered_coarse,
        dense_shape,
        dense_fixed,
        dense_values,
    )
    evaluation_substeps = args.evaluation_substeps
    if evaluation_substeps is None:
        evaluation_substeps = int(source_metadata.get("integration_substeps", 1))
    integrator = str(source_metadata.get("integrator", "rk4"))
    boundary_mode = str(source_metadata.get("boundary_mode", "terminate"))
    with torch.no_grad():
        trajectories_dense, projected_dense, validity_dense = _predict(
            recovered_dense,
            initial_positions,
            observation_times,
            projection,
            domain_bounds,
            velocity_times,
            integrator=integrator,
            substeps=evaluation_substeps,
            boundary_mode=boundary_mode,
        )
        if not bool(validity_dense.all()):
            raise RuntimeError(
                "selected recovered field produces invalid final trajectories"
            )

        zero_field = torch.zeros_like(recovered_dense)
        _, zero_projected, zero_validity = _predict(
            zero_field,
            initial_positions,
            observation_times,
            projection,
            domain_bounds,
            velocity_times,
            integrator=integrator,
            substeps=evaluation_substeps,
            boundary_mode=boundary_mode,
        )
        if not bool(zero_validity.all()):
            raise RuntimeError(
                "zero-field diagnostic unexpectedly produced invalid tracks"
            )
        boundary_only_coarse = torch.where(
            coarse_fixed[None, None],
            coarse_values[None],
            torch.zeros_like(result.recovered_coarse),
        )
        boundary_only = _upsample_with_boundaries(
            boundary_only_coarse,
            dense_shape,
            dense_fixed,
            dense_values,
        )
        _, boundary_projected, boundary_validity = _predict(
            boundary_only,
            initial_positions,
            observation_times,
            projection,
            domain_bounds,
            velocity_times,
            integrator=integrator,
            substeps=evaluation_substeps,
            boundary_mode=boundary_mode,
        )
        if not bool(boundary_validity.all()):
            raise RuntimeError("boundary-only diagnostic produced invalid tracks")

    train_rmse = _track_rmse(projected_dense, observations, mask, split.train)
    validation_rmse = _track_rmse(projected_dense, observations, mask, split.validation)
    test_rmse = _track_rmse(projected_dense, observations, mask, split.test)
    zero_test_rmse = _track_rmse(zero_projected, observations, mask, split.test)
    boundary_test_rmse = _track_rmse(boundary_projected, observations, mask, split.test)

    reference = torch.as_tensor(data["flow_field"], dtype=torch.float32, device=device)
    unknown_interior = dense_fluid & ~dense_fixed
    interior_metrics = _field_metrics(reference, recovered_dense, unknown_interior)
    inclusive_metrics = _field_metrics(reference, recovered_dense, dense_fluid)
    reference_energy_fluid = reference[:, :, dense_fluid].square().sum()
    reference_energy_prescribed = (
        reference[:, :, dense_fluid & dense_fixed].square().sum()
    )
    metrics = {
        "field_relative_l2_unknown_interior": interior_metrics[
            "field_relative_l2_fluid"
        ],
        "field_cosine_unknown_interior": interior_metrics["field_cosine_fluid"],
        "component_relative_l2_xyz_unknown_interior": interior_metrics[
            "component_relative_l2_xyz"
        ],
        "component_cosine_xyz_unknown_interior": interior_metrics[
            "component_cosine_xyz"
        ],
        "reference_component_rms_xyz_unknown_interior": interior_metrics[
            "reference_component_rms_xyz"
        ],
        "recovered_component_rms_xyz_unknown_interior": interior_metrics[
            "recovered_component_rms_xyz"
        ],
        "field_relative_l2_including_prescribed_boundary": inclusive_metrics[
            "field_relative_l2_fluid"
        ],
        "field_cosine_including_prescribed_boundary": inclusive_metrics[
            "field_cosine_fluid"
        ],
        "prescribed_boundary_reference_energy_fraction": float(
            (reference_energy_prescribed / reference_energy_fluid.clamp_min(1e-12))
            .detach()
            .cpu()
        ),
        "train_track_rmse_cells": train_rmse,
        "validation_track_rmse_cells": validation_rmse,
        "test_track_rmse_cells": test_rmse,
        "zero_field_test_rmse_cells": zero_test_rmse,
        "boundary_only_test_rmse_cells": boundary_test_rmse,
        "test_improvement_fraction_vs_zero": 1.0
        - test_rmse / max(zero_test_rmse, 1e-12),
        "test_improvement_fraction_vs_boundary_only": 1.0
        - test_rmse / max(boundary_test_rmse, 1e-12),
        "recovered_divergence_mse": float(
            divergence_mse(recovered_dense, domain_bounds, dense_fluid).detach().cpu()
        ),
        "reference_divergence_mse": float(
            divergence_mse(reference, domain_bounds, dense_fluid).detach().cpu()
        ),
        "all_final_trajectories_valid": bool(validity_dense.all().cpu()),
        "selected_iteration": result.best_iteration,
        "selected_iteration_is_maximum_budget": (
            result.best_iteration == args.max_iterations
        ),
        "optimization_elapsed_seconds": result.elapsed_seconds,
    }

    view_names = source_metadata.get("projection_names", [])
    selected_view_names = [
        str(view_names[index]) if index < len(view_names) else str(index)
        for index in views
    ]
    run_metadata = {
        "format_version": 1,
        "method": "regularized coarse-grid direct optimization baseline",
        "source_dataset": str(source_path),
        "source_dataset_sha256": source_sha256,
        "source_solver": source_metadata.get("solver", "unknown"),
        "source_upstream_commit": source_metadata.get("upstream_commit", "unknown"),
        "optimization_inputs": [
            "trajectories_2d",
            "projection_matrix",
            "observation_mask",
            "observation_times",
            "velocity_times",
            "domain_bounds",
            "source integrator and boundary mode",
            "known cavity geometry and moving-lid velocity",
        ],
        "reference_flow_used_during_optimization": False,
        "reference_flow_used_for_posthoc_metrics_only": True,
        "reference_3d_trajectories_used": False,
        "precomputed_hidden_depth_used": False,
        "initial_positions": "triangulated from synchronized selected views at t=0",
        "selected_views": selected_view_names,
        "coarse_grid_zyx": list(coarse_shape),
        "dense_grid_zyx": list(dense_shape),
        "max_speed": max_speed,
        "known_lid_boundary": args.known_lid_boundary,
        "train_particles": int(split.train.numel()),
        "validation_particles": int(split.validation.numel()),
        "test_particles": int(split.test.numel()),
        "split": "projected-motion-stratified particle split",
        "seed": args.seed,
        "optimizer": "Adam with cosine learning-rate decay",
        "initial_learning_rate": args.learning_rate,
        "maximum_iterations": args.max_iterations,
        "selected_iteration": result.best_iteration,
        "validation_every": args.validation_every,
        "patience": args.patience,
        "smoothness_weight": args.smoothness_weight,
        "divergence_weight": args.divergence_weight,
        "integrator": integrator,
        "training_substeps": args.training_substeps,
        "evaluation_substeps": evaluation_substeps,
        "training_forward_model": (
            "coarse-grid fast approximation; checkpoint selected with training "
            "substeps, final report decoded to dense grid with evaluation substeps"
        ),
        "boundary_mode": boundary_mode,
        "device": str(device),
        "torch_version": torch.__version__,
        "tensor_convention": "[T,component(x,y,z),D(z),H(y),W(x)]",
        "scope": (
            "multi-view, steady, known-boundary coarse-grid baseline; sparse "
            "tracks do not uniquely determine the dense unobserved volume"
        ),
    }

    _save_output(
        args.output,
        result,
        recovered_dense,
        projected_dense,
        trajectories_dense,
        validity_dense,
        initial_positions,
        view_indices,
        split,
        metrics,
        run_metadata,
    )
    _save_convergence(
        result,
        args.convergence_plot,
        smoothness_weight=args.smoothness_weight,
        divergence_weight=args.divergence_weight,
    )
    return metrics, args.output.expanduser(), args.convergence_plot.expanduser()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        metrics, output, plot = recover(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Wrote recovery archive to {output}")
    print(f"Wrote convergence plot to {plot}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
