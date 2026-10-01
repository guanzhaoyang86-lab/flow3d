#!/usr/bin/env python
"""Generate projected trajectories from a Taichi-LBM3D cavity flow.

Taichi-LBM3D is used only as an external CFD data generator. Its native
velocity layout is converted once at the solver boundary; particle advection,
projection, replay, and later likelihood calculations continue to use the
PyTorch implementation in this repository.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.advection import advect_particles
from flow_observation.projection import project_trajectories
from flow_observation.sparse_dataset import derive_physical_flow_group_id
from flow_observation.taichi_lbm3d import (
    make_lid_driven_cavity_geometry,
    taichi_solid_mask_to_canonical,
    taichi_velocity_to_canonical,
)


_UPSTREAM_URL = "https://github.com/yjhp1016/taichi_LBM3D"
_PROJECTIONS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "xy": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "xz": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "yz": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _grid_size(value: str) -> int:
    parsed = _positive_integer(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("must be at least 8")
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


def _projection_names(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one of xy, xz, yz")
    unknown = [name for name in names if name not in _PROJECTIONS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown projection(s) {', '.join(unknown)}; choose from xy, xz, yz"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("projection names must not be repeated")
    return names


def _nonempty_identifier(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise argparse.ArgumentTypeError("identifier must not be empty")
    return cleaned


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the professor-recommended Taichi-LBM3D cavity solver and "
            "write a canonical projected-trajectory NPZ data set."
        )
    )
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=_REPOSITORY_ROOT / "third_party" / "taichi_LBM3D",
        help="Checkout of github.com/yjhp1016/taichi_LBM3D.",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda", "vulkan"),
        default="cpu",
        help="Taichi backend. CPU is the correctness-first default.",
    )
    parser.add_argument("--grid-size", type=_grid_size, default=32)
    parser.add_argument("--warmup-steps", type=_positive_integer, default=600)
    parser.add_argument("--lid-speed", type=_positive_float, default=0.05)
    parser.add_argument(
        "--upstream-niu",
        "--viscosity",
        dest="upstream_niu",
        type=_positive_float,
        default=0.16667,
        help="Parameter passed unchanged to the upstream set_viscosity() method.",
    )
    parser.add_argument("--num-particles", type=_positive_integer, default=128)
    parser.add_argument("--num-observation-times", type=_positive_integer, default=20)
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=80.0,
        help="Particle-advection duration in LBM time steps.",
    )
    parser.add_argument(
        "--projections",
        type=_projection_names,
        default=_projection_names("xz,xy,yz"),
        help="Comma-separated 2D views. XZ is first because it shows the main roll.",
    )
    parser.add_argument("--integration-substeps", type=_positive_integer, default=4)
    parser.add_argument("--particle-margin", type=_nonnegative_float, default=2.0)
    parser.add_argument("--candidate-multiplier", type=_positive_integer, default=12)
    parser.add_argument("--minimum-path-length", type=_nonnegative_float, default=1e-3)
    parser.add_argument(
        "--particle-selection",
        choices=("diverse", "random"),
        default="diverse",
        help=(
            "Choose valid particles by spatial/path coverage (legacy default) or "
            "uniformly at random. Sparse two-particle studies should use random."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--case-id",
        type=_nonempty_identifier,
        default=None,
        help="Archive case identity; defaults to the output filename stem.",
    )
    parser.add_argument(
        "--flow-group-id",
        type=_nonempty_identifier,
        default=None,
        help=(
            "Physical simulation-run identity supplied by the collection driver. "
            "It is checked against the ID derived from physical metadata."
        ),
    )
    parser.add_argument("--progress-interval", type=_positive_integer, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lbm3d_cavity_dataset.npz"),
    )
    return parser


def _resolve_solver_directory(upstream_repo: Path) -> tuple[Path, Path]:
    repository = upstream_repo.expanduser().resolve()
    solver_directory = repository / "Single_phase"
    solver_file = solver_directory / "LBM_3D_SinglePhase_Solver.py"
    if not solver_file.is_file():
        raise FileNotFoundError(
            "Taichi-LBM3D solver not found at "
            f"{solver_file}. Clone {_UPSTREAM_URL} into {repository}."
        )
    return repository, solver_directory


def _upstream_commit(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def _load_solver_class(solver_directory: Path) -> type[Any]:
    path_string = str(solver_directory)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)
    module = importlib.import_module("LBM_3D_SinglePhase_Solver")
    solver_class = getattr(module, "LB3D_Solver_Single_Phase", None)
    if solver_class is None:
        raise ImportError("upstream module does not define LB3D_Solver_Single_Phase")
    return solver_class


def _run_cavity(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | str]]:
    try:
        import taichi as ti
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Taichi is not installed. Use the lbm3d optional environment described "
            "in README.md."
        ) from error

    repository, solver_directory = _resolve_solver_directory(args.upstream_repo)
    architectures = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
    }
    ti.init(
        arch=architectures[args.backend],
        enable_fallback=False,
        default_fp=ti.f32,
        kernel_profiler=False,
        print_ir=False,
        offline_cache=True,
    )
    if ti.lang.impl.current_cfg().arch != architectures[args.backend]:
        raise RuntimeError("Taichi selected a different backend; check TI_ARCH")
    solver_class = _load_solver_class(solver_directory)

    size = args.grid_size
    geometry_native = make_lid_driven_cavity_geometry(size).numpy()
    solver = solver_class(nx=size, ny=size, nz=size, sparse_storage=False)
    solver.solid.from_numpy(geometry_native)
    solver.set_bc_vel_x1([0.0, 0.0, args.lid_speed])
    solver.set_viscosity(args.upstream_niu)
    solver.init_simulation()

    start_time = time.perf_counter()
    previous_checkpoint: np.ndarray | None = None
    last_relative_change = float("nan")
    for iteration in range(1, args.warmup_steps + 1):
        solver.step()
        if iteration % args.progress_interval == 0 or iteration == args.warmup_steps:
            checkpoint = solver.v.to_numpy().astype(np.float32, copy=False)
            if previous_checkpoint is not None:
                denominator = max(
                    float(np.linalg.norm(checkpoint)),
                    np.finfo(np.float32).eps,
                )
                last_relative_change = float(
                    np.linalg.norm(checkpoint - previous_checkpoint) / denominator
                )
            previous_checkpoint = checkpoint.copy()
            print(
                f"LBM step {iteration}/{args.warmup_steps}: "
                f"max speed={np.linalg.norm(checkpoint, axis=-1).max():.6g}, "
                f"checkpoint rel. change={last_relative_change:.3e}",
                flush=True,
            )

    raw_velocity = solver.v.to_numpy().astype(np.float32, copy=False)
    raw_solid = solver.solid.to_numpy()
    raw_density = solver.rho.to_numpy().astype(np.float32, copy=False)
    elapsed = time.perf_counter() - start_time

    flow_field = taichi_velocity_to_canonical(raw_velocity).to(torch.float32)
    solid_mask = taichi_solid_mask_to_canonical(raw_solid)
    density_field = torch.from_numpy(raw_density.copy()).permute(2, 1, 0).contiguous()
    if not bool(torch.isfinite(density_field).all()):
        raise RuntimeError("Taichi-LBM3D produced a non-finite density field")

    run_metadata: dict[str, float | str] = {
        "upstream_commit": _upstream_commit(repository),
        "backend": args.backend,
        "elapsed_seconds": elapsed,
        "checkpoint_relative_change": last_relative_change,
    }
    return flow_field, solid_mask, density_field, run_metadata


def _component_statistics(
    flow_field: torch.Tensor, solid_mask: torch.Tensor
) -> dict[str, torch.Tensor | int]:
    fluid_values = flow_field[0, :, ~solid_mask].to(torch.float64)
    if fluid_values.shape[1] < 3:
        raise RuntimeError("cavity geometry contains too few fluid cells")

    component_rms = torch.sqrt(torch.mean(fluid_values.square(), dim=1))
    component_max_abs = fluid_values.abs().amax(dim=1)
    component_energy = torch.mean(fluid_values.square(), dim=1)
    energy_fraction = component_energy / component_energy.sum().clamp_min(
        torch.finfo(component_energy.dtype).eps
    )
    centered = fluid_values - fluid_values.mean(dim=1, keepdim=True)
    component_rank = int(torch.linalg.matrix_rank(centered).item())

    if bool((component_max_abs <= 1e-8).any()):
        raise RuntimeError(
            "at least one velocity component is numerically zero; "
            "the result is not a genuine three-component flow"
        )
    if bool((energy_fraction <= 1e-10).any()) or component_rank < 3:
        raise RuntimeError(
            "velocity components are degenerate; increase --warmup-steps "
            "before calling this a real 3D example"
        )
    return {
        "component_rms": component_rms.to(torch.float32),
        "component_max_abs": component_max_abs.to(torch.float32),
        "component_energy_fraction": energy_fraction.to(torch.float32),
        "component_rank": component_rank,
    }


def _select_spatially_diverse_indices(
    qualified_indices: torch.Tensor,
    candidates: torch.Tensor,
    path_lengths: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Greedily balance trajectory motion with coverage of the 3D domain."""
    if qualified_indices.numel() < count:
        raise ValueError("not enough qualified candidates for spatial selection")

    positions = candidates[qualified_indices]
    scale = (upper - lower).clamp_min(torch.finfo(positions.dtype).eps)
    normalized_positions = (positions - lower) / scale
    qualified_lengths = path_lengths[qualified_indices]
    length_quality = qualified_lengths / qualified_lengths.max().clamp_min(
        torch.finfo(qualified_lengths.dtype).eps
    )

    available = torch.ones(
        qualified_indices.numel(), dtype=torch.bool, device=qualified_indices.device
    )
    minimum_distance_squared = torch.full_like(qualified_lengths, torch.inf)
    selected_local: list[int] = []

    for _ in range(count):
        if not selected_local:
            score = length_quality
        else:
            newest = selected_local[-1]
            distance_squared = (
                (normalized_positions - normalized_positions[newest])
                .square()
                .sum(dim=1)
            )
            minimum_distance_squared = torch.minimum(
                minimum_distance_squared, distance_squared
            )
            available_maximum = (
                minimum_distance_squared[available]
                .max()
                .clamp_min(torch.finfo(minimum_distance_squared.dtype).eps)
            )
            distance_quality = minimum_distance_squared / available_maximum
            score = distance_quality * (0.25 + 0.75 * length_quality)

        score = score.masked_fill(~available, -torch.inf)
        chosen = int(torch.argmax(score).item())
        selected_local.append(chosen)
        available[chosen] = False

    selected = torch.tensor(
        selected_local, dtype=torch.long, device=qualified_indices.device
    )
    return qualified_indices[selected]


def _sample_trajectories(
    flow_field: torch.Tensor,
    solid_mask: torch.Tensor,
    observation_times: torch.Tensor,
    domain_bounds: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum_margin = 0.5 * float(args.grid_size - 1)
    if args.particle_margin >= maximum_margin:
        raise ValueError("--particle-margin leaves no fluid sampling region")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    candidate_count = args.num_particles * args.candidate_multiplier
    lower = domain_bounds[:, 0] + args.particle_margin
    upper = domain_bounds[:, 1] - args.particle_margin
    random_values = torch.rand((candidate_count, 3), generator=generator)
    candidates = lower + random_values * (upper - lower)

    with torch.no_grad():
        trajectories, validity = advect_particles(
            flow_field,
            candidates.unsqueeze(0),
            observation_times,
            integrator="rk4",
            num_substeps=args.integration_substeps,
            domain_bounds=domain_bounds,
            boundary_mode="terminate",
            velocity_times=torch.zeros((1,), dtype=flow_field.dtype),
            return_validity=True,
        )

    candidate_trajectories = trajectories[0]
    fluid_lower = domain_bounds[:, 0] + 1.001
    fluid_upper = domain_bounds[:, 1] - 1.001
    remains_in_fluid_interior = (
        (candidate_trajectories >= fluid_lower)
        & (candidate_trajectories <= fluid_upper)
    ).all(dim=(1, 2))
    path_lengths = torch.linalg.vector_norm(
        candidate_trajectories[:, 1:] - candidate_trajectories[:, :-1],
        dim=-1,
    ).sum(dim=-1)
    qualified = (
        validity[0].all(dim=1)
        & remains_in_fluid_interior
        & torch.isfinite(candidate_trajectories).all(dim=(1, 2))
        & (path_lengths >= args.minimum_path_length)
    )
    qualified_indices = torch.nonzero(qualified, as_tuple=False).flatten()
    if qualified_indices.numel() < args.num_particles:
        raise RuntimeError(
            f"only {qualified_indices.numel()} of {candidate_count} candidate "
            "particles produced valid moving trajectories; reduce --duration or "
            "--particle-margin, or increase --candidate-multiplier"
        )

    if args.particle_selection == "random":
        order = torch.randperm(qualified_indices.numel(), generator=generator)
        selected_indices = qualified_indices[order[: args.num_particles]]
    else:
        selected_indices = _select_spatially_diverse_indices(
            qualified_indices,
            candidates,
            path_lengths,
            lower,
            upper,
            args.num_particles,
        )
    selected_initial = candidates[selected_indices]
    selected_trajectories = candidate_trajectories[selected_indices]
    selected_lengths = path_lengths[selected_indices]

    axis_motion = (
        (selected_trajectories - selected_trajectories[:, :1]).abs().amax(dim=(0, 1))
    )
    if args.particle_selection == "diverse" and bool((axis_motion <= 1e-6).any()):
        raise RuntimeError(
            "selected particles do not move in all three coordinate directions"
        )
    if solid_mask.any():
        nearest = selected_trajectories.round().to(torch.long)
        occupied = solid_mask[nearest[..., 2], nearest[..., 1], nearest[..., 0]]
        if bool(occupied.any()):
            raise RuntimeError("a selected particle trajectory intersects a solid node")

    return selected_initial, selected_trajectories, selected_lengths


def _hidden_coordinate(
    initial_positions: torch.Tensor, projection_matrix: torch.Tensor
) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(projection_matrix, full_matrices=True)
    null_vector = vh[-1]
    pivot = null_vector[null_vector.abs().argmax()]
    null_vector = torch.where(pivot < 0, -null_vector, null_vector)
    return torch.einsum("nc,c->n", initial_positions, null_vector)


def generate_dataset(args: argparse.Namespace) -> dict[str, object]:
    if args.num_observation_times < 2:
        raise ValueError("--num-observation-times must be at least 2")

    flow_field, solid_mask, density_field, run_metadata = _run_cavity(args)
    statistics = _component_statistics(flow_field, solid_mask)
    size = args.grid_size
    domain_bounds = torch.tensor(((0.0, float(size - 1)),) * 3, dtype=torch.float32)
    observation_times = torch.linspace(
        0.0,
        args.duration,
        args.num_observation_times,
        dtype=torch.float32,
    )
    initial_positions, trajectory, path_lengths = _sample_trajectories(
        flow_field,
        solid_mask,
        observation_times,
        domain_bounds,
        args,
    )

    projection_names: list[str] = args.projections
    projection_matrix = torch.tensor(
        [_PROJECTIONS[name] for name in projection_names], dtype=torch.float32
    )
    trajectories_3d = trajectory.unsqueeze(0)
    projected_views = [
        project_trajectories(trajectories_3d, camera) for camera in projection_matrix
    ]
    trajectories_2d = torch.stack(projected_views, dim=1)
    hidden_views = [
        _hidden_coordinate(initial_positions, camera) for camera in projection_matrix
    ]
    initial_hidden_depth = torch.stack(hidden_views, dim=0).unsqueeze(0)
    observation_mask = torch.ones(
        (
            1,
            len(projection_names),
            args.num_particles,
            args.num_observation_times,
        ),
        dtype=torch.bool,
    )

    component_rms = statistics["component_rms"]
    component_max_abs = statistics["component_max_abs"]
    energy_fraction = statistics["component_energy_fraction"]
    assert isinstance(component_rms, torch.Tensor)
    assert isinstance(component_max_abs, torch.Tensor)
    assert isinstance(energy_fraction, torch.Tensor)
    metadata = {
        "format_version": 2,
        "field": "taichi_lbm3d_lid_driven_cavity",
        "reference_kind": "numerical 3D CFD simulation",
        "solver": "Taichi-LBM3D",
        "solver_method": "D3Q19 multi-relaxation-time lattice Boltzmann",
        "upstream_url": _UPSTREAM_URL,
        "upstream_commit": run_metadata["upstream_commit"],
        "backend": run_metadata["backend"],
        "grid_size_xyz": [size, size, size],
        "warmup_steps": args.warmup_steps,
        "snapshot_iteration": args.warmup_steps,
        "upstream_niu_parameter": args.upstream_niu,
        "lid_face": "x=max",
        "lid_velocity_xyz": [0.0, 0.0, args.lid_speed],
        "frozen_snapshot_used_for_advection": True,
        "velocity_units": "lattice cells per LBM step",
        "time_units": "LBM steps",
        "boundary_mode": "terminate",
        "preferred_field_plane": "xz",
        "integrator": "rk4",
        "integration_substeps": args.integration_substeps,
        "num_samples": 1,
        "max_num_particles": args.num_particles,
        "num_observation_times": args.num_observation_times,
        "duration": args.duration,
        "projection_names": projection_names,
        "noise_std": 0.0,
        "missing_probability": 0.0,
        "seed": args.seed,
        "particle_margin": args.particle_margin,
        "candidate_multiplier": args.candidate_multiplier,
        "minimum_path_length": args.minimum_path_length,
        "particle_selection": (
            "uniform random selection among valid fluid-interior candidates"
            if args.particle_selection == "random"
            else "valid fluid-interior candidates selected greedily for spatial "
            "coverage and Lagrangian path length"
        ),
        "component_rms_xyz": component_rms.tolist(),
        "component_max_abs_xyz": component_max_abs.tolist(),
        "component_energy_fraction_xyz": energy_fraction.tolist(),
        "component_matrix_rank": statistics["component_rank"],
        "checkpoint_relative_change": run_metadata["checkpoint_relative_change"],
        "solver_elapsed_seconds": run_metadata["elapsed_seconds"],
        "tensor_convention": (
            "coordinates/components are (x,y,z); flow grids are [T,3,D,H,W] "
            "with spatial axes (z,y,x)"
        ),
        "scope": (
            "reference CFD field plus projected-trajectory validation; "
            "this archive does not claim dense cavity-flow recovery"
        ),
    }
    derived_flow_group_id = derive_physical_flow_group_id(metadata)
    if (
        args.flow_group_id is not None
        and args.flow_group_id != derived_flow_group_id
    ):
        raise ValueError(
            "--flow-group-id does not match the physical solver configuration: "
            f"provided {args.flow_group_id!r}, derived {derived_flow_group_id!r}"
        )
    metadata["case_id"] = args.case_id or args.output.expanduser().stem
    metadata["flow_group_id"] = args.flow_group_id or derived_flow_group_id
    metadata["physical_flow_identity_version"] = 1
    metadata["particle_selection_mode"] = args.particle_selection
    return {
        "flow_field": flow_field,
        "velocity_times": torch.zeros((1,), dtype=torch.float32),
        "solid_mask": solid_mask,
        "density_field": density_field,
        "trajectories_3d": trajectories_3d,
        "trajectories_2d": trajectories_2d,
        "projection_matrix": projection_matrix,
        "initial_hidden_depth": initial_hidden_depth,
        "observation_mask": observation_mask,
        "observation_times": observation_times,
        "domain_bounds": domain_bounds,
        "metadata": json.dumps(metadata, sort_keys=True),
        "particle_counts": torch.tensor([args.num_particles], dtype=torch.int64),
        "selected_path_lengths": path_lengths,
        "component_rms": component_rms,
        "component_max_abs": component_max_abs,
        "component_energy_fraction": energy_fraction,
    }


def _save_npz(payload: dict[str, object], output: Path) -> None:
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized: dict[str, np.ndarray] = {}
    for name, value in payload.items():
        if isinstance(value, torch.Tensor):
            serialized[name] = value.detach().cpu().numpy()
        elif isinstance(value, str):
            serialized[name] = np.asarray(value)
        else:
            raise TypeError(
                f"unsupported payload type for {name}: {type(value).__name__}"
            )
    # A killed Slurm job must not leave a half-written .npz that prevents resume.
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **serialized)
    temporary.replace(output)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        payload = generate_dataset(args)
        _save_npz(payload, args.output)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote {args.output}")
    for key in (
        "flow_field",
        "solid_mask",
        "trajectories_3d",
        "trajectories_2d",
        "projection_matrix",
        "observation_mask",
    ):
        value = payload[key]
        assert isinstance(value, torch.Tensor)
        print(f"  {key}: {tuple(value.shape)}")
    print(f"  component RMS (x,y,z): {payload['component_rms'].tolist()}")
    print(f"  component max (x,y,z): {payload['component_max_abs'].tolist()}")
    print(
        "  selected path length range: "
        f"{payload['selected_path_lengths'].min().item():.6g} to "
        f"{payload['selected_path_lengths'].max().item():.6g}"
    )


if __name__ == "__main__":
    main()
