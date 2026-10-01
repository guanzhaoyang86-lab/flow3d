#!/usr/bin/env python
"""Create diagnostic plots for a generated projected-trajectory archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Any

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.advection import advect_particles
from flow_observation.projection import (
    lift_projected_initial_positions,
    project_trajectories,
)


_REQUIRED_KEYS = {
    "flow_field",
    "trajectories_3d",
    "trajectories_2d",
    "projection_matrix",
    "initial_hidden_depth",
    "observation_mask",
    "observation_times",
    "domain_bounds",
    "metadata",
}

_STORED_CANDIDATE_REPLAY_KEYS = {
    "predicted_trajectories_2d",
    "prediction_validity",
    "test_indices",
    "selected_view_indices",
}


@dataclass(frozen=True)
class _StoredCandidateReplay:
    trajectories_2d: np.ndarray
    validity: np.ndarray
    test_indices: np.ndarray
    selected_view_indices: np.ndarray


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an NPZ projected-trajectory data set."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--view",
        default="0",
        help="Projection name from metadata (xy/xz/yz) or zero-based view index.",
    )
    parser.add_argument("--max-particles", type=_positive_integer, default=24)
    parser.add_argument(
        "--field-time-index",
        type=_nonnegative_integer,
        default=0,
        help="Zero-based velocity snapshot rendered in the flow-field figures.",
    )
    parser.add_argument(
        "--field-plane",
        choices=("xy", "xz", "yz"),
        default=None,
        help="Mid-plane for single-slice plots; defaults to dataset metadata or xy.",
    )
    parser.add_argument(
        "--candidate-input",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing a recovered/candidate Taichi-LBM3D field. "
            "When omitted, only the reference field and trajectories are rendered."
        ),
    )
    parser.add_argument(
        "--candidate-key",
        default="recovered_field",
        help="Array key read from --candidate-input (default: recovered_field).",
    )
    parser.add_argument(
        "--candidate-label",
        default="Recovered field",
        help="Label used for an externally supplied candidate field.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lbm3d_cavity_reference_visualizations"),
    )
    return parser


def _load_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_KEYS.difference(archive.files))
        if missing:
            raise ValueError(
                f"archive is missing required key(s): {', '.join(missing)}"
            )
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}

    metadata_array = arrays["metadata"]
    if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
        raise ValueError("metadata must be a scalar JSON string (not an object array)")
    try:
        metadata = json.loads(str(metadata_array.item()))
    except json.JSONDecodeError as error:
        raise ValueError("metadata is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata JSON must encode an object")
    arrays["metadata"] = metadata
    _validate_shapes(arrays)
    return arrays


def _load_candidate_field(
    path: Path, key: str, reference_shape: tuple[int, ...]
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            available = ", ".join(sorted(archive.files))
            raise ValueError(
                f"candidate archive has no key {key!r}; available keys: {available}"
            )
        candidate = np.array(archive[key], copy=True)
    if candidate.shape != reference_shape:
        raise ValueError(
            "candidate velocity field must match reference shape "
            f"{reference_shape}; received {candidate.shape}"
        )
    if not np.issubdtype(candidate.dtype, np.floating):
        raise ValueError("candidate velocity field must have a floating-point dtype")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate velocity field must contain only finite values")
    return candidate


def _load_stored_candidate_replay(
    path: Path, data: dict[str, Any]
) -> _StoredCandidateReplay | None:
    with np.load(path, allow_pickle=False) as archive:
        if not _STORED_CANDIDATE_REPLAY_KEYS.issubset(archive.files):
            return None
        trajectories = np.array(archive["predicted_trajectories_2d"], copy=True)
        validity = np.array(archive["prediction_validity"], copy=True)
        test_indices = np.array(archive["test_indices"], copy=True)
        selected_views = np.array(archive["selected_view_indices"], copy=True)

    observations = data["trajectories_2d"]
    samples, dataset_views, particles, num_times, _ = observations.shape
    if trajectories.ndim != 5 or trajectories.shape[-1] != 2:
        raise ValueError(
            "candidate predicted_trajectories_2d must have shape "
            "[S,V_selected,N,T_obs,2]"
        )
    if trajectories.shape[0] != samples or trajectories.shape[2:] != (
        particles,
        num_times,
        2,
    ):
        raise ValueError(
            "candidate predicted_trajectories_2d sample, particle, and time "
            "dimensions must match the reference data"
        )
    if not np.issubdtype(trajectories.dtype, np.floating):
        raise ValueError("candidate predicted_trajectories_2d must be floating point")
    if not np.isfinite(trajectories).all():
        raise ValueError(
            "candidate predicted_trajectories_2d must contain only finite values"
        )

    selected_count = trajectories.shape[1]
    if selected_views.ndim != 1 or selected_views.shape[0] != selected_count:
        raise ValueError("candidate selected_view_indices must have shape [V_selected]")
    if not np.issubdtype(selected_views.dtype, np.integer):
        raise ValueError("candidate selected_view_indices must be integers")
    if (
        np.any(selected_views < 0)
        or np.any(selected_views >= dataset_views)
        or np.unique(selected_views).size != selected_views.size
    ):
        raise ValueError(
            "candidate selected_view_indices must contain unique reference-view "
            "indices"
        )

    if test_indices.ndim != 1 or not np.issubdtype(test_indices.dtype, np.integer):
        raise ValueError(
            "candidate test_indices must be a one-dimensional integer array"
        )
    if (
        np.any(test_indices < 0)
        or np.any(test_indices >= particles)
        or np.unique(test_indices).size != test_indices.size
    ):
        raise ValueError(
            "candidate test_indices must contain unique in-range particle indices"
        )

    shared_validity_shape = (samples, particles, num_times)
    view_validity_shape = (samples, selected_count, particles, num_times)
    if validity.shape not in {shared_validity_shape, view_validity_shape}:
        raise ValueError(
            "candidate prediction_validity must have shape [S,N,T_obs] or "
            "[S,V_selected,N,T_obs]"
        )
    if not np.issubdtype(validity.dtype, np.bool_):
        raise ValueError("candidate prediction_validity must have boolean dtype")

    return _StoredCandidateReplay(
        trajectories_2d=trajectories,
        validity=validity,
        test_indices=test_indices.astype(np.int64, copy=False),
        selected_view_indices=selected_views.astype(np.int64, copy=False),
    )


def _validate_shapes(data: dict[str, Any]) -> None:
    flow = data["flow_field"]
    trajectory_3d = data["trajectories_3d"]
    trajectory_2d = data["trajectories_2d"]
    cameras = data["projection_matrix"]
    hidden = data["initial_hidden_depth"]
    mask = data["observation_mask"]
    times = data["observation_times"]
    bounds = data["domain_bounds"]
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow_field must have shape [T_field, 3, D, H, W]")
    if trajectory_3d.ndim != 4 or trajectory_3d.shape[-1] != 3:
        raise ValueError("trajectories_3d must have shape [S, N, T_obs, 3]")
    if trajectory_2d.ndim != 5 or trajectory_2d.shape[-1] != 2:
        raise ValueError("trajectories_2d must have shape [S, V, N, T_obs, 2]")
    samples, particles, num_times, _ = trajectory_3d.shape
    if trajectory_2d.shape[:4] != (
        samples,
        trajectory_2d.shape[1],
        particles,
        num_times,
    ):
        raise ValueError("3D and projected trajectory dimensions are inconsistent")
    views = trajectory_2d.shape[1]
    if cameras.shape != (views, 2, 3):
        raise ValueError("projection_matrix must have shape [V, 2, 3]")
    if hidden.shape != (samples, views, particles):
        raise ValueError("initial_hidden_depth must have shape [S, V, N]")
    if mask.shape != (samples, views, particles, num_times):
        raise ValueError("observation_mask must have shape [S, V, N, T_obs]")
    if times.shape != (num_times,):
        raise ValueError("observation_times must have shape [T_obs]")
    if bounds.shape != (3, 2):
        raise ValueError("domain_bounds must have shape [3, 2]")
    if "velocity_times" in data:
        velocity_times = data["velocity_times"]
        if velocity_times.shape != (flow.shape[0],):
            raise ValueError("velocity_times must have shape [T_field]")
        if not np.isfinite(velocity_times).all():
            raise ValueError("velocity_times must contain only finite values")
        if len(velocity_times) > 1 and not np.all(
            velocity_times[1:] > velocity_times[:-1]
        ):
            raise ValueError("velocity_times must be strictly increasing")

    if "particle_counts" in data:
        counts = data["particle_counts"]
        if counts.shape != (samples,):
            raise ValueError("particle_counts must have shape [S]")
        if np.any(counts < 0) or np.any(counts > particles):
            raise ValueError("particle_counts contains an out-of-range count")


def _resolve_view(view: str, metadata: dict[str, Any], num_views: int) -> int:
    names = metadata.get("projection_names", [])
    if isinstance(names, list) and view in names:
        index = names.index(view)
    else:
        try:
            index = int(view)
        except ValueError as error:
            raise ValueError(
                f"unknown view {view!r}; use a projection name from metadata or an index"
            ) from error
    if index < 0 or index >= num_views:
        raise ValueError(f"view index {index} is outside [0, {num_views - 1}]")
    return index


def _resolve_field_plane(requested: str | None, metadata: dict[str, Any]) -> str:
    plane = requested
    if plane is None:
        preferred = metadata.get("preferred_field_plane", "xy")
        plane = preferred if isinstance(preferred, str) else "xy"
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("field plane must be one of xy, xz, or yz")
    return plane


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class _PlaneSlice:
    name: str
    horizontal_label: str
    vertical_label: str
    fixed_label: str
    fixed_value: float
    horizontal: np.ndarray
    vertical: np.ndarray
    horizontal_velocity: np.ndarray
    vertical_velocity: np.ndarray
    speed: np.ndarray


def _field_coordinates(
    data: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = data["flow_field"]
    bounds = data["domain_bounds"]
    metadata = data["metadata"]
    _, _, depth, height, width = flow.shape
    periodic = metadata.get("boundary_mode") == "periodic"
    x = np.linspace(bounds[0, 0], bounds[0, 1], width, endpoint=not periodic)
    y = np.linspace(bounds[1, 0], bounds[1, 1], height, endpoint=not periodic)
    z = np.linspace(bounds[2, 0], bounds[2, 1], depth, endpoint=not periodic)
    return x, y, z


def _midplane_slice(
    snapshot: np.ndarray,
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    plane: str,
) -> _PlaneSlice:
    x, y, z = coordinates
    _, depth, height, width = snapshot.shape
    if plane == "xy":
        index = depth // 2
        values = snapshot[:, index, :, :]
        return _PlaneSlice(
            name="XY",
            horizontal_label="x",
            vertical_label="y",
            fixed_label="z",
            fixed_value=float(z[index]),
            horizontal=x,
            vertical=y,
            horizontal_velocity=values[0],
            vertical_velocity=values[1],
            speed=np.linalg.norm(values, axis=0),
        )
    if plane == "xz":
        index = height // 2
        values = snapshot[:, :, index, :]
        return _PlaneSlice(
            name="XZ",
            horizontal_label="x",
            vertical_label="z",
            fixed_label="y",
            fixed_value=float(y[index]),
            horizontal=x,
            vertical=z,
            horizontal_velocity=values[0],
            vertical_velocity=values[2],
            speed=np.linalg.norm(values, axis=0),
        )
    if plane == "yz":
        index = width // 2
        values = snapshot[:, :, :, index]
        return _PlaneSlice(
            name="YZ",
            horizontal_label="y",
            vertical_label="z",
            fixed_label="x",
            fixed_value=float(x[index]),
            horizontal=y,
            vertical=z,
            horizontal_velocity=values[1],
            vertical_velocity=values[2],
            speed=np.linalg.norm(values, axis=0),
        )
    raise ValueError(f"unknown plane {plane!r}; expected xy, xz, or yz")


def _field_time_label(data: dict[str, Any], time_index: int) -> str:
    metadata = data.get("metadata", {})
    snapshot_iteration = (
        metadata.get("snapshot_iteration") if isinstance(metadata, dict) else None
    )
    if data["flow_field"].shape[0] == 1 and snapshot_iteration is not None:
        return f"LBM snapshot, iter. {snapshot_iteration}"

    velocity_times = data.get("velocity_times")
    if velocity_times is not None:
        return f"t = {float(velocity_times[time_index]):.3g}"
    if data["flow_field"].shape[0] == len(data["observation_times"]):
        return f"t = {float(data['observation_times'][time_index]):.3g}"
    return f"snapshot {time_index}"


def _draw_plane_field(
    ax: plt.Axes,
    plane: _PlaneSlice,
    *,
    magnitude_max: float,
    vector_max: float,
    cmap: str = "viridis",
    arrow_color: str = "white",
) -> Any:
    safe_magnitude_max = max(float(magnitude_max), np.finfo(float).eps)
    image = ax.imshow(
        plane.speed,
        origin="lower",
        extent=(
            plane.horizontal[0],
            plane.horizontal[-1],
            plane.vertical[0],
            plane.vertical[-1],
        ),
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=safe_magnitude_max,
    )

    horizontal_stride = max(1, len(plane.horizontal) // 16)
    vertical_stride = max(1, len(plane.vertical) // 16)
    horizontal_grid, vertical_grid = np.meshgrid(
        plane.horizontal, plane.vertical, indexing="xy"
    )
    sampled_u = plane.horizontal_velocity[::vertical_stride, ::horizontal_stride]
    sampled_v = plane.vertical_velocity[::vertical_stride, ::horizontal_stride]
    sampled_speed = np.hypot(sampled_u, sampled_v)
    if np.any(sampled_speed > np.finfo(float).eps):
        horizontal_count = len(plane.horizontal[::horizontal_stride])
        vertical_count = len(plane.vertical[::vertical_stride])
        horizontal_spacing = np.ptp(plane.horizontal) / max(horizontal_count - 1, 1)
        vertical_spacing = np.ptp(plane.vertical) / max(vertical_count - 1, 1)
        target_length = 0.62 * min(horizontal_spacing, vertical_spacing)
        quiver_scale = max(float(vector_max), np.finfo(float).eps) / max(
            target_length, np.finfo(float).eps
        )
        ax.quiver(
            horizontal_grid[::vertical_stride, ::horizontal_stride],
            vertical_grid[::vertical_stride, ::horizontal_stride],
            sampled_u,
            sampled_v,
            color=arrow_color,
            alpha=0.88,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            pivot="mid",
            width=0.004,
        )
    ax.set(
        xlabel=plane.horizontal_label,
        ylabel=plane.vertical_label,
        xlim=(plane.horizontal[0], plane.horizontal[-1]),
        ylim=(plane.vertical[0], plane.vertical[-1]),
    )
    return image


def _plot_velocity_field(
    data: dict[str, Any], output: Path, time_index: int, field_plane: str
) -> None:
    snapshot = data["flow_field"][time_index]
    plane = _midplane_slice(snapshot, _field_coordinates(data), field_plane)
    vector_max = float(
        np.hypot(plane.horizontal_velocity, plane.vertical_velocity).max()
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = _draw_plane_field(
        ax,
        plane,
        magnitude_max=float(plane.speed.max()),
        vector_max=vector_max,
    )
    fig.colorbar(image, ax=ax, label=r"3D speed $\|\mathbf{u}\|_2$")
    ax.set_title(
        f"Velocity field: {plane.name} at {plane.fixed_label} = "
        f"{plane.fixed_value:.3g} ({_field_time_label(data, time_index)})"
    )
    _save_figure(fig, output / "velocity_field.png")


def _plot_velocity_streamlines(
    data: dict[str, Any], output: Path, time_index: int, field_plane: str
) -> None:
    snapshot = data["flow_field"][time_index]
    plane = _midplane_slice(snapshot, _field_coordinates(data), field_plane)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(
        plane.speed,
        origin="lower",
        extent=(
            plane.horizontal[0],
            plane.horizontal[-1],
            plane.vertical[0],
            plane.vertical[-1],
        ),
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(plane.speed.max()), np.finfo(float).eps),
    )
    in_plane_speed = np.hypot(plane.horizontal_velocity, plane.vertical_velocity)
    maximum = float(in_plane_speed.max())
    if maximum > np.finfo(float).eps:
        linewidth = 0.55 + 1.75 * in_plane_speed / maximum
        ax.streamplot(
            plane.horizontal,
            plane.vertical,
            plane.horizontal_velocity,
            plane.vertical_velocity,
            color="white",
            density=1.25,
            linewidth=linewidth,
            arrowsize=0.9,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No in-plane velocity",
            ha="center",
            va="center",
            color="white",
            transform=ax.transAxes,
        )
    fig.colorbar(image, ax=ax, label=r"3D speed $\|\mathbf{u}\|_2$")
    ax.set(
        xlabel=plane.horizontal_label,
        ylabel=plane.vertical_label,
        xlim=(plane.horizontal[0], plane.horizontal[-1]),
        ylim=(plane.vertical[0], plane.vertical[-1]),
        title=(
            f"In-plane streamlines: {plane.name} at {plane.fixed_label} = "
            f"{plane.fixed_value:.3g} ({_field_time_label(data, time_index)})"
        ),
    )
    _save_figure(fig, output / "velocity_field_streamlines.png")


def _plot_velocity_field_slices(
    data: dict[str, Any], output: Path, time_index: int
) -> None:
    snapshot = data["flow_field"][time_index]
    coordinates = _field_coordinates(data)
    planes = [
        _midplane_slice(snapshot, coordinates, name) for name in ("xy", "xz", "yz")
    ]
    magnitude_max = max(float(plane.speed.max()) for plane in planes)
    vector_max = max(
        float(np.hypot(plane.horizontal_velocity, plane.vertical_velocity).max())
        for plane in planes
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    images = []
    for ax, plane in zip(axes, planes, strict=True):
        images.append(
            _draw_plane_field(
                ax,
                plane,
                magnitude_max=magnitude_max,
                vector_max=vector_max,
            )
        )
        ax.set_title(f"{plane.name} at {plane.fixed_label} = {plane.fixed_value:.3g}")
    fig.colorbar(
        images[0],
        ax=axes,
        shrink=0.82,
        pad=0.025,
        label=r"3D speed $\|\mathbf{u}\|_2$",
    )
    fig.suptitle(
        f"Orthogonal velocity-field slices ({_field_time_label(data, time_index)})"
    )
    _save_figure(fig, output / "velocity_field_slices.png")


def _plot_velocity_field_3d(
    data: dict[str, Any], output: Path, time_index: int
) -> None:
    snapshot = data["flow_field"][time_index]
    x, y, z = _field_coordinates(data)
    z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")
    _, depth, height, width = snapshot.shape
    z_stride = max(1, int(np.ceil(depth / 7)))
    y_stride = max(1, int(np.ceil(height / 7)))
    x_stride = max(1, int(np.ceil(width / 7)))
    selection = (
        slice(None, None, z_stride),
        slice(None, None, y_stride),
        slice(None, None, x_stride),
    )
    u, v, w = (component[selection] for component in snapshot)
    speed = np.sqrt(u * u + v * v + w * w)
    nonzero = speed > np.finfo(float).eps

    fig = plt.figure(figsize=(8.0, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    speed_max = max(float(speed.max()), np.finfo(float).eps)
    normalization = matplotlib.colors.Normalize(vmin=0.0, vmax=speed_max)
    color_map = matplotlib.colormaps["viridis"]
    if np.any(nonzero):
        extents = data["domain_bounds"][:, 1] - data["domain_bounds"][:, 0]
        samples_per_axis = np.array(
            [len(x[::x_stride]), len(y[::y_stride]), len(z[::z_stride])]
        )
        spacing = extents / np.maximum(samples_per_axis - 1, 1)
        ax.quiver(
            x_grid[selection][nonzero],
            y_grid[selection][nonzero],
            z_grid[selection][nonzero],
            u[nonzero],
            v[nonzero],
            w[nonzero],
            color=color_map(normalization(speed[nonzero])),
            length=0.58 * float(spacing.min()),
            normalize=True,
            linewidth=0.7,
            alpha=0.9,
        )
    else:
        ax.text2D(0.5, 0.5, "Zero velocity field", ha="center", transform=ax.transAxes)
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalization, cmap=color_map)
    fig.colorbar(
        scalar_mappable,
        ax=ax,
        shrink=0.72,
        pad=0.08,
        label=r"speed $\|\mathbf{u}\|_2$",
    )
    bounds = data["domain_bounds"]
    extents = bounds[:, 1] - bounds[:, 0]
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xlim=tuple(bounds[0]),
        ylim=tuple(bounds[1]),
        zlim=tuple(bounds[2]),
        title=(
            "Subsampled 3D velocity directions "
            f"({_field_time_label(data, time_index)}; color = speed)"
        ),
    )
    ax.set_box_aspect(tuple(extents))
    ax.view_init(elev=25, azim=-55)
    _save_figure(fig, output / "velocity_field_3d.png")


def _recovery_interior_mask(
    data: dict[str, Any], spatial_shape: tuple[int, int, int]
) -> np.ndarray:
    mask = np.ones(spatial_shape, dtype=bool)
    if "solid_mask" in data:
        solid = np.asarray(data["solid_mask"], dtype=bool)
        if solid.shape != spatial_shape:
            raise ValueError("solid_mask must match the velocity-field spatial shape")
        mask &= ~solid

    lid_face = data["metadata"].get("lid_face")
    if lid_face is not None:
        try:
            axis_name, side = str(lid_face).lower().split("=", maxsplit=1)
            axis = {"x": 2, "y": 1, "z": 0}[axis_name]
        except (KeyError, ValueError) as error:
            raise ValueError(f"unsupported metadata lid_face {lid_face!r}") from error
        if side not in {"min", "max"}:
            raise ValueError(f"unsupported metadata lid_face {lid_face!r}")
        selection = [slice(None), slice(None), slice(None)]
        selection[axis] = 0 if side == "min" else spatial_shape[axis] - 1
        mask[tuple(selection)] = False

    if not bool(mask.any()):
        raise ValueError("no unknown interior cells remain for field comparison")
    return mask


def _interior_relative_l2(
    data: dict[str, Any],
    reference_field: np.ndarray,
    candidate_field: np.ndarray,
) -> float:
    interior = _recovery_interior_mask(data, tuple(reference_field.shape[-3:]))
    reference_values = reference_field[:, :, interior]
    residual_values = (candidate_field - reference_field)[:, :, interior]
    denominator = float(np.linalg.norm(reference_values))
    return float(np.linalg.norm(residual_values)) / max(
        denominator, np.finfo(float).eps
    )


def _plot_velocity_field_comparison(
    data: dict[str, Any],
    reference_field: np.ndarray,
    candidate_field: np.ndarray,
    output: Path,
    time_index: int,
    *,
    candidate_label: str = "Candidate field",
    field_plane: str = "xy",
    filename: str = "velocity_field_comparison.png",
) -> None:
    if candidate_field.shape != reference_field.shape:
        raise ValueError(
            "candidate and reference velocity fields must have the same shape"
        )
    coordinates = _field_coordinates(data)
    reference = _midplane_slice(reference_field[time_index], coordinates, field_plane)
    candidate = _midplane_slice(candidate_field[time_index], coordinates, field_plane)
    residual = _midplane_slice(
        candidate_field[time_index] - reference_field[time_index],
        coordinates,
        field_plane,
    )

    shared_magnitude_max = max(
        float(reference.speed.max()), float(candidate.speed.max())
    )
    shared_vector_max = max(
        float(
            np.hypot(reference.horizontal_velocity, reference.vertical_velocity).max()
        ),
        float(
            np.hypot(candidate.horizontal_velocity, candidate.vertical_velocity).max()
        ),
    )
    residual_vector_max = float(
        np.hypot(residual.horizontal_velocity, residual.vertical_velocity).max()
    )
    interior_relative_l2 = _interior_relative_l2(data, reference_field, candidate_field)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    reference_image = _draw_plane_field(
        axes[0],
        reference,
        magnitude_max=shared_magnitude_max,
        vector_max=shared_vector_max,
    )
    _draw_plane_field(
        axes[1],
        candidate,
        magnitude_max=shared_magnitude_max,
        vector_max=shared_vector_max,
    )
    residual_image = _draw_plane_field(
        axes[2],
        residual,
        magnitude_max=float(residual.speed.max()),
        vector_max=residual_vector_max,
        cmap="magma",
    )
    axes[0].set_title("Reference field")
    axes[1].set_title(candidate_label)
    axes[2].set_title(r"Vector error $\|\hat{\mathbf{u}}-\mathbf{u}\|_2$")
    if float(residual.speed.max()) <= np.finfo(np.float32).eps:
        axes[2].text(
            0.5,
            0.5,
            "zero at float32 precision",
            ha="center",
            va="center",
            color="white",
            transform=axes[2].transAxes,
        )
    fig.colorbar(
        reference_image,
        ax=axes[:2],
        shrink=0.82,
        pad=0.025,
        label=r"3D speed $\|\mathbf{u}\|_2$",
    )
    fig.colorbar(
        residual_image,
        ax=axes[2],
        shrink=0.82,
        pad=0.025,
        label="absolute vector error",
    )
    fig.suptitle(
        f"{field_plane.upper()} mid-plane comparison "
        f"({_field_time_label(data, time_index)}; "
        f"interior relative L2 = {interior_relative_l2:.3e})"
    )
    _save_figure(fig, output / filename)


def _break_periodic_wraps(
    trajectory: np.ndarray, bounds: np.ndarray, is_periodic: bool
) -> np.ndarray:
    displayed = trajectory.copy()
    if not is_periodic or len(displayed) < 2:
        return displayed
    extent = bounds[:, 1] - bounds[:, 0]
    wraps = (np.abs(np.diff(displayed, axis=0)) > 0.5 * extent).any(axis=1)
    displayed[1:][wraps] = np.nan
    return displayed


def _break_projected_periodic_wraps(
    predicted: np.ndarray, data: dict[str, Any], view: int
) -> np.ndarray:
    displayed = predicted.copy()
    if data["metadata"].get("boundary_mode") != "periodic" or len(displayed) < 2:
        return displayed
    domain_extent = data["domain_bounds"][:, 1] - data["domain_bounds"][:, 0]
    projected_extent = np.abs(data["projection_matrix"][view]) @ domain_extent
    wraps = (np.abs(np.diff(predicted, axis=0)) > 0.5 * projected_extent).any(axis=1)
    displayed[1:][wraps] = np.nan
    return displayed


def _plot_3d_trajectories(
    data: dict[str, Any], sample: int, indices: np.ndarray, output: Path
) -> None:
    trajectories = data["trajectories_3d"][sample]
    bounds = data["domain_bounds"]
    periodic = data["metadata"].get("boundary_mode") == "periodic"
    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for index in indices:
        path = _break_periodic_wraps(trajectories[index], bounds, periodic)
        (line,) = ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.3)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], s=13, color=line.get_color())
    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=f"3D particle trajectories (sample {sample})",
        xlim=tuple(bounds[0]),
        ylim=tuple(bounds[1]),
        zlim=tuple(bounds[2]),
    )
    _save_figure(fig, output / "trajectories_3d.png")


def _masked_observation(observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask[..., None], observation, np.nan)


def _projection_label(metadata: dict[str, Any], view: int) -> str:
    names = metadata.get("projection_names", [])
    if isinstance(names, list) and view < len(names):
        return str(names[view])
    return f"view {view}"


def _plot_projected_trajectories(
    data: dict[str, Any], sample: int, view: int, indices: np.ndarray, output: Path
) -> None:
    observations = data["trajectories_2d"][sample, view]
    masks = data["observation_mask"][sample, view].astype(bool)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for index in indices:
        shown = _masked_observation(observations[index], masks[index])
        shown = _break_projected_periodic_wraps(shown, data, view)
        (line,) = ax.plot(shown[:, 0], shown[:, 1], "o-", markersize=2.5, linewidth=1.0)
        if masks[index, 0]:
            ax.scatter(
                observations[index, 0, 0],
                observations[index, 0, 1],
                color=line.get_color(),
                marker="x",
                s=22,
            )
    ax.set(
        xlabel="projected coordinate 1",
        ylabel="projected coordinate 2",
        title=f"Observed projected trajectories ({_projection_label(data['metadata'], view)})",
        aspect="equal",
    )
    ax.grid(alpha=0.25)
    _save_figure(fig, output / "projected_trajectories.png")


def _replay(
    data: dict[str, Any], sample: int, view: int, velocity_field: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = data["observation_mask"][sample, view].astype(bool)
    count = (
        int(data["particle_counts"][sample])
        if "particle_counts" in data
        else masks.shape[0]
    )
    replay_indices = np.flatnonzero(masks[:count, 0])
    if replay_indices.size == 0:
        empty_trajectory = np.empty((0, masks.shape[-1], 2), dtype=np.float32)
        empty_validity = np.empty((0, masks.shape[-1]), dtype=bool)
        return replay_indices, empty_trajectory, empty_validity

    projected_x0 = torch.as_tensor(
        data["trajectories_2d"][sample, view, replay_indices, 0],
        dtype=velocity_field.dtype,
    ).unsqueeze(0)
    camera = torch.as_tensor(
        data["projection_matrix"][view], dtype=velocity_field.dtype
    )
    hidden_depth = torch.as_tensor(
        data["initial_hidden_depth"][sample, view, replay_indices],
        dtype=velocity_field.dtype,
    ).unsqueeze(0)
    observation_times = torch.as_tensor(
        data["observation_times"], dtype=velocity_field.dtype
    )
    domain_bounds = torch.as_tensor(data["domain_bounds"], dtype=velocity_field.dtype)
    metadata = data["metadata"]
    initial_positions = lift_projected_initial_positions(
        projected_x0, camera, hidden_depth
    )
    if "velocity_times" in data:
        velocity_times = torch.as_tensor(
            data["velocity_times"], dtype=velocity_field.dtype
        )
    else:
        velocity_times = (
            observation_times
            if velocity_field.shape[0] == observation_times.numel()
            else None
        )
    trajectories_3d, validity = advect_particles(
        velocity_field,
        initial_positions,
        observation_times,
        integrator=str(metadata.get("integrator", "rk4")),
        num_substeps=int(metadata.get("integration_substeps", 1)),
        domain_bounds=domain_bounds,
        boundary_mode=str(metadata.get("boundary_mode", "clamp")),
        velocity_times=velocity_times,
        return_validity=True,
    )
    projected = project_trajectories(trajectories_3d, camera)[0]
    return (
        replay_indices,
        projected.detach().cpu().numpy(),
        validity[0].detach().cpu().numpy(),
    )


def _stored_candidate_view(
    stored: _StoredCandidateReplay, sample: int, view: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_views = np.flatnonzero(stored.selected_view_indices == view)
    if local_views.size != 1:
        available = ", ".join(str(int(index)) for index in stored.selected_view_indices)
        raise ValueError(
            f"candidate has no saved predictions for reference view {view}; "
            f"selected_view_indices contains [{available}]"
        )
    local_view = int(local_views[0])
    indices = stored.test_indices
    trajectories = stored.trajectories_2d[sample, local_view, indices]
    if stored.validity.ndim == 3:
        validity = stored.validity[sample, indices]
    else:
        validity = stored.validity[sample, local_view, indices]
    return indices, trajectories, validity


def _plot_replay_comparison(
    data: dict[str, Any],
    sample: int,
    view: int,
    plot_indices: np.ndarray,
    replay_indices: np.ndarray,
    replay: np.ndarray,
    replay_validity: np.ndarray,
    output: Path,
    *,
    candidate_label: str | None = None,
    filename: str | None = None,
    title_override: str | None = None,
) -> None:
    observations = data["trajectories_2d"][sample, view]
    masks = data["observation_mask"][sample, view].astype(bool)
    replay_lookup = {int(index): offset for offset, index in enumerate(replay_indices)}
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    plotted_replay = False
    replay_label = (
        f"{candidate_label} replay"
        if candidate_label is not None
        else "ground-truth replay"
    )
    for color_index, index in enumerate(plot_indices):
        color = plt.cm.tab20(color_index % 20)
        shown = _masked_observation(observations[index], masks[index])
        ax.plot(
            shown[:, 0],
            shown[:, 1],
            "o",
            markersize=2.5,
            color=color,
            alpha=0.8,
            label="observed" if color_index == 0 else None,
        )
        if int(index) in replay_lookup:
            offset = replay_lookup[int(index)]
            predicted = np.where(
                replay_validity[offset, :, None], replay[offset], np.nan
            )
            predicted = _break_projected_periodic_wraps(predicted, data, view)
            ax.plot(
                predicted[:, 0],
                predicted[:, 1],
                "--" if candidate_label is not None else "-",
                linewidth=1.25,
                color=color,
                label=replay_label if not plotted_replay else None,
            )
            plotted_replay = True
    if not plotted_replay:
        ax.text(
            0.5,
            0.5,
            "No particles have an observed initial position",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    title = title_override
    if title is None:
        title = (
            f"Observed vs. {candidate_label} replay"
            if candidate_label is not None
            else "Observed vs. ground-truth-field replay"
        )
    ax.set(
        xlabel="projected coordinate 1",
        ylabel="projected coordinate 2",
        title=f"{title} ({_projection_label(data['metadata'], view)})",
        aspect="equal",
    )
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save_figure(fig, output / (filename or "ground_truth_replay.png"))


def create_diagnostics(
    data: dict[str, Any],
    sample: int,
    view: int,
    max_particles: int,
    output: Path,
    field_time_index: int = 0,
    candidate_field: torch.Tensor | None = None,
    candidate_label: str = "Recovered field",
    field_plane: str | None = None,
    stored_candidate_replay: _StoredCandidateReplay | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    samples, max_count = data["trajectories_3d"].shape[:2]
    if sample < 0 or sample >= samples:
        raise ValueError(f"sample index {sample} is outside [0, {samples - 1}]")
    count = (
        int(data["particle_counts"][sample]) if "particle_counts" in data else max_count
    )
    masks = data["observation_mask"][sample, view, :count].astype(bool)
    observed_indices = np.flatnonzero(masks.any(axis=1))
    if observed_indices.size:
        plot_indices = observed_indices[:max_particles]
    else:
        plot_indices = np.arange(min(count, max_particles))

    field_time_count = data["flow_field"].shape[0]
    if field_time_index < 0 or field_time_index >= field_time_count:
        raise ValueError(
            f"field time index {field_time_index} is outside "
            f"[0, {field_time_count - 1}]"
        )
    resolved_field_plane = _resolve_field_plane(field_plane, data["metadata"])

    reference_field = torch.as_tensor(data["flow_field"], dtype=torch.float32)
    if candidate_field is not None:
        candidate_field = torch.as_tensor(candidate_field, dtype=torch.float32)
        if candidate_field.shape != reference_field.shape:
            raise ValueError(
                "candidate and reference velocity fields must have the same shape"
            )

    _plot_velocity_field(data, output, field_time_index, resolved_field_plane)
    _plot_velocity_streamlines(data, output, field_time_index, resolved_field_plane)
    _plot_velocity_field_slices(data, output, field_time_index)
    _plot_velocity_field_3d(data, output, field_time_index)
    if candidate_field is not None:
        _plot_velocity_field_comparison(
            data,
            reference_field.numpy(),
            candidate_field.numpy(),
            output,
            field_time_index,
            candidate_label=candidate_label,
            filename="recovered_field_comparison.png",
            field_plane=resolved_field_plane,
        )
    _plot_3d_trajectories(data, sample, plot_indices, output)
    _plot_projected_trajectories(data, sample, view, plot_indices, output)

    replay_indices, replay, replay_validity = _replay(
        data, sample, view, reference_field
    )
    _plot_replay_comparison(
        data,
        sample,
        view,
        plot_indices,
        replay_indices,
        replay,
        replay_validity,
        output,
    )

    if candidate_field is not None:
        if stored_candidate_replay is not None:
            candidate_indices, candidate_replay, candidate_validity = (
                _stored_candidate_view(stored_candidate_replay, sample, view)
            )
            if np.any(candidate_indices >= count):
                raise ValueError(
                    "candidate test_indices contains a padded particle for this sample"
                )
            candidate_masks = data["observation_mask"][
                sample, view, candidate_indices
            ].astype(bool)
            candidate_plot_indices = candidate_indices[candidate_masks.any(axis=1)][
                :max_particles
            ]
            candidate_title = f"Held-out test tracks: observed vs. {candidate_label}"
        else:
            candidate_indices, candidate_replay, candidate_validity = _replay(
                data, sample, view, candidate_field
            )
            candidate_plot_indices = plot_indices
            candidate_title = f"Stored-depth diagnostic: observed vs. {candidate_label}"
        _plot_replay_comparison(
            data,
            sample,
            view,
            candidate_plot_indices,
            candidate_indices,
            candidate_replay,
            candidate_validity,
            output,
            candidate_label=candidate_label,
            filename="recovered_field_replay.png",
            title_override=candidate_title,
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        data = _load_archive(args.input.expanduser())
        num_views = data["projection_matrix"].shape[0]
        view = _resolve_view(args.view, data["metadata"], num_views)
        candidate_field = None
        stored_candidate_replay = None
        if args.candidate_input is not None:
            candidate_path = args.candidate_input.expanduser()
            candidate_array = _load_candidate_field(
                candidate_path,
                args.candidate_key,
                tuple(data["flow_field"].shape),
            )
            candidate_field = torch.as_tensor(candidate_array, dtype=torch.float32)
            stored_candidate_replay = _load_stored_candidate_replay(
                candidate_path, data
            )
        create_diagnostics(
            data,
            args.sample_index,
            view,
            args.max_particles,
            args.output_dir.expanduser(),
            field_time_index=args.field_time_index,
            candidate_field=candidate_field,
            candidate_label=args.candidate_label,
            field_plane=args.field_plane,
            stored_candidate_replay=stored_candidate_replay,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Wrote diagnostics to {args.output_dir}")
    filenames = [
        "velocity_field.png",
        "velocity_field_streamlines.png",
        "velocity_field_slices.png",
        "velocity_field_3d.png",
        "trajectories_3d.png",
        "projected_trajectories.png",
        "ground_truth_replay.png",
    ]
    if args.candidate_input is not None:
        filenames.insert(4, "recovered_field_comparison.png")
        filenames.append("recovered_field_replay.png")
    for filename in filenames:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
