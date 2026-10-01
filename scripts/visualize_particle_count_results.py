"""Create report-ready figures for the sparse particle-count experiment.

The script consumes the completed N=2/N=32 sweep without rerunning inference.
It selects a representative case by the median paired field-relative-L2 score,
uses shared color scales for fair visual comparison, and writes a small figure
manifest documenting every selection and metric shown in the plots.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PARTICLE_COUNTS = (2, 32)
PARTICLE_COLORS = {2: "#0072B2", 32: "#D55E00"}
VIEW_NAMES = ("XZ", "XY", "YZ")
VIEW_AXIS_LABELS = (("x", "z"), ("x", "y"), ("y", "z"))
METRIC_SPECS = (
    ("field_relative_l2_unknown_interior", "Field relative L2", False),
    ("field_cosine_unknown_interior", "Velocity cosine similarity", True),
    ("observed_track_rmse_cells", "Observed-track RMSE (cells)", False),
    ("probe_track_rmse_cells", "Held-out probe RMSE (cells)", False),
)


@dataclass(frozen=True)
class Plane:
    name: str
    horizontal_label: str
    vertical_label: str
    fixed_label: str
    fixed_value: float
    horizontal: np.ndarray
    vertical: np.ndarray
    horizontal_velocity: np.ndarray
    vertical_velocity: np.ndarray
    scalar: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the completed N=2 versus N=32 diffusion sweep."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/particle_count_3seed"),
    )
    parser.add_argument("--training-seed", type=int, default=31)
    parser.add_argument("--representative-index", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected an object on {path}:{line_number}")
        records.append(value)
    return records


def _load_paired_runs(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    paired: dict[int, dict[int, dict[str, Any]]] = {}
    for record in _read_json_lines(path):
        if record.get("status") != "completed" or int(record.get("returncode", 1)) != 0:
            continue
        count = int(record["num_particles"])
        if count not in PARTICLE_COUNTS:
            continue
        index = int(record["test_case_index"])
        if count in paired.setdefault(index, {}):
            raise ValueError(f"duplicate N={count} record for test index {index}")
        paired[index][count] = record

    incomplete = [index for index, values in paired.items() if set(values) != set(PARTICLE_COUNTS)]
    if incomplete:
        raise ValueError(f"unpaired test indices: {incomplete[:10]}")
    if not paired:
        raise ValueError(f"no completed paired runs found in {path}")
    for index, values in paired.items():
        if values[2]["case_id"] != values[32]["case_id"]:
            raise ValueError(f"case mismatch at test index {index}")
        if int(values[2]["seed"]) != int(values[32]["seed"]):
            raise ValueError(f"sampling-seed mismatch at test index {index}")
    return paired


def _case_score(pair: dict[int, dict[str, Any]]) -> float:
    return float(
        np.mean(
            [
                pair[count]["metrics"]["field_relative_l2_unknown_interior"]
                for count in PARTICLE_COUNTS
            ]
        )
    )


def _select_cases(
    paired: dict[int, dict[int, dict[str, Any]]], representative_index: int | None
) -> dict[str, int]:
    ranked = sorted((_case_score(pair), index) for index, pair in paired.items())
    if representative_index is not None:
        if representative_index not in paired:
            raise ValueError(f"test index {representative_index} is not available")
        median_index = representative_index
    else:
        median_score = float(np.median([score for score, _ in ranked]))
        # For an even number of cases, use the smaller test index as a
        # deterministic tie-break between the two central order statistics.
        median_index = min(
            ranked, key=lambda item: (abs(item[0] - median_score), item[1])
        )[1]
    return {"best": ranked[0][1], "median": median_index, "worst": ranked[-1][1]}


def _load_case(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["output"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    result["metrics_json"] = json.loads(str(result["metrics"].item()))
    result["metadata_json"] = json.loads(str(result["metadata"].item()))
    result["source_path"] = path
    return result


def _coordinates(bounds: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, ...]:
    _, depth, height, width = field.shape
    return (
        np.linspace(bounds[0, 0], bounds[0, 1], width),
        np.linspace(bounds[1, 0], bounds[1, 1], height),
        np.linspace(bounds[2, 0], bounds[2, 1], depth),
    )


def _plane(
    field: np.ndarray,
    bounds: np.ndarray,
    name: str,
    scalar_volume: np.ndarray | None = None,
) -> Plane:
    if field.shape[0] != 3 or field.ndim != 4:
        raise ValueError(f"expected field [3,D,H,W], got {field.shape}")
    x, y, z = _coordinates(bounds, field)
    _, depth, height, width = field.shape
    scalar_source = np.linalg.norm(field, axis=0) if scalar_volume is None else scalar_volume
    if scalar_source.shape != (depth, height, width):
        raise ValueError("scalar volume shape does not match the field grid")
    def center_values(
        values: np.ndarray, axis: int, coordinates: np.ndarray
    ) -> tuple[np.ndarray, float]:
        size = values.shape[axis]
        upper = size // 2
        if size % 2:
            return np.take(values, upper, axis=axis), float(coordinates[upper])
        lower = upper - 1
        averaged = 0.5 * (
            np.take(values, lower, axis=axis)
            + np.take(values, upper, axis=axis)
        )
        return averaged, float(0.5 * (coordinates[lower] + coordinates[upper]))

    if name == "xy":
        values, fixed = center_values(field, 1, z)
        scalar, _ = center_values(scalar_source, 0, z)
        return Plane(
            "XY", "x", "y", "z", fixed, x, y,
            values[0], values[1], scalar,
        )
    if name == "xz":
        values, fixed = center_values(field, 2, y)
        scalar, _ = center_values(scalar_source, 1, y)
        return Plane(
            "XZ", "x", "z", "y", fixed, x, z,
            values[0], values[2], scalar,
        )
    if name == "yz":
        values, fixed = center_values(field, 3, x)
        scalar, _ = center_values(scalar_source, 2, x)
        return Plane(
            "YZ", "y", "z", "x", fixed, y, z,
            values[1], values[2], scalar,
        )
    raise ValueError(f"unknown plane {name}")


def _draw_plane(
    ax: plt.Axes,
    plane: Plane,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    draw_vectors: bool,
    vector_max: float | None = None,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(
        plane.scalar,
        origin="lower",
        extent=(
            float(plane.horizontal[0]),
            float(plane.horizontal[-1]),
            float(plane.vertical[0]),
            float(plane.vertical[-1]),
        ),
        cmap=cmap,
        vmin=vmin,
        vmax=max(vmax, np.finfo(float).eps),
        aspect="equal",
        interpolation="nearest",
    )
    if draw_vectors:
        hs = max(1, len(plane.horizontal) // 12)
        vs = max(1, len(plane.vertical) // 12)
        horizontal_grid, vertical_grid = np.meshgrid(
            plane.horizontal, plane.vertical, indexing="xy"
        )
        u = plane.horizontal_velocity[::vs, ::hs]
        v = plane.vertical_velocity[::vs, ::hs]
        local_vector_max = float(np.hypot(u, v).max())
        shared_vector_max = max(
            float(vector_max if vector_max is not None else local_vector_max),
            np.finfo(float).eps,
        )
        horizontal_spacing = np.ptp(plane.horizontal) / max(len(plane.horizontal[::hs]) - 1, 1)
        vertical_spacing = np.ptp(plane.vertical) / max(len(plane.vertical[::vs]) - 1, 1)
        target = 0.58 * min(horizontal_spacing, vertical_spacing)
        ax.quiver(
            horizontal_grid[::vs, ::hs],
            vertical_grid[::vs, ::hs],
            u,
            v,
            color="white",
            alpha=0.86,
            angles="xy",
            scale_units="xy",
            scale=shared_vector_max / max(target, np.finfo(float).eps),
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


def _save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def _plot_pipeline_overview(output_dir: Path, dpi: int) -> list[str]:
    fig, ax = plt.subplots(figsize=(15.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float, y: float, width: float, height: float, text: str,
        *, face: str, edge: str = "#23364d", fontsize: float = 10.5,
    ) -> tuple[float, float, float, float]:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
        return (x, y, width, height)

    def arrow(
        source: tuple[float, float, float, float],
        target: tuple[float, float, float, float],
        *, color: str = "#435466", dashed: bool = False,
    ) -> None:
        start = (source[0] + source[2], source[1] + source[3] / 2)
        end = (target[0], target[1] + target[3] / 2)
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=14,
                linewidth=1.5, color=color,
                linestyle="--" if dashed else "-",
                connectionstyle="arc3,rad=0.0",
            )
        )

    reference = box(
        0.025, 0.66, 0.15, 0.18,
        "Taichi-LBM3D\n3D reference flow",
        face="#dceef8", edge="#0072B2",
    )
    advection = box(
        0.215, 0.66, 0.15, 0.18,
        "RK4 particle\nadvection",
        face="#e8f4fb", edge="#0072B2",
    )
    projection = box(
        0.405, 0.66, 0.16, 0.18,
        "Three orthographic views\nXZ · XY · YZ",
        face="#e8f4fb", edge="#0072B2",
    )
    tracks = box(
        0.605, 0.66, 0.15, 0.18,
        "Projected tracks\nN=2 or N=32 · T=20",
        face="#fff0df", edge="#D55E00",
    )
    arrow(reference, advection)
    arrow(advection, projection)
    arrow(projection, tracks)

    encoder = box(
        0.115, 0.27, 0.16, 0.18,
        "Permutation-invariant\ntrack-set encoder",
        face="#fff0df", edge="#D55E00",
    )
    diffusion = box(
        0.325, 0.27, 0.17, 0.18,
        "Conditional 3D\ndiffusion model",
        face="#f7e9f3", edge="#CC79A7",
    )
    posterior = box(
        0.545, 0.27, 0.17, 0.18,
        "16 posterior samples\n32³ × 3 velocity fields",
        face="#eee8f8", edge="#7B61A8",
    )
    outputs = box(
        0.765, 0.27, 0.20, 0.18,
        "Posterior mean + uncertainty\nReplay tracks + 32 probes",
        face="#e6f4ea", edge="#009E73",
    )
    # Route the observation-to-encoder connection through the whitespace
    # between rows so it never crosses a node.
    track_x = tracks[0] + tracks[2] / 2
    encoder_x = encoder[0] + encoder[2] / 2
    route_y = 0.535
    ax.plot(
        [track_x, track_x, encoder_x],
        [tracks[1], route_y, route_y],
        color="#D55E00", linewidth=1.5,
    )
    ax.add_patch(
        FancyArrowPatch(
            (encoder_x, route_y),
            (encoder_x, encoder[1] + encoder[3]),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.5,
            color="#D55E00",
        )
    )
    arrow(encoder, diffusion)
    arrow(diffusion, posterior)
    arrow(posterior, outputs)
    ax.text(
        0.49, 0.10,
        "Ground truth is a training target and an evaluation reference; it is hidden from the model at test time",
        ha="center", va="center", color="#4b5661", fontsize=10.2,
    )
    ax.text(
        0.5, 0.96,
        "Sparse multi-view particle tracks → posterior over 3D flow fields",
        ha="center", va="top", fontsize=16, fontweight="bold",
    )
    ax.text(
        0.5, 0.895,
        "Single training seed · 100 paired unseen flows · N=2 and N=32 models evaluated on identical cases",
        ha="center", va="top", fontsize=10.5, color="#4b5661",
    )
    return _save(fig, output_dir, "00_pipeline_overview", dpi)


def _validate_pair(cases: dict[int, dict[str, Any]]) -> None:
    first, second = cases[2], cases[32]
    for key in (
        "reference_field",
        "projection_matrix",
        "observation_times",
        "domain_bounds",
        "probe_particle_indices",
        "probe_observed_tracks",
    ):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"paired N=2/N=32 cases disagree on {key}")
    n2_indices = first["particle_indices"][0]
    n32_indices = second["particle_indices"][0]
    if not np.array_equal(n2_indices, n32_indices[: len(n2_indices)]):
        raise ValueError("N=2 observed particles are not nested inside N=32")


def _plot_flow_slices(
    cases: dict[int, dict[str, Any]], case_id: str, output_dir: Path, dpi: int
) -> list[str]:
    reference = cases[2]["reference_field"][0]
    fields = {
        "Ground truth": reference,
        "2 particles": cases[2]["posterior_mean"][0],
        "32 particles": cases[32]["posterior_mean"][0],
    }
    bounds = cases[2]["domain_bounds"][0]
    planes = ("xz", "xy", "yz")
    plane_values = {
        (label, name): _plane(field, bounds, name)
        for label, field in fields.items()
        for name in planes
    }
    vmax = max(float(value.scalar.max()) for value in plane_values.values())
    vector_max = max(
        float(np.hypot(value.horizontal_velocity, value.vertical_velocity).max())
        for value in plane_values.values()
    )
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 11.4), constrained_layout=True)
    image = None
    for row, (label, field) in enumerate(fields.items()):
        for col, name in enumerate(planes):
            ax = axes[row, col]
            plane = plane_values[(label, name)]
            image = _draw_plane(
                ax,
                plane,
                vmin=0.0,
                vmax=vmax,
                cmap="viridis",
                draw_vectors=True,
                vector_max=vector_max,
            )
            panel_label = "GT"
            metric = ""
            if label != "Ground truth":
                count = 2 if label.startswith("2 ") else 32
                value = cases[count]["metrics_json"]["field_relative_l2_unknown_interior"]
                panel_label = f"N={count}"
                metric = f"\nrelative L2 = {value:.3f}"
            ax.set_title(
                f"{panel_label} · {plane.name} ({plane.fixed_label}={plane.fixed_value:g}){metric}"
            )
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.72, pad=0.02, label=r"Speed $\|\mathbf{u}\|_2$")
    fig.suptitle(
        f"3D velocity-field reconstruction on a paired median case ({case_id})\n"
        "Shared color and vector scales across ground truth and both reconstructions"
    )
    return _save(fig, output_dir, "01_flow_slices_median", dpi)


def _plot_error_slices(
    cases: dict[int, dict[str, Any]], case_id: str, output_dir: Path, dpi: int
) -> list[str]:
    reference = cases[2]["reference_field"][0]
    bounds = cases[2]["domain_bounds"][0]
    planes = ("xz", "xy", "yz")
    errors: dict[tuple[int, str], Plane] = {}
    for count in PARTICLE_COUNTS:
        difference = cases[count]["posterior_mean"][0] - reference
        magnitude = np.linalg.norm(difference, axis=0)
        magnitude[[0, -1], :, :] = np.nan
        magnitude[:, [0, -1], :] = np.nan
        magnitude[:, :, [0, -1]] = np.nan
        for name in planes:
            errors[(count, name)] = _plane(difference, bounds, name, magnitude)
    finite_values = np.concatenate(
        [value.scalar[np.isfinite(value.scalar)] for value in errors.values()]
    )
    vmax = float(np.quantile(finite_values, 0.995))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), constrained_layout=True)
    image = None
    for row, count in enumerate(PARTICLE_COUNTS):
        for col, name in enumerate(planes):
            plane = errors[(count, name)]
            image = _draw_plane(
                axes[row, col], plane, vmin=0.0, vmax=vmax, cmap="magma",
                draw_vectors=False,
            )
            value = cases[count]["metrics_json"]["field_relative_l2_unknown_interior"]
            axes[row, col].set_title(
                f"N={count} · {plane.name} ({plane.fixed_label}={plane.fixed_value:g})\n"
                f"vector error | relative L2={value:.3f}"
            )
            axes[row, col].set_facecolor("#d9d9d9")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02, label=r"$\|\hat{\mathbf{u}}-\mathbf{u}_{GT}\|_2$")
    fig.suptitle(
        f"Reconstruction-error slices ({case_id})\n"
        "Prescribed boundary cells are masked; error scale is shared between N=2 and N=32"
    )
    return _save(fig, output_dir, "02_flow_error_slices_median", dpi)


def _plot_track_set(
    cases: dict[int, dict[str, Any]],
    *,
    observed_key: str,
    replay_key: str,
    mask_key: str,
    validity_key: str,
    title: str,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), constrained_layout=True)
    for row, count in enumerate(PARTICLE_COUNTS):
        case = cases[count]
        observed = case[observed_key][0]
        replay = case[replay_key][0]
        masks = case[mask_key][0]
        validity = case[validity_key][0]
        bounds = case["domain_bounds"][0]
        projection = case["projection_matrix"][0]
        view_ranges = []
        for view_index in range(3):
            projected_corners = np.stack(
                (
                    projection[view_index, :, 0] * bounds[0, 1]
                    + projection[view_index, :, 1] * bounds[1, 1]
                    + projection[view_index, :, 2] * bounds[2, 1],
                    projection[view_index, :, 0] * bounds[0, 0]
                    + projection[view_index, :, 1] * bounds[1, 0]
                    + projection[view_index, :, 2] * bounds[2, 0],
                )
            )
            view_ranges.append((projected_corners.min(axis=0), projected_corners.max(axis=0)))
        for view_index, ax in enumerate(axes[row]):
            num_particles = observed.shape[1]
            if num_particles <= 4:
                highlight = set(range(num_particles))
            elif observed_key.startswith("probe_"):
                highlight = set(np.linspace(0, num_particles - 1, 8, dtype=int).tolist())
            else:
                # The N=2 tracks are exactly the first two tracks in N=32.
                # Highlight the shared nested pair and keep extra tracks faint.
                highlight = {0, 1}
            for particle in range(num_particles):
                valid = masks[view_index, particle] & validity[particle]
                if not bool(valid.any()):
                    continue
                obs_line = observed[view_index, particle].copy()
                pred_line = replay[view_index, particle].copy()
                obs_line[~valid] = np.nan
                pred_line[~valid] = np.nan
                emphasized = particle in highlight
                alpha = 0.96 if emphasized else 0.24
                width = 1.8 if emphasized else 0.65
                observed_color = "#0072B2" if emphasized else "#9aa0a6"
                replay_color = "#D55E00" if emphasized else "#d8a06e"
                ax.plot(
                    obs_line[:, 0], obs_line[:, 1], color=observed_color, alpha=alpha,
                    linewidth=width,
                )
                ax.plot(
                    pred_line[:, 0], pred_line[:, 1], color=replay_color, alpha=alpha,
                    linewidth=width, linestyle="--",
                )
                if emphasized:
                    finite = np.flatnonzero(valid)
                    first, last = int(finite[0]), int(finite[-1])
                    ax.scatter(
                        obs_line[first, 0], obs_line[first, 1], s=18,
                        color="#0072B2", marker="o", alpha=alpha, zorder=5,
                    )
                    ax.scatter(
                        obs_line[last, 0], obs_line[last, 1], s=23,
                        color="#0072B2", marker="^", alpha=alpha, zorder=5,
                    )
            low, high = view_ranges[view_index]
            padding = 0.02 * max(float(np.max(high - low)), 1.0)
            metric_key = (
                "probe_track_rmse_cells"
                if observed_key.startswith("probe_")
                else "observed_track_rmse_cells"
            )
            rmse = float(case["metrics_json"][metric_key])
            ax.set(
                xlabel=VIEW_AXIS_LABELS[view_index][0],
                ylabel=VIEW_AXIS_LABELS[view_index][1],
                xlim=(low[0] - padding, high[0] + padding),
                ylim=(low[1] - padding, high[1] + padding),
                aspect="equal",
                title=f"N={count} · {VIEW_NAMES[view_index]} | RMSE={rmse:.3f}",
            )
            ax.grid(alpha=0.18, linewidth=0.5)
    handles = [
        plt.Line2D([0], [0], color="#0072B2", linewidth=2, label="Observed / GT track"),
        plt.Line2D([0], [0], color="#D55E00", linewidth=2, linestyle="--", label="Replay from posterior mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(title)
    return _save(fig, output_dir, stem, dpi)


def _track_error_statistics(
    case: dict[str, Any], *, probe: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = "probe_" if probe else ""
    observed = case[f"{prefix}observed_tracks"][0]
    replay = case[f"{prefix}replay_tracks"][0]
    observation_mask = case[f"{prefix}observation_mask"][0]
    replay_validity = case[f"{prefix}replay_validity"][0]
    valid = observation_mask & replay_validity[None, :, :]
    error = np.linalg.norm(replay - observed, axis=-1)
    times = case["observation_times"][0]
    mean = np.empty(len(times), dtype=float)
    lower = np.empty(len(times), dtype=float)
    upper = np.empty(len(times), dtype=float)
    for time_index in range(len(times)):
        values = error[:, :, time_index][valid[:, :, time_index]]
        mean[time_index] = float(values.mean())
        lower[time_index], upper[time_index] = np.quantile(values, (0.25, 0.75))
    return times, mean, lower, upper


def _plot_track_error_over_time(
    cases: dict[int, dict[str, Any]], case_id: str, output_dir: Path, dpi: int
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True, constrained_layout=True)
    for ax, probe, title in (
        (axes[0], False, "Conditioning particles"),
        (axes[1], True, "Held-out probe particles"),
    ):
        for count in PARTICLE_COUNTS:
            times, mean, lower, upper = _track_error_statistics(cases[count], probe=probe)
            color = PARTICLE_COLORS[count]
            ax.plot(times, mean, color=color, linewidth=2.1, label=f"N={count} mean")
            ax.fill_between(times, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.set(
            xlabel="Observation time (LBM steps)",
            title=title,
            xlim=(float(times[0]), float(times[-1])),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Projected position error (cells)\nmean line with interquartile band")
    fig.suptitle(
        f"Trajectory-replay error over time on the paired median case ({case_id})"
    )
    return _save(fig, output_dir, "04b_track_error_over_time_median", dpi)


def _jittered_x(count: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(1729 + count)
    center = 0.0 if count == 2 else 1.0
    return center + rng.normal(0.0, 0.035, size=size)


def _plot_aggregate_metrics(
    paired: dict[int, dict[int, dict[str, Any]]], output_dir: Path, dpi: int
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    for ax, (key, label, higher_is_better) in zip(axes.flat, METRIC_SPECS, strict=True):
        values = {
            count: np.array(
                [float(paired[index][count]["metrics"][key]) for index in sorted(paired)],
                dtype=float,
            )
            for count in PARTICLE_COUNTS
        }
        for left, right in zip(values[2], values[32], strict=True):
            ax.plot([0, 1], [left, right], color="#7f8c8d", alpha=0.10, linewidth=0.7, zorder=1)
        boxes = ax.boxplot(
            [values[2], values[32]], positions=[0, 1], widths=0.38,
            patch_artist=True, showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.7},
            whiskerprops={"color": "#555555"}, capprops={"color": "#555555"},
        )
        for patch, count in zip(boxes["boxes"], PARTICLE_COUNTS, strict=True):
            patch.set_facecolor(PARTICLE_COLORS[count])
            patch.set_alpha(0.26)
            patch.set_edgecolor(PARTICLE_COLORS[count])
        for count in PARTICLE_COUNTS:
            ax.scatter(
                _jittered_x(count, len(values[count])), values[count], s=15,
                color=PARTICLE_COLORS[count], alpha=0.48, edgecolors="none", zorder=3,
            )
            ax.scatter(
                0 if count == 2 else 1, values[count].mean(), marker="D", s=54,
                color=PARTICLE_COLORS[count], edgecolor="white", linewidth=0.9, zorder=5,
            )
        n32_better = int(
            np.sum(values[32] > values[2]) if higher_is_better else np.sum(values[32] < values[2])
        )
        direction = "higher is better" if higher_is_better else "lower is better"
        ax.set_title(label)
        ax.set_xticks([0, 1], ["N=2", "N=32"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
        ax.text(
            0.02, 0.98,
            f"Means: {values[2].mean():.3f} vs {values[32].mean():.3f}\n"
            f"N=32 better in {n32_better}/{len(values[2])} paired cases\n({direction})",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.88, "pad": 4},
        )
    fig.suptitle(
        "Paired test performance across 100 unseen Taichi-LBM3D flows\n"
        "Thin lines connect the same test flow; diamonds mark means"
    )
    return _save(fig, output_dir, "05_aggregate_metrics_paired", dpi)


def _plot_paired_identity(
    paired: dict[int, dict[int, dict[str, Any]]], output_dir: Path, dpi: int
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.0), constrained_layout=True)
    for ax, (key, label, higher_is_better) in zip(axes.flat, METRIC_SPECS, strict=True):
        x_values = np.array(
            [float(paired[index][2]["metrics"][key]) for index in sorted(paired)]
        )
        y_values = np.array(
            [float(paired[index][32]["metrics"][key]) for index in sorted(paired)]
        )
        n32_better = y_values > x_values if higher_is_better else y_values < x_values
        colors = np.where(n32_better, PARTICLE_COLORS[32], PARTICLE_COLORS[2])
        combined = np.concatenate((x_values, y_values))
        span = max(float(np.ptp(combined)), 1e-8)
        low = float(combined.min() - 0.06 * span)
        high = float(combined.max() + 0.06 * span)
        ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1.1)
        ax.scatter(
            x_values, y_values, c=colors, s=29, alpha=0.70,
            edgecolors="white", linewidths=0.35,
        )
        ax.set(
            xlim=(low, high), ylim=(low, high),
            xlabel=f"N=2 {label}", ylabel=f"N=32 {label}",
            title=label,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16)
        ax.text(
            0.03, 0.97,
            f"means: {x_values.mean():.3f} vs {y_values.mean():.3f}\n"
            f"N=32 better: {int(n32_better.sum())}/{len(x_values)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.88, "pad": 4},
        )
    fig.suptitle(
        "Strictly paired comparison on 100 unseen flows\n"
        "Blue: N=2 better; orange: N=32 better; dashed diagonal: equal performance"
    )
    return _save(fig, output_dir, "05b_aggregate_metrics_identity", dpi)


def _interior_mask(shape: tuple[int, int, int]) -> np.ndarray:
    if min(shape) < 3:
        raise ValueError("grid must have an interior after removing six boundary faces")
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    return mask


def _plot_uncertainty(
    cases: dict[int, dict[str, Any]], case_id: str, output_dir: Path, dpi: int
) -> list[str]:
    reference = cases[2]["reference_field"][0]
    bounds = cases[2]["domain_bounds"][0]
    errors: dict[int, np.ndarray] = {}
    uncertainty: dict[int, np.ndarray] = {}
    for count in PARTICLE_COUNTS:
        estimate = cases[count]["posterior_mean"][0]
        variance = cases[count]["posterior_variance"][0]
        errors[count] = np.linalg.norm(estimate - reference, axis=0)
        uncertainty[count] = np.sqrt(np.maximum(variance.sum(axis=0), 0.0))
        for volume in (errors[count], uncertainty[count]):
            volume[[0, -1], :, :] = np.nan
            volume[:, [0, -1], :] = np.nan
            volume[:, :, [0, -1]] = np.nan
    error_vmax = float(
        np.quantile(np.concatenate([value[np.isfinite(value)] for value in errors.values()]), 0.995)
    )
    uncertainty_vmax = float(
        np.quantile(
            np.concatenate([value[np.isfinite(value)] for value in uncertainty.values()]), 0.995
        )
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2), constrained_layout=True)
    rng = np.random.default_rng(20260921)
    for row, count in enumerate(PARTICLE_COUNTS):
        estimate = cases[count]["posterior_mean"][0]
        variance = cases[count]["posterior_variance"][0]
        error_squared = (estimate - reference) ** 2
        error_plane = _plane(estimate - reference, bounds, "xz", errors[count])
        uncertainty_plane = _plane(estimate, bounds, "xz", uncertainty[count])
        error_image = _draw_plane(
            axes[row, 0], error_plane, vmin=0.0, vmax=error_vmax, cmap="magma",
            draw_vectors=False,
        )
        uncertainty_image = _draw_plane(
            axes[row, 1], uncertainty_plane, vmin=0.0, vmax=uncertainty_vmax,
            cmap="cividis", draw_vectors=False,
        )
        axes[row, 0].set_title(f"N={count} · XZ vector error")
        axes[row, 1].set_title(f"N={count} · XZ posterior std")
        axes[row, 0].set_facecolor("#d9d9d9")
        axes[row, 1].set_facecolor("#d9d9d9")

        mask = _interior_mask(reference.shape[1:])
        x_values = variance[:, mask].reshape(-1)
        y_values = error_squared[:, mask].reshape(-1)
        keep = (x_values > 0) & (y_values > 0) & np.isfinite(x_values) & np.isfinite(y_values)
        x_values, y_values = x_values[keep], y_values[keep]
        if len(x_values) > 12000:
            selected = rng.choice(len(x_values), size=12000, replace=False)
            x_values, y_values = x_values[selected], y_values[selected]
        axes[row, 2].scatter(
            x_values, y_values, s=4, alpha=0.12, color=PARTICLE_COLORS[count],
            edgecolors="none", rasterized=True,
        )
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_yscale("log")
        axes[row, 2].set(
            xlabel="Posterior component variance",
            ylabel="Squared component error",
            title=(
                f"N={count} | uncertainty vs error\n"
                f"Pearson r={cases[count]['metrics_json']['uncertainty_error_pearson_unknown_interior']:.3f}"
            ),
        )
        axes[row, 2].grid(alpha=0.16)
    fig.colorbar(error_image, ax=axes[:, 0], shrink=0.77, pad=0.02, label="Vector error magnitude")
    fig.colorbar(
        uncertainty_image, ax=axes[:, 1], shrink=0.77, pad=0.02,
        label="Posterior standard deviation",
    )
    fig.suptitle(
        f"Posterior uncertainty and actual reconstruction error ({case_id})\n"
        "Maps mask prescribed boundaries; scatter uses unknown interior velocity components"
    )
    return _save(fig, output_dir, "06_uncertainty_vs_error_median", dpi)


def _load_log_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if isinstance(value, dict) and "epoch" in value:
            rows.append(value)
    if not rows:
        raise ValueError(f"no epoch records in {path}")
    return rows


def _plot_training_curves(experiment_root: Path, seed: int, output_dir: Path, dpi: int) -> list[str]:
    all_rows: dict[int, list[dict[str, float]]] = {}
    physics_rows: dict[int, list[dict[str, float]]] = {}
    for count in PARTICLE_COUNTS:
        base = experiment_root / "models" / f"training_seed_{seed}" / f"N{count:03d}"
        pretrain = _load_log_rows(base / "pretrain" / "run.log")
        physics = _load_log_rows(base / "physics" / "run.log")
        all_rows[count] = pretrain + physics
        physics_rows[count] = physics

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.1), constrained_layout=True)
    for count in PARTICLE_COUNTS:
        rows = all_rows[count]
        epochs = np.array([row["epoch"] for row in rows])
        train = np.array([row["train_total"] for row in rows])
        validation = np.array([row["validation_total"] for row in rows])
        validation_diffusion = np.array([row["validation_diffusion"] for row in rows])
        color = PARTICLE_COLORS[count]
        axes[0, 0].plot(
            epochs, train, color=color, alpha=0.34, linewidth=1.0,
            linestyle=":", label=f"N={count} train",
        )
        axes[0, 0].plot(
            epochs, validation, color=color, linewidth=2.0,
            label=f"N={count} validation",
        )
        axes[0, 1].plot(
            epochs, validation_diffusion, color=color, linewidth=2.0,
            label=f"N={count}",
        )
    for ax in axes[0]:
        ax.axvline(100.5, color="#555555", linestyle="--", linewidth=1.1)
        ax.text(
            0.69, 0.93, "physics fine-tuning", transform=ax.transAxes,
            fontsize=8.7, va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
        )
        ax.grid(alpha=0.18)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
    axes[0, 0].set(
        xlabel="Epoch",
        ylabel="Total loss",
        title="Training and validation convergence",
    )
    axes[0, 0].legend(ncol=2, fontsize=8.3, loc="lower left")
    axes[0, 1].set(
        ylabel="Validation diffusion loss",
        title="Validation DDPM objective",
    )
    axes[0, 1].legend(fontsize=8.8, loc="lower left")

    component_colors = ("#0072B2", "#CC79A7", "#009E73", "#E69F00")
    all_component_values: list[np.ndarray] = []
    for column, count in enumerate(PARTICLE_COUNTS):
        rows = physics_rows[count]
        epochs = np.array([row["epoch"] for row in rows])
        components = {
            "Diffusion": np.array([row["validation_diffusion"] for row in rows]),
            "0.1 × track": 0.1 * np.array([row["validation_track"] for row in rows]),
            "0.01 × divergence": 0.01 * np.array([row["validation_divergence"] for row in rows]),
            "0.01 × boundary": 0.01 * np.array([row["validation_boundary"] for row in rows]),
        }
        all_component_values.extend(components.values())
        for (label, values), color in zip(components.items(), component_colors, strict=True):
            axes[1, column].plot(epochs, values, label=label, color=color, linewidth=1.8)
        axes[1, column].set(
            xlabel="Epoch",
            title=f"N={count} weighted physics-stage contributions",
            yscale="log",
        )
        axes[1, column].grid(alpha=0.18)
        axes[1, column].legend(fontsize=8.1, loc="center right")
    positive = np.concatenate([values[values > 0] for values in all_component_values])
    common_limits = (float(positive.min() / 2.0), float(positive.max() * 2.0))
    for ax in axes[1]:
        ax.set_ylim(common_limits)
    axes[1, 0].set_ylabel("Weighted validation-loss contribution")
    axes[1, 0].text(
        0.03, 0.04,
        "Explicit track and physics terms are many orders below diffusion",
        transform=axes[1, 0].transAxes, fontsize=8.7,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9, "pad": 3},
    )
    fig.suptitle("Optimization diagnostics for the single training seed (seed 31)")
    return _save(fig, output_dir, "07_training_convergence_and_loss_scale", dpi)


def _plot_3d_flow(
    cases: dict[int, dict[str, Any]], case_id: str, output_dir: Path, dpi: int
) -> list[str]:
    fields = (
        ("Ground truth", cases[2]["reference_field"][0]),
        ("N=2 posterior mean", cases[2]["posterior_mean"][0]),
        ("N=32 posterior mean", cases[32]["posterior_mean"][0]),
    )
    bounds = cases[2]["domain_bounds"][0]
    x, y, z = _coordinates(bounds, fields[0][1])
    z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")
    selection = (slice(1, -1, 5), slice(1, -1, 5), slice(1, -1, 5))
    speeds = [np.linalg.norm(field[:, selection[0], selection[1], selection[2]], axis=0) for _, field in fields]
    vmax = max(float(speed.max()) for speed in speeds)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, np.finfo(float).eps))
    cmap = matplotlib.colormaps["viridis"]
    fig = plt.figure(figsize=(15.2, 5.6), constrained_layout=True)
    for panel, ((label, field), speed) in enumerate(zip(fields, speeds, strict=True), 1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        u = field[0][selection]
        v = field[1][selection]
        w = field[2][selection]
        nonzero = speed > np.finfo(float).eps
        ax.quiver(
            x_grid[selection][nonzero], y_grid[selection][nonzero], z_grid[selection][nonzero],
            u[nonzero], v[nonzero], w[nonzero],
            color=cmap(norm(speed[nonzero])), length=2.15, normalize=True,
            linewidth=0.65, alpha=0.84,
        )
        ax.set(
            xlabel="x", ylabel="y", zlabel="z", title=label,
            xlim=tuple(bounds[0]), ylim=tuple(bounds[1]), zlim=tuple(bounds[2]),
        )
        ax.set_box_aspect(tuple(bounds[:, 1] - bounds[:, 0]))
        ax.view_init(elev=24, azim=-52)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(scalar, ax=fig.axes, shrink=0.70, pad=0.03, label=r"Speed $\|\mathbf{u}\|_2$")
    fig.suptitle(
        f"Subsampled 3D velocity directions on the paired median case ({case_id})\n"
        "Arrow color encodes speed; prescribed boundary vectors are omitted for clarity"
    )
    return _save(fig, output_dir, "08_flow_field_3d_median", dpi)


def _plot_case_gallery(
    paired: dict[int, dict[int, dict[str, Any]]],
    selections: dict[str, int],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    rows: list[tuple[str, int, dict[int, dict[str, Any]]]] = []
    all_fields: list[np.ndarray] = []
    for label in ("best", "median", "worst"):
        index = selections[label]
        cases = {count: _load_case(paired[index][count]) for count in PARTICLE_COUNTS}
        _validate_pair(cases)
        rows.append((label, index, cases))
        all_fields.extend(
            [
                cases[2]["reference_field"][0],
                cases[2]["posterior_mean"][0],
                cases[32]["posterior_mean"][0],
            ]
        )
    vmax = max(float(np.linalg.norm(field, axis=0).max()) for field in all_fields)
    fig, axes = plt.subplots(3, 3, figsize=(12.3, 11.0), constrained_layout=True)
    image = None
    for row, (label, index, cases) in enumerate(rows):
        bounds = cases[2]["domain_bounds"][0]
        case_id = paired[index][2]["case_id"]
        fields = (
            ("Ground truth", cases[2]["reference_field"][0], None),
            ("N=2", cases[2]["posterior_mean"][0], cases[2]["metrics_json"]["field_relative_l2_unknown_interior"]),
            ("N=32", cases[32]["posterior_mean"][0], cases[32]["metrics_json"]["field_relative_l2_unknown_interior"]),
        )
        vector_max = max(
            float(np.hypot(_plane(field, bounds, "xz").horizontal_velocity, _plane(field, bounds, "xz").vertical_velocity).max())
            for _, field, _ in fields
        )
        for col, (field_label, field, metric) in enumerate(fields):
            plane = _plane(field, bounds, "xz")
            image = _draw_plane(
                axes[row, col], plane, vmin=0.0, vmax=vmax, cmap="viridis",
                draw_vectors=True, vector_max=vector_max,
            )
            metric_text = "" if metric is None else f" | L2={metric:.3f}"
            axes[row, col].set_title(
                f"{label.capitalize()} case {case_id}\n{field_label}{metric_text}"
            )
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02, label=r"Speed $\|\mathbf{u}\|_2$")
    fig.suptitle(
        "Best, median, and worst paired test cases (XZ mid-plane)\n"
        "Cases are ranked by the average N=2/N=32 field-relative-L2 score"
    )
    return _save(fig, output_dir, "09_best_median_worst_gallery", dpi)


def _write_readme(
    output_dir: Path,
    selections: dict[str, int],
    paired: dict[int, dict[int, dict[str, Any]]],
) -> None:
    median = selections["median"]
    case_id = paired[median][2]["case_id"]
    content = f"""# Particle-count experiment figures

Generated from the completed single-training-seed (`seed=31`) experiment.
No model was retrained and no new posterior samples were generated.

## Representative-case selection

The main figures use test index `{median}` (`{case_id}`). It is selected
deterministically as the median of the 100 paired cases after ranking by the
average of the N=2 and N=32 field-relative-L2 errors. This avoids choosing a
visually favorable example. The gallery also shows the paired best and worst
cases under the same ranking rule.

## Files

- `00_pipeline_overview`: end-to-end data, inference, and evaluation protocol.
- `01_flow_slices_median`: GT, N=2, and N=32 orthogonal velocity slices.
- `02_flow_error_slices_median`: matched absolute vector-error slices.
- `03_observed_track_replay_median`: conditioning tracks versus replay tracks.
- `04_probe_track_replay_median`: held-out probe tracks versus predictions.
- `04b_track_error_over_time_median`: projected replay error over time.
- `05_aggregate_metrics_paired`: distributions and paired changes over 100 flows.
- `05b_aggregate_metrics_identity`: N=2 versus N=32 for every paired flow.
- `06_uncertainty_vs_error_median`: posterior uncertainty versus actual error.
- `07_training_convergence_and_loss_scale`: convergence and weighted loss scales.
- `08_flow_field_3d_median`: subsampled 3D velocity directions.
- `09_best_median_worst_gallery`: qualitative range across test difficulty.

Every figure is provided as both PNG and vector PDF. Error bars/distributions
are across 100 test flows, not across training seeds. The experiment contains
one training seed only.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    _configure_style()
    experiment_root = args.experiment_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else experiment_root / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = (
        experiment_root
        / "sweeps"
        / f"training_seed_{args.training_seed}"
        / "runs.jsonl"
    )
    paired = _load_paired_runs(runs_path)
    selections = _select_cases(paired, args.representative_index)
    representative_index = selections["median"]
    cases = {
        count: _load_case(paired[representative_index][count])
        for count in PARTICLE_COUNTS
    }
    _validate_pair(cases)
    case_id = str(paired[representative_index][2]["case_id"])

    generated: list[str] = []
    generated += _plot_pipeline_overview(output_dir, args.dpi)
    generated += _plot_flow_slices(cases, case_id, output_dir, args.dpi)
    generated += _plot_error_slices(cases, case_id, output_dir, args.dpi)
    generated += _plot_track_set(
        cases,
        observed_key="observed_tracks",
        replay_key="replay_tracks",
        mask_key="observation_mask",
        validity_key="replay_validity",
        title=(
            f"Observed conditioning tracks versus posterior-mean replay ({case_id})\n"
            "N=2 particles are nested inside the N=32 observation set"
        ),
        stem="03_observed_track_replay_median",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    generated += _plot_track_set(
        cases,
        observed_key="probe_observed_tracks",
        replay_key="probe_replay_tracks",
        mask_key="probe_observation_mask",
        validity_key="probe_replay_validity",
        title=(
            f"Held-out probe tracks versus posterior-mean prediction ({case_id})\n"
            "These 32 particles are excluded from conditioning and shared by both models"
        ),
        stem="04_probe_track_replay_median",
        output_dir=output_dir,
        dpi=args.dpi,
    )
    generated += _plot_track_error_over_time(cases, case_id, output_dir, args.dpi)
    generated += _plot_aggregate_metrics(paired, output_dir, args.dpi)
    generated += _plot_paired_identity(paired, output_dir, args.dpi)
    generated += _plot_uncertainty(cases, case_id, output_dir, args.dpi)
    generated += _plot_training_curves(
        experiment_root, args.training_seed, output_dir, args.dpi
    )
    generated += _plot_3d_flow(cases, case_id, output_dir, args.dpi)
    generated += _plot_case_gallery(paired, selections, output_dir, args.dpi)
    _write_readme(output_dir, selections, paired)

    manifest = {
        "format_version": 1,
        "experiment_root": str(experiment_root),
        "training_seed": args.training_seed,
        "num_paired_test_cases": len(paired),
        "representative_selection_rule": (
            "closest to numeric median of mean N=2/N=32 field_relative_l2; "
            "smaller test index breaks the even-sample tie"
        ),
        "selected_indices": selections,
        "selected_case_ids": {
            label: paired[index][2]["case_id"] for label, index in selections.items()
        },
        "representative_metrics": {
            str(count): cases[count]["metrics_json"] for count in PARTICLE_COUNTS
        },
        "generated_files": generated,
        "notes": [
            "Single training seed only; distributions are across test cases.",
            "N=2 observed particles are nested in the N=32 set.",
            "Held-out probe particles are identical for N=2 and N=32.",
            "Prescribed boundary cells are masked in error/uncertainty maps.",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(generated)} figure files to {output_dir}")
    print(f"Representative case: index={representative_index}, case_id={case_id}")


if __name__ == "__main__":
    main()
