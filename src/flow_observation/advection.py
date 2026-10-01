"""Differentiable Euler and RK4 integration of Lagrangian particles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import torch
from torch import Tensor

from .interpolation import BoundaryMode, _domain_tensor, sample_velocity

Integrator = Literal["euler", "rk4"]


def _substeps_per_interval(
    num_substeps: int | Sequence[int], num_intervals: int
) -> list[int]:
    if isinstance(num_substeps, int):
        values = [num_substeps] * num_intervals
    else:
        values = [int(value) for value in num_substeps]
        if len(values) != num_intervals:
            raise ValueError(
                "a num_substeps sequence must have one value per observation interval"
            )
    if any(value < 1 for value in values):
        raise ValueError("num_substeps values must all be positive")
    return values


def _inside_domain(positions: Tensor, bounds: Tensor) -> Tensor:
    return torch.isfinite(positions).all(dim=-1) & (
        (positions >= bounds[:, 0]) & (positions <= bounds[:, 1])
    ).all(dim=-1)


def _wrap_positions(positions: Tensor, bounds: Tensor) -> Tensor:
    lower = bounds[:, 0]
    extent = bounds[:, 1] - lower
    return torch.remainder(positions - lower, extent) + lower


def _clamp_positions(positions: Tensor, bounds: Tensor) -> Tensor:
    return torch.maximum(torch.minimum(positions, bounds[:, 1]), bounds[:, 0])


@overload
def advect_particles(
    velocity_field: Tensor,
    x0: Tensor,
    observation_times: Tensor | Sequence[float],
    integrator: Integrator = "rk4",
    *,
    num_substeps: int | Sequence[int] = 1,
    domain_bounds: Tensor | Sequence[Sequence[float]] | None = None,
    boundary_mode: BoundaryMode = "clamp",
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: Literal[False] = False,
) -> Tensor: ...


@overload
def advect_particles(
    velocity_field: Tensor,
    x0: Tensor,
    observation_times: Tensor | Sequence[float],
    integrator: Integrator = "rk4",
    *,
    num_substeps: int | Sequence[int] = 1,
    domain_bounds: Tensor | Sequence[Sequence[float]] | None = None,
    boundary_mode: BoundaryMode = "clamp",
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def advect_particles(
    velocity_field: Tensor,
    x0: Tensor,
    observation_times: Tensor | Sequence[float],
    integrator: Integrator = "rk4",
    *,
    num_substeps: int | Sequence[int] = 1,
    domain_bounds: Tensor | Sequence[Sequence[float]] | None = None,
    boundary_mode: BoundaryMode = "clamp",
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate particles through a gridded velocity field.

    Args:
        velocity_field: Velocity snapshots ``[T_field, 3, D, H, W]``.
        x0: Initial physical positions ``[B, N, 3]``.
        observation_times: Strictly increasing one-dimensional physical times.
        integrator: ``"euler"`` or the default ``"rk4"``.
        num_substeps: One positive integer for every interval, or a sequence
            with one value per observation interval.
        domain_bounds: Physical x, y, z bounds ``[3, 2]``. ``None`` uses the
            unit cube.
        boundary_mode: ``periodic`` wraps every accepted step; ``clamp`` clips
            it to the box; ``terminate`` freezes a particle at its last valid
            position after any RK stage or proposed step leaves the box.
        velocity_times: Times associated with velocity snapshots. If omitted
            and T_field > 1, snapshots are spread uniformly from the first to
            last observation time. A single snapshot is treated as steady.
        return_validity: Also return cumulative validity ``[B, N, T_obs]``.

    Returns:
        Trajectories ``[B, N, T_obs, 3]``, optionally followed by the boolean
        validity history. Clamp and terminate never silently discard a particle:
        their validity becomes false once it attempts to leave the domain.

    Notes:
        The implementation builds a regular PyTorch computation graph through
        every interpolation and integration stage. Boolean boundary decisions
        are necessarily piecewise differentiable, while in-domain positions and
        velocity values retain their gradients.
    """
    if integrator not in {"euler", "rk4"}:
        raise ValueError("integrator must be 'euler' or 'rk4'")
    if boundary_mode not in {"periodic", "clamp", "terminate"}:
        raise ValueError(
            "boundary_mode must be one of 'periodic', 'clamp', or 'terminate'"
        )
    if velocity_field.ndim != 5 or velocity_field.shape[1] != 3:
        raise ValueError("velocity_field must have shape [T, 3, D, H, W]")
    if x0.ndim != 3 or x0.shape[-1] != 3:
        raise ValueError("x0 must have shape [B, N, 3]")
    if velocity_field.device != x0.device:
        raise ValueError("velocity_field and x0 must be on one device")

    positions = x0.to(dtype=velocity_field.dtype)
    times = torch.as_tensor(
        observation_times, dtype=positions.dtype, device=positions.device
    )
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("observation_times must be a non-empty 1D sequence")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1]).detach()):
        raise ValueError("observation_times must be strictly increasing")
    interval_substeps = _substeps_per_interval(
        num_substeps, max(int(times.numel()) - 1, 0)
    )
    if domain_bounds is None:
        domain_bounds = ((0.0, 1.0),) * 3
    bounds = _domain_tensor(domain_bounds, like=positions)

    if velocity_times is None:
        if velocity_field.shape[0] == 1:
            frame_times = times[:1]
        else:
            frame_times = torch.linspace(
                times[0],
                times[-1],
                velocity_field.shape[0],
                dtype=positions.dtype,
                device=positions.device,
            )
    else:
        frame_times = torch.as_tensor(
            velocity_times, dtype=positions.dtype, device=positions.device
        )

    initial_inside = _inside_domain(positions, bounds)
    if boundary_mode == "periodic":
        valid = torch.isfinite(positions).all(dim=-1)
        positions = _wrap_positions(positions, bounds)
    elif boundary_mode == "clamp":
        positions = _clamp_positions(positions, bounds)
        valid = initial_inside
    else:
        valid = initial_inside

    trajectory = [positions]
    validity_history = [valid]

    def evaluate(state: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        sampled, stage_valid = sample_velocity(
            velocity_field,
            state,
            time,
            bounds,
            boundary_mode,
            velocity_times=frame_times,
            return_validity=True,
        )
        if boundary_mode == "terminate":
            sampled = sampled * valid[..., None].to(sampled.dtype)
            stage_valid = stage_valid & valid
        return sampled, stage_valid

    for interval_index, steps in enumerate(interval_substeps):
        interval_start = times[interval_index]
        step_size = (times[interval_index + 1] - interval_start) / steps
        for substep_index in range(steps):
            time = interval_start + substep_index * step_size
            if integrator == "euler":
                velocity, stage_valid = evaluate(positions, time)
                proposed = positions + step_size * velocity
            else:
                k1, valid1 = evaluate(positions, time)
                k2, valid2 = evaluate(
                    positions + 0.5 * step_size * k1, time + 0.5 * step_size
                )
                k3, valid3 = evaluate(
                    positions + 0.5 * step_size * k2, time + 0.5 * step_size
                )
                k4, valid4 = evaluate(
                    positions + step_size * k3, time + step_size
                )
                proposed = positions + (step_size / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
                stage_valid = valid1 & valid2 & valid3 & valid4

            step_valid = stage_valid & _inside_domain(proposed, bounds)
            if boundary_mode == "periodic":
                valid = valid & torch.isfinite(proposed).all(dim=-1)
                positions = _wrap_positions(proposed, bounds)
            elif boundary_mode == "clamp":
                positions = _clamp_positions(proposed, bounds)
                valid = valid & step_valid
            else:
                accepted = valid & step_valid
                positions = torch.where(accepted[..., None], proposed, positions)
                valid = accepted

        trajectory.append(positions)
        validity_history.append(valid)

    trajectories = torch.stack(trajectory, dim=2)
    validity = torch.stack(validity_history, dim=2)
    return (trajectories, validity) if return_validity else trajectories
