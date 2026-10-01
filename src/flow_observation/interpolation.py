"""Differentiable interpolation of gridded, time-dependent velocity fields.

Positions always use physical ``(x, y, z)`` order.  A velocity tensor uses
``[T, 3, D, H, W]`` order, so its last three axes correspond to ``(z, y, x)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import torch
from torch import Tensor

BoundaryMode = Literal["periodic", "clamp", "terminate"]


def _domain_tensor(
    domain_bounds: Tensor | Sequence[Sequence[float]], *, like: Tensor
) -> Tensor:
    bounds = torch.as_tensor(domain_bounds, dtype=like.dtype, device=like.device)
    if bounds.shape != (3, 2):
        raise ValueError(
            "domain_bounds must have shape [3, 2] in ((xmin, xmax), "
            "(ymin, ymax), (zmin, zmax)) order"
        )
    if not bool(torch.all(bounds[:, 1] > bounds[:, 0]).detach()):
        raise ValueError("every domain upper bound must be larger than its lower bound")
    return bounds


def _broadcast_particle_times(particle_times: Tensor | float, positions: Tensor) -> Tensor:
    batch_size, num_particles, _ = positions.shape
    times = torch.as_tensor(
        particle_times, dtype=positions.dtype, device=positions.device
    )
    if times.ndim == 0:
        return times.expand(batch_size, num_particles)
    if times.ndim == 1:
        if times.shape[0] == batch_size:
            return times[:, None].expand(batch_size, num_particles)
        if times.shape[0] == num_particles:
            return times[None, :].expand(batch_size, num_particles)
    if times.ndim <= 2:
        try:
            return torch.broadcast_to(times, (batch_size, num_particles))
        except RuntimeError as error:
            raise ValueError(
                "particle_times must be scalar or broadcastable to [B, N]"
            ) from error
    raise ValueError("particle_times must be scalar or broadcastable to [B, N]")


def _time_indices_and_weights(
    particle_times: Tensor, velocity_times: Tensor, num_frames: int
) -> tuple[Tensor, Tensor, Tensor]:
    if num_frames == 1:
        zeros = torch.zeros_like(particle_times, dtype=torch.long)
        return zeros, zeros, torch.zeros_like(particle_times)

    query = particle_times.clamp(velocity_times[0], velocity_times[-1]).contiguous()
    upper = torch.searchsorted(velocity_times, query, right=False)
    upper = upper.clamp(1, num_frames - 1)
    lower = upper - 1
    t_lower = velocity_times[lower]
    t_upper = velocity_times[upper]
    weight = (query - t_lower) / (t_upper - t_lower)
    return lower, upper, weight


def _axis_indices(
    coordinate: Tensor,
    lower: Tensor,
    upper: Tensor,
    size: int,
    boundary_mode: BoundaryMode,
) -> tuple[Tensor, Tensor, Tensor]:
    if size < 1:
        raise ValueError("spatial velocity dimensions must be non-empty")
    if size == 1:
        zero_index = torch.zeros_like(coordinate, dtype=torch.long)
        return zero_index, zero_index, torch.zeros_like(coordinate)

    extent = upper - lower
    if boundary_mode == "periodic":
        # Periodic snapshots contain samples at lower + j * extent / size,
        # j=0,...,size-1; the duplicated upper endpoint is intentionally absent.
        wrapped = torch.remainder(coordinate - lower, extent)
        continuous_index = wrapped * (size / extent)
        index0_unwrapped = torch.floor(continuous_index).to(torch.long)
        index0 = torch.remainder(index0_unwrapped, size)
        index1 = torch.remainder(index0 + 1, size)
        weight = continuous_index - index0_unwrapped.to(continuous_index.dtype)
        return index0, index1, weight

    # For clamp/terminate, samples include both physical endpoints.  This is
    # equivalent to the align_corners=True normalized coordinate relation
    # g = 2 * (x - lower) / (upper - lower) - 1 and
    # continuous_index = (g + 1) * (size - 1) / 2.
    safe_coordinate = coordinate.clamp(lower, upper)
    continuous_index = (safe_coordinate - lower) * ((size - 1) / extent)
    index0 = torch.floor(continuous_index).to(torch.long)
    index1 = (index0 + 1).clamp_max(size - 1)
    weight = continuous_index - index0.to(continuous_index.dtype)
    return index0, index1, weight


def _gather_velocity(
    flattened_field: Tensor,
    time_index: Tensor,
    z_index: Tensor,
    y_index: Tensor,
    x_index: Tensor,
    height: int,
    width: int,
) -> Tensor:
    spatial_index = z_index * (height * width) + y_index * width + x_index
    return flattened_field[time_index, spatial_index]


def _trilinear_at_time(
    flattened_field: Tensor,
    time_index: Tensor,
    z0: Tensor,
    z1: Tensor,
    y0: Tensor,
    y1: Tensor,
    x0: Tensor,
    x1: Tensor,
    wz: Tensor,
    wy: Tensor,
    wx: Tensor,
    height: int,
    width: int,
) -> Tensor:
    c000 = _gather_velocity(
        flattened_field, time_index, z0, y0, x0, height, width
    )
    c001 = _gather_velocity(
        flattened_field, time_index, z0, y0, x1, height, width
    )
    c010 = _gather_velocity(
        flattened_field, time_index, z0, y1, x0, height, width
    )
    c011 = _gather_velocity(
        flattened_field, time_index, z0, y1, x1, height, width
    )
    c100 = _gather_velocity(
        flattened_field, time_index, z1, y0, x0, height, width
    )
    c101 = _gather_velocity(
        flattened_field, time_index, z1, y0, x1, height, width
    )
    c110 = _gather_velocity(
        flattened_field, time_index, z1, y1, x0, height, width
    )
    c111 = _gather_velocity(
        flattened_field, time_index, z1, y1, x1, height, width
    )

    wx = wx[:, None]
    wy = wy[:, None]
    wz = wz[:, None]
    c00 = c000 + wx * (c001 - c000)
    c01 = c010 + wx * (c011 - c010)
    c10 = c100 + wx * (c101 - c100)
    c11 = c110 + wx * (c111 - c110)
    c0 = c00 + wy * (c01 - c00)
    c1 = c10 + wy * (c11 - c10)
    return c0 + wz * (c1 - c0)


@overload
def sample_velocity(
    velocity_field: Tensor,
    particle_positions: Tensor,
    particle_times: Tensor | float,
    domain_bounds: Tensor | Sequence[Sequence[float]],
    boundary_mode: BoundaryMode = "clamp",
    *,
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: Literal[False] = False,
) -> Tensor: ...


@overload
def sample_velocity(
    velocity_field: Tensor,
    particle_positions: Tensor,
    particle_times: Tensor | float,
    domain_bounds: Tensor | Sequence[Sequence[float]],
    boundary_mode: BoundaryMode = "clamp",
    *,
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: Literal[True],
) -> tuple[Tensor, Tensor]: ...


def sample_velocity(
    velocity_field: Tensor,
    particle_positions: Tensor,
    particle_times: Tensor | float,
    domain_bounds: Tensor | Sequence[Sequence[float]],
    boundary_mode: BoundaryMode = "clamp",
    *,
    velocity_times: Tensor | Sequence[float] | None = None,
    return_validity: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample a gridded velocity using linear time and trilinear space interpolation.

    Args:
        velocity_field: Tensor ``[T, 3, D, H, W]``.
        particle_positions: Physical ``(x, y, z)`` positions ``[B, N, 3]``.
        particle_times: Scalar or tensor broadcastable to ``[B, N]``.
        domain_bounds: ``[3, 2]`` bounds in x, y, z order.
        boundary_mode: ``periodic`` wraps coordinates, ``clamp`` samples at the
            closest boundary point, and ``terminate`` returns zero velocity for
            an out-of-domain particle.  Both nonperiodic modes report such a
            particle as invalid when ``return_validity=True``.
        velocity_times: Physical times of the T snapshots.  If omitted, integer
            snapshot times ``0, ..., T-1`` are used. Queries outside this range
            use the nearest temporal endpoint.
        return_validity: Also return a boolean ``[B, N]`` in-domain mask.

    Returns:
        Interpolated velocities ``[B, N, 3]``, optionally with a validity mask.

    Notes:
        For nonperiodic axes, physical position x maps to normalized grid
        coordinate ``g_x = 2 (x-xmin)/(xmax-xmin)-1`` and continuous W-index
        ``(g_x+1)(W-1)/2``.  y maps to H and z maps to D.  Periodic grids omit
        the duplicated upper endpoint and therefore use W, H, or D rather than
        ``size-1`` in the corresponding continuous-index formula.

        All interpolation arithmetic is implemented with PyTorch tensor
        operations. Gradients flow to both ``velocity_field`` and in-domain
        ``particle_positions`` (piecewise linearly at cell boundaries).
    """
    if boundary_mode not in {"periodic", "clamp", "terminate"}:
        raise ValueError(
            "boundary_mode must be one of 'periodic', 'clamp', or 'terminate'"
        )
    if velocity_field.ndim != 5 or velocity_field.shape[1] != 3:
        raise ValueError("velocity_field must have shape [T, 3, D, H, W]")
    if velocity_field.shape[0] < 1:
        raise ValueError("velocity_field must contain at least one time snapshot")
    if not velocity_field.is_floating_point():
        raise TypeError("velocity_field must be floating point")
    if particle_positions.ndim != 3 or particle_positions.shape[-1] != 3:
        raise ValueError("particle_positions must have shape [B, N, 3]")
    if not particle_positions.is_floating_point():
        raise TypeError("particle_positions must be floating point")
    if particle_positions.device != velocity_field.device:
        raise ValueError("velocity_field and particle_positions must be on one device")

    positions = particle_positions.to(dtype=velocity_field.dtype)
    bounds = _domain_tensor(domain_bounds, like=positions)
    lower, upper = bounds[:, 0], bounds[:, 1]
    finite = torch.isfinite(positions).all(dim=-1)
    in_domain = finite & ((positions >= lower) & (positions <= upper)).all(dim=-1)
    # Every finite coordinate is valid on a periodic domain because it wraps;
    # non-finite inputs remain explicitly invalid rather than being hidden by
    # the safe indexing coordinates below.
    validity = finite if boundary_mode == "periodic" else in_domain

    # Avoid invalid integer indices for NaN/Inf inputs. They remain invalid and
    # terminate mode zeros the associated result.
    safe_positions = torch.where(
        torch.isfinite(positions), positions, lower.view(1, 1, 3)
    )
    batch_size, num_particles, _ = safe_positions.shape
    num_frames, _, depth, height, width = velocity_field.shape

    times = _broadcast_particle_times(particle_times, safe_positions)
    if velocity_times is None:
        frame_times = torch.arange(
            num_frames, dtype=positions.dtype, device=positions.device
        )
    else:
        frame_times = torch.as_tensor(
            velocity_times, dtype=positions.dtype, device=positions.device
        )
        if frame_times.shape != (num_frames,):
            raise ValueError("velocity_times must have shape [T]")
        if num_frames > 1 and not bool(
            torch.all(frame_times[1:] > frame_times[:-1]).detach()
        ):
            raise ValueError("velocity_times must be strictly increasing")

    flat_times = times.reshape(-1)
    t0, t1, wt = _time_indices_and_weights(flat_times, frame_times, num_frames)
    flat_positions = safe_positions.reshape(-1, 3)
    x0, x1, wx = _axis_indices(
        flat_positions[:, 0], lower[0], upper[0], width, boundary_mode
    )
    y0, y1, wy = _axis_indices(
        flat_positions[:, 1], lower[1], upper[1], height, boundary_mode
    )
    z0, z1, wz = _axis_indices(
        flat_positions[:, 2], lower[2], upper[2], depth, boundary_mode
    )

    # [T, D*H*W, 3], ordered consistently with z-major, then y, then x.
    flattened_field = velocity_field.permute(0, 2, 3, 4, 1).reshape(
        num_frames, depth * height * width, 3
    )
    velocity0 = _trilinear_at_time(
        flattened_field,
        t0,
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
        wz,
        wy,
        wx,
        height,
        width,
    )
    velocity1 = _trilinear_at_time(
        flattened_field,
        t1,
        z0,
        z1,
        y0,
        y1,
        x0,
        x1,
        wz,
        wy,
        wx,
        height,
        width,
    )
    sampled = velocity0 + wt[:, None] * (velocity1 - velocity0)
    sampled = sampled.reshape(batch_size, num_particles, 3)
    if boundary_mode == "terminate":
        sampled = sampled * validity[..., None].to(sampled.dtype)
    return (sampled, validity) if return_validity else sampled
