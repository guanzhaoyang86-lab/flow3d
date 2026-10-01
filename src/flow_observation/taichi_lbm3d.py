"""Axis and geometry adapters for the external Taichi-LBM3D solver.

The upstream solver stores velocity as `[x, y, z, component]`. The
differentiable observation code deliberately uses a different, explicit
convention: `[time, component, z, y, x]`. Keeping this conversion in one
small module makes the external-solver boundary testable without importing
Taichi or requiring a GPU.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to a torch.Tensor") from error
    return tensor


def taichi_velocity_to_canonical(raw_velocity: Any) -> torch.Tensor:
    """Convert Taichi-LBM3D velocity to `[T, 3, D, H, W]`.

    Args:
        raw_velocity: One upstream snapshot `[nx, ny, nz, 3]` or a sequence
            of snapshots `[T, nx, ny, nz, 3]`. Components are `(u_x,
            u_y, u_z)` and velocities remain in lattice cells per LBM step.

    Returns:
        A contiguous floating tensor with spatial axes `(z, y, x)`. A
        single input snapshot receives a leading time dimension of length one.
    """

    velocity = _as_tensor(raw_velocity, "raw_velocity")
    if velocity.ndim not in (4, 5):
        raise ValueError(
            "raw_velocity must have shape [nx, ny, nz, 3] or "
            f"[T, nx, ny, nz, 3]; received {tuple(velocity.shape)}"
        )
    if velocity.shape[-1] != 3:
        raise ValueError(
            "the final raw_velocity axis must contain (u_x, u_y, u_z); "
            f"received size {velocity.shape[-1]}"
        )
    spatial_shape = velocity.shape[-4:-1]
    if any(size < 2 for size in spatial_shape):
        raise ValueError("nx, ny, and nz must each be at least 2")
    if not velocity.is_floating_point():
        raise TypeError("raw_velocity must have a floating-point dtype")
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError("raw_velocity must contain only finite values")

    if velocity.ndim == 4:
        canonical = velocity.permute(3, 2, 1, 0).unsqueeze(0)
    else:
        canonical = velocity.permute(0, 4, 3, 2, 1)
    return canonical.contiguous()


def taichi_solid_mask_to_canonical(raw_solid: Any) -> torch.Tensor:
    """Convert an upstream `[nx, ny, nz]` solid array to `[D, H, W]`.

    The returned Boolean mask is true for solid lattice nodes.
    """

    solid = _as_tensor(raw_solid, "raw_solid")
    if solid.ndim != 3:
        raise ValueError(
            "raw_solid must have shape [nx, ny, nz]; " f"received {tuple(solid.shape)}"
        )
    if any(size < 2 for size in solid.shape):
        raise ValueError("nx, ny, and nz must each be at least 2")
    if solid.is_floating_point() and not bool(torch.isfinite(solid).all()):
        raise ValueError("raw_solid must contain only finite values")
    if not bool(((solid == 0) | (solid == 1)).all()):
        raise ValueError("raw_solid must be binary with 0=fluid and 1=solid")
    return solid.to(dtype=torch.bool).permute(2, 1, 0).contiguous()


def make_lid_driven_cavity_geometry(
    nx: int,
    ny: int | None = None,
    nz: int | None = None,
) -> torch.Tensor:
    """Build the upstream cavity geometry in native `[x, y, z]` order.

    Five walls are solid. The interior of the `x=max` face is fluid so the
    solver's fixed-velocity boundary condition can act as the moving lid. Its
    perimeter remains solid, matching the repository's bundled 50-cubed
    `geo_cavity.dat` example.
    """

    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    sizes = (nx, ny, nz)
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise TypeError("nx, ny, and nz must be integers")
    if min(sizes) < 4:
        raise ValueError("nx, ny, and nz must each be at least 4")

    geometry = torch.zeros((nx, ny, nz), dtype=torch.int8)
    geometry[0, :, :] = 1
    geometry[-1, :, :] = 1
    geometry[:, 0, :] = 1
    geometry[:, -1, :] = 1
    geometry[:, :, 0] = 1
    geometry[:, :, -1] = 1
    geometry[-1, 1:-1, 1:-1] = 0
    return geometry


__all__ = [
    "make_lid_driven_cavity_geometry",
    "taichi_solid_mask_to_canonical",
    "taichi_velocity_to_canonical",
]
