"""Building blocks for coarse-grid velocity-field recovery baselines."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _spatial_shape(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in values)
    if len(shape) != 3 or any(value < 2 for value in shape):
        raise ValueError(f"{name} must contain three integers, each at least 2")
    return shape


class CoarseVelocityField(nn.Module):
    """Bounded coarse velocity parameters decoded onto a dense CFD grid.

    The trainable tensor follows ``[1, 3, D, H, W]`` order. Trilinear
    upsampling uses ``align_corners=True`` so the coarse and dense endpoints
    describe the same physical box. Optional fixed voxels provide hard wall or
    known moving-lid boundary conditions without adding penalty tuning.
    """

    def __init__(
        self,
        dense_shape: Sequence[int],
        coarse_shape: Sequence[int] = (8, 8, 8),
        *,
        max_speed: float = 0.1,
        fixed_mask: Tensor | None = None,
        fixed_values: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dense_shape = _spatial_shape(dense_shape, name="dense_shape")
        self.coarse_shape = _spatial_shape(coarse_shape, name="coarse_shape")
        if not math.isfinite(max_speed) or max_speed <= 0.0:
            raise ValueError("max_speed must be finite and strictly positive")
        self.max_speed = float(max_speed)

        parameter = torch.zeros((1, 3, *self.coarse_shape), dtype=dtype, device=device)
        self.raw_velocity = nn.Parameter(parameter)

        if fixed_mask is None:
            mask = torch.zeros(self.dense_shape, dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(fixed_mask, dtype=torch.bool, device=device)
            if tuple(mask.shape) != self.dense_shape:
                raise ValueError(
                    "fixed_mask must match dense_shape; "
                    f"expected {self.dense_shape}, got {tuple(mask.shape)}"
                )

        if fixed_values is None:
            values = torch.zeros((1, 3, *self.dense_shape), dtype=dtype, device=device)
        else:
            values = torch.as_tensor(fixed_values, dtype=dtype, device=device)
            if values.ndim == 4:
                values = values.unsqueeze(0)
            expected = (1, 3, *self.dense_shape)
            if tuple(values.shape) != expected:
                raise ValueError(
                    "fixed_values must have shape [3,D,H,W] or [1,3,D,H,W]; "
                    f"expected {expected}, got {tuple(values.shape)}"
                )
            if not bool(torch.isfinite(values).all().detach()):
                raise ValueError("fixed_values must contain only finite values")

        self.register_buffer("fixed_mask", mask)
        self.register_buffer("fixed_values", values)

    def coarse_velocity(self) -> Tensor:
        """Return bounded physical coarse-grid velocities."""

        return self.max_speed * torch.tanh(self.raw_velocity)

    def forward(self) -> Tensor:
        dense = F.interpolate(
            self.coarse_velocity(),
            size=self.dense_shape,
            mode="trilinear",
            align_corners=True,
        )
        return torch.where(
            self.fixed_mask[None, None],
            self.fixed_values,
            dense,
        )


def project_trajectory_views(
    trajectories_3d: Tensor, projection_matrices: Tensor
) -> Tensor:
    """Project trajectories into shared views.

    Args:
        trajectories_3d: Positions ``[B,N,T,3]``.
        projection_matrices: Orthographic camera matrices ``[V,2,3]``.

    Returns:
        Projected positions ``[B,V,N,T,2]``.
    """

    if trajectories_3d.ndim != 4 or trajectories_3d.shape[-1] != 3:
        raise ValueError("trajectories_3d must have shape [B,N,T,3]")
    if not trajectories_3d.is_floating_point():
        raise TypeError("trajectories_3d must be floating point")
    matrices = torch.as_tensor(
        projection_matrices,
        dtype=trajectories_3d.dtype,
        device=trajectories_3d.device,
    )
    if matrices.ndim != 3 or tuple(matrices.shape[-2:]) != (2, 3):
        raise ValueError("projection_matrices must have shape [V,2,3]")
    return torch.einsum("bntc,voc->bvnto", trajectories_3d, matrices)


def masked_trajectory_mse(
    predicted: Tensor, observed: Tensor, observation_mask: Tensor | None = None
) -> Tensor:
    """Return mean squared error per observed projected coordinate."""

    if predicted.shape != observed.shape or predicted.ndim != 5:
        raise ValueError("predicted and observed must share shape [B,V,N,T,2]")
    if predicted.shape[-1] != 2:
        raise ValueError("the final predicted/observed dimension must be 2")
    if not predicted.is_floating_point() or not observed.is_floating_point():
        raise TypeError("predicted and observed trajectories must be floating point")
    observed = observed.to(device=predicted.device, dtype=predicted.dtype)
    target_shape = predicted.shape[:-1]
    if observation_mask is None:
        weights = torch.ones(
            target_shape, dtype=predicted.dtype, device=predicted.device
        )
    else:
        weights = torch.as_tensor(observation_mask, device=predicted.device)
        try:
            weights = torch.broadcast_to(weights, target_shape)
        except RuntimeError as error:
            raise ValueError(
                "observation_mask must be broadcastable to [B,V,N,T]"
            ) from error
        if weights.dtype == torch.bool:
            weights = weights.to(dtype=predicted.dtype)
        elif weights.is_floating_point() or weights.dtype in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            weights = weights.to(dtype=predicted.dtype)
        else:
            raise TypeError("observation_mask must be boolean or numeric")
        if not bool(torch.isfinite(weights).all().detach()) or bool(
            torch.any(weights < 0).detach()
        ):
            raise ValueError("observation_mask must be finite and non-negative")

    weight_sum = weights.sum()
    if not bool((weight_sum > 0).detach()):
        raise ValueError("observation_mask must retain at least one observation")
    squared_error = (predicted - observed).square()
    squared_error = torch.where(
        weights[..., None] > 0,
        squared_error,
        torch.zeros_like(squared_error),
    )
    return (squared_error * weights[..., None]).sum() / (2.0 * weight_sum)


def spatial_smoothness_mse(
    velocity_field: Tensor, spatial_mask: Tensor | None = None
) -> Tensor:
    """Mean squared first difference over valid neighboring grid nodes."""

    if velocity_field.ndim != 5 or velocity_field.shape[1] != 3:
        raise ValueError("velocity_field must have shape [T,3,D,H,W]")
    mask = _validate_spatial_mask(spatial_mask, velocity_field)
    numerator = velocity_field.new_zeros(())
    denominator = velocity_field.new_zeros(())
    for axis in range(3):
        tensor_axis = axis + 2
        differences = torch.diff(velocity_field, dim=tensor_axis)
        if mask is None:
            numerator = numerator + differences.square().sum()
            denominator = denominator + differences.new_tensor(differences.numel())
            continue
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair_mask = mask[tuple(left)] & mask[tuple(right)]
        weights = pair_mask[None, None].to(dtype=differences.dtype)
        numerator = numerator + (differences.square() * weights).sum()
        denominator = denominator + weights.sum() * (
            velocity_field.shape[0] * velocity_field.shape[1]
        )
    if not bool((denominator > 0).detach()):
        raise ValueError("spatial_mask contains no valid neighboring nodes")
    return numerator / denominator


def divergence_mse(
    velocity_field: Tensor,
    domain_bounds: Tensor | Sequence[Sequence[float]],
    spatial_mask: Tensor | None = None,
) -> Tensor:
    """Mean squared central-difference divergence on interior grid nodes."""

    if velocity_field.ndim != 5 or velocity_field.shape[1] != 3:
        raise ValueError("velocity_field must have shape [T,3,D,H,W]")
    depth, height, width = velocity_field.shape[-3:]
    if min(depth, height, width) < 3:
        raise ValueError("each spatial velocity dimension must be at least 3")
    bounds = torch.as_tensor(
        domain_bounds, dtype=velocity_field.dtype, device=velocity_field.device
    )
    if bounds.shape != (3, 2) or not bool(
        torch.all(bounds[:, 1] > bounds[:, 0]).detach()
    ):
        raise ValueError("domain_bounds must contain increasing [x,y,z] bounds")
    dx = (bounds[0, 1] - bounds[0, 0]) / (width - 1)
    dy = (bounds[1, 1] - bounds[1, 0]) / (height - 1)
    dz = (bounds[2, 1] - bounds[2, 0]) / (depth - 1)

    ux = velocity_field[:, 0]
    uy = velocity_field[:, 1]
    uz = velocity_field[:, 2]
    divergence = (
        (ux[:, 1:-1, 1:-1, 2:] - ux[:, 1:-1, 1:-1, :-2]) / (2.0 * dx)
        + (uy[:, 1:-1, 2:, 1:-1] - uy[:, 1:-1, :-2, 1:-1]) / (2.0 * dy)
        + (uz[:, 2:, 1:-1, 1:-1] - uz[:, :-2, 1:-1, 1:-1]) / (2.0 * dz)
    )

    mask = _validate_spatial_mask(spatial_mask, velocity_field)
    if mask is None:
        return divergence.square().mean()
    valid = mask[1:-1, 1:-1, 1:-1].clone()
    valid &= mask[1:-1, 1:-1, 2:] & mask[1:-1, 1:-1, :-2]
    valid &= mask[1:-1, 2:, 1:-1] & mask[1:-1, :-2, 1:-1]
    valid &= mask[2:, 1:-1, 1:-1] & mask[:-2, 1:-1, 1:-1]
    if not bool(valid.any().detach()):
        raise ValueError("spatial_mask contains no valid divergence stencil")
    weights = valid[None].to(dtype=divergence.dtype)
    return (divergence.square() * weights).sum() / (
        weights.sum() * velocity_field.shape[0]
    )


def _validate_spatial_mask(mask: Tensor | None, field: Tensor) -> Tensor | None:
    if mask is None:
        return None
    result = torch.as_tensor(mask, dtype=torch.bool, device=field.device)
    if tuple(result.shape) != tuple(field.shape[-3:]):
        raise ValueError("spatial_mask must match velocity spatial dimensions")
    return result


__all__ = [
    "CoarseVelocityField",
    "divergence_mse",
    "masked_trajectory_mse",
    "project_trajectory_views",
    "spatial_smoothness_mse",
]
