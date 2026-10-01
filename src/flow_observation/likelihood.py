"""Projected Lagrangian-trajectory observation likelihood."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .advection import advect_particles
from .projection import lift_projected_initial_positions, project_trajectories


def _prepare_observation_mask(mask: Tensor, target: Tensor) -> Tensor:
    """Return a mask broadcast to target.shape == [B, N, T_obs]."""

    mask = torch.as_tensor(mask, device=target.device)
    if mask.ndim == 4:
        if mask.shape[-1] != 1:
            raise ValueError(
                "observation_mask describes whole 2D observations and must "
                "not have a non-singleton coordinate dimension"
            )
        mask = mask.squeeze(-1)
    if mask.ndim == 1:
        mask = mask.reshape(1, 1, -1)
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 3:
        raise ValueError(
            "observation_mask must have shape [T_obs], [N, T_obs], "
            "[B, N, T_obs], or [B, N, T_obs, 1]; "
            f"got {tuple(mask.shape)}"
        )
    try:
        mask = torch.broadcast_to(mask, target.shape)
    except RuntimeError as error:
        raise ValueError(
            "observation_mask is not broadcastable to projected observations "
            f"with shape {tuple(target.shape)}; got {tuple(mask.shape)}"
        ) from error

    if mask.dtype == torch.bool:
        return mask.to(dtype=target.dtype)
    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if not (mask.is_floating_point() or mask.dtype in integer_dtypes):
        raise TypeError("observation_mask must be boolean or numeric")
    mask = mask.to(dtype=target.dtype)
    if not bool(torch.all(torch.isfinite(mask)).detach()):
        raise ValueError("observation_mask must contain only finite values")
    if bool(torch.any(mask < 0).detach()):
        raise ValueError("observation_mask weights must be non-negative")
    return mask


def projected_trajectory_nll(
    velocity_field: Tensor,
    projected_trajectories: Tensor,
    projection_matrix: Tensor,
    hidden_depth: Tensor,
    observation_times: Tensor,
    sigma: float | Tensor,
    observation_mask: Tensor | None = None,
    integrator: str = "rk4",
    num_substeps: int | Sequence[int] = 1,
    domain_bounds: Tensor | Sequence[Sequence[float]] | None = None,
    boundary_mode: str = "clamp",
    velocity_times: Tensor | None = None,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Evaluate the projected-trajectory Gaussian negative log-likelihood.

    This function composes the complete differentiable observation operator:
    projected initial position -> 3D lift -> particle advection -> camera
    projection -> masked squared error. The returned scalar is exactly

        sum(mask * ||y_obs - y_pred||^2) / (2 * sigma**2).

    No mean reduction or Gaussian normalizing constant is added. Invalid
    particles reported by advect_particles are deliberately not hidden by an
    implicit mask; callers can inspect validity_mask in diagnostics and decide
    explicitly how those observations should be handled.

    Args:
        velocity_field: Eulerian snapshots with shape [T_v, 3, D, H, W].
        projected_trajectories: Observations of shape [B, N, T_obs, 2].
        projection_matrix: Shared [2, 3] or batched [B, 2, 3] camera.
        hidden_depth: Initial null-space coordinate with shape [B, N] or
            [B, N, 1]. A tensor with requires_grad=True is supported.
        observation_times: Times corresponding to the observation axis.
        sigma: Positive scalar localization-noise standard deviation.
        observation_mask: Optional non-negative mask/weight with shape
            [B, N, T_obs] (or a broadcastable abbreviated shape).
        integrator: "euler" or "rk4"; passed to advection.
        num_substeps: Integration substeps per observation interval.
        domain_bounds: Physical lower/upper bounds used by advection.
        boundary_mode: "periodic", "clamp", or "terminate".
        velocity_times: Optional times corresponding to velocity snapshots.
        return_diagnostics: If true, return (loss, diagnostics).

    Returns:
        A scalar loss, or (loss, diagnostics). Diagnostics contains
        trajectories_3d, trajectories_2d, and validity_mask.
    """

    if not isinstance(velocity_field, Tensor):
        raise TypeError("velocity_field must be a torch.Tensor")
    if not velocity_field.is_floating_point():
        raise TypeError("velocity_field must have a floating-point dtype")
    if not isinstance(projected_trajectories, Tensor):
        raise TypeError("projected_trajectories must be a torch.Tensor")
    if projected_trajectories.ndim != 4 or projected_trajectories.shape[-1] != 2:
        raise ValueError(
            "projected_trajectories must have shape [B, N, T_obs, 2]; "
            f"got {tuple(projected_trajectories.shape)}"
        )
    if not projected_trajectories.is_floating_point():
        raise TypeError("projected_trajectories must have a floating-point dtype")
    if projected_trajectories.shape[2] < 1:
        raise ValueError("projected_trajectories must contain at least one time")

    observations = projected_trajectories.to(
        device=velocity_field.device, dtype=velocity_field.dtype
    )
    times = torch.as_tensor(
        observation_times, device=velocity_field.device, dtype=velocity_field.dtype
    )
    if times.ndim != 1:
        raise ValueError(
            "observation_times must be one-dimensional; "
            f"got shape {tuple(times.shape)}"
        )
    if times.numel() != observations.shape[2]:
        raise ValueError(
            "observation_times length must equal T_obs = "
            f"{observations.shape[2]}; got {times.numel()}"
        )

    sigma_tensor = torch.as_tensor(
        sigma, device=velocity_field.device, dtype=velocity_field.dtype
    )
    if sigma_tensor.numel() != 1:
        raise ValueError(f"sigma must be scalar; got shape {tuple(sigma_tensor.shape)}")
    sigma_tensor = sigma_tensor.reshape(())
    if not bool(torch.isfinite(sigma_tensor).detach()) or bool(
        (sigma_tensor <= 0).detach()
    ):
        raise ValueError("sigma must be finite and strictly positive")

    x0 = lift_projected_initial_positions(
        observations[:, :, 0, :], projection_matrix, hidden_depth
    )
    advection_result = advect_particles(
        velocity_field,
        x0,
        times,
        integrator=integrator,
        num_substeps=num_substeps,
        domain_bounds=domain_bounds,
        boundary_mode=boundary_mode,
        velocity_times=velocity_times,
        return_validity=True,
    )
    if not isinstance(advection_result, tuple) or len(advection_result) != 2:
        raise RuntimeError(
            "advect_particles(..., return_validity=True) must return "
            "(trajectories_3d, validity_mask)"
        )
    trajectories_3d, validity_mask = advection_result
    trajectories_2d = project_trajectories(trajectories_3d, projection_matrix)

    squared_distance = (observations - trajectories_2d).square().sum(dim=-1)
    if observation_mask is not None:
        mask = _prepare_observation_mask(observation_mask, squared_distance)
        squared_distance = squared_distance * mask
    loss = squared_distance.sum() / (2 * sigma_tensor.square())

    if not return_diagnostics:
        return loss
    diagnostics = {
        "trajectories_3d": trajectories_3d,
        "trajectories_2d": trajectories_2d,
        "validity_mask": validity_mask,
        "predicted_trajectories_3d": trajectories_3d,
        "predicted_trajectories_2d": trajectories_2d,
    }
    return loss, diagnostics


__all__ = ["projected_trajectory_nll"]
