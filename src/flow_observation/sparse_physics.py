"""Physics helpers for sparse, multi-view flow reconstruction.

The core observation package represents one steady velocity field as
``[1, 3, D, H, W]``.  A diffusion model, however, naturally emits a batch of
steady fields with shape ``[B, 3, D, H, W]``.  This module provides the small
batch adapters needed by that use case without changing the established
single-field interpolation and RK4 implementations.

No reference 3D trajectory or precomputed hidden depth is consumed here.
Initial particle positions are triangulated from synchronized projected
observations at time zero for every item in the batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from .advection import advect_particles
from .multiview import triangulate_initial_positions
from .recovery import (
    divergence_mse,
    masked_trajectory_mse,
    project_trajectory_views,
    spatial_smoothness_mse,
)


Reduction = Literal["none", "mean", "sum"]


def _validate_batched_field(velocity_fields: Tensor) -> tuple[int, int, int, int]:
    if not isinstance(velocity_fields, Tensor):
        raise TypeError("velocity_fields must be a torch.Tensor")
    if velocity_fields.ndim != 5 or velocity_fields.shape[1] != 3:
        raise ValueError("velocity_fields must have shape [B,3,D,H,W]")
    if velocity_fields.shape[0] < 1 or min(velocity_fields.shape[-3:]) < 2:
        raise ValueError("velocity_fields must contain a non-empty batch and grid")
    if not velocity_fields.is_floating_point():
        raise TypeError("velocity_fields must be floating point")
    return (
        velocity_fields.shape[0],
        velocity_fields.shape[-3],
        velocity_fields.shape[-2],
        velocity_fields.shape[-1],
    )


def _reduce_per_sample(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _broadcast_boundary_mask(mask: Tensor, field: Tensor) -> Tensor:
    batch, _, depth, height, width = field.shape
    spatial_shape = (depth, height, width)
    result = torch.as_tensor(mask, device=field.device)
    if result.dtype != torch.bool:
        if not (result.is_floating_point() or result.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }):
            raise TypeError("fixed_mask must be boolean or numeric")
        if not bool(torch.isfinite(result).all().detach()) or not bool(
            ((result == 0) | (result == 1)).all().detach()
        ):
            raise ValueError("numeric fixed_mask entries must be finite zeros or ones")
        result = result.to(dtype=torch.bool)

    if tuple(result.shape) == spatial_shape:
        result = result.reshape(1, 1, *spatial_shape)
    elif result.ndim == 4 and tuple(result.shape[-3:]) == spatial_shape:
        if result.shape[0] == 3 and batch == 3:
            raise ValueError(
                "rank-4 fixed_mask [3,D,H,W] is ambiguous when B == 3; "
                "use [1,3,D,H,W] for a shared component mask or "
                "[B,1,D,H,W] for per-sample spatial masks"
            )
        if result.shape[0] == 3:
            # Shared component-specific mask.  When B == 3, callers needing
            # three distinct spatial masks can disambiguate with [B,1,D,H,W].
            result = result[None]
        elif result.shape[0] in (1, batch):
            result = result[:, None]
        else:
            raise ValueError(
                "a rank-4 fixed_mask must have leading size 1, 3, or B"
            )
    elif result.ndim == 5 and tuple(result.shape[-3:]) == spatial_shape:
        if result.shape[0] not in (1, batch) or result.shape[1] not in (1, 3):
            raise ValueError(
                "a rank-5 fixed_mask must be broadcastable to [B,3,D,H,W]"
            )
    else:
        raise ValueError(
            "fixed_mask must have shape [D,H,W], [B,D,H,W], or a shape "
            "broadcastable to [B,3,D,H,W]"
        )
    return torch.broadcast_to(result, field.shape)


def _broadcast_boundary_values(values: Tensor, field: Tensor) -> Tensor:
    batch, _, depth, height, width = field.shape
    spatial_shape = (depth, height, width)
    result = torch.as_tensor(values, dtype=field.dtype, device=field.device)
    if result.ndim == 1 and tuple(result.shape) == (3,):
        result = result.reshape(1, 3, 1, 1, 1)
    elif result.ndim == 2 and result.shape[0] in (1, batch) and result.shape[1] == 3:
        result = result.reshape(result.shape[0], 3, 1, 1, 1)
    elif result.ndim == 4 and tuple(result.shape) == (3, *spatial_shape):
        result = result.unsqueeze(0)
    elif result.ndim == 5 and result.shape[0] in (1, batch) and result.shape[1] == 3:
        if any(
            actual not in (1, expected)
            for actual, expected in zip(result.shape[-3:], spatial_shape)
        ):
            raise ValueError(
                "fixed_values spatial dimensions must be 1 or match D,H,W"
            )
    else:
        raise ValueError(
            "fixed_values must contain three components and be broadcastable "
            "from [3], [B,3], [3,D,H,W], or [B,3,D,H,W]"
        )
    if not bool(torch.isfinite(result).all().detach()):
        raise ValueError("fixed_values must contain only finite values")
    try:
        return torch.broadcast_to(result, field.shape)
    except RuntimeError as error:
        raise ValueError(
            "fixed_values must be broadcastable to velocity_fields"
        ) from error


def apply_hard_boundary_conditions(
    velocity_fields: Tensor,
    fixed_mask: Tensor,
    fixed_values: Tensor,
) -> Tensor:
    """Overwrite known velocity nodes in a batch exactly.

    Args:
        velocity_fields: Candidate steady fields ``[B,3,D,H,W]``.
        fixed_mask: Shared ``[D,H,W]`` or batched spatial mask.  Component-wise
            masks broadcastable to the field are also accepted.  When the
            batch size is three, rank-4 ``[3,D,H,W]`` is rejected as ambiguous;
            use explicit ``[1,3,D,H,W]`` or ``[B,1,D,H,W]`` instead.
        fixed_values: A shared velocity vector ``[3]``, shared field
            ``[3,D,H,W]``, or batched values ``[B,3,D,H,W]``.

    Returns:
        A new tensor with exact prescribed values at every fixed entry.  The
        result remains differentiable on all non-fixed entries, while gradients
        through fixed entries are zero by construction.
    """

    _validate_batched_field(velocity_fields)
    mask = _broadcast_boundary_mask(fixed_mask, velocity_fields)
    values = _broadcast_boundary_values(fixed_values, velocity_fields)
    return torch.where(mask, values, velocity_fields)


def _sample_spatial_mask(
    spatial_mask: Tensor | None,
    sample_index: int,
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> Tensor | None:
    if spatial_mask is None:
        return None
    mask = torch.as_tensor(spatial_mask, dtype=torch.bool, device=device)
    if tuple(mask.shape) == spatial_shape:
        return mask
    if mask.ndim == 4 and tuple(mask.shape[1:]) == spatial_shape:
        if mask.shape[0] == 1:
            return mask[0]
        if mask.shape[0] == batch_size:
            return mask[sample_index]
    raise ValueError("spatial_mask must have shape [D,H,W] or [B,D,H,W]")


def batched_spatial_smoothness_mse(
    velocity_fields: Tensor,
    spatial_mask: Tensor | None = None,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Evaluate the existing smoothness penalty independently per field."""

    batch, depth, height, width = _validate_batched_field(velocity_fields)
    spatial_shape = (depth, height, width)
    losses = []
    for index in range(batch):
        mask = _sample_spatial_mask(
            spatial_mask, index, batch, spatial_shape, velocity_fields.device
        )
        losses.append(
            spatial_smoothness_mse(velocity_fields[index : index + 1], mask)
        )
    return _reduce_per_sample(torch.stack(losses), reduction)


def _sample_domain_bounds(
    domain_bounds: Tensor | Sequence[Sequence[float]],
    sample_index: int,
    batch_size: int,
    field: Tensor,
) -> Tensor:
    bounds = torch.as_tensor(
        domain_bounds, dtype=field.dtype, device=field.device
    )
    if tuple(bounds.shape) == (3, 2):
        return bounds
    if bounds.ndim == 3 and tuple(bounds.shape[1:]) == (3, 2):
        if bounds.shape[0] == 1:
            return bounds[0]
        if bounds.shape[0] == batch_size:
            return bounds[sample_index]
    raise ValueError("domain_bounds must have shape [3,2] or [B,3,2]")


def batched_divergence_mse(
    velocity_fields: Tensor,
    domain_bounds: Tensor | Sequence[Sequence[float]],
    spatial_mask: Tensor | None = None,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Evaluate central-difference divergence independently per field."""

    batch, depth, height, width = _validate_batched_field(velocity_fields)
    if min(depth, height, width) < 3:
        raise ValueError("each spatial field dimension must be at least 3")
    spatial_shape = (depth, height, width)
    losses = []
    for index in range(batch):
        bounds = _sample_domain_bounds(
            domain_bounds, index, batch, velocity_fields
        )
        mask = _sample_spatial_mask(
            spatial_mask, index, batch, spatial_shape, velocity_fields.device
        )
        losses.append(
            divergence_mse(velocity_fields[index : index + 1], bounds, mask)
        )
    return _reduce_per_sample(torch.stack(losses), reduction)


def _sample_projection_matrices(
    projection_matrices: Tensor,
    sample_index: int,
    batch_size: int,
    num_views: int,
    reference: Tensor,
) -> Tensor:
    matrices = torch.as_tensor(
        projection_matrices, dtype=reference.dtype, device=reference.device
    )
    if tuple(matrices.shape) == (num_views, 2, 3):
        return matrices
    if matrices.ndim == 4 and tuple(matrices.shape[1:]) == (num_views, 2, 3):
        if matrices.shape[0] == 1:
            return matrices[0]
        if matrices.shape[0] == batch_size:
            return matrices[sample_index]
    raise ValueError(
        "projection_matrices must have shape [V,2,3] or [B,V,2,3]"
    )


def _sample_times(
    values: Tensor | Sequence[float],
    sample_index: int,
    batch_size: int,
    expected_length: int | None,
    reference: Tensor,
    *,
    name: str,
) -> Tensor:
    times = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if times.ndim == 1:
        result = times
    elif times.ndim == 2 and times.shape[0] in (1, batch_size):
        result = times[0 if times.shape[0] == 1 else sample_index]
    else:
        raise ValueError(f"{name} must have shape [T] or [B,T]")
    if expected_length is not None and result.shape != (expected_length,):
        raise ValueError(f"{name} must contain {expected_length} entries")
    return result


def batched_multiview_replay(
    velocity_fields: Tensor,
    observed_trajectories: Tensor,
    projection_matrices: Tensor,
    observation_times: Tensor | Sequence[float],
    domain_bounds: Tensor | Sequence[Sequence[float]],
    *,
    observation_mask: Tensor | None = None,
    integrator: str = "rk4",
    num_substeps: int | Sequence[int] = 1,
    boundary_mode: str = "terminate",
    velocity_times: Tensor | Sequence[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Replay sparse synchronized observations through batched steady fields.

    The underlying advection routine accepts one field at a time, so this
    adapter deliberately loops over the model batch.  That preserves the
    already-tested interpolation/RK4 implementation and its autograd graph.

    Args:
        velocity_fields: Candidate fields ``[B,3,D,H,W]``.
        observed_trajectories: Synchronized tracks ``[B,V,N,T,2]``.  ``N=2``
            is fully supported.
        projection_matrices: Shared ``[V,2,3]`` or batched ``[B,V,2,3]``.
        observation_times: Shared ``[T]`` or batched ``[B,T]``.
        domain_bounds: Shared ``[3,2]`` or batched ``[B,3,2]``.
        observation_mask: Optional mask broadcastable to ``[B,V,N,T]``.  The
            time-zero slice determines which views triangulate each particle.

    Returns:
        Predicted projected trajectories ``[B,V,N,T,2]`` and cumulative
        particle validity ``[B,N,T]``.
    """

    batch, _, _, _ = _validate_batched_field(velocity_fields)
    if not isinstance(observed_trajectories, Tensor):
        raise TypeError("observed_trajectories must be a torch.Tensor")
    if observed_trajectories.ndim != 5 or observed_trajectories.shape[-1] != 2:
        raise ValueError(
            "observed_trajectories must have shape [B,V,N,T,2]"
        )
    if observed_trajectories.shape[0] != batch:
        raise ValueError("field and observation batch dimensions must match")
    if not observed_trajectories.is_floating_point():
        raise TypeError("observed_trajectories must be floating point")
    observations = observed_trajectories.to(
        device=velocity_fields.device, dtype=velocity_fields.dtype
    )
    _, num_views, _, num_observation_times, _ = observations.shape
    replay_mask = None
    if observation_mask is not None:
        replay_mask = torch.as_tensor(observation_mask, device=observations.device)
        if replay_mask.dtype != torch.bool:
            raise TypeError("observation_mask must be boolean")
        try:
            replay_mask = torch.broadcast_to(
                replay_mask, observations.shape[:-1]
            ).clone()
        except RuntimeError as error:
            raise ValueError(
                "observation_mask must be broadcastable to [B,V,N,T]"
            ) from error

    predicted_batches: list[Tensor] = []
    validity_batches: list[Tensor] = []
    for index in range(batch):
        cameras = _sample_projection_matrices(
            projection_matrices,
            index,
            batch,
            num_views,
            velocity_fields,
        )
        times = _sample_times(
            observation_times,
            index,
            batch,
            num_observation_times,
            velocity_fields,
            name="observation_times",
        )
        bounds = _sample_domain_bounds(
            domain_bounds, index, batch, velocity_fields
        )
        frame_times = None
        if velocity_times is not None:
            frame_times = _sample_times(
                velocity_times,
                index,
                batch,
                1,
                velocity_fields,
                name="velocity_times",
            )

        sample_observations = observations[index : index + 1]
        initial_positions = triangulate_initial_positions(
            sample_observations,
            cameras,
            None if replay_mask is None else replay_mask[index : index + 1],
        )
        trajectories_3d, validity = advect_particles(
            velocity_fields[index : index + 1],
            initial_positions,
            times,
            integrator=integrator,
            num_substeps=num_substeps,
            domain_bounds=bounds,
            boundary_mode=boundary_mode,
            velocity_times=frame_times,
            return_validity=True,
        )
        predicted_batches.append(project_trajectory_views(trajectories_3d, cameras))
        validity_batches.append(validity)

    return torch.cat(predicted_batches, dim=0), torch.cat(validity_batches, dim=0)


def batched_multiview_trajectory_consistency(
    velocity_fields: Tensor,
    observed_trajectories: Tensor,
    projection_matrices: Tensor,
    observation_times: Tensor | Sequence[float],
    domain_bounds: Tensor | Sequence[Sequence[float]],
    *,
    observation_mask: Tensor | None = None,
    exclude_initial_observation: bool = True,
    integrator: str = "rk4",
    num_substeps: int | Sequence[int] = 1,
    boundary_mode: str = "terminate",
    velocity_times: Tensor | Sequence[float] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return trajectory-consistency loss, replay tracks, and validity.

    Invalid trajectories are intentionally *not* removed automatically from
    the loss: otherwise a candidate could reduce its objective by pushing
    particles outside the domain.  Callers receive the validity history and
    may add an explicit invalid-particle penalty if desired.
    """

    predicted, validity = batched_multiview_replay(
        velocity_fields,
        observed_trajectories,
        projection_matrices,
        observation_times,
        domain_bounds,
        observation_mask=observation_mask,
        integrator=integrator,
        num_substeps=num_substeps,
        boundary_mode=boundary_mode,
        velocity_times=velocity_times,
    )
    observed = observed_trajectories.to(
        device=predicted.device, dtype=predicted.dtype
    )
    if observation_mask is None:
        mask = torch.ones(
            observed.shape[:-1], dtype=torch.bool, device=observed.device
        )
    else:
        mask = torch.as_tensor(observation_mask, device=observed.device)
        try:
            mask = torch.broadcast_to(mask, observed.shape[:-1]).clone()
        except RuntimeError as error:
            raise ValueError(
                "observation_mask must be broadcastable to [B,V,N,T]"
            ) from error
    if exclude_initial_observation:
        mask[..., 0] = 0
        if not bool((mask != 0).any().detach()):
            raise ValueError(
                "excluding time zero leaves no trajectory observation in the loss"
            )
    loss = masked_trajectory_mse(predicted, observed, mask)
    return loss, predicted, validity


# Short alias for training code that does not need the geometry qualifier.
batched_trajectory_consistency = batched_multiview_trajectory_consistency


__all__ = [
    "apply_hard_boundary_conditions",
    "batched_divergence_mse",
    "batched_multiview_replay",
    "batched_multiview_trajectory_consistency",
    "batched_spatial_smoothness_mse",
    "batched_trajectory_consistency",
]
