"""Differentiable camera projection and projected-position lifting.

The camera convention used throughout this package is y = P @ x, where x is
a column vector in R^3 and P has shape [2, 3]. Tensor storage follows the
usual PyTorch convention with coordinates in the last dimension, so
projection is implemented as a right multiplication by P.T.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _projection_for_batch(
    projection_matrix: Tensor,
    *,
    batch_size: int,
    reference: Tensor,
) -> Tensor:
    """Validate projection_matrix and return it as [B, 2, 3]."""

    projection_matrix = torch.as_tensor(
        projection_matrix, dtype=reference.dtype, device=reference.device
    )
    if projection_matrix.ndim == 2:
        if projection_matrix.shape != (2, 3):
            raise ValueError(
                "projection_matrix must have shape [2, 3] or [B, 2, 3]; "
                f"got {tuple(projection_matrix.shape)}"
            )
        projection_matrix = projection_matrix.unsqueeze(0)
    elif projection_matrix.ndim == 3:
        if projection_matrix.shape[-2:] != (2, 3):
            raise ValueError(
                "projection_matrix must have shape [2, 3] or [B, 2, 3]; "
                f"got {tuple(projection_matrix.shape)}"
            )
    else:
        raise ValueError(
            "projection_matrix must have shape [2, 3] or [B, 2, 3]; "
            f"got a rank-{projection_matrix.ndim} tensor"
        )

    matrix_batch = projection_matrix.shape[0]
    if matrix_batch == 1:
        return projection_matrix.expand(batch_size, -1, -1)
    if matrix_batch != batch_size:
        raise ValueError(
            "The projection-matrix batch dimension must be 1 or match the "
            f"trajectory batch dimension ({batch_size}); got {matrix_batch}"
        )
    return projection_matrix


def project_trajectories(
    trajectories_3d: Tensor,
    projection_matrix: Tensor,
) -> Tensor:
    """Project batched 3D trajectories into the camera plane.

    Args:
        trajectories_3d: Particle positions with shape [B, N, T_obs, 3].
        projection_matrix: A shared [2, 3] matrix or per-example matrices
            with shape [B, 2, 3].

    Returns:
        Projected trajectories with shape [B, N, T_obs, 2].

    The operation is differentiable with respect to both arguments.
    """

    if not isinstance(trajectories_3d, Tensor):
        raise TypeError("trajectories_3d must be a torch.Tensor")
    if trajectories_3d.ndim != 4 or trajectories_3d.shape[-1] != 3:
        raise ValueError(
            "trajectories_3d must have shape [B, N, T_obs, 3]; "
            f"got {tuple(trajectories_3d.shape)}"
        )
    if not trajectories_3d.is_floating_point():
        raise TypeError("trajectories_3d must have a floating-point dtype")

    projection_matrix = _projection_for_batch(
        projection_matrix,
        batch_size=trajectories_3d.shape[0],
        reference=trajectories_3d,
    )
    return torch.einsum("bntc,boc->bnto", trajectories_3d, projection_matrix)


def lift_projected_initial_positions(
    projected_x0: Tensor,
    projection_matrix: Tensor,
    hidden_depth: Tensor,
) -> Tensor:
    """Lift projected initial positions to 3D using one hidden coordinate.

    The lift is x0 = pinv(P) @ y0 + n(P) * z, where n(P) is the unit
    vector spanning the one-dimensional null space of a full-row-rank 2 x 3
    projection matrix. n(P) is obtained from the final right singular vector
    of P. Since singular vectors are defined only up to sign, its
    largest-magnitude component is oriented to be non-negative. Consequently
    hidden_depth is a coordinate in this deterministic null-space basis; for
    the standard xy camera it is the z coordinate.

    Args:
        projected_x0: Initial image-plane locations, shape [B, N, 2].
        projection_matrix: Shared [2, 3] or batched [B, 2, 3] matrix.
        hidden_depth: Null-space coordinate, shape [B, N] or [B, N, 1].
            It may be a trainable tensor.

    Returns:
        Lifted positions with shape [B, N, 3].

    Projection matrices must have rank two because a scalar hidden depth can
    parameterize only a one-dimensional null space. All numerical work uses
    PyTorch, preserving gradients to projected_x0 and hidden_depth.
    """

    if not isinstance(projected_x0, Tensor):
        raise TypeError("projected_x0 must be a torch.Tensor")
    if projected_x0.ndim != 3 or projected_x0.shape[-1] != 2:
        raise ValueError(
            "projected_x0 must have shape [B, N, 2]; "
            f"got {tuple(projected_x0.shape)}"
        )
    if not projected_x0.is_floating_point():
        raise TypeError("projected_x0 must have a floating-point dtype")

    batch_size, num_particles, _ = projected_x0.shape
    projection_matrix = _projection_for_batch(
        projection_matrix,
        batch_size=batch_size,
        reference=projected_x0,
    )

    hidden_depth = torch.as_tensor(
        hidden_depth, dtype=projected_x0.dtype, device=projected_x0.device
    )
    if hidden_depth.ndim == 3:
        if hidden_depth.shape[-1] != 1:
            raise ValueError(
                "hidden_depth must have shape [B, N] or [B, N, 1]; "
                f"got {tuple(hidden_depth.shape)}"
            )
        hidden_depth = hidden_depth.squeeze(-1)
    if hidden_depth.ndim != 2:
        raise ValueError(
            "hidden_depth must have shape [B, N] or [B, N, 1]; "
            f"got {tuple(hidden_depth.shape)}"
        )
    if hidden_depth.shape[0] not in (1, batch_size) or hidden_depth.shape[1] not in (
        1,
        num_particles,
    ):
        raise ValueError(
            "hidden_depth dimensions must be broadcastable to [B, N] = "
            f"[{batch_size}, {num_particles}]; got {tuple(hidden_depth.shape)}"
        )
    hidden_depth = hidden_depth.expand(batch_size, num_particles)

    # pinv/SVD do not support every low-precision dtype. Promote just the
    # small camera calculation and cast the result back afterwards.
    original_dtype = projected_x0.dtype
    linalg_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    p_work = projection_matrix.to(dtype=linalg_dtype)
    y_work = projected_x0.to(dtype=linalg_dtype)
    depth_work = hidden_depth.to(dtype=linalg_dtype)

    # full_matrices=True is essential: for a 2 x 3 matrix the final row of Vh
    # is the right-null vector and is omitted by the reduced decomposition.
    _, singular_values, vh = torch.linalg.svd(p_work, full_matrices=True)
    eps = torch.finfo(linalg_dtype).eps
    rank_tolerance = 3 * eps * singular_values[..., :1]
    if bool(torch.any(singular_values[..., -1:] <= rank_tolerance).detach()):
        raise ValueError(
            "projection_matrix must have row rank 2 so hidden_depth describes "
            "its one-dimensional null space"
        )

    null_vector = vh[..., -1, :]
    pivot_index = null_vector.abs().argmax(dim=-1, keepdim=True)
    pivot = null_vector.gather(dim=-1, index=pivot_index)
    orientation = torch.where(
        pivot < 0,
        -torch.ones_like(pivot),
        torch.ones_like(pivot),
    )
    null_vector = null_vector * orientation

    pseudoinverse = torch.linalg.pinv(p_work)
    minimum_norm_lift = torch.einsum("bci,bni->bnc", pseudoinverse, y_work)
    lifted = minimum_norm_lift + depth_work.unsqueeze(-1) * null_vector.unsqueeze(1)
    return lifted.to(dtype=original_dtype)


__all__ = ["lift_projected_initial_positions", "project_trajectories"]
