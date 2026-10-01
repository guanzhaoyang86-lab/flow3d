"""Synchronized multi-view geometry used by inverse baselines."""

from __future__ import annotations

import torch
from torch import Tensor


def triangulate_initial_positions(
    projected_trajectories: Tensor,
    projection_matrices: Tensor,
    observation_mask: Tensor | None = None,
) -> Tensor:
    """Triangulate time-zero positions from each particle's valid views.

    Args:
        projected_trajectories: Synchronized observations ``[B,V,N,T,2]``.
        projection_matrices: Camera matrices ``[V,2,3]``.
        observation_mask: Optional validity mask broadcastable to
            ``[B,V,N,T]``.  Only views valid at time zero are used for each
            individual particle.

    Returns:
        Least-squares initial 3D positions ``[B,N,3]``.

    Only projected observations are used. In particular, this operation does
    not consume reference 3D trajectories or precomputed hidden depths.
    """

    if projected_trajectories.ndim != 5 or projected_trajectories.shape[-1] != 2:
        raise ValueError("projected_trajectories must have shape [B,V,N,T,2]")
    if not projected_trajectories.is_floating_point():
        raise TypeError("projected_trajectories must be floating point")
    matrices = torch.as_tensor(
        projection_matrices,
        dtype=projected_trajectories.dtype,
        device=projected_trajectories.device,
    )
    views = projected_trajectories.shape[1]
    if tuple(matrices.shape) != (views, 2, 3):
        raise ValueError(
            "projection_matrices must have shape [V,2,3] matching observations"
        )
    if observation_mask is None:
        mask = torch.ones(
            projected_trajectories.shape[:-1],
            dtype=torch.bool,
            device=projected_trajectories.device,
        )
    else:
        mask = torch.as_tensor(observation_mask, device=projected_trajectories.device)
        if mask.dtype != torch.bool:
            raise TypeError("observation_mask must be boolean")
        try:
            mask = torch.broadcast_to(mask, projected_trajectories.shape[:-1])
        except RuntimeError as error:
            raise ValueError(
                "observation_mask must be broadcastable to [B,V,N,T]"
            ) from error

    initial_2d = projected_trajectories[:, :, :, 0, :]
    initial_mask = mask[:, :, :, 0]
    positions: list[Tensor] = []
    for batch_index in range(projected_trajectories.shape[0]):
        particle_positions: list[Tensor] = []
        for particle_index in range(projected_trajectories.shape[2]):
            valid_views = initial_mask[batch_index, :, particle_index]
            selected_matrices = matrices[valid_views].reshape(-1, 3)
            rank = int(torch.linalg.matrix_rank(selected_matrices).detach())
            if rank != 3:
                raise ValueError(
                    "valid time-zero views must jointly constrain all 3D axes "
                    f"for batch {batch_index}, particle {particle_index}; got rank {rank}"
                )
            right_hand_side = initial_2d[
                batch_index, valid_views, particle_index
            ].reshape(-1)
            if not bool(torch.isfinite(right_hand_side).all().detach()):
                raise ValueError(
                    "valid time-zero observations must be finite for "
                    f"batch {batch_index}, particle {particle_index}"
                )
            particle_positions.append(
                torch.linalg.lstsq(selected_matrices, right_hand_side).solution
            )
        positions.append(torch.stack(particle_positions, dim=0))
    return torch.stack(positions, dim=0)


__all__ = ["triangulate_initial_positions"]
