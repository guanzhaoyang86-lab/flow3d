from __future__ import annotations

import pytest
import torch

from flow_observation.advection import advect_particles
from flow_observation.multiview import triangulate_initial_positions
from flow_observation.recovery import (
    CoarseVelocityField,
    divergence_mse,
    masked_trajectory_mse,
    project_trajectory_views,
    spatial_smoothness_mse,
)


DTYPE = torch.float64


def _orthographic_views() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=DTYPE,
    )


def test_multiview_triangulation_recovers_initial_positions() -> None:
    trajectories = torch.tensor(
        [
            [
                [[0.2, 0.3, 0.4], [0.25, 0.35, 0.45]],
                [[0.7, 0.1, 0.8], [0.65, 0.15, 0.75]],
            ]
        ],
        dtype=DTYPE,
    )
    projected = project_trajectory_views(trajectories, _orthographic_views())
    recovered = triangulate_initial_positions(projected, _orthographic_views())
    torch.testing.assert_close(recovered, trajectories[:, :, 0], atol=1e-12, rtol=0)


def test_multiview_triangulation_rejects_rank_deficient_views() -> None:
    observations = torch.zeros((1, 2, 3, 2, 2), dtype=DTYPE)
    repeated = _orthographic_views()[0].repeat(2, 1, 1)
    with pytest.raises(ValueError, match="jointly constrain"):
        triangulate_initial_positions(observations, repeated)


def test_coarse_velocity_field_enforces_bounds_and_fixed_values() -> None:
    fixed_mask = torch.zeros((5, 6, 7), dtype=torch.bool)
    fixed_mask[:, :, 0] = True
    fixed_values = torch.zeros((3, 5, 6, 7), dtype=DTYPE)
    fixed_values[2, :, :, 0] = 0.05
    model = CoarseVelocityField(
        (5, 6, 7),
        (3, 3, 3),
        max_speed=0.1,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        dtype=DTYPE,
    )
    with torch.no_grad():
        model.raw_velocity.fill_(20.0)
    dense = model()
    assert dense.shape == (1, 3, 5, 6, 7)
    assert torch.all(dense[:, :, :, :, 1:] <= 0.1)
    torch.testing.assert_close(
        dense[0, :, fixed_mask], fixed_values[:, fixed_mask], atol=0, rtol=0
    )

    model.raw_velocity.data.zero_()
    model().sum().backward()
    assert model.raw_velocity.grad is not None
    assert torch.isfinite(model.raw_velocity.grad).all()
    assert model.raw_velocity.grad.abs().sum() > 0


def test_recovery_regularizers_use_xyz_derivatives_correctly() -> None:
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=DTYPE)
    constant = torch.ones((1, 3, 5, 6, 7), dtype=DTYPE)
    assert spatial_smoothness_mse(constant).item() == 0.0
    assert divergence_mse(constant, bounds).item() == 0.0

    x = torch.linspace(0.0, 1.0, 7, dtype=DTYPE)
    expanding = torch.zeros_like(constant)
    expanding[:, 0] = x.reshape(1, 1, 1, 7)
    torch.testing.assert_close(
        divergence_mse(expanding, bounds),
        torch.tensor(1.0, dtype=DTYPE),
        atol=1e-12,
        rtol=0,
    )
    assert spatial_smoothness_mse(expanding) > 0


def test_coarse_grid_gradient_step_reduces_multiview_track_loss() -> None:
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=DTYPE)
    times = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
    x0 = torch.tensor(
        [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.2], [0.7, 0.2, 0.6]]],
        dtype=DTYPE,
    )
    reference = torch.zeros((1, 3, 4, 4, 4), dtype=DTYPE)
    reference[:, 0] = 0.08
    observed_3d = advect_particles(
        reference,
        x0,
        times,
        integrator="euler",
        domain_bounds=bounds,
    )
    views = _orthographic_views()
    observed = project_trajectory_views(observed_3d, views)
    triangulated = triangulate_initial_positions(observed, views)

    model = CoarseVelocityField((4, 4, 4), (3, 3, 3), max_speed=0.2, dtype=DTYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    candidate_3d = advect_particles(
        model(),
        triangulated,
        times,
        integrator="euler",
        domain_bounds=bounds,
    )
    candidate_2d = project_trajectory_views(candidate_3d, views)
    initial_loss = masked_trajectory_mse(candidate_2d, observed)
    initial_loss.backward()
    optimizer.step()

    updated_3d = advect_particles(
        model(),
        triangulated,
        times,
        integrator="euler",
        domain_bounds=bounds,
    )
    updated_loss = masked_trajectory_mse(
        project_trajectory_views(updated_3d, views), observed
    )
    assert updated_loss < initial_loss
