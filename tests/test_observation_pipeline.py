"""Numerical and autograd tests for the differentiable observation pipeline.

The grids are deliberately small and all tests run on CPU.  Double precision is
used in the integration accuracy tests so that the asserted differences measure
the integrators rather than float32 round-off.  Spatial axes in a velocity tensor
are ``[D(z), H(y), W(x)]``, while the final coordinate axis is always ``(x,y,z)``.
"""

from __future__ import annotations

import pytest
import torch

from flow_observation.advection import advect_particles
from flow_observation.interpolation import sample_velocity
from flow_observation.likelihood import projected_trajectory_nll
from flow_observation.projection import (
    lift_projected_initial_positions,
    project_trajectories,
)


DTYPE = torch.float64


def uniform_velocity(
    positions: torch.Tensor,
    times: torch.Tensor | None = None,
    *,
    velocity: tuple[float, float, float],
) -> torch.Tensor:
    """Test-only constant field used to verify the numerical integrator."""
    del times
    value = torch.as_tensor(velocity, dtype=positions.dtype, device=positions.device)
    return value.expand_as(positions) + positions * 0.0


def solid_body_rotation_velocity(
    positions: torch.Tensor,
    times: torch.Tensor | None = None,
    *,
    angular_velocity: tuple[float, float, float],
) -> torch.Tensor:
    """Test-only rigid rotation with an analytical trajectory."""
    del times
    omega = torch.as_tensor(
        angular_velocity, dtype=positions.dtype, device=positions.device
    )
    return torch.linalg.cross(omega.expand_as(positions), positions, dim=-1)


def generate_velocity_snapshots(
    field,
    velocity_times: torch.Tensor,
    domain_bounds: torch.Tensor,
    grid_size: int,
    **field_kwargs,
) -> torch.Tensor:
    """Grid a small analytic fixture without exposing a production flow source."""
    axes = [
        torch.linspace(
            domain_bounds[axis, 0],
            domain_bounds[axis, 1],
            grid_size,
            dtype=domain_bounds.dtype,
            device=domain_bounds.device,
        )
        for axis in range(3)
    ]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    coordinates = torch.stack((xx, yy, zz), dim=-1)
    snapshots = [
        field(coordinates, time, **field_kwargs).movedim(-1, 0)
        for time in velocity_times
    ]
    return torch.stack(snapshots, dim=0)


def _domain(low: float = -2.0, high: float = 2.0) -> torch.Tensor:
    """Return bounds in the public ``[3, 2]`` (xyz, lower/upper) convention."""

    return torch.tensor([[low, high], [low, high], [low, high]], dtype=DTYPE)


def _snapshots(
    field,
    velocity_times: torch.Tensor,
    domain_bounds: torch.Tensor,
    *,
    grid_size: int = 13,
    **field_kwargs,
) -> torch.Tensor:
    return generate_velocity_snapshots(
        field,
        velocity_times,
        domain_bounds,
        grid_size,
        **field_kwargs,
    ).to(dtype=DTYPE)


def _rotation_exact(
    x0: torch.Tensor, times: torch.Tensor, omega: float
) -> torch.Tensor:
    """Analytical rotation about the z axis, shaped ``[B,N,T,3]``."""

    theta = omega * times
    cos_theta = torch.cos(theta)[None, None, :]
    sin_theta = torch.sin(theta)[None, None, :]
    x = x0[..., 0, None] * cos_theta - x0[..., 1, None] * sin_theta
    y = x0[..., 0, None] * sin_theta + x0[..., 1, None] * cos_theta
    z = x0[..., 2, None].expand_as(x)
    return torch.stack((x, y, z), dim=-1)


def _xy_projection(*, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)


def _depth_shear_velocity(
    positions: torch.Tensor,
    times: torch.Tensor | None = None,
    *,
    shear_rate: float = 0.4,
) -> torch.Tensor:
    """A linear field whose visible x motion depends on the hidden z depth."""

    del times
    velocity = torch.zeros_like(positions)
    velocity[..., 0] = shear_rate * positions[..., 2]
    return velocity


def _affine_velocity_snapshots(
    domain_bounds: torch.Tensor, grid_size: int = 7
) -> torch.Tensor:
    """Create an affine field, which trilinear interpolation reproduces exactly."""

    x = torch.linspace(domain_bounds[0, 0], domain_bounds[0, 1], grid_size, dtype=DTYPE)
    y = torch.linspace(domain_bounds[1, 0], domain_bounds[1, 1], grid_size, dtype=DTYPE)
    z = torch.linspace(domain_bounds[2, 0], domain_bounds[2, 1], grid_size, dtype=DTYPE)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    # Components are stored first; spatial storage remains D(z), H(y), W(x).
    one_snapshot = torch.stack(
        (
            xx + 2.0 * yy - 0.5 * zz,
            -0.25 * xx + yy + zz,
            0.5 * xx - yy + 0.75 * zz,
        ),
        dim=0,
    )
    return torch.stack((one_snapshot, one_snapshot), dim=0)


def test_trilinear_interpolation_is_exact_for_affine_field_and_differentiable() -> None:
    """Affine fields give an exact interpolation check and nontrivial gradients."""

    bounds = _domain(-1.0, 1.0)
    velocity_field = _affine_velocity_snapshots(bounds).requires_grad_()
    positions = torch.tensor(
        [[[-0.61, 0.12, 0.43], [0.27, -0.38, 0.09], [0.72, 0.51, -0.44]]],
        dtype=DTYPE,
        requires_grad=True,
    )
    particle_times = torch.tensor([[0.2, 0.5, 0.8]], dtype=DTYPE)

    sampled = sample_velocity(
        velocity_field,
        positions,
        particle_times,
        bounds,
        "clamp",
        velocity_times=torch.tensor([0.0, 1.0], dtype=DTYPE),
    )
    x, y, z = positions.detach().unbind(dim=-1)
    expected = torch.stack(
        (
            x + 2.0 * y - 0.5 * z,
            -0.25 * x + y + z,
            0.5 * x - y + 0.75 * z,
        ),
        dim=-1,
    )
    torch.testing.assert_close(sampled, expected, rtol=1e-11, atol=1e-11)

    sampled.square().sum().backward()
    assert velocity_field.grad is not None
    assert positions.grad is not None
    assert torch.isfinite(velocity_field.grad).all()
    assert torch.isfinite(positions.grad).all()
    assert velocity_field.grad.abs().sum() > 0
    assert positions.grad.abs().sum() > 0


def test_linear_temporal_interpolation() -> None:
    """Snapshot values at t=0 and t=2 should average exactly at t=1."""

    bounds = _domain(-1.0, 1.0)
    velocity_field = torch.zeros((2, 3, 3, 3, 3), dtype=DTYPE)
    velocity_field[0, 0] = -1.0
    velocity_field[1, 0] = 3.0
    positions = torch.zeros((1, 3, 3), dtype=DTYPE)
    particle_times = torch.tensor([[0.0, 1.0, 2.0]], dtype=DTYPE)

    sampled = sample_velocity(
        velocity_field,
        positions,
        particle_times,
        bounds,
        "clamp",
        velocity_times=torch.tensor([0.0, 2.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        sampled[0, :, 0],
        torch.tensor([-1.0, 1.0, 3.0], dtype=DTYPE),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(sampled[..., 1:], torch.zeros_like(sampled[..., 1:]))


def test_periodic_and_clamp_boundary_sampling_are_explicit() -> None:
    """Periodic wraps positions, whereas clamp evaluates at the nearest face."""

    bounds = _domain(0.0, 1.0)
    velocity_field = _affine_velocity_snapshots(bounds, grid_size=9)
    outside = torch.tensor([[[1.2, 0.4, 0.6]]], dtype=DTYPE)
    wrapped = torch.tensor([[[0.2, 0.4, 0.6]]], dtype=DTYPE)
    face = torch.tensor([[[1.0, 0.4, 0.6]]], dtype=DTYPE)
    time = torch.zeros((1, 1), dtype=DTYPE)
    velocity_times = torch.tensor([0.0, 1.0], dtype=DTYPE)

    periodic_outside = sample_velocity(
        velocity_field,
        outside,
        time,
        bounds,
        "periodic",
        velocity_times=velocity_times,
    )
    periodic_wrapped = sample_velocity(
        velocity_field,
        wrapped,
        time,
        bounds,
        "periodic",
        velocity_times=velocity_times,
    )
    clamped_outside = sample_velocity(
        velocity_field,
        outside,
        time,
        bounds,
        "clamp",
        velocity_times=velocity_times,
    )
    sampled_face = sample_velocity(
        velocity_field,
        face,
        time,
        bounds,
        "clamp",
        velocity_times=velocity_times,
    )
    torch.testing.assert_close(periodic_outside, periodic_wrapped, atol=1e-11, rtol=0.0)
    torch.testing.assert_close(clamped_outside, sampled_face, atol=1e-11, rtol=0.0)


def test_uniform_flow_matches_analytical_trajectories() -> None:
    bounds = _domain(-2.0, 2.0)
    velocity_times = torch.tensor([0.0, 1.0], dtype=DTYPE)
    observation_times = torch.tensor([0.0, 0.2, 0.7, 1.0], dtype=DTYPE)
    velocity = (0.17, -0.11, 0.08)
    field = _snapshots(
        uniform_velocity,
        velocity_times,
        bounds,
        grid_size=7,
        velocity=velocity,
    )
    x0 = torch.tensor(
        [[[0.2, -0.4, 0.1], [-0.8, 0.3, -0.2], [0.5, 0.6, 0.4]]],
        dtype=DTYPE,
    )

    trajectories = advect_particles(
        field,
        x0,
        observation_times,
        domain_bounds=bounds,
        velocity_times=velocity_times,
        num_substeps=2,
    )
    expected = x0[:, :, None, :] + observation_times[
        None, None, :, None
    ] * torch.tensor(velocity, dtype=DTYPE)
    assert trajectories.shape == (1, 3, 4, 3)
    # A constant RHS is integrated exactly by both Euler and RK4.
    torch.testing.assert_close(trajectories, expected, rtol=1e-11, atol=1e-11)


def test_solid_body_rotation_follows_circular_trajectories() -> None:
    bounds = _domain(-2.0, 2.0)
    omega = 0.8
    velocity_times = torch.tensor([0.0, 1.5], dtype=DTYPE)
    observation_times = torch.linspace(0.0, 1.5, 7, dtype=DTYPE)
    field = _snapshots(
        solid_body_rotation_velocity,
        velocity_times,
        bounds,
        grid_size=15,
        angular_velocity=(0.0, 0.0, omega),
    )
    x0 = torch.tensor(
        [[[0.6, 0.0, 0.15], [-0.25, 0.45, -0.3], [0.3, -0.4, 0.0]]],
        dtype=DTYPE,
    )

    trajectories = advect_particles(
        field,
        x0,
        observation_times,
        integrator="rk4",
        num_substeps=8,
        domain_bounds=bounds,
        velocity_times=velocity_times,
    )
    expected = _rotation_exact(x0, observation_times, omega)
    # h=0.03125 and RK4's O(h^4) global error make 2e-7 conservative.
    torch.testing.assert_close(trajectories, expected, rtol=2e-7, atol=2e-7)
    radii = torch.linalg.vector_norm(trajectories[..., :2], dim=-1)
    initial_radii = torch.linalg.vector_norm(x0[..., :2], dim=-1)[..., None]
    torch.testing.assert_close(
        radii, initial_radii.expand_as(radii), rtol=2e-7, atol=2e-7
    )


def test_rk4_converges_and_is_more_accurate_than_euler() -> None:
    bounds = _domain(-3.0, 3.0)
    omega = 1.0
    observation_times = torch.tensor([0.0, 2.0], dtype=DTYPE)
    field = _snapshots(
        solid_body_rotation_velocity,
        observation_times,
        bounds,
        grid_size=13,
        angular_velocity=(0.0, 0.0, omega),
    )
    x0 = torch.tensor([[[0.65, -0.2, 0.1]]], dtype=DTYPE)
    exact_final = _rotation_exact(x0, observation_times, omega)[:, :, -1]

    def error(integrator: str, substeps: int) -> torch.Tensor:
        prediction = advect_particles(
            field,
            x0,
            observation_times,
            integrator=integrator,
            num_substeps=substeps,
            domain_bounds=bounds,
            velocity_times=observation_times,
        )
        return torch.linalg.vector_norm(prediction[:, :, -1] - exact_final)

    euler_coarse = error("euler", 4)
    euler_fine = error("euler", 16)
    rk4_coarse = error("rk4", 4)
    rk4_fine = error("rk4", 16)
    assert euler_fine < euler_coarse
    assert rk4_fine < rk4_coarse
    assert rk4_coarse < euler_coarse
    # The ratio leaves ample margin around the theoretical orders (1 and 4).
    assert rk4_fine < 0.02 * rk4_coarse


def test_terminate_boundary_returns_and_propagates_validity() -> None:
    bounds = _domain(0.0, 1.0)
    velocity_times = torch.tensor([0.0, 1.0], dtype=DTYPE)
    field = _snapshots(
        uniform_velocity,
        velocity_times,
        bounds,
        grid_size=5,
        velocity=(0.9, 0.0, 0.0),
    )
    x0 = torch.tensor([[[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]]], dtype=DTYPE)

    trajectories, validity = advect_particles(
        field,
        x0,
        torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE),
        integrator="euler",
        num_substeps=4,
        domain_bounds=bounds,
        boundary_mode="terminate",
        velocity_times=velocity_times,
        return_validity=True,
    )
    assert validity.shape == trajectories.shape[:-1]
    assert validity.dtype == torch.bool
    assert validity[:, :, 0].all()
    assert not validity[0, 1, 1]
    # Once invalid, a particle may not silently become valid again.
    assert not validity[0, 1, 1:].any()
    assert not validity[0, 0, -1]


def test_xy_projection_selects_x_and_y_components() -> None:
    generator = torch.Generator().manual_seed(7)
    trajectories = torch.randn((2, 4, 5, 3), generator=generator, dtype=DTYPE)
    projected = project_trajectories(trajectories, _xy_projection())
    assert projected.shape == (2, 4, 5, 2)
    torch.testing.assert_close(projected, trajectories[..., :2])


def test_lift_reprojects_to_observed_coordinates_for_batched_cameras() -> None:
    projected_x0 = torch.tensor(
        [
            [[0.1, -0.2], [0.7, 0.3], [-0.4, 0.5]],
            [[-0.3, 0.2], [0.4, -0.8], [0.6, 0.1]],
        ],
        dtype=DTYPE,
    )
    projection_matrix = torch.stack(
        (
            _xy_projection(),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        )
    )
    hidden_depth = torch.tensor(
        [[[0.2], [-0.1], [0.8]], [[-0.4], [0.5], [0.3]]], dtype=DTYPE
    )

    lifted = lift_projected_initial_positions(
        projected_x0, projection_matrix, hidden_depth
    )
    reprojected = torch.einsum("bij,bnj->bni", projection_matrix, lifted)
    assert lifted.shape == (2, 3, 3)
    torch.testing.assert_close(reprojected, projected_x0, rtol=1e-11, atol=1e-11)


def _make_uniform_likelihood_case() -> dict[str, torch.Tensor]:
    bounds = _domain(-2.0, 2.0)
    observation_times = torch.tensor([0.0, 0.3, 0.65, 1.0], dtype=DTYPE)
    velocity_times = torch.tensor([0.0, 1.0], dtype=DTYPE)
    projection = _xy_projection()
    hidden_depth = torch.tensor([[[0.2], [-0.35], [0.5]]], dtype=DTYPE)
    projected_x0 = torch.tensor([[[0.1, -0.2], [-0.6, 0.4], [0.7, 0.15]]], dtype=DTYPE)
    x0 = lift_projected_initial_positions(projected_x0, projection, hidden_depth)
    field = _snapshots(
        uniform_velocity,
        velocity_times,
        bounds,
        grid_size=7,
        velocity=(0.16, -0.09, 0.04),
    )
    trajectories_3d = advect_particles(
        field,
        x0,
        observation_times,
        integrator="rk4",
        num_substeps=2,
        domain_bounds=bounds,
        velocity_times=velocity_times,
    )
    trajectories_2d = project_trajectories(trajectories_3d, projection)
    return {
        "bounds": bounds,
        "observation_times": observation_times,
        "velocity_times": velocity_times,
        "projection": projection,
        "hidden_depth": hidden_depth,
        "field": field,
        "trajectories_3d": trajectories_3d,
        "trajectories_2d": trajectories_2d,
    }


def _trajectory_nll(
    case: dict[str, torch.Tensor],
    field: torch.Tensor,
    *,
    hidden_depth: torch.Tensor | None = None,
    observation_mask: torch.Tensor | None = None,
    return_diagnostics: bool = False,
):
    return projected_trajectory_nll(
        field,
        case["trajectories_2d"],
        case["projection"],
        case["hidden_depth"] if hidden_depth is None else hidden_depth,
        case["observation_times"],
        sigma=0.05,
        observation_mask=observation_mask,
        integrator="rk4",
        num_substeps=2,
        domain_bounds=case["bounds"],
        boundary_mode="clamp",
        velocity_times=case["velocity_times"],
        return_diagnostics=return_diagnostics,
    )


def test_ground_truth_velocity_has_near_zero_projected_likelihood() -> None:
    case = _make_uniform_likelihood_case()
    loss = _trajectory_nll(case, case["field"])
    # Data and replay use the identical deterministic path; this mainly allows
    # a few ulps for repeated SVD/pseudoinverse calculations in the lift.
    assert loss.item() < 1e-20


def test_masked_observations_do_not_contribute_to_likelihood() -> None:
    case = _make_uniform_likelihood_case()
    corrupted = case["trajectories_2d"].clone()
    corrupted[0, 1, 2] += torch.tensor([10.0, -7.0], dtype=DTYPE)
    mask = torch.ones(corrupted.shape[:-1], dtype=torch.bool)
    mask[0, 1, 2] = False
    case["trajectories_2d"] = corrupted
    loss = _trajectory_nll(case, case["field"], observation_mask=mask)
    assert loss.item() < 1e-20


def test_velocity_perturbation_increases_trajectory_loss() -> None:
    case = _make_uniform_likelihood_case()
    true_loss = _trajectory_nll(case, case["field"])
    perturbed = case["field"].clone()
    perturbed[:, 0] += 0.08
    perturbed_loss = _trajectory_nll(case, perturbed)
    assert perturbed_loss > true_loss
    assert perturbed_loss.item() > 0.1


def test_likelihood_gradient_with_respect_to_velocity_is_finite_and_nonzero() -> None:
    case = _make_uniform_likelihood_case()
    candidate = case["field"].clone()
    candidate[:, 0] += 0.05
    candidate.requires_grad_()

    loss = _trajectory_nll(case, candidate)
    loss.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum().item() > 0.0


def test_likelihood_gradient_with_respect_to_hidden_depth_is_finite() -> None:
    bounds = _domain(-1.5, 1.5)
    observation_times = torch.tensor([0.0, 0.4, 0.8, 1.2], dtype=DTYPE)
    velocity_times = torch.tensor([0.0, 1.2], dtype=DTYPE)
    projection = _xy_projection()
    field = _snapshots(
        _depth_shear_velocity,
        velocity_times,
        bounds,
        grid_size=9,
        shear_rate=0.4,
    )
    projected_x0 = torch.tensor([[[0.1, 0.2], [-0.3, 0.1]]], dtype=DTYPE)
    true_depth = torch.tensor([[[0.45], [-0.25]]], dtype=DTYPE)
    true_x0 = lift_projected_initial_positions(projected_x0, projection, true_depth)
    true_3d = advect_particles(
        field,
        true_x0,
        observation_times,
        integrator="rk4",
        num_substeps=2,
        domain_bounds=bounds,
        velocity_times=velocity_times,
    )
    observed_2d = project_trajectories(true_3d, projection)
    candidate_depth = torch.tensor([[[0.1], [0.15]]], dtype=DTYPE, requires_grad=True)

    loss = projected_trajectory_nll(
        field,
        observed_2d,
        projection,
        candidate_depth,
        observation_times,
        sigma=0.1,
        integrator="rk4",
        num_substeps=2,
        domain_bounds=bounds,
        velocity_times=velocity_times,
    )
    loss.backward()
    assert candidate_depth.grad is not None
    assert torch.isfinite(candidate_depth.grad).all()
    # This shear makes projected x motion explicitly dependent on depth.
    assert candidate_depth.grad.abs().sum().item() > 0.0


def test_batch_particle_time_shapes_and_diagnostics() -> None:
    bounds = _domain(-2.0, 2.0)
    batch_size, num_particles, num_times = 2, 4, 5
    observation_times = torch.linspace(0.0, 1.0, num_times, dtype=DTYPE)
    velocity_times = torch.tensor([0.0, 1.0], dtype=DTYPE)
    field = _snapshots(
        uniform_velocity,
        velocity_times,
        bounds,
        grid_size=7,
        velocity=(0.1, -0.07, 0.04),
    )
    projections = torch.stack(
        (
            _xy_projection(),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE),
        )
    )
    generator = torch.Generator().manual_seed(11)
    projected_x0 = 0.5 * torch.randn(
        (batch_size, num_particles, 2), generator=generator, dtype=DTYPE
    )
    hidden_depth = 0.3 * torch.randn(
        (batch_size, num_particles, 1), generator=generator, dtype=DTYPE
    )
    x0 = lift_projected_initial_positions(projected_x0, projections, hidden_depth)
    trajectories_3d, validity = advect_particles(
        field,
        x0,
        observation_times,
        domain_bounds=bounds,
        velocity_times=velocity_times,
        num_substeps=2,
        return_validity=True,
    )
    trajectories_2d = project_trajectories(trajectories_3d, projections)
    mask = torch.ones((batch_size, num_particles, num_times), dtype=torch.bool)

    loss, diagnostics = projected_trajectory_nll(
        field,
        trajectories_2d,
        projections,
        hidden_depth,
        observation_times,
        sigma=0.05,
        observation_mask=mask,
        num_substeps=2,
        domain_bounds=bounds,
        velocity_times=velocity_times,
        return_diagnostics=True,
    )
    assert trajectories_3d.shape == (batch_size, num_particles, num_times, 3)
    assert trajectories_2d.shape == (batch_size, num_particles, num_times, 2)
    assert validity.shape == (batch_size, num_particles, num_times)
    assert loss.ndim == 0
    assert diagnostics["trajectories_3d"].shape == trajectories_3d.shape
    assert diagnostics["trajectories_2d"].shape == trajectories_2d.shape
    assert diagnostics["validity_mask"].shape == validity.shape
    assert loss.item() < 1e-20


@pytest.mark.parametrize("bad_integrator", ["midpoint", "bogus"])
def test_unknown_integrator_is_rejected(bad_integrator: str) -> None:
    """Typos must not silently select a different numerical method."""

    bounds = _domain(-1.0, 1.0)
    times = torch.tensor([0.0, 1.0], dtype=DTYPE)
    field = _snapshots(
        uniform_velocity, times, bounds, grid_size=3, velocity=(0.1, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="integrator"):
        advect_particles(
            field,
            torch.zeros((1, 1, 3), dtype=DTYPE),
            times,
            integrator=bad_integrator,
            domain_bounds=bounds,
            velocity_times=times,
        )
