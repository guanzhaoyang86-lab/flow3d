from __future__ import annotations

import pytest
import torch

from flow_observation.models import ConditionalUNet3D, TrackSetEncoder


def _track_inputs(
    *, batch: int = 1, particles: int = 3
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    tracks = torch.rand((batch, 2, particles, 4, 2), generator=generator)
    mask = torch.ones((batch, 2, particles, 4), dtype=torch.bool)
    cameras = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    times = torch.tensor([0.0, 0.2, 0.6, 1.0])
    bounds = torch.tensor(((0.0, 1.0),) * 3)
    return tracks, mask, cameras, times, bounds


def test_track_set_encoder_is_particle_permutation_invariant() -> None:
    encoder = TrackSetEncoder(condition_dim=24, token_dim=16, particle_dim=20)
    inputs = _track_inputs(particles=3)
    reference = encoder(*inputs)

    permutation = torch.tensor([2, 0, 1])
    tracks, mask, cameras, times, bounds = inputs
    permuted = encoder(
        tracks[:, :, permutation],
        mask[:, :, permutation],
        cameras,
        times,
        bounds,
    )
    torch.testing.assert_close(permuted, reference, atol=2e-6, rtol=2e-6)


def test_track_set_encoder_ignores_masked_padding_particle() -> None:
    encoder = TrackSetEncoder(condition_dim=24, token_dim=16, particle_dim=20)
    tracks, mask, cameras, times, bounds = _track_inputs(particles=2)
    reference = encoder(tracks, mask, cameras, times, bounds)

    padding = torch.full((1, 2, 1, 4, 2), torch.nan)
    padded_tracks = torch.cat((tracks, padding), dim=2)
    padded_mask = torch.cat(
        (mask, torch.zeros((1, 2, 1, 4), dtype=torch.bool)), dim=2
    )
    particle_mask = torch.tensor([[True, True, False]])
    padded = encoder(
        padded_tracks,
        padded_mask,
        cameras,
        times,
        bounds,
        particle_mask,
    )
    torch.testing.assert_close(padded, reference, atol=2e-6, rtol=2e-6)


def test_track_set_encoder_has_finite_gradients() -> None:
    encoder = TrackSetEncoder(condition_dim=24, token_dim=16, particle_dim=20)
    tracks, mask, cameras, times, bounds = _track_inputs(batch=2, particles=2)
    tracks.requires_grad_(True)
    encoded = encoder(tracks, mask, cameras, times, bounds)
    assert encoded.shape == (2, 24)
    encoded.square().mean().backward()
    assert tracks.grad is not None
    assert torch.isfinite(tracks.grad).all()
    gradients = [parameter.grad for parameter in encoder.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_track_set_encoder_validates_shapes() -> None:
    encoder = TrackSetEncoder(condition_dim=16, token_dim=8, particle_dim=12)
    tracks, mask, cameras, times, bounds = _track_inputs(particles=2)
    with pytest.raises(ValueError, match="projection_matrices"):
        encoder(tracks, mask, cameras[:1], times, bounds)
    with pytest.raises(ValueError, match="strictly increasing"):
        encoder(tracks, mask, cameras, torch.tensor([0.0, 0.2, 0.2, 1.0]), bounds)


def test_conditional_unet3d_preserves_shape_and_backpropagates() -> None:
    torch.manual_seed(3)
    model = ConditionalUNet3D(
        condition_dim=24,
        base_channels=8,
        channel_multipliers=(1, 2, 4),
        time_embedding_dim=32,
    )
    field = torch.randn((2, 3, 8, 8, 8), requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.long)
    condition = torch.randn((2, 24), requires_grad=True)
    prediction = model(field, timesteps, condition)
    assert prediction.shape == field.shape
    assert torch.isfinite(prediction).all()

    prediction.square().mean().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()
    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
        if gradient is not None
    )


def test_conditional_unet3d_supports_unconditional_branch_and_validates_shape() -> None:
    model = ConditionalUNet3D(
        condition_dim=12,
        base_channels=8,
        channel_multipliers=(1, 2, 4),
        time_embedding_dim=16,
    )
    field = torch.randn((1, 3, 8, 8, 8))
    result = model(field, torch.tensor([2]), None)
    assert result.shape == field.shape
    with pytest.raises(ValueError, match="condition"):
        model(field, torch.tensor([2]), torch.randn((1, 11)))
    with pytest.raises(ValueError, match="spatial"):
        model(torch.randn((1, 3, 4, 8, 8)), torch.tensor([2]), None)
