from __future__ import annotations

import pytest
import torch

from flow_observation.diffusion import GaussianDiffusion, cosine_beta_schedule


DTYPE = torch.float64


def test_cosine_schedule_is_valid_and_monotone_in_noise() -> None:
    betas = cosine_beta_schedule(16, dtype=DTYPE)
    assert betas.shape == (16,)
    assert torch.all((betas > 0) & (betas < 1))
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])


def test_q_sample_and_epsilon_loss_reconstruct_clean_field() -> None:
    diffusion = GaussianDiffusion(12).to(dtype=DTYPE)
    generator = torch.Generator().manual_seed(4)
    clean = torch.randn((2, 3, 4, 4, 4), generator=generator, dtype=DTYPE)
    noise = torch.randn(clean.shape, generator=generator, dtype=DTYPE)
    timesteps = torch.tensor([1, 9], dtype=torch.long)

    result = diffusion.training_loss(
        lambda noisy, steps, condition: condition,
        clean,
        noise,
        timesteps=timesteps,
        noise=noise,
    )
    torch.testing.assert_close(result["loss"], torch.zeros((), dtype=DTYPE))
    torch.testing.assert_close(result["predicted_x0"], clean, atol=1e-10, rtol=1e-10)
    assert result["noisy"].shape == clean.shape


def test_ddim_eta_zero_is_deterministic_from_fixed_noise() -> None:
    diffusion = GaussianDiffusion(8).to(dtype=DTYPE)
    initial = torch.randn(
        (1, 3, 4, 4, 4), generator=torch.Generator().manual_seed(11), dtype=DTYPE
    )

    def zero_denoiser(noisy, timesteps, condition):
        del timesteps, condition
        return torch.zeros_like(noisy)

    first = diffusion.ddim_sample(
        zero_denoiser,
        tuple(initial.shape),
        None,
        sampling_steps=4,
        eta=0.0,
        initial_noise=initial,
    )
    second = diffusion.ddim_sample(
        zero_denoiser,
        tuple(initial.shape),
        None,
        sampling_steps=4,
        eta=0.0,
        initial_noise=initial,
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.isfinite(first).all()


def test_ddim_oracle_noise_prediction_recovers_clean_field() -> None:
    diffusion = GaussianDiffusion(13).to(dtype=DTYPE)
    generator = torch.Generator().manual_seed(19)
    clean = torch.randn((1, 3, 3, 4, 5), generator=generator, dtype=DTYPE)
    noise = torch.randn(clean.shape, generator=generator, dtype=DTYPE)
    final_timestep = torch.tensor([diffusion.num_steps - 1], dtype=torch.long)
    initial = diffusion.q_sample(clean, final_timestep, noise=noise)

    def oracle_denoiser(noisy, timesteps, condition):
        del condition
        alpha_bar = diffusion.alpha_bars.to(dtype=noisy.dtype)[timesteps].reshape(
            noisy.shape[0], 1, 1, 1, 1
        )
        return (noisy - alpha_bar.sqrt() * clean) / (1.0 - alpha_bar).sqrt()

    recovered = diffusion.ddim_sample(
        oracle_denoiser,
        tuple(initial.shape),
        None,
        sampling_steps=7,
        eta=0.0,
        initial_noise=initial,
    )
    # The schedule is constructed in float32 before the module is promoted to
    # float64, so the end-to-end round trip is limited by schedule precision.
    torch.testing.assert_close(recovered, clean, atol=5e-8, rtol=1e-6)


def test_ddim_guidance_and_projection_are_applied() -> None:
    diffusion = GaussianDiffusion(6).to(dtype=DTYPE)
    initial = torch.zeros((1, 3, 3, 3, 3), dtype=DTYPE)

    def zero_denoiser(noisy, timesteps, condition):
        del timesteps, condition
        return torch.zeros_like(noisy)

    def guidance_loss(clean, timesteps):
        del timesteps
        return (clean - 1.0).square().mean()

    def zero_boundary(clean):
        result = clean.clone()
        result[..., 0, :, :] = 0.0
        return result

    sample = diffusion.ddim_sample(
        zero_denoiser,
        tuple(initial.shape),
        None,
        sampling_steps=3,
        initial_noise=initial,
        guidance_loss=guidance_loss,
        guidance_strength=0.2,
        projection=zero_boundary,
    )
    assert torch.isfinite(sample).all()
    torch.testing.assert_close(sample[..., 0, :, :], torch.zeros_like(sample[..., 0, :, :]))
    assert sample[..., 1:, :, :].mean() > 0


def test_q_sample_rejects_invalid_external_timestep_and_noise() -> None:
    diffusion = GaussianDiffusion(8).to(dtype=DTYPE)
    clean = torch.zeros((1, 3, 2, 2, 2), dtype=DTYPE)

    with pytest.raises(ValueError, match="outside the diffusion schedule"):
        diffusion.q_sample(clean, torch.tensor([8]), noise=torch.zeros_like(clean))
    with pytest.raises(ValueError, match="device and dtype"):
        diffusion.q_sample(
            clean,
            torch.tensor([1]),
            noise=torch.zeros_like(clean, dtype=torch.float32),
        )
    invalid = torch.zeros_like(clean)
    invalid[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        diffusion.q_sample(clean, torch.tensor([1]), noise=invalid)
