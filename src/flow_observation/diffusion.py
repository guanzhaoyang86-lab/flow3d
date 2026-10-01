"""Small, dependency-free Gaussian diffusion utilities for 3D flow fields.

The diffusion variable is a normalized velocity volume with shape
``[B, 3, D, H, W]``.  Conditioning is intentionally opaque to this module:
the supplied denoiser receives ``(noisy_field, diffusion_step, condition)``.
This keeps the schedule/sampler reusable with either sparse-track conditions
or an unconditional flow prior.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch
from torch import Tensor, nn


Condition = Tensor | None
Denoiser = Callable[[Tensor, Tensor, Condition], Tensor]
GuidanceLoss = Callable[[Tensor, Tensor], Tensor]
Projection = Callable[[Tensor], Tensor]


def cosine_beta_schedule(
    num_steps: int, *, offset: float = 0.008, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Return the improved-DDPM cosine variance schedule."""

    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError("offset must be finite and non-negative")
    steps = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    cumulative = torch.cos(
        ((steps / num_steps + offset) / (1.0 + offset)) * math.pi * 0.5
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-8, 0.999).to(dtype=dtype)


def linear_beta_schedule(
    num_steps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a simple linear variance schedule."""

    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not (0.0 < beta_start < beta_end < 1.0):
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
    return torch.linspace(beta_start, beta_end, num_steps, dtype=dtype)


def _extract(values: Tensor, timesteps: Tensor, reference: Tensor) -> Tensor:
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("schedule values must be a non-empty vector")
    if not isinstance(timesteps, Tensor):
        raise TypeError("timesteps must be a torch.Tensor")
    if timesteps.ndim != 1 or timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps must have shape [B]")
    if timesteps.dtype not in (torch.int32, torch.int64):
        raise TypeError("timesteps must contain integer indices")
    if bool(((timesteps < 0) | (timesteps >= values.numel())).any().detach()):
        raise ValueError("timesteps contain an index outside the diffusion schedule")
    indices = timesteps.to(device=reference.device, dtype=torch.long)
    gathered = values.to(device=reference.device, dtype=reference.dtype).gather(
        0, indices
    )
    return gathered.reshape(reference.shape[0], *((1,) * (reference.ndim - 1)))


class GaussianDiffusion(nn.Module):
    """Forward noising process, epsilon loss, and DDIM sampling.

    This object has buffers but no trainable parameters.  Registering the
    schedule as buffers makes checkpoints and device transfers unambiguous.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        *,
        schedule: str = "cosine",
        clip_x0: float | None = None,
    ) -> None:
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        elif schedule == "linear":
            betas = linear_beta_schedule(num_steps)
        else:
            raise ValueError("schedule must be 'cosine' or 'linear'")
        if clip_x0 is not None and (
            not math.isfinite(clip_x0) or clip_x0 <= 0.0
        ):
            raise ValueError("clip_x0 must be finite and positive")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        self.num_steps = int(num_steps)
        self.schedule_name = schedule
        self.clip_x0 = clip_x0
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(
        self,
        clean: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample ``q(x_t | x_0)`` for a batch of normalized fields."""

        self._validate_field(clean)
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        if noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("noise must share clean's device and dtype")
        if not bool(torch.isfinite(noise).all().detach()):
            raise ValueError("noise must contain only finite values")
        return _extract(self.sqrt_alpha_bars, timesteps, clean) * clean + _extract(
            self.sqrt_one_minus_alpha_bars, timesteps, clean
        ) * noise

    def predict_x0_from_noise(
        self, noisy: Tensor, timesteps: Tensor, predicted_noise: Tensor
    ) -> Tensor:
        """Convert an epsilon prediction into a clean-field prediction."""

        self._validate_field(noisy)
        if predicted_noise.shape != noisy.shape:
            raise ValueError("predicted_noise must have the same shape as noisy")
        clean = (
            noisy
            - _extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy)
            * predicted_noise
        ) / _extract(self.sqrt_alpha_bars, timesteps, noisy).clamp_min(1e-12)
        if self.clip_x0 is not None:
            clean = clean.clamp(-self.clip_x0, self.clip_x0)
        return clean

    def training_loss(
        self,
        denoiser: Denoiser,
        clean: Tensor,
        condition: Condition,
        *,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return epsilon-MSE and tensors needed by auxiliary losses."""

        self._validate_field(clean)
        batch_size = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                self.num_steps, (batch_size,), device=clean.device
            )
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise=noise)
        predicted_noise = denoiser(noisy, timesteps, condition)
        if predicted_noise.shape != clean.shape:
            raise ValueError("denoiser output must match the input field shape")
        per_example = (predicted_noise - noise).square().flatten(1).mean(dim=1)
        predicted_x0 = self.predict_x0_from_noise(
            noisy, timesteps, predicted_noise
        )
        return {
            "loss": per_example.mean(),
            "per_example_loss": per_example,
            "predicted_noise": predicted_noise,
            "predicted_x0": predicted_x0,
            "noisy": noisy,
            "noise": noise,
            "timesteps": timesteps,
        }

    def ddim_sample(
        self,
        denoiser: Denoiser,
        shape: tuple[int, int, int, int, int],
        condition: Condition,
        *,
        sampling_steps: int | None = None,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        unconditional_condition: Condition = None,
        guidance_loss: GuidanceLoss | None = None,
        guidance_strength: float = 0.0,
        projection: Projection | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Generate normalized fields with deterministic or stochastic DDIM.

        ``guidance_loss`` receives the current clean prediction and the current
        integer timestep batch.  Its gradient is applied directly to the clean
        prediction before the DDIM update.  It is intended for differentiable
        trajectory/physics consistency. ``projection`` is a non-learned hard
        constraint such as overwriting known cavity boundaries.
        """

        if len(shape) != 5 or shape[1] != 3 or any(value < 1 for value in shape):
            raise ValueError("shape must be [B,3,D,H,W] with positive dimensions")
        steps = self.num_steps if sampling_steps is None else int(sampling_steps)
        if not 1 <= steps <= self.num_steps:
            raise ValueError("sampling_steps must be in [1, num_steps]")
        if not math.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if not math.isfinite(guidance_strength) or guidance_strength < 0.0:
            raise ValueError("guidance_strength must be finite and non-negative")
        if guidance_strength > 0.0 and guidance_loss is None:
            raise ValueError("guidance_loss is required when guidance_strength > 0")

        if initial_noise is not None:
            if tuple(initial_noise.shape) != shape:
                raise ValueError("initial_noise does not match shape")
            if not initial_noise.is_floating_point():
                raise TypeError("initial_noise must be floating point")
            if not bool(torch.isfinite(initial_noise).all().detach()):
                raise ValueError("initial_noise must contain only finite values")
            sample = initial_noise.clone()
            device = sample.device
            dtype = sample.dtype
        else:
            if device is None:
                device = self.betas.device
            if dtype is None:
                dtype = self.betas.dtype
            sample = torch.randn(
                shape, device=device, dtype=dtype, generator=generator
            )

        schedule = torch.linspace(
            self.num_steps - 1,
            0,
            steps,
            device=sample.device,
            dtype=torch.float64,
        ).round().to(torch.long)
        # Rounding can duplicate indices only when steps > num_steps, rejected above.
        schedule = torch.unique_consecutive(schedule)
        previous = torch.cat(
            (schedule[1:], torch.full((1,), -1, device=sample.device, dtype=torch.long))
        )

        for step_value, previous_value in zip(schedule.tolist(), previous.tolist()):
            timesteps = torch.full(
                (shape[0],), step_value, device=sample.device, dtype=torch.long
            )
            with torch.no_grad():
                predicted_noise = denoiser(sample, timesteps, condition)
                if guidance_scale != 1.0:
                    unconditioned = denoiser(
                        sample, timesteps, unconditional_condition
                    )
                    predicted_noise = unconditioned + guidance_scale * (
                        predicted_noise - unconditioned
                    )
                clean = self.predict_x0_from_noise(
                    sample, timesteps, predicted_noise
                )

            if guidance_loss is not None and guidance_strength > 0.0:
                with torch.enable_grad():
                    clean_for_guidance = clean.detach().requires_grad_(True)
                    loss = guidance_loss(clean_for_guidance, timesteps)
                    if loss.ndim != 0:
                        raise ValueError("guidance_loss must return a scalar")
                    gradient = torch.autograd.grad(loss, clean_for_guidance)[0]
                    if not bool(torch.isfinite(gradient).all().detach()):
                        raise RuntimeError("guidance_loss produced a non-finite gradient")
                    clean = (
                        clean_for_guidance - guidance_strength * gradient
                    ).detach()

            if projection is not None:
                clean = projection(clean)
                if clean.shape != sample.shape:
                    raise ValueError("projection must preserve the field shape")

            alpha_bar = self.alpha_bars[step_value].to(
                device=sample.device, dtype=sample.dtype
            )
            if previous_value < 0:
                sample = clean
                continue
            alpha_bar_previous = self.alpha_bars[previous_value].to(
                device=sample.device, dtype=sample.dtype
            )
            reconstructed_noise = (
                sample - alpha_bar.sqrt() * clean
            ) / (1.0 - alpha_bar).sqrt().clamp_min(1e-12)
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_bar_previous) / (1.0 - alpha_bar))
                * (1.0 - alpha_bar / alpha_bar_previous).clamp_min(0.0)
            )
            direction_scale = (1.0 - alpha_bar_previous - sigma.square()).clamp_min(
                0.0
            ).sqrt()
            if eta > 0.0:
                random_noise = torch.randn(
                    sample.shape,
                    device=sample.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
            else:
                random_noise = torch.zeros_like(sample)
            sample = (
                alpha_bar_previous.sqrt() * clean
                + direction_scale * reconstructed_noise
                + sigma * random_noise
            )
        return sample

    @staticmethod
    def _validate_field(field: Tensor) -> None:
        if not isinstance(field, Tensor):
            raise TypeError("flow fields must be torch.Tensor instances")
        if field.ndim != 5 or field.shape[1] != 3:
            raise ValueError("flow fields must have shape [B,3,D,H,W]")
        if not field.is_floating_point():
            raise TypeError("flow fields must be floating point")
        if not bool(torch.isfinite(field).all().detach()):
            raise ValueError("flow fields must contain only finite values")

    def extra_repr(self) -> str:
        return (
            f"num_steps={self.num_steps}, schedule={self.schedule_name!r}, "
            f"clip_x0={self.clip_x0}"
        )


__all__ = [
    "GaussianDiffusion",
    "cosine_beta_schedule",
    "linear_beta_schedule",
]
