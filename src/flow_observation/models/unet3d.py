"""A compact conditional 3D U-Net for diffusion over velocity volumes."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("time embedding dimension must be at least 4")
        self.dimension = int(dimension)

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B]")
        half = self.dimension // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / denominator
        )
        angles = timesteps.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class FiLMResidualBlock3D(nn.Module):
    """Residual block modulated by a shared time/trajectory embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or embedding_dim < 1:
            raise ValueError("block dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding_projection = nn.Linear(embedding_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.embedding_projection(F.silu(embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + self.skip(values)


class ConditionalUNet3D(nn.Module):
    """Predict diffusion noise for a conditional 3D velocity field.

    The denoiser consumes ``x`` with shape ``[B,3,D,H,W]``, diffusion steps
    ``t`` with shape ``[B]``, and a global trajectory condition
    ``[B,condition_dim]``.  Three normalized ``(x,y,z)`` coordinate channels
    are appended internally so absolute particle locations can influence a
    convolutional model.  Passing ``condition=None`` supplies a zero condition
    for classifier-free guidance.
    """

    def __init__(
        self,
        condition_dim: int = 128,
        *,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        time_embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        multipliers = tuple(int(value) for value in channel_multipliers)
        if condition_dim < 1 or base_channels < 1:
            raise ValueError("condition_dim and base_channels must be positive")
        if not multipliers or any(value < 1 for value in multipliers):
            raise ValueError("channel_multipliers must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.condition_dim = int(condition_dim)
        self.base_channels = int(base_channels)
        self.channel_multipliers = multipliers
        self.time_embedding_dim = int(time_embedding_dim)
        self.minimum_spatial_size = 2 ** len(multipliers)
        channels = tuple(self.base_channels * value for value in multipliers)
        embedding_dim = max(4 * self.base_channels, self.time_embedding_dim)

        self.time_embedding = SinusoidalTimeEmbedding(self.time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Three velocity plus x/y/z coordinate channels.
        self.input_convolution = nn.Conv3d(6, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels[0]
        for level, level_channels in enumerate(channels):
            self.down_blocks.append(
                FiLMResidualBlock3D(
                    current_channels,
                    level_channels,
                    embedding_dim,
                    dropout=dropout,
                )
            )
            current_channels = level_channels
            if level < len(channels) - 1:
                self.downsamples.append(
                    nn.Conv3d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )

        self.middle_block1 = FiLMResidualBlock3D(
            channels[-1], channels[-1], embedding_dim, dropout=dropout
        )
        self.middle_block2 = FiLMResidualBlock3D(
            channels[-1], channels[-1], embedding_dim, dropout=dropout
        )

        self.upsample_convolutions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = channels[-1]
        for level in reversed(range(len(channels))):
            level_channels = channels[level]
            if level == len(channels) - 1:
                self.upsample_convolutions.append(nn.Identity())
            else:
                self.upsample_convolutions.append(
                    nn.Conv3d(
                        current_channels, level_channels, kernel_size=3, padding=1
                    )
                )
                current_channels = level_channels
            self.up_blocks.append(
                FiLMResidualBlock3D(
                    current_channels + level_channels,
                    level_channels,
                    embedding_dim,
                    dropout=dropout,
                )
            )
            current_channels = level_channels

        self.output_normalization = nn.GroupNorm(
            _group_count(channels[0]), channels[0]
        )
        self.output_convolution = nn.Conv3d(
            channels[0], 3, kernel_size=3, padding=1
        )

    def forward(
        self, x: Tensor, timesteps: Tensor, condition: Tensor | None
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("x must have shape [B,3,D,H,W]")
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        batch = x.shape[0]
        if batch < 1:
            raise ValueError("x batch dimension must be non-empty")
        if min(x.shape[-3:]) < self.minimum_spatial_size:
            raise ValueError(
                "each spatial dimension must be at least "
                f"{self.minimum_spatial_size} for this U-Net depth"
            )
        if not isinstance(timesteps, Tensor) or tuple(timesteps.shape) != (batch,):
            raise ValueError("timesteps must be a tensor with shape [B]")
        if timesteps.device != x.device:
            raise ValueError("timesteps and x must be on the same device")
        if timesteps.is_floating_point() and not bool(
            torch.isfinite(timesteps).all().detach()
        ):
            raise ValueError("timesteps must be finite")
        if bool((timesteps < 0).any().detach()):
            raise ValueError("timesteps must be non-negative")

        if condition is None:
            condition_values = torch.zeros(
                (batch, self.condition_dim), dtype=x.dtype, device=x.device
            )
        else:
            if not isinstance(condition, Tensor) or tuple(condition.shape) != (
                batch,
                self.condition_dim,
            ):
                raise ValueError(
                    f"condition must have shape [B,{self.condition_dim}]"
                )
            if condition.device != x.device:
                raise ValueError("condition and x must be on the same device")
            if not condition.is_floating_point():
                raise TypeError("condition must be floating point")
            if not bool(torch.isfinite(condition).all().detach()):
                raise ValueError("condition must be finite")
            condition_values = condition.to(dtype=x.dtype)

        time_values = self.time_embedding(timesteps).to(dtype=x.dtype)
        embedding = self.time_mlp(time_values) + self.condition_mlp(condition_values)
        coordinates = self._coordinate_grid(x)
        hidden = self.input_convolution(torch.cat((x, coordinates), dim=1))

        skips: list[Tensor] = []
        for level, block in enumerate(self.down_blocks):
            hidden = block(hidden, embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        hidden = self.middle_block1(hidden, embedding)
        hidden = self.middle_block2(hidden, embedding)

        for stage, level in enumerate(reversed(range(len(skips)))):
            skip = skips[level]
            if level < len(skips) - 1:
                hidden = F.interpolate(
                    hidden,
                    size=skip.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            hidden = self.upsample_convolutions[stage](hidden)
            hidden = self.up_blocks[stage](torch.cat((hidden, skip), dim=1), embedding)

        return self.output_convolution(F.silu(self.output_normalization(hidden)))

    @staticmethod
    def _coordinate_grid(reference: Tensor) -> Tensor:
        depth, height, width = reference.shape[-3:]
        z_axis = torch.linspace(-1.0, 1.0, depth, device=reference.device, dtype=reference.dtype)
        y_axis = torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
        x_axis = torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
        z_grid, y_grid, x_grid = torch.meshgrid(
            z_axis, y_axis, x_axis, indexing="ij"
        )
        coordinates = torch.stack((x_grid, y_grid, z_grid), dim=0)
        return coordinates.unsqueeze(0).expand(reference.shape[0], -1, -1, -1, -1)

    def extra_repr(self) -> str:
        return (
            f"condition_dim={self.condition_dim}, base_channels={self.base_channels}, "
            f"channel_multipliers={self.channel_multipliers}"
        )


__all__ = ["ConditionalUNet3D", "FiLMResidualBlock3D", "SinusoidalTimeEmbedding"]
