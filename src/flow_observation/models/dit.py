"""Small conditional diffusion transformers for volumes and aligned Tucker data.

The adaLN-Zero architecture follows Peebles and Xie, *Scalable Diffusion
Models with Transformers* (https://arxiv.org/abs/2212.09748). This is an
independent PyTorch implementation for this project's trajectory conditioning;
it does not import or distribute the reference implementation or image weights.
The Tucker representation and its alignment are supplied by the data codec,
not estimated inside the denoiser.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn

from .unet3d import SinusoidalTimeEmbedding


def _spatial_shape(values: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(values)
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in shape
    ):
        raise ValueError("spatial_shape must contain three positive integers")
    return shape


def patchify_3d(values: Tensor, patch_size: int) -> Tensor:
    """Return [B, patches, channels * patch_size**3], without losing voxels."""
    if values.ndim != 5:
        raise ValueError("values must have shape [B,C,D,H,W]")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    batch, channels, depth, height, width = values.shape
    if any(size % patch_size for size in (depth, height, width)):
        raise ValueError("each spatial dimension must be divisible by patch_size")
    p = patch_size
    return values.reshape(batch, channels, depth // p, p, height // p, p, width // p, p).permute(
        0, 2, 4, 6, 1, 3, 5, 7
    ).reshape(batch, (depth // p) * (height // p) * (width // p), channels * p**3)


def unpatchify_3d(
    tokens: Tensor, spatial_shape: Sequence[int], patch_size: int, channels: int = 3
) -> Tensor:
    """Invert :func:`patchify_3d`, including non-cubic spatial shapes."""
    depth, height, width = _spatial_shape(spatial_shape)
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
        raise ValueError("channels must be a positive integer")
    p = patch_size
    if any(size % p for size in (depth, height, width)):
        raise ValueError("each spatial dimension must be divisible by patch_size")
    expected = ((depth // p) * (height // p) * (width // p), channels * p**3)
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"tokens must have shape [B,{expected[0]},{expected[1]}]")
    return tokens.reshape(tokens.shape[0], depth // p, height // p, width // p, channels, p, p, p).permute(
        0, 4, 1, 5, 2, 6, 3, 7
    ).reshape(tokens.shape[0], channels, depth, height, width)


def _grid_coordinates(shape: tuple[int, int, int]) -> Tensor:
    axes = [torch.linspace(-1.0, 1.0, size) if size > 1 else torch.zeros(1) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


class _AdaptiveBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.feedforward_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        attention_shift, attention_scale, attention_gate, ff_shift, ff_scale, ff_gate = (
            self.modulation(condition).unsqueeze(1).chunk(6, dim=-1)
        )
        normalized = self.attention_norm(tokens) * (1 + attention_scale) + attention_shift
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attention_gate * attended
        normalized = self.feedforward_norm(tokens) * (1 + ff_scale) + ff_shift
        return tokens + ff_gate * self.feedforward(normalized)


class _ConditionalTransformer(nn.Module):
    def __init__(
        self, hidden_dim: int, depth: int, num_heads: int, condition_dim: int, dropout: float
    ) -> None:
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("depth", depth), ("num_heads", num_heads), ("condition_dim", condition_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim < 4 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be at least 4 and divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.condition_projection = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.position_projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_AdaptiveBlock(hidden_dim, num_heads, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))

    def _initialize(self, output_layers: Sequence[nn.Linear]) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
        for layer in [block.modulation[-1] for block in self.blocks] + [self.output_modulation[-1]] + list(output_layers):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _condition(self, x: Tensor, timesteps: Tensor, condition: Tensor | None) -> Tensor:
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        batch = x.shape[0]
        if batch < 1:
            raise ValueError("x batch dimension must be non-empty")
        if not isinstance(timesteps, Tensor) or tuple(timesteps.shape) != (batch,):
            raise ValueError("timesteps must be a tensor with shape [B]")
        if timesteps.device != x.device:
            raise ValueError("timesteps and x must be on the same device")
        if not bool(torch.isfinite(timesteps).all()) or bool((timesteps < 0).any()):
            raise ValueError("timesteps must be finite and non-negative")
        if condition is None:
            condition = x.new_zeros(batch, self.condition_dim)
        else:
            if not isinstance(condition, Tensor) or tuple(condition.shape) != (batch, self.condition_dim):
                raise ValueError(f"condition must have shape [B,{self.condition_dim}]")
            if condition.device != x.device:
                raise ValueError("condition and x must be on the same device")
            if not condition.is_floating_point():
                raise TypeError("condition must be floating point")
            condition = condition.to(dtype=x.dtype)
        time = self.time_embedding(timesteps).to(dtype=x.dtype)
        return self.time_projection(time) + self.condition_projection(condition)

    def _transform(self, tokens: Tensor, condition: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.output_modulation(condition).unsqueeze(1).chunk(2, dim=-1)
        return self.output_norm(tokens) * (1 + scale) + shift


class ConditionalDiT3D(_ConditionalTransformer):
    """Volume-space noise prediction with non-overlapping 3D patches.

    ``condition=None`` supplies a zero trajectory embedding for classifier-free
    guidance, matching this project's existing U-Net interface. Output predicts
    only noise (three channels); no learned-variance channels are appended.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int] = (32, 32, 32),
        patch_size: int = 4,
        hidden_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        condition_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(hidden_dim, depth, num_heads, condition_dim, dropout)
        self.spatial_shape = _spatial_shape(spatial_shape)
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError("patch_size must be a positive integer")
        if any(size % patch_size for size in self.spatial_shape):
            raise ValueError("each spatial dimension must be divisible by patch_size")
        self.patch_size = patch_size
        patch_shape = tuple(size // patch_size for size in self.spatial_shape)
        self.num_tokens = math.prod(patch_shape)
        self.register_buffer("positions", _grid_coordinates(patch_shape))
        patch_dimension = 3 * patch_size**3
        self.input_projection = nn.Linear(patch_dimension, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, patch_dimension)
        self._initialize([self.output_projection])

    def patchify(self, values: Tensor) -> Tensor:
        if values.ndim != 5 or tuple(values.shape[1:]) != (3, *self.spatial_shape):
            raise ValueError(f"x must have shape [B,3,{','.join(map(str, self.spatial_shape))}]")
        return patchify_3d(values, self.patch_size)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        return unpatchify_3d(tokens, self.spatial_shape, self.patch_size)

    def forward(self, x: Tensor, timesteps: Tensor, condition: Tensor | None) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        patches = self.patchify(x)
        context = self._condition(x, timesteps, condition)
        tokens = self.input_projection(patches) + self.position_projection(self.positions.to(dtype=x.dtype)).unsqueeze(0)
        return self.unpatchify(self.output_projection(self._transform(tokens, context)))


class TensorDiT(_ConditionalTransformer):
    """Joint noise prediction for an aligned spatial Tucker core and factors.

    Packed input/output is ``[core(3,r,r,r), Z(D,r), Y(H,r), X(W,r)]`` flattened
    in row-major order. Core patches are tokens; each factor row is another
    token. Separate projections and learned type embeddings distinguish core,
    Z, Y and X. Absolute positions distinguish the rows/patches within a type.
    All groups attend jointly, so core and factors can influence each other.

    The core uses 2-cubed patches at common even ranks up to 16. For other ranks
    an exact divisor is selected to keep at most 512 core tokens. No padding,
    truncated coefficients, or reconstructed-volume input is required.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int] = (32, 32, 32),
        rank: int = 8,
        hidden_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        condition_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(hidden_dim, depth, num_heads, condition_dim, dropout)
        self.spatial_shape = _spatial_shape(spatial_shape)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(self.spatial_shape):
            raise ValueError("rank must be a positive integer no larger than each spatial dimension")
        self.rank = rank
        self.core_size = 3 * rank**3
        self.packed_dim = self.core_size + sum(self.spatial_shape) * rank
        self.latent_dim = self.packed_dim
        preferred_patch = 2 if rank % 2 == 0 else 1
        self.core_patch_size = next(p for p in range(preferred_patch, rank + 1) if rank % p == 0 and (rank // p)**3 <= 512)
        core_shape = (rank // self.core_patch_size,) * 3
        self.group_token_counts = (math.prod(core_shape), *self.spatial_shape)
        self.num_tokens = sum(self.group_token_counts)
        self.input_projections = nn.ModuleList([nn.Linear(3 * self.core_patch_size**3, hidden_dim)] + [nn.Linear(rank, hidden_dim) for _ in range(3)])
        self.output_projections = nn.ModuleList([nn.Linear(hidden_dim, 3 * self.core_patch_size**3)] + [nn.Linear(hidden_dim, rank) for _ in range(3)])
        self.type_embedding = nn.Parameter(torch.empty(4, hidden_dim))
        positions = [_grid_coordinates(core_shape)]
        for axis, size in enumerate(self.spatial_shape):
            coordinates = torch.zeros(size, 3)
            coordinates[:, axis] = torch.linspace(-1.0, 1.0, size) if size > 1 else 0.0
            positions.append(coordinates)
        self.register_buffer("positions", torch.cat(positions))
        self.register_buffer("token_types", torch.cat([torch.full((count,), i, dtype=torch.long) for i, count in enumerate(self.group_token_counts)]))
        self._initialize(self.output_projections)
        nn.init.normal_(self.type_embedding, std=0.02)

    def split_tokens(self, packed: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if packed.ndim != 2 or packed.shape[1] != self.packed_dim:
            raise ValueError(f"x must have shape [B,{self.packed_dim}]")
        sizes = (self.core_size, *(size * self.rank for size in self.spatial_shape))
        pieces = packed.split(sizes, dim=1)
        core = pieces[0].reshape(packed.shape[0], 3, self.rank, self.rank, self.rank)
        return (patchify_3d(core, self.core_patch_size), *(piece.reshape(packed.shape[0], size, self.rank) for piece, size in zip(pieces[1:], self.spatial_shape)))

    def merge_tokens(self, pieces: Sequence[Tensor]) -> Tensor:
        if len(pieces) != 4:
            raise ValueError("expected core and three factor token tensors")
        core = unpatchify_3d(pieces[0], (self.rank,) * 3, self.core_patch_size)
        batch = core.shape[0]
        for piece, size in zip(pieces[1:], self.spatial_shape):
            if tuple(piece.shape) != (batch, size, self.rank):
                raise ValueError("factor tokens do not match the spatial shape and rank")
        return torch.cat([core.reshape(batch, -1), *(piece.reshape(batch, -1) for piece in pieces[1:])], dim=1)

    def forward(self, x: Tensor, timesteps: Tensor, condition: Tensor | None) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        pieces = self.split_tokens(x)
        context = self._condition(x, timesteps, condition)
        tokens = torch.cat([projection(piece) for projection, piece in zip(self.input_projections, pieces)], dim=1)
        tokens = tokens + self.position_projection(self.positions.to(dtype=x.dtype)).unsqueeze(0)
        tokens = tokens + self.type_embedding[self.token_types].unsqueeze(0)
        groups = self._transform(tokens, context).split(self.group_token_counts, dim=1)
        return self.merge_tokens([projection(group) for projection, group in zip(self.output_projections, groups)])
