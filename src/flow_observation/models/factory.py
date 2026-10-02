"""Checkpoint-compatible construction of flow diffusion denoisers.

Missing architecture metadata identifies the original 3D U-Net checkpoints.
DiT implementations are imported lazily so those checkpoints remain usable.
"""
from __future__ import annotations

from typing import Any

from torch import nn

from .unet3d import ConditionalUNet3D


def build_denoiser(config: dict[str, Any]) -> nn.Module:
    architecture = config.get("architecture", "unet3d")
    common = {"condition_dim": int(config["condition_dim"]),
              "dropout": float(config.get("dropout", 0.0))}
    if architecture == "unet3d":
        return ConditionalUNet3D(
            **common, base_channels=int(config["base_channels"]),
            channel_multipliers=tuple(config.get("channel_multipliers", (1, 2, 4))),
            time_embedding_dim=int(config["time_embedding_dim"]),
        )
    from .dit import ConditionalDiT3D, TensorDiT
    common.update(spatial_shape=tuple(config["spatial_shape"]),
                  hidden_dim=int(config["dit_hidden_dim"]),
                  depth=int(config["dit_depth"]), num_heads=int(config["dit_heads"]))
    if architecture == "dit3d":
        return ConditionalDiT3D(**common, patch_size=int(config["patch_size"]))
    if architecture == "tensor-dit":
        return TensorDiT(**common, rank=int(config["tensor_rank"]))
    raise ValueError(f"unknown diffusion architecture: {architecture}")


def diffusion_clip(config: dict[str, Any]) -> float | None:
    # Componentwise clipping of aligned factors would distort the reconstructed
    # tensor. Preserve the old field clipping only for full-volume diffusion.
    return None if config.get("architecture") == "tensor-dit" else 8.0
