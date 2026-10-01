"""Adapter boundary for external CFD velocity snapshots.

Numerical flow solvers and file formats should expose their data through the
small :class:`CFDVelocityAdapter` protocol.  Downstream observation code can
then consume a validated :class:`VelocitySnapshotData` without depending on a
particular CFD package.

NumPy is intentionally confined to :meth:`NPZCFDAdapter.load`, the file I/O
boundary.  The returned data are PyTorch tensors, so no NumPy operation is
introduced into a differentiable observation path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


_BOUNDARY_MODES = frozenset({"periodic", "clamp", "terminate"})


@dataclass(frozen=True, slots=True)
class VelocitySnapshotData:
    """Validated Eulerian velocity snapshots and their physical coordinates.

    Args:
        velocity_field: Floating Tensor ``[T, 3, D, H, W]``. Components are in
            ``(x, y, z)`` order and spatial axes index ``(z, y, x)``.
        snapshot_times: Strictly increasing physical times ``[T]``.
        domain_bounds: Bounds ``[3, 2]`` ordered ``x, y, z``. Each row is the
            inclusive physical interval ``(lower, upper)``; periodic storage
            may omit the sample at the upper bound even though the bound still
            describes the full period.
        boundary_mode: One of ``"periodic"``, ``"clamp"``, or ``"terminate"``.
        metadata: Optional solver- and dataset-specific descriptive values.

    ``snapshot_times`` and ``domain_bounds`` are normalized to the velocity
    Tensor's dtype and device. The velocity Tensor itself is not detached or
    copied, so a caller can deliberately retain an autograd connection.
    """

    velocity_field: torch.Tensor
    snapshot_times: torch.Tensor
    domain_bounds: torch.Tensor
    boundary_mode: str = "clamp"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        velocity = self.velocity_field
        if not isinstance(velocity, torch.Tensor):
            raise TypeError("velocity_field must be a torch.Tensor")
        if velocity.ndim != 5:
            raise ValueError(
                "velocity_field must have shape [T, 3, D, H, W]; "
                f"received {tuple(velocity.shape)}"
            )
        if velocity.shape[0] < 1 or velocity.shape[1] != 3:
            raise ValueError(
                "velocity_field must have shape [T, 3, D, H, W] with T >= 1; "
                f"received {tuple(velocity.shape)}"
            )
        if any(size < 2 for size in velocity.shape[2:]):
            raise ValueError(
                "D, H, and W must each be at least 2 for trilinear interpolation"
            )
        if not velocity.is_floating_point():
            raise TypeError("velocity_field must have a floating-point dtype")
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError("velocity_field must contain only finite values")

        if not isinstance(self.snapshot_times, torch.Tensor):
            raise TypeError("snapshot_times must be a torch.Tensor")
        times = self.snapshot_times.to(device=velocity.device, dtype=velocity.dtype)
        if times.ndim != 1 or times.shape[0] != velocity.shape[0]:
            raise ValueError(
                "snapshot_times must have shape [T] matching velocity_field; "
                f"received {tuple(times.shape)} for T={velocity.shape[0]}"
            )
        if not bool(torch.isfinite(times).all()):
            raise ValueError("snapshot_times must contain only finite values")
        if times.numel() > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValueError("snapshot_times must be strictly increasing")

        if not isinstance(self.domain_bounds, torch.Tensor):
            raise TypeError("domain_bounds must be a torch.Tensor")
        bounds = self.domain_bounds.to(device=velocity.device, dtype=velocity.dtype)
        if bounds.shape != (3, 2):
            raise ValueError(
                "domain_bounds must have shape [3, 2] in x, y, z order; "
                f"received {tuple(bounds.shape)}"
            )
        if not bool(torch.isfinite(bounds).all()):
            raise ValueError("domain_bounds must contain only finite values")
        if not bool((bounds[:, 1] > bounds[:, 0]).all()):
            raise ValueError("each domain upper bound must exceed its lower bound")

        boundary_mode = self.boundary_mode.strip().lower()
        if boundary_mode not in _BOUNDARY_MODES:
            choices = ", ".join(sorted(_BOUNDARY_MODES))
            raise ValueError(
                f"boundary_mode must be one of {choices}; received {self.boundary_mode!r}"
            )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "snapshot_times", times)
        object.__setattr__(self, "domain_bounds", bounds)
        object.__setattr__(self, "boundary_mode", boundary_mode)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def flow_field(self) -> torch.Tensor:
        """Alias matching the canonical dataset field key."""

        return self.velocity_field

    @property
    def observation_times(self) -> torch.Tensor:
        """Alias used by particle-advection APIs."""

        return self.snapshot_times

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "VelocitySnapshotData":
        """Return a snapshot bundle converted with :meth:`torch.Tensor.to`."""

        target_dtype = self.velocity_field.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; received {target_dtype}")
        return VelocitySnapshotData(
            velocity_field=self.velocity_field.to(device=device, dtype=target_dtype),
            snapshot_times=self.snapshot_times.to(device=device, dtype=target_dtype),
            domain_bounds=self.domain_bounds.to(device=device, dtype=target_dtype),
            boundary_mode=self.boundary_mode,
            metadata=self.metadata,
        )


@runtime_checkable
class CFDVelocityAdapter(Protocol):
    """Interface implemented by external CFD snapshot loaders.

    An adapter should perform any solver-specific I/O and axis conversion in
    :meth:`load`, returning the canonical ``[T, 3, D, H, W]`` representation.
    """

    def load(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> VelocitySnapshotData:
        """Load and normalize snapshots as PyTorch tensors."""


def _first_existing_key(
    keys: set[str],
    preferred: str,
    alternatives: tuple[str, ...],
    description: str,
) -> str:
    if preferred in keys:
        return preferred
    for alternative in alternatives:
        if alternative in keys:
            return alternative
    expected = ", ".join(repr(key) for key in (preferred, *alternatives))
    raise KeyError(
        f"NPZ archive does not contain {description}; expected one of {expected}"
    )


def _decode_metadata(raw_metadata: Any) -> dict[str, Any]:
    """Decode the prototype's scalar JSON metadata without enabling pickle."""

    if getattr(raw_metadata, "size", None) != 1:
        raise ValueError("NPZ metadata must be a scalar JSON string")
    value = raw_metadata.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(
            "NPZ metadata must be stored as a JSON string, not a pickled Python object"
        )
    decoded = json.loads(value)
    if decoded is None:
        return {}
    if not isinstance(decoded, dict):
        raise TypeError("decoded NPZ metadata must be a JSON object")
    return decoded


@dataclass(frozen=True, slots=True)
class NPZCFDAdapter:
    """Load canonical CFD snapshots from a portable ``.npz`` archive.

    The preferred keys are ``flow_field`` and ``velocity_times``.
    Legacy and format-neutral time keys remain accepted as fallbacks. Metadata,
    when present, must be a scalar JSON string so loading
    remains safe with ``allow_pickle=False``.

    Args:
        path: Input archive.
        velocity_key: Preferred key for ``[T, 3, D, H, W]`` data.
        time_key: Preferred key for snapshot times ``[T]``.
        bounds_key: Key for domain bounds ``[3, 2]``.
        metadata_key: Optional scalar-JSON metadata key.
        boundary_mode: Explicit boundary-mode override. When omitted, the
            loader checks metadata and otherwise defaults to ``"clamp"``.
    """

    path: str | Path
    velocity_key: str = "flow_field"
    time_key: str = "velocity_times"
    bounds_key: str = "domain_bounds"
    metadata_key: str = "metadata"
    boundary_mode: str | None = None

    def load(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> VelocitySnapshotData:
        """Read the archive at the I/O boundary and return validated tensors."""

        # NumPy is deliberately imported and used only within this I/O method.
        import numpy as np

        path = Path(self.path)
        if not path.is_file():
            raise FileNotFoundError(f"CFD snapshot archive does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            velocity_key = _first_existing_key(
                keys,
                self.velocity_key,
                ("velocity_field",),
                "a velocity field",
            )
            time_key = _first_existing_key(
                keys,
                self.time_key,
                ("snapshot_times", "observation_times", "times"),
                "snapshot times",
            )
            bounds_key = _first_existing_key(
                keys,
                self.bounds_key,
                (),
                "domain bounds",
            )

            # Copies detach the arrays from the archive before its file handle
            # closes and ensure torch receives writable contiguous storage.
            velocity_array = np.array(archive[velocity_key], copy=True)
            times_array = np.array(archive[time_key], copy=True)
            bounds_array = np.array(archive[bounds_key], copy=True)
            if self.metadata_key in keys:
                try:
                    metadata = _decode_metadata(archive[self.metadata_key])
                except ValueError as error:
                    if "Object arrays cannot be loaded" in str(error):
                        raise ValueError(
                            "NPZ metadata uses a pickled object; store metadata as a JSON string"
                        ) from error
                    raise
            else:
                metadata = {}

        velocity = torch.from_numpy(velocity_array)
        if not velocity.is_floating_point():
            velocity = velocity.to(dtype=torch.get_default_dtype())
        target_dtype = velocity.dtype if dtype is None else dtype
        if not target_dtype.is_floating_point:
            raise TypeError(f"dtype must be floating point; received {target_dtype}")
        velocity = velocity.to(device=device, dtype=target_dtype)
        times = torch.from_numpy(times_array).to(device=device, dtype=target_dtype)
        bounds = torch.from_numpy(bounds_array).to(device=device, dtype=target_dtype)

        boundary_mode = self.boundary_mode
        if boundary_mode is None:
            metadata_boundary = metadata.get("boundary_mode")
            if isinstance(metadata_boundary, str):
                boundary_mode = metadata_boundary
            elif metadata.get("periodic") is True:
                boundary_mode = "periodic"
            else:
                boundary_mode = "clamp"

        return VelocitySnapshotData(
            velocity_field=velocity,
            snapshot_times=times,
            domain_bounds=bounds,
            boundary_mode=boundary_mode,
            metadata=metadata,
        )


def load_velocity_snapshots(
    source: CFDVelocityAdapter | str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> VelocitySnapshotData:
    """Load snapshots from an adapter or directly from an NPZ path."""

    adapter: CFDVelocityAdapter
    if isinstance(source, (str, Path)):
        adapter = NPZCFDAdapter(source)
    else:
        adapter = source
    result = adapter.load(device=device, dtype=dtype)
    if not isinstance(result, VelocitySnapshotData):
        raise TypeError("CFD adapter load() must return VelocitySnapshotData")
    return result


# Descriptive aliases for callers that prefer format-neutral names.
VelocitySnapshotAdapter = CFDVelocityAdapter
NPZVelocitySnapshotAdapter = NPZCFDAdapter


__all__ = [
    "CFDVelocityAdapter",
    "NPZCFDAdapter",
    "NPZVelocitySnapshotAdapter",
    "VelocitySnapshotAdapter",
    "VelocitySnapshotData",
    "load_velocity_snapshots",
]
