"""Case-disjoint datasets for sparse projected-particle flow reconstruction.

The existing Taichi-LBM3D archives store one steady velocity field together
with projected trajectories for a pool of physical particles.  This module
turns a directory of those independent *flow cases* into training examples.
Each example exposes exactly ``num_particles`` projected trajectories while
keeping the complete velocity field as the reconstruction target.

The manifest split is deliberately based on ``flow_group_id``. Particle
subsets from one physical flow must never be divided across train, validation,
and test: doing so would measure memorization of a flow rather than
generalization to an unseen flow.
The only exception is a manifest explicitly marked as a single-case smoke
test.  Such a manifest is useful for exercising code paths, but is not valid
scientific evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


_SPLIT_NAMES = ("train", "validation", "test")
_REQUIRED_CASE_KEYS = frozenset(
    {
        "flow_field",
        "trajectories_2d",
        "projection_matrix",
        "observation_mask",
        "observation_times",
        "domain_bounds",
        "solid_mask",
    }
)


def _canonical_command_float(value: object, *, name: str) -> str:
    """Return the precision used by the collection generator command line."""

    if isinstance(value, bool):
        raise ValueError(f"physical metadata {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"physical metadata {name} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"physical metadata {name} must be finite")
    return format(number, ".12g")


def physical_flow_identity(metadata: Mapping[str, Any]) -> dict[str, object]:
    """Extract the physical simulation identity used for leakage-safe splits.

    Snapshot/warm-up iteration and particle seed are intentionally absent:
    they select a time or an observation from one deterministic simulation run,
    rather than defining a new physical run.
    """

    field = metadata.get("field")
    grid = metadata.get("grid_size_xyz")
    lid_face = metadata.get("lid_face")
    lid_velocity = metadata.get("lid_velocity_xyz")
    niu = metadata.get("upstream_niu_parameter")
    if not isinstance(field, str) or not field.strip():
        raise ValueError("physical metadata field must be a non-empty string")
    if (
        not isinstance(grid, (list, tuple))
        or len(grid) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in grid
        )
        or any(int(value) < 2 for value in grid)
    ):
        raise ValueError("physical metadata grid_size_xyz must contain three integers >= 2")
    if not isinstance(lid_face, str) or not lid_face.strip():
        raise ValueError("physical metadata lid_face must be a non-empty string")
    if not isinstance(lid_velocity, (list, tuple)) or len(lid_velocity) != 3:
        raise ValueError("physical metadata lid_velocity_xyz must contain three values")
    return {
        "field": field.strip(),
        "grid_size_xyz": [int(value) for value in grid],
        "lid_face": lid_face.strip().lower(),
        "lid_velocity_xyz": [
            _canonical_command_float(value, name="lid_velocity_xyz")
            for value in lid_velocity
        ],
        "upstream_niu_parameter": _canonical_command_float(
            niu, name="upstream_niu_parameter"
        ),
    }


def derive_physical_flow_group_id(metadata: Mapping[str, Any]) -> str:
    """Derive a stable group ID from physical metadata only."""

    encoded = json.dumps(
        physical_flow_identity(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "flow_" + hashlib.sha256(encoded).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SparseFlowCase:
    """One manifest entry resolved to an absolute archive path."""

    case_id: str
    path: Path
    flow_group_id: str = ""

    def __post_init__(self) -> None:
        # Older hand-written manifests have no group field.  Treating a case
        # as its own group preserves their original, leakage-safe semantics.
        if not self.flow_group_id:
            object.__setattr__(self, "flow_group_id", self.case_id)


@dataclass(frozen=True, slots=True)
class FlowNormalizationStats:
    """Per-velocity-component normalization computed from training flows."""

    mean: Tensor
    std: Tensor
    count: int
    fluid_only: bool = True

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(self.std, dtype=torch.float32).detach().clone()
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("normalization mean and std must each have shape [3]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("normalization statistics must be finite")
        if not bool((std > 0).all()):
            raise ValueError("normalization std must be strictly positive")
        if int(self.count) < 1:
            raise ValueError("normalization count must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "fluid_only", bool(self.fluid_only))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
            "fluid_only": self.fluid_only,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "FlowNormalizationStats":
        """Construct statistics previously produced by :meth:`to_dict`."""

        return cls(
            mean=torch.as_tensor(values["mean"], dtype=torch.float32),
            std=torch.as_tensor(values["std"], dtype=torch.float32),
            count=int(values["count"]),
            fluid_only=bool(values.get("fluid_only", True)),
        )


def _decode_metadata(raw: np.ndarray | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if raw.shape != ():
        raise ValueError("metadata must be a scalar JSON string")
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError("metadata must be stored as a JSON string")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError("metadata JSON must decode to an object")
    return decoded


def _normalise_entry(
    entry: object, *, split: str, base_dir: Path, require_flow_group: bool
) -> SparseFlowCase:
    if not isinstance(entry, Mapping):
        raise TypeError(f"manifest {split} entries must be objects")
    case_id = entry.get("case_id")
    raw_path = entry.get("path")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"manifest {split} case_id must be a non-empty string")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"manifest {split} path must be a non-empty string")
    if "flow_group_id" in entry:
        flow_group_id = entry.get("flow_group_id")
        if not isinstance(flow_group_id, str) or not flow_group_id.strip():
            raise ValueError(
                f"manifest {split} flow_group_id must be a non-empty string"
            )
        flow_group_id = flow_group_id.strip()
    elif require_flow_group:
        raise ValueError(
            f"manifest {split} entries in scientific mode must explicitly provide "
            "flow_group_id"
        )
    else:
        flow_group_id = case_id.strip()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return SparseFlowCase(
        case_id=case_id.strip(),
        path=path.resolve(),
        flow_group_id=flow_group_id,
    )


def validate_sparse_flow_manifest(
    manifest: Mapping[str, object],
    *,
    base_dir: str | Path = ".",
    require_files: bool = True,
) -> dict[str, list[SparseFlowCase]]:
    """Validate a manifest and return normalized case entries.

    Scientific manifests must have three non-empty lists that are disjoint by
    physical flow group. Case IDs, resolved paths, and ``flow_group_id`` are
    checked, so a different particle seed or filename cannot hide reuse of the
    same ground-truth flow. Multiple cases from one group are allowed within a
    split. Overlap is accepted only when the manifest contains both
    ``mode: \"smoke_test\"`` and
    ``allow_overlap_for_smoke_test: true``.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a JSON object")
    mode = manifest.get("mode", "scientific")
    if mode not in {"scientific", "smoke_test"}:
        raise ValueError("manifest mode must be 'scientific' or 'smoke_test'")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("manifest must contain a splits object")

    root = Path(base_dir).expanduser().resolve()
    normalized: dict[str, list[SparseFlowCase]] = {}
    for split in _SPLIT_NAMES:
        entries = splits.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest split {split!r} must be a non-empty list")
        cases = [
            _normalise_entry(
                entry,
                split=split,
                base_dir=root,
                require_flow_group=mode == "scientific",
            )
            for entry in entries
        ]
        ids = [case.case_id for case in cases]
        paths = [case.path for case in cases]
        if len(set(ids)) != len(ids):
            raise ValueError(f"manifest split {split!r} contains duplicate case IDs")
        if len(set(paths)) != len(paths):
            raise ValueError(f"manifest split {split!r} contains duplicate paths")
        if require_files:
            missing = [str(case.path) for case in cases if not case.path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"manifest split {split!r} references missing case(s): "
                    + ", ".join(missing)
                )
        normalized[split] = cases

    overlap_messages: list[str] = []
    for left_index, left_name in enumerate(_SPLIT_NAMES):
        for right_name in _SPLIT_NAMES[left_index + 1 :]:
            left_ids = {case.case_id for case in normalized[left_name]}
            right_ids = {case.case_id for case in normalized[right_name]}
            repeated_ids = sorted(left_ids & right_ids)
            left_paths = {case.path for case in normalized[left_name]}
            right_paths = {case.path for case in normalized[right_name]}
            repeated_paths = sorted(str(path) for path in left_paths & right_paths)
            left_groups = {
                case.flow_group_id for case in normalized[left_name]
            }
            right_groups = {
                case.flow_group_id for case in normalized[right_name]
            }
            repeated_groups = sorted(left_groups & right_groups)
            if repeated_ids:
                overlap_messages.append(
                    f"{left_name}/{right_name} IDs: {', '.join(repeated_ids)}"
                )
            if repeated_paths:
                overlap_messages.append(
                    f"{left_name}/{right_name} paths: {', '.join(repeated_paths)}"
                )
            if repeated_groups:
                overlap_messages.append(
                    f"{left_name}/{right_name} flow groups: "
                    + ", ".join(repeated_groups)
                )

    smoke_overlap = (
        manifest.get("mode") == "smoke_test"
        and manifest.get("allow_overlap_for_smoke_test") is True
    )
    if overlap_messages and not smoke_overlap:
        raise ValueError("case split leakage detected: " + "; ".join(overlap_messages))
    if smoke_overlap and not overlap_messages:
        raise ValueError(
            "a smoke-test overlap marker is only valid when the splits overlap"
        )
    if not overlap_messages and mode != "scientific":
        raise ValueError("a disjoint manifest must use mode 'scientific'")
    return normalized


def load_sparse_flow_manifest(
    path: str | Path, *, require_files: bool = True
) -> tuple[dict[str, object], dict[str, list[SparseFlowCase]]]:
    """Load and validate a sparse-flow manifest."""

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases = validate_sparse_flow_manifest(
        manifest,
        base_dir=manifest_path.parent,
        require_files=require_files,
    )
    return manifest, cases


def _validate_case_arrays(
    arrays: Mapping[str, np.ndarray], *, path: Path
) -> dict[str, Any]:
    missing = sorted(_REQUIRED_CASE_KEYS.difference(arrays))
    if missing:
        raise ValueError(f"case {path} is missing key(s): {', '.join(missing)}")

    flow = arrays["flow_field"]
    tracks = arrays["trajectories_2d"]
    projection = arrays["projection_matrix"]
    mask = arrays["observation_mask"]
    times = arrays["observation_times"]
    bounds = arrays["domain_bounds"]
    solid = arrays["solid_mask"]
    if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"case {path} flow_field must be [1,3,D,H,W]")
    if min(flow.shape[-3:]) < 2 or not np.issubdtype(flow.dtype, np.floating):
        raise ValueError(f"case {path} flow_field must be floating with D,H,W >= 2")
    if not np.isfinite(flow).all():
        raise ValueError(f"case {path} flow_field contains non-finite values")
    if tracks.ndim != 5 or tracks.shape[-1] != 2:
        raise ValueError(f"case {path} trajectories_2d must be [S,V,N,T,2]")
    if any(int(value) < 1 for value in tracks.shape[:-1]):
        raise ValueError(f"case {path} trajectories_2d dimensions S,V,N,T must be non-empty")
    if tracks.shape[3] < 2:
        raise ValueError(f"case {path} trajectories_2d must contain at least two times")
    if not np.issubdtype(tracks.dtype, np.floating) or not np.isfinite(tracks).all():
        raise ValueError(f"case {path} trajectories_2d must be finite floating data")
    if projection.shape != (tracks.shape[1], 2, 3):
        raise ValueError(f"case {path} projection_matrix must be [V,2,3]")
    if mask.shape != tracks.shape[:-1]:
        raise ValueError(f"case {path} observation_mask must be [S,V,N,T]")
    if times.shape != (tracks.shape[3],):
        raise ValueError(f"case {path} observation_times must match T")
    if (
        bounds.shape != (3, 2)
        or not np.issubdtype(bounds.dtype, np.number)
        or not np.isfinite(bounds).all()
        or not np.all(bounds[:, 1] > bounds[:, 0])
    ):
        raise ValueError(f"case {path} domain_bounds must be increasing [3,2]")
    if solid.shape != flow.shape[-3:]:
        raise ValueError(f"case {path} solid_mask must match flow spatial shape")
    if (
        not np.issubdtype(projection.dtype, np.number)
        or not np.issubdtype(times.dtype, np.number)
        or not np.isfinite(projection).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError(f"case {path} projection/times contain non-finite values")
    if not np.all(np.diff(times.astype(np.float64, copy=False)) > 0.0):
        raise ValueError(f"case {path} observation_times must be strictly increasing")
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number) or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"case {path} observation_mask must contain only 0/1")
    if solid.dtype != np.bool_:
        if not np.issubdtype(solid.dtype, np.number) or not np.isin(solid, (0, 1)).all():
            raise ValueError(f"case {path} solid_mask must contain only 0/1")

    sample_count, _, maximum_particles, _, _ = tracks.shape
    counts = arrays.get("particle_counts")
    if counts is None:
        validated_counts = np.full(sample_count, maximum_particles, dtype=np.int64)
    else:
        raw_counts = np.asarray(counts)
        if raw_counts.dtype == np.bool_ or not np.issubdtype(raw_counts.dtype, np.integer):
            raise ValueError(f"case {path} particle_counts must contain integers")
        flat_counts = raw_counts.reshape(-1)
        if flat_counts.size == 1:
            validated_counts = np.full(sample_count, int(flat_counts[0]), dtype=np.int64)
        elif flat_counts.size == sample_count:
            validated_counts = flat_counts.astype(np.int64, copy=False)
        else:
            raise ValueError(
                f"case {path} particle_counts must be scalar or contain one value per sample"
            )
        if np.any(validated_counts < 1) or np.any(validated_counts > maximum_particles):
            raise ValueError(f"case {path} particle_counts contains a value outside stored bounds")
    mask_bool = mask.astype(bool, copy=False)
    for sample_index, count in enumerate(validated_counts.tolist()):
        valid_particles = mask_bool[sample_index, :, :count, :].any(axis=(0, 2))
        if not bool(valid_particles.all()):
            raise ValueError(
                f"case {path} observation_mask leaves a counted particle entirely unobserved"
            )
        if count < maximum_particles and bool(mask_bool[sample_index, :, count:, :].any()):
            raise ValueError(
                f"case {path} observation_mask marks padded particles as observed"
            )
    metadata = _decode_metadata(arrays.get("metadata"))
    return metadata


def inspect_sparse_flow_case(path: str | Path) -> dict[str, object]:
    """Validate one archive without reading hidden 3D trajectory/depth inputs."""

    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise FileNotFoundError(f"sparse flow case does not exist: {case_path}")
    with np.load(case_path, allow_pickle=False) as archive:
        arrays = {
            key: np.array(archive[key], copy=False)
            for key in _REQUIRED_CASE_KEYS
            if key in archive.files
        }
        if "metadata" in archive.files:
            arrays["metadata"] = np.array(archive["metadata"], copy=False)
        if "particle_counts" in archive.files:
            arrays["particle_counts"] = np.array(
                archive["particle_counts"], copy=False
            )
        metadata = _validate_case_arrays(arrays, path=case_path)
        tracks_shape = tuple(int(value) for value in arrays["trajectories_2d"].shape)
        flow_shape = tuple(int(value) for value in arrays["flow_field"].shape)
    return {
        "path": case_path,
        "flow_shape": flow_shape,
        "tracks_shape": tracks_shape,
        "projection_matrix": np.array(arrays["projection_matrix"], copy=True),
        "observation_times": np.array(arrays["observation_times"], copy=True),
        "domain_bounds": np.array(arrays["domain_bounds"], copy=True),
        "metadata": metadata,
    }


def _face_mask(shape: Sequence[int], face: str) -> Tensor:
    axis_map = {"x": 2, "y": 1, "z": 0}
    try:
        axis_name, side = face.lower().split("=", maxsplit=1)
        axis = axis_map[axis_name]
    except (KeyError, ValueError) as error:
        raise ValueError(f"unsupported boundary face {face!r}") from error
    if side not in {"min", "max"}:
        raise ValueError(f"unsupported boundary face {face!r}")
    result = torch.zeros(tuple(int(value) for value in shape), dtype=torch.bool)
    selection = [slice(None), slice(None), slice(None)]
    selection[axis] = 0 if side == "min" else result.shape[axis] - 1
    result[tuple(selection)] = True
    return result


def _boundary_tensors(
    solid_mask: Tensor, metadata: Mapping[str, Any], dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    shape = tuple(int(value) for value in solid_mask.shape)
    fixed = solid_mask.to(dtype=torch.bool).clone()
    values = torch.zeros((3, *shape), dtype=dtype)
    lid_face = metadata.get("lid_face")
    lid_velocity = metadata.get("lid_velocity_xyz")
    if lid_face is None and lid_velocity is None:
        return fixed, values
    if lid_face is None or lid_velocity is None:
        raise ValueError(
            "metadata must provide lid_face and lid_velocity_xyz together"
        )
    if not isinstance(lid_face, str):
        raise ValueError("metadata lid_face must be a string")
    velocity = torch.as_tensor(lid_velocity, dtype=dtype)
    if velocity.shape != (3,) or not bool(torch.isfinite(velocity).all()):
        raise ValueError("metadata lid_velocity_xyz must contain three finite values")
    lid = _face_mask(shape, lid_face) & ~fixed
    fixed |= lid
    values[:, lid] = velocity[:, None]
    return fixed, values


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def _particle_count(arrays: Mapping[str, np.ndarray], sample_index: int) -> int:
    maximum = int(arrays["trajectories_2d"].shape[2])
    counts = arrays.get("particle_counts")
    if counts is None:
        return maximum
    flat = np.asarray(counts).reshape(-1)
    if flat.size == 1:
        count = int(flat[0])
    elif flat.size == arrays["trajectories_2d"].shape[0]:
        count = int(flat[sample_index])
    else:
        raise ValueError("particle_counts must be scalar or contain one value per sample")
    if count < 1 or count > maximum:
        raise ValueError("particle_counts contains a value outside stored bounds")
    return count


class SparseFlowDataset(Dataset[dict[str, object]]):
    """Select deterministic sparse particle sets from independent flow cases.

    Calling :meth:`set_epoch` changes the sampled particle set in a reproducible
    way.  Repeating an epoch, process, or worker with the same seed yields the
    same source sample and ordered particle indices.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        *,
        num_particles: int = 2,
        num_probe_particles: int = 0,
        observations_per_flow: int = 1,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        normalize: bool = False,
        normalization_stats: FlowNormalizationStats | None = None,
    ) -> None:
        if split not in _SPLIT_NAMES:
            raise ValueError(f"split must be one of {', '.join(_SPLIT_NAMES)}")
        if (
            isinstance(num_particles, bool)
            or int(num_particles) != num_particles
            or int(num_particles) < 1
        ):
            raise ValueError("num_particles must be positive")
        if (
            isinstance(observations_per_flow, bool)
            or int(observations_per_flow) != observations_per_flow
            or int(observations_per_flow) < 1
        ):
            raise ValueError("observations_per_flow must be positive")
        if (
            isinstance(num_probe_particles, bool)
            or int(num_probe_particles) != num_probe_particles
            or int(num_probe_particles) < 0
        ):
            raise ValueError("num_probe_particles must be non-negative")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest, splits = load_sparse_flow_manifest(self.manifest_path)
        self.cases = splits[split]
        self.split = split
        self.num_particles = int(num_particles)
        self.num_probe_particles = int(num_probe_particles)
        self.observations_per_flow = int(observations_per_flow)
        self.seed = int(seed)
        self.dtype = dtype
        self.epoch = 0
        self.is_smoke_test = self.manifest.get("mode") == "smoke_test"
        if normalize and normalization_stats is None:
            normalization_stats = compute_train_flow_stats(self.manifest_path)
        self.normalization_stats = normalization_stats
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.cases) * self.observations_per_flow

    def set_epoch(self, epoch: int) -> None:
        """Choose the deterministic random particle pairing for an epoch."""

        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _load_case(self, case: SparseFlowCase) -> dict[str, object]:
        with np.load(case.path, allow_pickle=False) as archive:
            arrays: dict[str, np.ndarray] = {
                key: np.array(archive[key], copy=True)
                for key in _REQUIRED_CASE_KEYS
                if key in archive.files
            }
            if "metadata" in archive.files:
                arrays["metadata"] = np.array(archive["metadata"], copy=True)
            if "particle_counts" in archive.files:
                arrays["particle_counts"] = np.array(
                    archive["particle_counts"], copy=True
                )
        metadata = _validate_case_arrays(arrays, path=case.path)
        return {"arrays": arrays, "metadata": metadata}

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        case_index = index // self.observations_per_flow
        observation_index = index % self.observations_per_flow
        case = self.cases[case_index]
        loaded = self._load_case(case)
        arrays = loaded["arrays"]
        metadata = loaded["metadata"]
        assert isinstance(arrays, dict)
        assert isinstance(metadata, dict)

        tracks_all = arrays["trajectories_2d"]
        sample_count = int(tracks_all.shape[0])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                self.seed,
                self.epoch,
                self.split,
                case.case_id,
                observation_index,
            )
        )
        source_sample_index = int(
            torch.randint(sample_count, (1,), generator=generator).item()
        )
        count = _particle_count(arrays, source_sample_index)
        if self.num_particles + self.num_probe_particles > count:
            raise ValueError(
                f"case {case.case_id!r} stores {count} valid particles, "
                "fewer than requested observed + probe particles "
                f"({self.num_particles} + {self.num_probe_particles})"
            )
        permutation = torch.randperm(count, generator=generator)
        particle_indices = permutation[: self.num_particles]
        probe_particle_indices = (
            permutation[-self.num_probe_particles :]
            if self.num_probe_particles
            else permutation[:0]
        )
        indices_numpy = particle_indices.numpy()
        probe_indices_numpy = probe_particle_indices.numpy()

        flow = torch.from_numpy(arrays["flow_field"][0]).to(dtype=self.dtype)
        if self.normalize:
            assert self.normalization_stats is not None
            flow = normalize_flow(flow, self.normalization_stats)
        tracks = torch.from_numpy(
            tracks_all[source_sample_index][:, indices_numpy]
        ).to(dtype=self.dtype)
        mask = torch.from_numpy(
            arrays["observation_mask"][source_sample_index][:, indices_numpy]
        ).to(dtype=torch.bool)
        projection = torch.from_numpy(arrays["projection_matrix"]).to(dtype=self.dtype)
        times = torch.from_numpy(arrays["observation_times"]).to(dtype=self.dtype)
        bounds = torch.from_numpy(arrays["domain_bounds"]).to(dtype=self.dtype)
        solid = torch.from_numpy(arrays["solid_mask"]).to(dtype=torch.bool)
        boundary_mask, boundary_values = _boundary_tensors(
            solid, metadata, self.dtype
        )

        result: dict[str, object] = {
            "flow": flow,
            "tracks": tracks,
            "mask": mask,
            "projection": projection,
            "times": times,
            "bounds": bounds,
            "solid_mask": solid,
            "boundary_mask": boundary_mask,
            "boundary_values": boundary_values,
            "case_id": case.case_id,
            "flow_group_id": case.flow_group_id,
            "case_index": torch.tensor(case_index, dtype=torch.long),
            "observation_index": torch.tensor(observation_index, dtype=torch.long),
            "source_sample_index": torch.tensor(
                source_sample_index, dtype=torch.long
            ),
            "particle_indices": particle_indices,
        }
        if self.num_probe_particles:
            result.update(
                {
                    "probe_tracks": torch.from_numpy(
                        tracks_all[source_sample_index][:, probe_indices_numpy]
                    ).to(dtype=self.dtype),
                    "probe_mask": torch.from_numpy(
                        arrays["observation_mask"][source_sample_index][
                            :, probe_indices_numpy
                        ]
                    ).to(dtype=torch.bool),
                    "probe_particle_indices": probe_particle_indices,
                }
            )
        return result


def _component_shape(values: Tensor) -> tuple[int, ...]:
    if values.ndim < 4 or values.shape[-4] != 3:
        raise ValueError("flow must end with [3,D,H,W]")
    return (1,) * (values.ndim - 4) + (3, 1, 1, 1)


def normalize_flow(flow: Tensor, stats: FlowNormalizationStats) -> Tensor:
    """Normalize ``[...,3,D,H,W]`` velocity fields component wise."""

    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")
    shape = _component_shape(flow)
    mean = stats.mean.to(device=flow.device, dtype=flow.dtype).reshape(shape)
    std = stats.std.to(device=flow.device, dtype=flow.dtype).reshape(shape)
    return (flow - mean) / std


def denormalize_flow(flow: Tensor, stats: FlowNormalizationStats) -> Tensor:
    """Invert :func:`normalize_flow` for ``[...,3,D,H,W]`` tensors."""

    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")
    shape = _component_shape(flow)
    mean = stats.mean.to(device=flow.device, dtype=flow.dtype).reshape(shape)
    std = stats.std.to(device=flow.device, dtype=flow.dtype).reshape(shape)
    return flow * std + mean


def compute_train_flow_stats(
    manifest_path: str | Path,
    *,
    fluid_only: bool = True,
    minimum_std: float = 1e-6,
) -> FlowNormalizationStats:
    """Compute population mean/std from train cases only.

    By default solid nodes are excluded, preventing no-slip padding from
    dominating the learned velocity scale.  The returned count is the number
    of spatial values contributing to each component.
    """

    if not math.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError("minimum_std must be finite and positive")
    _, splits = load_sparse_flow_manifest(manifest_path)
    total = torch.zeros(3, dtype=torch.float64)
    total_square = torch.zeros(3, dtype=torch.float64)
    count = 0
    for case in splits["train"]:
        with np.load(case.path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = sorted({"flow_field", "solid_mask"}.difference(keys))
            if missing:
                raise ValueError(
                    f"case {case.path} is missing key(s): {', '.join(missing)}"
                )
            flow = torch.from_numpy(np.array(archive["flow_field"], copy=True))
            solid = torch.from_numpy(np.array(archive["solid_mask"], copy=True))
        if flow.ndim != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
            raise ValueError(f"case {case.path} flow_field must be [1,3,D,H,W]")
        values = flow[0].to(torch.float64)
        if fluid_only:
            if tuple(solid.shape) != tuple(values.shape[-3:]):
                raise ValueError(f"case {case.path} solid_mask shape is invalid")
            valid = ~solid.to(torch.bool)
            if not bool(valid.any()):
                raise ValueError(f"case {case.path} has no fluid nodes")
            values = values[:, valid]
        else:
            values = values.reshape(3, -1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"case {case.path} flow_field contains non-finite values")
        total += values.sum(dim=1)
        total_square += values.square().sum(dim=1)
        count += int(values.shape[1])
    if count < 1:
        raise ValueError("training split contains no velocity values")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(minimum_std)
    return FlowNormalizationStats(
        mean=mean.to(torch.float32),
        std=std.to(torch.float32),
        count=count,
        fluid_only=fluid_only,
    )


__all__ = [
    "FlowNormalizationStats",
    "SparseFlowCase",
    "SparseFlowDataset",
    "compute_train_flow_stats",
    "denormalize_flow",
    "derive_physical_flow_group_id",
    "inspect_sparse_flow_case",
    "load_sparse_flow_manifest",
    "normalize_flow",
    "physical_flow_identity",
    "validate_sparse_flow_manifest",
]
