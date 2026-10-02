"""Aligned spatial Tucker coordinates for three-component velocity fields.

This is an independent, task-specific implementation inspired by DiffATS
(https://arxiv.org/abs/2605.09275). The component axis is NOT compressed.
Spatial factors are aligned to train-only Grassmann chordal medoids. A generated
core and generated factors are decoded together, without a QR operation that
would change the field unless its triangular factor were propagated to the core.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .recovery import divergence_mse
from .sparse_dataset import (
    FlowNormalizationStats,
    compute_train_flow_stats,
    denormalize_flow,
    inspect_sparse_flow_case,
    load_sparse_flow_manifest,
    normalize_flow,
)


FORMAT_VERSION = 1
_SPLITS = ("train", "validation", "test")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(values: Tensor, name: str) -> None:
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def _mode_product(values: Tensor, matrix: Tensor, mode: int) -> Tensor:
    """Multiply an unbatched tensor by ``matrix[out,in]`` along one axis."""
    return torch.tensordot(matrix, values, dims=([1], [mode])).movedim(0, mode)


def _spatial_shape(shape: Sequence[int], rank: int) -> tuple[int, int, int]:
    if len(shape) != 3 or any(isinstance(n, bool) or int(n) != n or n < 2 for n in shape):
        raise ValueError("spatial_shape must contain three integers >= 2")
    result = tuple(int(n) for n in shape)
    if isinstance(rank, bool) or int(rank) != rank or not 1 <= rank <= min(result):
        raise ValueError("rank must be an integer between 1 and the smallest spatial axis")
    return result


@torch.no_grad()
def spatial_tucker(
    field: Tensor, rank: int, *, hooi_iterations: int = 2
) -> tuple[Tensor, list[Tensor]]:
    """HOSVD initialization followed by a fixed, recorded number of HOOI sweeps.

    ``field`` is [3,D,H,W]. No claim of convergence is made after a fixed number
    of sweeps. SVD operations run in float32, including inside autocast callers.
    """
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError("field must have shape [3,D,H,W]")
    _spatial_shape(field.shape[1:], rank)
    if isinstance(hooi_iterations, bool) or int(hooi_iterations) != hooi_iterations or hooi_iterations < 0:
        raise ValueError("hooi_iterations must be a non-negative integer")
    _finite(field, "field")
    with torch.autocast(device_type=field.device.type, enabled=False):
        field = field.float()
        factors = []
        for mode in (1, 2, 3):
            unfolding = field.movedim(mode, 0).reshape(field.shape[mode], -1)
            u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
            factors.append(u[:, :rank])
        for _ in range(hooi_iterations):
            for axis in range(3):
                projected = field
                for other in range(3):
                    if other != axis:
                        projected = _mode_product(projected, factors[other].T, other + 1)
                unfolding = projected.movedim(axis + 1, 0).reshape(field.shape[axis + 1], -1)
                u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
                factors[axis] = u[:, :rank]
        core = field
        for axis, factor in enumerate(factors):
            core = _mode_product(core, factor.T, axis + 1)
    return core, factors


@torch.no_grad()
def align_tucker(
    core: Tensor, factors: Sequence[Tensor], anchors: Sequence[Tensor]
) -> tuple[Tensor, list[Tensor], list[float]]:
    """Apply orthogonal Procrustes and consistently rotate the core.

    The minimum singular value of each factor/anchor overlap is returned as a
    diagnostic: near-zero overlap makes the gauge alignment ill-conditioned.
    """
    if len(factors) != 3 or len(anchors) != 3:
        raise ValueError("three spatial factors and anchors are required")
    aligned = []
    overlaps = []
    with torch.autocast(device_type=core.device.type, enabled=False):
        result = core.float()
        for axis, (factor, anchor) in enumerate(zip(factors, anchors)):
            if factor.shape != anchor.shape or factor.ndim != 2:
                raise ValueError("factor and anchor shapes must match")
            u, singular, vh = torch.linalg.svd(factor.float().T @ anchor.float(), full_matrices=False)
            rotation = u @ vh
            aligned.append(factor.float() @ rotation)
            result = _mode_product(result, rotation.T, axis + 1)
            overlaps.append(float(singular.min()))
    return result, aligned, overlaps


@torch.no_grad()
def select_medoid_anchors(
    training_factors: Sequence[Sequence[Tensor]],
) -> tuple[list[Tensor], list[int]]:
    """Select observed train factors minimizing summed squared chordal distance.

    For orthonormal U, its projector UU^T is invariant to sign/rotation. Using
    the mean projector computes the exact medoid objective without a quadratic
    number of pairwise SVDs. Deterministic ties choose the first training case.
    """
    if not training_factors or any(len(factors) != 3 for factors in training_factors):
        raise ValueError("training_factors must contain at least one triplet")
    anchors, indices = [], []
    for axis in range(3):
        stack = torch.stack([factors[axis].detach().cpu().double() for factors in training_factors])
        projectors = stack @ stack.transpose(-1, -2)
        centroid = projectors.mean(0)
        distances = (projectors - centroid).square().sum((-1, -2))
        index = int(distances.argmin())
        anchors.append(stack[index].float())
        indices.append(index)
    return anchors, indices


class TensorSpaceCodec(nn.Module):
    """Normalize, pack and explicitly reconstruct aligned Tucker primitives.

    Input fields and decoded fields are in the dataset's component-normalized
    coordinates. The packed latent additionally has train-only mean/std scaling.
    Packing order is core [3,r,r,r], then depth, height and width factors.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        rank: int,
        anchors: Sequence[Tensor],
        latent_mean: Tensor | None = None,
        latent_std: Tensor | None = None,
        *,
        hooi_iterations: int = 2,
    ) -> None:
        super().__init__()
        self.spatial_shape = _spatial_shape(spatial_shape, rank)
        self.rank = int(rank)
        if isinstance(hooi_iterations, bool) or int(hooi_iterations) != hooi_iterations or hooi_iterations < 0:
            raise ValueError("hooi_iterations must be a non-negative integer")
        self.hooi_iterations = int(hooi_iterations)
        self.latent_dim = 3 * self.rank**3 + sum(self.spatial_shape) * self.rank
        if len(anchors) != 3:
            raise ValueError("three anchor matrices are required")
        for axis, (size, anchor) in enumerate(zip(self.spatial_shape, anchors)):
            anchor = torch.as_tensor(anchor).detach().float().clone()
            if anchor.shape != (size, self.rank):
                raise ValueError("anchor shape does not match spatial_shape and rank")
            _finite(anchor, "anchor")
            identity = torch.eye(self.rank, device=anchor.device)
            if not torch.allclose(anchor.T @ anchor, identity, atol=2e-4, rtol=2e-4):
                raise ValueError("anchor columns must be orthonormal")
            self.register_buffer(f"anchor_{axis}", anchor)
        mean = torch.zeros(self.latent_dim) if latent_mean is None else torch.as_tensor(latent_mean).detach().float().clone()
        std = torch.ones(self.latent_dim) if latent_std is None else torch.as_tensor(latent_std).detach().float().clone()
        for name, values in (("latent_mean", mean), ("latent_std", std)):
            if values.shape != (self.latent_dim,):
                raise ValueError(f"{name} must have shape [latent_dim]")
            _finite(values, name)
        if not bool((std > 0).all()):
            raise ValueError("latent_std must be strictly positive")
        self.register_buffer("latent_mean", mean)
        self.register_buffer("latent_std", std)

    @property
    def anchors(self) -> list[Tensor]:
        return [getattr(self, f"anchor_{axis}") for axis in range(3)]

    def config(self) -> dict[str, Any]:
        return {"format_version": FORMAT_VERSION, "spatial_shape": list(self.spatial_shape),
                "rank": self.rank, "hooi_iterations": self.hooi_iterations}

    def to_artifact(self) -> dict[str, Any]:
        """Compact checkpoint payload, excluding cached dataset targets."""
        return {"codec_config": self.config(), "codec_state_dict": {
            key: value.detach().cpu().clone() for key, value in self.state_dict().items()
        }}

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "TensorSpaceCodec":
        config = artifact["codec_config"]
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported tensor-space codec format")
        state = artifact["codec_state_dict"]
        codec = cls(config["spatial_shape"], config["rank"],
                    [state[f"anchor_{axis}"] for axis in range(3)],
                    state["latent_mean"], state["latent_std"],
                    hooi_iterations=config["hooi_iterations"])
        codec.load_state_dict(state, strict=True)
        return codec

    def pack(self, core: Tensor, factors: Sequence[Tensor]) -> Tensor:
        if core.shape != (3, self.rank, self.rank, self.rank):
            raise ValueError("core shape does not match rank")
        if len(factors) != 3 or any(f.shape != (n, self.rank) for f, n in zip(factors, self.spatial_shape)):
            raise ValueError("factor shapes do not match codec")
        raw = torch.cat([core.reshape(-1), *(factor.reshape(-1) for factor in factors)]).float()
        return (raw - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def encode(self, normalized_flow: Tensor) -> Tensor:
        if normalized_flow.ndim != 5 or tuple(normalized_flow.shape[1:]) != (3, *self.spatial_shape):
            raise ValueError("normalized_flow must have shape [B,3,D,H,W] matching codec")
        if normalized_flow.shape[0] < 1:
            raise ValueError("batch must be non-empty")
        _finite(normalized_flow, "normalized_flow")
        result = []
        for field in normalized_flow:
            core, factors = spatial_tucker(field, self.rank, hooi_iterations=self.hooi_iterations)
            core, factors, _ = align_tucker(core, factors, self.anchors)
            result.append(self.pack(core, factors))
        return torch.stack(result)

    def decode(self, standardized_latent: Tensor) -> Tensor:
        """Differentiable float32 multilinear reconstruction, including under AMP."""
        if standardized_latent.ndim != 2 or standardized_latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [B,latent_dim]")
        _finite(standardized_latent, "latent")
        with torch.autocast(device_type=standardized_latent.device.type, enabled=False):
            raw = standardized_latent.float() * self.latent_std + self.latent_mean
            offset = 3 * self.rank**3
            core = raw[:, :offset].reshape(-1, 3, self.rank, self.rank, self.rank)
            factors = []
            for size in self.spatial_shape:
                end = offset + size * self.rank
                factors.append(raw[:, offset:end].reshape(-1, size, self.rank))
                offset = end
            # Staged contractions avoid a large opt_einsum dependency/path search.
            value = torch.einsum("bcijk,bdi->bcdjk", core, factors[0])
            value = torch.einsum("bcdjk,bhj->bcdhk", value, factors[1])
            value = torch.einsum("bcdhk,bwk->bcdhw", value, factors[2])
        _finite(value, "decoded flow")
        return value.float()


def _read_field(path: Path, stats: FlowNormalizationStats, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        field = torch.from_numpy(np.array(archive["flow_field"], copy=True)).to(device=device, dtype=torch.float32)
        solid = torch.from_numpy(np.array(archive["solid_mask"], copy=True)).to(device=device, dtype=torch.bool)
        bounds = torch.from_numpy(np.array(archive["domain_bounds"], copy=True)).to(device=device, dtype=torch.float32)
    return normalize_flow(field, stats), solid, bounds


def _reconstruction_metrics(reference: Tensor, prediction: Tensor, solid: Tensor, bounds: Tensor) -> dict[str, Any]:
    valid = ~solid
    if not bool(valid.any()):
        raise ValueError("case contains no fluid nodes")
    truth = reference[0, :, valid].double()
    error = (prediction - reference)[0, :, valid].double()
    eps = 1e-20
    result: dict[str, Any] = {
        "relative_l2_fluid": float(error.norm() / truth.norm().clamp_min(eps)),
        "component_relative_l2_fluid": (error.norm(dim=1) / truth.norm(dim=1).clamp_min(eps)).tolist(),
        "component_rmse_fluid": error.square().mean(dim=1).sqrt().tolist(),
    }
    try:
        result["reference_divergence_mse"] = float(divergence_mse(reference, bounds, valid))
        result["reconstruction_divergence_mse"] = float(divergence_mse(prediction, bounds, valid))
        result["divergence_error_mse"] = float(divergence_mse(prediction - reference, bounds, valid))
    except ValueError as error_message:
        if "at least 3" not in str(error_message) and "no valid divergence stencil" not in str(error_message):
            raise
        result["divergence_unavailable"] = str(error_message)
    for key, value in result.items():
        if isinstance(value, (int, float, list)) and not np.isfinite(value).all():
            raise ValueError(f"non-finite reconstruction metric {key}")
    return result


@torch.no_grad()
def build_tensor_space_artifact(
    manifest_path: str | Path,
    rank: int,
    *,
    device: str | torch.device = "cpu",
    hooi_iterations: int = 2,
    minimum_latent_std: float = 1e-4,
) -> dict[str, Any]:
    """Fit a train-only codec and cache targets for every manifest split.

    Held-out targets are encoded using the already fixed anchors and statistics.
    Only validation reconstruction metrics are reported, so the test split is
    not used for rank selection. CUDA preprocessing must run within Slurm.
    """
    device = torch.device(device)
    if device.type == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CUDA tensor preparation must run in a Slurm allocation")
    if not np.isfinite(minimum_latent_std) or minimum_latent_std <= 0:
        raise ValueError("minimum_latent_std must be finite and positive")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest, splits = load_sparse_flow_manifest(manifest_path)
    original_manifest_hash = sha256_file(manifest_path)
    # Validate before reading stats; all original archives are hashed, and their
    # hashes are checked again after preparation to reject concurrent mutation.
    provenance_files: dict[str, dict[str, Any]] = {}
    spatial_shape = None
    for split, cases in splits.items():
        provenance_files[split] = {}
        for case in cases:
            shape = tuple(inspect_sparse_flow_case(case.path)["flow_shape"][-3:])
            if spatial_shape is None:
                spatial_shape = _spatial_shape(shape, rank)
            elif shape != spatial_shape:
                raise ValueError("all flow cases must have the same spatial shape")
            provenance_files[split][case.case_id] = {
                "path": str(case.path), "sha256": sha256_file(case.path),
                "flow_group_id": case.flow_group_id,
            }
    assert spatial_shape is not None
    stats = compute_train_flow_stats(manifest_path)
    train_decompositions = []
    for case in splits["train"]:
        fields, _, _ = _read_field(case.path, stats, device)
        core, factors = spatial_tucker(fields[0], rank, hooi_iterations=hooi_iterations)
        train_decompositions.append((core.cpu(), [factor.cpu() for factor in factors]))
    anchors, anchor_indices = select_medoid_anchors([factors for _, factors in train_decompositions])
    codec = TensorSpaceCodec(spatial_shape, rank, anchors, hooi_iterations=hooi_iterations).to(device)
    latents: dict[str, dict[str, Tensor]] = {split: {} for split in _SPLITS}
    overlaps: dict[str, list[list[float]]] = {split: [] for split in _SPLITS}
    validation_metrics = {}
    for split, cases in splits.items():
        for index, case in enumerate(cases):
            field, solid, bounds = _read_field(case.path, stats, device)
            if split == "train":
                core_cpu, factors_cpu = train_decompositions[index]
                core = core_cpu.to(device)
                factors = [factor.to(device) for factor in factors_cpu]
            else:
                core, factors = spatial_tucker(field[0], rank, hooi_iterations=hooi_iterations)
            aligned_core, aligned_factors, singular = align_tucker(core, factors, codec.anchors)
            latent = codec.pack(aligned_core, aligned_factors)
            _finite(latent, "aligned latent")
            latents[split][case.case_id] = latent.cpu()
            overlaps[split].append(singular)
            if split == "validation":
                prediction = denormalize_flow(codec.decode(latent[None]), stats)
                reference = denormalize_flow(field, stats)
                validation_metrics[case.case_id] = _reconstruction_metrics(reference, prediction, solid, bounds)
    train_codes = torch.stack(list(latents["train"].values())).double()
    latent_mean = train_codes.mean(0).float()
    latent_std = train_codes.std(0, correction=0).float().clamp_min(minimum_latent_std)
    codec.latent_mean.copy_(latent_mean)
    codec.latent_std.copy_(latent_std)
    for split in _SPLITS:
        for case_id, latent in latents[split].items():
            standardized = (latent - latent_mean) / latent_std
            _finite(standardized, "standardized latent")
            latents[split][case_id] = standardized
    for split, cases in splits.items():
        for case in cases:
            if sha256_file(case.path) != provenance_files[split][case.case_id]["sha256"]:
                raise ValueError(f"case changed during preparation: {case.path}")
    if sha256_file(manifest_path) != original_manifest_hash:
        raise ValueError("manifest changed during preparation")
    relative_l2 = [metric["relative_l2_fluid"] for metric in validation_metrics.values()]
    report = {
        "rank": rank, "spatial_shape": list(spatial_shape), "latent_dim": codec.latent_dim,
        "raw_field_elements": 3 * int(np.prod(spatial_shape)),
        "compression_ratio": 3 * int(np.prod(spatial_shape)) / codec.latent_dim,
        "hooi_iterations": hooi_iterations, "convergence_claimed": False,
        "anchor_metric": "squared_chordal_projector_distance_medoid",
        "anchor_train_case_ids": [splits["train"][i].case_id for i in anchor_indices],
        "validation_mean_relative_l2_fluid": float(np.mean(relative_l2)),
        "validation_max_relative_l2_fluid": float(np.max(relative_l2)),
        "validation_cases": validation_metrics,
        "anchor_overlap_min_singular_value": {
            split: torch.tensor(overlaps[split]).amin(dim=0).tolist() for split in ("train", "validation")
        },
        "low_anchor_overlap_threshold": 1e-5,
        "low_anchor_overlap_case_count": {
            split: sum(min(values) < 1e-5 for values in overlaps[split]) for split in ("train", "validation")
        },
        "rank_selection_split": "validation", "test_metrics_computed": False,
    }
    artifact = codec.to_artifact()
    artifact.update({
        "format_version": FORMAT_VERSION, "normalization_stats": stats.to_dict(),
        "latents": latents, "splits": {split: [case.case_id for case in cases] for split, cases in splits.items()},
        "provenance": {"manifest_path": str(manifest_path), "manifest_sha256": original_manifest_hash,
                       "manifest_mode": manifest.get("mode", "scientific"), "files": provenance_files,
                       "fitted_split": "train", "anchor_train_case_ids": report["anchor_train_case_ids"],
                       "minimum_latent_std": minimum_latent_std,
                       "torch_version": str(torch.__version__)},
        "reconstruction_report": report,
    })
    return artifact


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported tensor-space artifact format")
    codec = TensorSpaceCodec.from_artifact(artifact)
    FlowNormalizationStats.from_dict(artifact["normalization_stats"])
    if artifact["provenance"].get("fitted_split") != "train":
        raise ValueError("codec must be fitted on the training split only")
    for case_id in artifact["provenance"]["anchor_train_case_ids"]:
        if case_id not in artifact["splits"]["train"]:
            raise ValueError("anchor is not a training case")
    for split in _SPLITS:
        if not artifact["splits"][split] or set(artifact["latents"][split]) != set(artifact["splits"][split]):
            raise ValueError("cached latent case IDs do not match manifest split")
        for latent in artifact["latents"][split].values():
            if not isinstance(latent, Tensor) or latent.shape != (codec.latent_dim,):
                raise ValueError("cached latent shape does not match codec")
            _finite(latent, "cached latent")


def load_tensor_space_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Load tensors with weights_only and reject stale manifest/data caches."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate_artifact(artifact)
    provenance = artifact["provenance"]
    manifest_path = Path(manifest_path or provenance["manifest_path"]).expanduser().resolve()
    if sha256_file(manifest_path) != provenance["manifest_sha256"]:
        raise ValueError("tensor artifact manifest SHA256 mismatch")
    _, splits = load_sparse_flow_manifest(manifest_path, require_files=verify_files)
    for split, cases in splits.items():
        if [case.case_id for case in cases] != artifact["splits"][split]:
            raise ValueError("tensor artifact split differs from current manifest")
        for case in cases:
            record = provenance["files"][split][case.case_id]
            if record["flow_group_id"] != case.flow_group_id:
                raise ValueError("tensor artifact flow-group mismatch")
            if verify_files and sha256_file(case.path) != record["sha256"]:
                raise ValueError(f"tensor artifact case SHA256 mismatch: {case.case_id}")
    return artifact


# Short alias for callers that use the same naming as training artifact options.
load_tensor_artifact = load_tensor_space_artifact


def atomic_write_new(path: str | Path, writer: Any) -> None:
    """Publish a fully written file atomically, failing if its name exists.

    The hard-link publication is atomic and non-replacing on Linux and NTFS;
    the temporary file is on the same filesystem as the destination.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_tensor_space_artifact(artifact: Mapping[str, Any], path: str | Path) -> None:
    _validate_artifact(artifact)
    atomic_write_new(path, lambda handle: torch.save(dict(artifact), handle))


__all__ = ["TensorSpaceCodec", "spatial_tucker", "align_tucker", "select_medoid_anchors",
           "build_tensor_space_artifact", "save_tensor_space_artifact", "load_tensor_space_artifact",
           "load_tensor_artifact", "atomic_write_new", "sha256_file"]
