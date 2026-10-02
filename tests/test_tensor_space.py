from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flow_observation.tensor_space import (
    TensorSpaceCodec,
    align_tucker,
    atomic_write_new,
    build_tensor_space_artifact,
    load_tensor_space_artifact,
    save_tensor_space_artifact,
    select_medoid_anchors,
    spatial_tucker,
)


@pytest.fixture(autouse=True)
def one_torch_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _orthonormal(size: int, rank: int, generator: torch.Generator) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(size, rank, generator=generator), mode="reduced").Q


def _low_rank_field(shape=(4, 5, 6), rank=2):
    generator = torch.Generator().manual_seed(37)
    factors = [_orthonormal(size, rank, generator) for size in shape]
    core = torch.randn(3, rank, rank, rank, generator=generator)
    field = torch.einsum("cijk,di,hj,wk->cdhw", core, *factors)
    return field, core, factors


def _manifest(tmp_path: Path) -> Path:
    generator = torch.Generator().manual_seed(9)
    _, _, factors = _low_rank_field(shape=(4, 4, 4))
    entries = []
    for index in range(5):
        core = torch.randn(3, 2, 2, 2, generator=generator) + 0.15 * index
        field = torch.einsum("cijk,di,hj,wk->cdhw", core, *factors)[None].numpy()
        tracks = np.zeros((1, 3, 4, 3, 2), dtype=np.float32)
        path = tmp_path / f"case_{index}.npz"
        np.savez_compressed(
            path, flow_field=field, trajectories_2d=tracks,
            observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
            projection_matrix=np.asarray([[[1, 0, 0], [0, 1, 0]],
                                          [[1, 0, 0], [0, 0, 1]],
                                          [[0, 1, 0], [0, 0, 1]]], dtype=np.float32),
            observation_times=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            domain_bounds=np.asarray([[0, 3], [0, 3], [0, 3]], dtype=np.float32),
            solid_mask=np.zeros((4, 4, 4), dtype=bool),
        )
        entries.append({"case_id": f"case_{index}", "flow_group_id": f"flow_{index}", "path": path.name})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format_version": 1, "mode": "scientific",
                               "splits": {"train": entries[:3], "validation": entries[3:4],
                                          "test": entries[4:]}}), encoding="utf-8")
    return path


def _modify_case(path: Path, factor: float) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["flow_field"] *= factor
    np.savez_compressed(path, **arrays)


def test_low_rank_round_trip_and_decoder_gradients():
    field, _, factors = _low_rank_field()
    codec = TensorSpaceCodec(field.shape[1:], 2, factors, hooi_iterations=3)
    latent = codec.encode(field[None])
    assert latent.shape == (1, 3 * 2**3 + (4 + 5 + 6) * 2)
    assert torch.allclose(codec.decode(latent), field[None], atol=2e-6, rtol=2e-6)
    latent.requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        decoded = codec.decode(latent)
        loss = decoded.square().mean()
    loss.backward()
    assert decoded.dtype == torch.float32
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0
    restored = TensorSpaceCodec.from_artifact(codec.to_artifact())
    assert torch.equal(restored.decode(latent.detach()), codec.decode(latent.detach()))


def test_procrustes_is_invariant_to_rotated_decomposition_and_preserves_field():
    field, core, factors = _low_rank_field()
    generator = torch.Generator().manual_seed(81)
    rotations = [_orthonormal(2, 2, generator) for _ in range(3)]
    rotated_factors = [factor @ rotation for factor, rotation in zip(factors, rotations)]
    rotated_core = torch.einsum("cijk,ia,jb,kd->cabd", core, *rotations)
    aligned_core, aligned_factors, overlaps = align_tucker(rotated_core, rotated_factors, factors)
    assert min(overlaps) > 0.99999
    assert torch.allclose(aligned_core, core, atol=2e-6, rtol=2e-6)
    for actual, expected in zip(aligned_factors, factors):
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    codec = TensorSpaceCodec(field.shape[1:], 2, factors)
    assert torch.allclose(codec.decode(codec.pack(aligned_core, aligned_factors)[None]), field[None], atol=3e-6, rtol=3e-6)


def test_medoid_is_rotation_invariant_and_selects_an_observed_factor():
    generator = torch.Generator().manual_seed(51)
    triplets = [[_orthonormal(size, 2, generator) for size in (4, 5, 6)] for _ in range(5)]
    anchors, indices = select_medoid_anchors(triplets)
    rotations = [_orthonormal(2, 2, generator) for _ in range(5)]
    rotated = [[factor @ rotations[i] for factor in triplet] for i, triplet in enumerate(triplets)]
    rotated_anchors, rotated_indices = select_medoid_anchors(rotated)
    assert indices == rotated_indices
    for axis, index in enumerate(indices):
        assert torch.allclose(anchors[axis], triplets[index][axis])
        assert torch.allclose(anchors[axis] @ anchors[axis].T,
                              rotated_anchors[axis] @ rotated_anchors[axis].T, atol=1e-6)


def test_full_rank_exact_and_lower_rank_loss_is_reported():
    field = torch.randn(3, 4, 4, 4, generator=torch.Generator().manual_seed(1))
    core, factors = spatial_tucker(field, 4, hooi_iterations=2)
    codec = TensorSpaceCodec((4, 4, 4), 4, factors)
    reconstructed = codec.decode(codec.pack(core, factors)[None])[0]
    assert torch.allclose(reconstructed, field, atol=6e-6, rtol=6e-6)
    small_core, small_factors = spatial_tucker(field, 1, hooi_iterations=2)
    small_codec = TensorSpaceCodec((4, 4, 4), 1, small_factors)
    assert (small_codec.decode(small_codec.pack(small_core, small_factors)[None])[0] - field).norm() > 1


@pytest.mark.parametrize("rank", [0, 5, 1.5, True])
def test_invalid_ranks_rejected(rank):
    with pytest.raises(ValueError, match="rank"):
        spatial_tucker(torch.zeros(3, 4, 4, 4), rank)


def test_heldout_values_do_not_change_anchors_or_statistics(tmp_path):
    manifest = _manifest(tmp_path)
    before = build_tensor_space_artifact(manifest, 2)
    _modify_case(tmp_path / "case_3.npz", 100)
    _modify_case(tmp_path / "case_4.npz", -200)
    after = build_tensor_space_artifact(manifest, 2)
    assert before["normalization_stats"] == after["normalization_stats"]
    assert before["provenance"]["anchor_train_case_ids"] == after["provenance"]["anchor_train_case_ids"]
    for key, value in before["codec_state_dict"].items():
        assert torch.equal(value, after["codec_state_dict"][key])
    for case_id in before["latents"]["train"]:
        assert torch.equal(before["latents"]["train"][case_id], after["latents"]["train"][case_id])
    assert before["reconstruction_report"]["test_metrics_computed"] is False
    assert set(before["reconstruction_report"]["validation_cases"]) == {"case_3"}
    assert "test" not in before["reconstruction_report"]["anchor_overlap_min_singular_value"]


def test_saved_cache_checks_manifest_and_archive_provenance(tmp_path):
    manifest = _manifest(tmp_path)
    artifact = build_tensor_space_artifact(manifest, 2)
    artifact_path = tmp_path / "rank_2.pt"
    save_tensor_space_artifact(artifact, artifact_path)
    restored = load_tensor_space_artifact(artifact_path, manifest_path=manifest)
    assert restored["splits"] == artifact["splits"]
    with pytest.raises(FileExistsError):
        save_tensor_space_artifact(artifact, artifact_path)
    _modify_case(tmp_path / "case_4.npz", 2)
    with pytest.raises(ValueError, match="case SHA256 mismatch"):
        load_tensor_space_artifact(artifact_path, manifest_path=manifest)
    load_tensor_space_artifact(artifact_path, manifest_path=manifest, verify_files=False)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_tensor_space_artifact(artifact_path, manifest_path=manifest, verify_files=False)


def test_corrupt_cache_is_rejected(tmp_path):
    artifact = build_tensor_space_artifact(_manifest(tmp_path), 2)
    bad = copy.deepcopy(artifact)
    bad["latents"]["validation"]["case_3"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_tensor_space_artifact(bad, tmp_path / "corrupt.pt")
    bad = copy.deepcopy(artifact)
    bad["provenance"]["anchor_train_case_ids"][0] = "case_4"
    with pytest.raises(ValueError, match="not a training case"):
        save_tensor_space_artifact(bad, tmp_path / "corrupt.pt")


def test_failed_atomic_write_publishes_nothing(tmp_path):
    target = tmp_path / "artifact.pt"
    def fail(handle):
        handle.write(b"partial")
        raise RuntimeError("simulated failure")
    with pytest.raises(RuntimeError, match="simulated failure"):
        atomic_write_new(target, fail)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_cuda_preparation_requires_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        build_tensor_space_artifact(tmp_path / "unused.json", 2, device="cuda")


def test_preparation_cli_writes_report_and_refuses_overwrite(tmp_path):
    manifest = _manifest(tmp_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_tensor_space.py"
    spec = importlib.util.spec_from_file_location("prepare_tensor_space", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "prepared"
    args = ["--manifest", str(manifest), "--output-dir", str(output_dir), "--ranks", "2", "--device", "cpu"]
    module.main(args)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["ranks"]["2"]["latent_dim"] == 48
    assert report["ranks"]["2"]["artifact_sha256"]
    with pytest.raises(FileExistsError):
        module.main(args)
