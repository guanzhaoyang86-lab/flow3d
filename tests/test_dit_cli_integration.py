"""Small CPU integrations for the two new architectures and their checkpoint API."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_dit_cli_helpers", Path(__file__).with_name("test_sparse_diffusion_cli.py")
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPERS)


@pytest.mark.parametrize("architecture", ["dit3d", "tensor-dit"])
def test_dit_prepare_train_resume_and_guided_sample_cli_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    # Avoid oversubscribing tiny CPU operations in subprocesses and CI hosts.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    case = tmp_path / "tiny_case.npz"
    manifest = tmp_path / "manifest.json"
    _HELPERS._write_eight_cubed_case(case)
    _HELPERS._write_smoke_manifest(manifest, case)
    artifact_options = []
    artifact_path = None
    if architecture == "tensor-dit":
        prepared = tmp_path / "prepared"
        _HELPERS._run_cli([
            str(_ROOT / "scripts/prepare_tensor_space.py"), "--manifest", str(manifest),
            "--output-dir", str(prepared), "--ranks", "2", "--hooi-iterations", "1",
            "--device", "cpu",
        ])
        artifact_path = prepared / "rank_2.pt"
        artifact_options = ["--tensor-artifact", str(artifact_path)]
        report = json.loads((prepared / "report.json").read_text())
        assert report["rank_selection_split"] == "validation"
        assert report["test_metrics_computed"] is False
        assert report["ranks"]["2"]["latent_dim"] == 3 * 2**3 + 3 * 8 * 2

    common = [
        str(_ROOT / "scripts/train_sparse_track_diffusion.py"),
        "--manifest", str(manifest), "--architecture", architecture,
        "--dit-hidden-dim", "24", "--dit-depth", "1", "--dit-heads", "3",
        "--condition-dim", "16", "--patch-size", "4", "--diffusion-steps", "4",
        "--num-particles", "2", "--batch-size", "1", "--num-workers", "0",
        "--condition-dropout", "0.1", "--dropout", "0.1", "--seed", "31",
        "--boundary-loss-weight", "0.01", "--divergence-loss-weight", "0.001",
        "--track-loss-weight", "0.01", "--auxiliary-max-t-fraction", "1",
        "--trajectory-substeps", "1", "--save-every", "1", "--device", "cpu",
        "--no-amp", *artifact_options,
    ]
    uninterrupted = tmp_path / "uninterrupted"
    _HELPERS._run_cli([*common, "--output-dir", str(uninterrupted), "--epochs", "2"])
    partial = tmp_path / "partial"
    _HELPERS._run_cli([*common, "--output-dir", str(partial), "--epochs", "1"])
    resumed = tmp_path / "resumed"
    output = _HELPERS._run_cli([
        *common, "--output-dir", str(resumed), "--epochs", "2",
        "--resume", str(partial / "latest.pt"),
    ])
    assert "at epoch 2" in output.stdout
    checkpoints = [
        torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
        for directory in (uninterrupted, resumed)
    ]
    for checkpoint in checkpoints:
        assert checkpoint["epoch"] == 2
        assert checkpoint["model_config"]["architecture"] == architecture
        assert checkpoint["model_config"]["spatial_shape"] == [8, 8, 8]
        assert np.isfinite(checkpoint["history"][-1]["validation_total"])
    for group in ("encoder_state", "unet_state", "ema_encoder_state", "ema_unet_state"):
        assert checkpoints[0][group].keys() == checkpoints[1][group].keys()
        for key, expected in checkpoints[0][group].items():
            torch.testing.assert_close(checkpoints[1][group][key], expected, rtol=0, atol=0)
    if artifact_path is not None:
        assert checkpoints[1]["tensor_codec"]["codec_config"]["rank"] == 2
        assert "latents" not in checkpoints[1]["tensor_codec"]
        assert len(checkpoints[1]["tensor_artifact_sha256"]) == 64
        # Sampling must depend only on the compact codec embedded in the model.
        artifact_path.rename(artifact_path.with_suffix(".hidden"))
    else:
        assert checkpoints[1]["tensor_codec"] is None

    posterior = tmp_path / "posterior.npz"
    sampled = _HELPERS._run_cli([
        str(_ROOT / "scripts/sample_sparse_track_diffusion.py"),
        "--checkpoint", str(resumed / "best.pt"), "--manifest", str(manifest),
        "--output", str(posterior), "--split", "test", "--num-particles", "2",
        "--num-probe-particles", "2", "--num-samples", "1", "--sampling-steps", "2",
        "--eta", "0", "--cfg-scale", "1.5", "--boundary-projection", "final",
        "--trajectory-guidance-strength", "0.0001", "--divergence-guidance-weight", "0.01",
        "--trajectory-substeps", "1", "--device", "cpu",
    ])
    assert "sample 1/1 complete" in sampled.stdout
    with np.load(posterior, allow_pickle=False) as archive:
        assert archive["posterior_samples"].shape == (1, 3, 8, 8, 8)
        assert np.isfinite(archive["posterior_samples"]).all()
        assert np.isfinite(archive["replay_tracks"]).all()
        assert not np.intersect1d(archive["particle_indices"], archive["probe_particle_indices"]).size
        metadata = _HELPERS._json_scalar(archive["metadata"])
        assert metadata["model_config"]["architecture"] == architecture
        assert metadata["boundary_projection"] == "final"
        assert metadata["scientific_result"] is False
