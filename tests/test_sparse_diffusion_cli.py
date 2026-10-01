from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import torch
import numpy as np
import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _REPOSITORY_ROOT / "scripts" / "train_sparse_track_diffusion.py"
_SAMPLE_SCRIPT = _REPOSITORY_ROOT / "scripts" / "sample_sparse_track_diffusion.py"
_SWEEP_SCRIPT = _REPOSITORY_ROOT / "scripts" / "run_sparse_particle_sweep.py"
_SAMPLE_SPEC = importlib.util.spec_from_file_location("_sparse_sampler", _SAMPLE_SCRIPT)
assert _SAMPLE_SPEC is not None and _SAMPLE_SPEC.loader is not None
_SAMPLE = importlib.util.module_from_spec(_SAMPLE_SPEC)
_SAMPLE_SPEC.loader.exec_module(_SAMPLE)
_SWEEP_SPEC = importlib.util.spec_from_file_location("_sparse_sweep", _SWEEP_SCRIPT)
assert _SWEEP_SPEC is not None and _SWEEP_SPEC.loader is not None
_SWEEP = importlib.util.module_from_spec(_SWEEP_SPEC)
_SWEEP_SPEC.loader.exec_module(_SWEEP)


def _write_eight_cubed_case(path: Path) -> None:
    size = 8
    z, y, x = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    # All three components are finite, spatially varying, and have non-zero
    # training variance.  Magnitudes remain small enough for the replay check.
    flow = np.stack(
        (
            0.010 + 0.0010 * x + 0.0002 * y,
            -0.006 + 0.0008 * y + 0.0001 * z,
            0.004 + 0.0007 * z - 0.0001 * x,
        ),
        axis=0,
    )[None].astype(np.float32)

    initial = np.asarray(
        (
            (2.0, 2.0, 2.0),
            (3.0, 4.0, 2.5),
            (4.0, 2.5, 5.0),
            (5.0, 5.0, 4.0),
        ),
        dtype=np.float32,
    )
    times = np.asarray((0.0, 0.5, 1.0), dtype=np.float32)
    drift = np.asarray((0.04, -0.02, 0.03), dtype=np.float32)
    trajectories_3d = initial[:, None, :] + times[None, :, None] * drift
    projection = np.asarray(
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        dtype=np.float32,
    )
    # [S,V,N,T,2].  XY and XZ together have rank three.
    tracks = np.einsum("ntc,voc->vnto", trajectories_3d, projection)[None]
    assert np.linalg.matrix_rank(projection.reshape(-1, 3)) == 3

    solid = np.zeros((size, size, size), dtype=bool)
    solid[:, :, 0] = True
    metadata = {
        "case_id": "eight-cubed-smoke-case",
        "lid_face": "x=max",
        "lid_velocity_xyz": [0.0, 0.0, 0.05],
        "scope": "CLI smoke test only",
    }
    np.savez_compressed(
        path,
        flow_field=flow,
        trajectories_2d=tracks.astype(np.float32),
        projection_matrix=projection,
        observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
        observation_times=times,
        domain_bounds=np.asarray(((0.0, 7.0),) * 3, dtype=np.float32),
        solid_mask=solid,
        particle_counts=np.asarray([4], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _write_smoke_manifest(path: Path, case_path: Path) -> None:
    entry = {"case_id": "eight-cubed-smoke-case", "path": case_path.name}
    manifest = {
        "format_version": 1,
        "mode": "smoke_test",
        "allow_overlap_for_smoke_test": True,
        "scientific_result": False,
        "splits": {
            "train": [entry],
            "validation": [entry],
            "test": [entry],
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_cli(arguments: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(arguments)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _json_scalar(value: np.ndarray) -> dict[str, object]:
    decoded = json.loads(str(value.reshape(()).item()))
    assert isinstance(decoded, dict)
    return decoded


def test_sparse_diffusion_train_and_sample_cli_cpu_smoke(tmp_path: Path) -> None:
    case_path = tmp_path / "eight_cubed_case.npz"
    manifest_path = tmp_path / "smoke_manifest.json"
    training_dir = tmp_path / "training"
    posterior_path = tmp_path / "posterior.npz"
    _write_eight_cubed_case(case_path)
    _write_smoke_manifest(manifest_path, case_path)

    training = _run_cli(
        [
            str(_TRAIN_SCRIPT),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(training_dir),
            "--num-particles",
            "2",
            "--observations-per-flow",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--num-workers",
            "0",
            "--diffusion-steps",
            "4",
            "--base-channels",
            "8",
            "--condition-dim",
            "32",
            "--time-embedding-dim",
            "16",
            "--condition-dropout",
            "0",
            "--device",
            "cpu",
            "--no-amp",
            "--save-every",
            "1",
        ]
    )
    assert "not scientific results" in training.stdout

    best_checkpoint = training_dir / "best.pt"
    latest_checkpoint = training_dir / "latest.pt"
    assert best_checkpoint.is_file()
    assert latest_checkpoint.is_file()
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    expected_checkpoint_keys = {
        "format_version",
        "encoder_state",
        "unet_state",
        "ema_encoder_state",
        "ema_unet_state",
        "diffusion_state",
        "normalization",
        "model_config",
        "data_config",
    }
    assert expected_checkpoint_keys.issubset(checkpoint)
    assert checkpoint["model_config"]["diffusion_steps"] == 4
    assert checkpoint["model_config"]["base_channels"] == 8
    assert checkpoint["model_config"]["condition_dim"] == 32
    assert checkpoint["data_config"]["num_particles"] == 2

    summary = json.loads((training_dir / "training_summary.json").read_text())
    assert summary["scientific_result"] is False
    assert summary["manifest_mode"] == "smoke_test"

    sampling = _run_cli(
        [
            str(_SAMPLE_SCRIPT),
            "--checkpoint",
            str(best_checkpoint),
            "--manifest",
            str(manifest_path),
            "--split",
            "test",
            "--index",
            "0",
            "--num-particles",
            "2",
            "--num-samples",
            "1",
            "--num-probe-particles",
            "2",
            "--sampling-steps",
            "2",
            "--eta",
            "0",
            "--cfg-scale",
            "1",
            "--device",
            "cpu",
            "--output",
            str(posterior_path),
        ]
    )
    assert "sample 1/1 complete" in sampling.stdout
    assert posterior_path.is_file()

    with np.load(posterior_path, allow_pickle=False) as posterior:
        expected_posterior_keys = {
            "posterior_samples",
            "posterior_mean",
            "posterior_variance",
            "reference_field",
            "observed_tracks",
            "observation_mask",
            "replay_tracks",
            "replay_validity",
            "projection_matrix",
            "observation_times",
            "domain_bounds",
            "particle_indices",
            "metrics",
            "metadata",
            "probe_observed_tracks",
            "probe_observation_mask",
            "probe_replay_tracks",
            "probe_replay_validity",
            "probe_particle_indices",
        }
        assert expected_posterior_keys.issubset(posterior.files)
        assert posterior["posterior_samples"].shape == (1, 3, 8, 8, 8)
        assert posterior["posterior_mean"].shape == (1, 3, 8, 8, 8)
        assert posterior["posterior_variance"].shape == (1, 3, 8, 8, 8)
        assert posterior["reference_field"].shape == (1, 3, 8, 8, 8)
        assert posterior["observed_tracks"].shape == (1, 2, 2, 3, 2)
        assert posterior["replay_tracks"].shape == (1, 2, 2, 3, 2)
        assert posterior["replay_validity"].shape == (1, 2, 3)
        assert posterior["particle_indices"].shape == (1, 2)
        assert posterior["probe_observed_tracks"].shape == (1, 2, 2, 3, 2)
        assert posterior["probe_replay_tracks"].shape == (1, 2, 2, 3, 2)
        assert posterior["probe_replay_validity"].shape == (1, 2, 3)
        assert posterior["probe_particle_indices"].shape == (1, 2)
        assert not np.intersect1d(
            posterior["particle_indices"], posterior["probe_particle_indices"]
        ).size
        assert np.isfinite(posterior["posterior_samples"]).all()
        metadata = _json_scalar(posterior["metadata"])
        metrics = _json_scalar(posterior["metrics"])

    assert metadata["smoke_test_manifest"] is True
    assert metadata["scientific_result"] is False
    assert metadata["method"] == "2-particle conditional 3D flow diffusion"
    assert metadata["probe_particles"] == 2
    assert metadata["checkpoint_has_training_provenance"] is True
    assert metadata["evaluation_manifest_provenance"]["manifest_mode"] == "smoke_test"
    assert metadata["scientific_result_reasons"]
    assert metadata["sampled_particles"] == 2
    assert metadata["sampling_steps"] == 2
    assert metrics["num_posterior_samples"] == 1
    assert metrics["num_observed_particles"] == 2
    assert metrics["num_probe_particles"] == 2
    assert metrics["probe_track_rmse_cells"] >= 0.0
    assert metrics["uncertainty_error_pearson_unknown_interior"] == 0.0
    assert metrics["uncertainty_error_pearson_defined"] is False


def test_scientific_assessment_requires_provenance_test_split_and_matching_n(
    tmp_path: Path,
) -> None:
    case_paths = []
    for name in ("train", "validation", "test"):
        case_path = tmp_path / f"{name}.npz"
        _write_eight_cubed_case(case_path)
        case_paths.append(case_path)
    manifest_path = tmp_path / "scientific.json"
    manifest = {
        "format_version": 1,
        "mode": "scientific",
        "splits": {
            name: [
                {
                    "case_id": f"{name}-case",
                    "flow_group_id": f"{name}-flow",
                    "path": f"{name}.npz",
                }
            ]
            for name in ("train", "validation", "test")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    provenance = _SAMPLE._manifest_provenance(manifest_path)
    checkpoint = {"training_provenance": dict(provenance)}

    valid, reasons, _ = _SAMPLE._scientific_sampling_assessment(
        checkpoint,
        manifest_path,
        split="test",
        trained_particles=2,
        sampled_particles=2,
    )
    assert valid is True
    assert reasons == []

    for altered_checkpoint, split, sampled in (
        ({}, "test", 2),
        (checkpoint, "validation", 2),
        (checkpoint, "test", 4),
    ):
        valid, reasons, _ = _SAMPLE._scientific_sampling_assessment(
            altered_checkpoint,
            manifest_path,
            split=split,
            trained_particles=2,
            sampled_particles=sampled,
        )
        assert valid is False
        assert reasons


def test_guidance_configuration_rejects_silent_divergence_noop(tmp_path: Path) -> None:
    parser = _SWEEP._build_parser()
    args = parser.parse_args(
        [
            "--manifest",
            str(tmp_path / "unused.json"),
            "--allow-untrained-count-extrapolation",
            "--shared-checkpoint",
            str(tmp_path / "unused.pt"),
            "--output-dir",
            str(tmp_path / "out"),
            "--divergence-guidance-weight",
            "1",
            "--trajectory-guidance-strength",
            "0",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="total guidance is disabled"):
        _SWEEP.run_sweep(args)


def _sweep_resume_arguments(
    tmp_path: Path, *additional: str
) -> tuple[argparse.Namespace, Path]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "shared.pt"
    torch.save({"data_config": {"num_particles": 2}}, checkpoint_path)
    output_dir = tmp_path / "sweep"
    parser = _SWEEP._build_parser()
    arguments = [
        "--manifest",
        str(manifest_path),
        "--allow-untrained-count-extrapolation",
        "--shared-checkpoint",
        str(checkpoint_path),
        "--num-test-cases",
        "1",
        "--num-samples",
        "1",
        "--sampling-steps",
        "1",
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
        *additional,
    ]
    return parser.parse_args(arguments), output_dir


def _completed_sweep_record(
    *,
    run_id: str,
    count: int,
    output: Path,
    checkpoint: Path,
    marker: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": run_id,
        "num_particles": count,
        "test_case_index": 0,
        "seed": 47,
        "checkpoint": str(checkpoint.resolve()),
        "trained_particles": 2,
        "scientific_design": False,
        "output": str(output.resolve()),
        "log": str(output.with_suffix(".log").resolve()),
        "command": [],
        "command_line": "",
        "status": "completed",
        "returncode": 0,
        "elapsed_seconds": 0.0,
        "case_id": "retained-case",
        "source_scientific_result": False,
        "scientific_result": False,
        "scientific_result_reasons": ["test fixture"],
        "metrics": {
            "num_observed_particles": count,
            "score": float(count),
        },
    }
    if marker is not None:
        record["marker"] = marker
    return record


def test_sweep_resume_skips_completed_and_retries_failed_or_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output_dir = _sweep_resume_arguments(tmp_path, "--resume")
    checkpoint_path = Path(args.shared_checkpoint)
    retained_output = output_dir / "N002" / "case_0000_seed_47.npz"
    retained_output.parent.mkdir(parents=True)
    retained_output.write_bytes(b"retained")
    missing_output = output_dir / "N008" / "case_0000_seed_47.npz"
    failed_output = output_dir / "N004" / "case_0000_seed_47.npz"
    failed_output.parent.mkdir(parents=True)
    failed_output.write_bytes(b"failed output must not be retained")
    runs_path = output_dir / "runs.jsonl"
    existing_records = [
        _completed_sweep_record(
            run_id="N002_case_0000",
            count=2,
            output=retained_output,
            checkpoint=checkpoint_path,
            marker="keep-me",
        ),
        {
            **_completed_sweep_record(
                run_id="N004_case_0000",
                count=4,
                output=failed_output,
                checkpoint=checkpoint_path,
            ),
            "status": "failed",
        },
        _completed_sweep_record(
            run_id="N008_case_0000",
            count=8,
            output=missing_output,
            checkpoint=checkpoint_path,
        ),
    ]
    runs_path.write_text(
        "".join(json.dumps(record) + "\n" for record in existing_records),
        encoding="utf-8",
    )

    called_counts: list[int] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        count = int(command[command.index("--num-particles") + 1])
        output = Path(command[command.index("--output") + 1])
        called_counts.append(count)
        if len(called_counts) == 1:
            compacted = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert [record["run_id"] for record in compacted] == ["N002_case_0000"]
        metrics = {
            "num_observed_particles": count,
            "score": float(count),
        }
        metadata = {
            "trained_particles": 2,
            "sampled_particles": count,
            "split": "test",
            "method": f"{count}-particle conditional 3D flow diffusion",
            "scientific_result": False,
            "scientific_result_reasons": ["shared checkpoint test fixture"],
            "checkpoint_has_training_provenance": False,
            "evaluation_manifest_provenance": {"manifest_mode": "smoke_test"},
            "case_id": f"case-{count}",
        }
        np.savez_compressed(
            output,
            metrics=np.asarray(json.dumps(metrics, sort_keys=True)),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    summary = _SWEEP.run_sweep(args)

    assert called_counts == [4, 8, 32, 64, 128]
    final_records = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(final_records) == 6
    assert len({record["run_id"] for record in final_records}) == 6
    assert all(record["status"] == "completed" for record in final_records)
    assert final_records[0]["marker"] == "keep-me"
    assert summary["complete"] is True
    assert summary["completed_runs"] == 6
    assert summary["failed_runs"] == 0
    assert summary["planned_runs"] == 0
    assert all(
        summary["per_particle_count"][str(count)]["completed_runs"] == 1
        for count in (2, 4, 8, 32, 64, 128)
    )


def test_sweep_resume_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    parser = _SWEEP._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--allow-untrained-count-extrapolation",
                "--shared-checkpoint",
                str(tmp_path / "shared.pt"),
                "--output-dir",
                str(tmp_path / "sweep"),
                "--resume",
                "--overwrite",
            ]
        )


def test_sweep_particle_subset_runs_and_summarizes_only_requested_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _sweep_resume_arguments(
        tmp_path, "--particle-counts", "2,32"
    )
    called_counts: list[int] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        count = int(command[command.index("--num-particles") + 1])
        output = Path(command[command.index("--output") + 1])
        called_counts.append(count)
        metrics = {
            "num_observed_particles": count,
            "score": float(count),
        }
        metadata = {
            "trained_particles": 2,
            "sampled_particles": count,
            "split": "test",
            "method": f"{count}-particle conditional 3D flow diffusion",
            "scientific_result": False,
            "scientific_result_reasons": ["shared checkpoint test fixture"],
            "checkpoint_has_training_provenance": False,
            "evaluation_manifest_provenance": {"manifest_mode": "smoke_test"},
            "case_id": f"case-{count}",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            metrics=np.asarray(json.dumps(metrics, sort_keys=True)),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    summary = _SWEEP.run_sweep(args)

    assert called_counts == [2, 32]
    assert summary["particle_counts"] == [2, 32]
    assert summary["expected_runs"] == 2
    assert summary["completed_runs"] == 2
    assert set(summary["per_particle_count"]) == {"2", "32"}


def test_scientific_subset_requires_exact_checkpoint_set(tmp_path: Path) -> None:
    parser = _SWEEP._build_parser()
    base = [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--particle-counts",
        "2,32",
        "--output-dir",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    missing = parser.parse_args(
        [*base, "--checkpoint", f"2={tmp_path / 'n2.pt'}"]
    )
    with pytest.raises(ValueError, match="missing"):
        _SWEEP._resolve_checkpoint_policy(missing)

    unexpected = parser.parse_args(
        [
            *base,
            "--checkpoint",
            f"2={tmp_path / 'n2.pt'}",
            "--checkpoint",
            f"32={tmp_path / 'n32.pt'}",
            "--checkpoint",
            f"4={tmp_path / 'n4.pt'}",
        ]
    )
    with pytest.raises(ValueError, match="unrequested"):
        _SWEEP._resolve_checkpoint_policy(unexpected)
