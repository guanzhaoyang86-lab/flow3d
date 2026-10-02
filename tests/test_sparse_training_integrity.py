from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPOSITORY_ROOT / "scripts" / "train_sparse_track_diffusion.py"
_SPEC = importlib.util.spec_from_file_location("sparse_diffusion_trainer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_TRAINER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINER)

_CLI_TEST = _REPOSITORY_ROOT / "tests" / "test_sparse_diffusion_cli.py"
_CLI_SPEC = importlib.util.spec_from_file_location("sparse_cli_test_helpers", _CLI_TEST)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI_HELPERS = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(_CLI_HELPERS)


def test_validation_is_fixed_and_selects_on_full_enabled_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDiffusion:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def training_loss(
            self,
            _unet: nn.Module,
            clean: torch.Tensor,
            _condition: torch.Tensor,
            *,
            timesteps: torch.Tensor,
            noise: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            self.calls.append((timesteps.clone(), noise.clone()))
            return {
                "loss": clean.new_tensor(2.0),
                "predicted_x0": torch.zeros_like(clean),
                "timesteps": timesteps,
            }

    monkeypatch.setattr(
        _TRAINER,
        "_encode_condition",
        lambda _encoder, batch: batch["flow"].new_zeros((batch["flow"].shape[0], 4)),
    )
    monkeypatch.setattr(
        _TRAINER,
        "_auxiliary_losses",
        lambda *_args, **_kwargs: {
            "track": torch.tensor(1.0),
            "divergence": torch.tensor(2.0),
            "boundary": torch.tensor(3.0),
        },
    )
    args = argparse.Namespace(
        diffusion_steps=11,
        seed=17,
        track_loss_weight=2.0,
        divergence_loss_weight=3.0,
        boundary_loss_weight=4.0,
    )
    batch = {"flow": torch.zeros(2, 3, 4, 4, 4)}
    diffusion = FakeDiffusion()
    stats = _TRAINER.FlowNormalizationStats(
        mean=torch.zeros(3), std=torch.ones(3), count=1
    )

    first = _TRAINER._run_validation(
        nn.Identity(), nn.Identity(), diffusion, [batch], torch.device("cpu"), stats, args
    )
    second = _TRAINER._run_validation(
        nn.Identity(), nn.Identity(), diffusion, [batch], torch.device("cpu"), stats, args
    )

    assert first == second
    assert first == {
        "total": pytest.approx(22.0),
        "diffusion": pytest.approx(2.0),
        "track": pytest.approx(1.0),
        "divergence": pytest.approx(2.0),
        "boundary": pytest.approx(3.0),
    }
    assert torch.equal(diffusion.calls[0][0], diffusion.calls[1][0])
    assert torch.equal(diffusion.calls[0][1], diffusion.calls[1][1])


def test_training_provenance_hashes_original_bytes_and_records_split_groups(
    tmp_path: Path,
) -> None:
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        paths[split] = tmp_path / f"{split}.npz"
        paths[split].write_bytes(b"placeholder")
    manifest = {
        "format_version": 1,
        "mode": "scientific",
        "splits": {
            split: [
                {
                    "case_id": f"{split}-case",
                    "flow_group_id": f"{split}-group",
                    "path": paths[split].name,
                }
            ]
            for split in paths
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    provenance = _TRAINER._training_provenance(manifest_path)

    assert provenance == {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_mode": "scientific",
        "train_flow_group_ids": ["train-group"],
        "validation_flow_group_ids": ["validation-group"],
        "test_flow_group_ids": ["test-group"],
        "smoke_test": False,
    }


def test_rng_loader_and_scaler_state_restore_exactly() -> None:
    _TRAINER._seed_everything(123)
    loader_generator = torch.Generator().manual_seed(456)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    state = _TRAINER._capture_training_state(loader_generator, scaler)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
        torch.rand(5, generator=loader_generator),
    )
    random.random()
    np.random.random()
    torch.rand(9)
    torch.rand(9, generator=loader_generator)

    assert _TRAINER._restore_training_state(
        state, loader_generator, scaler, use_cuda=False
    )
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
        torch.rand(5, generator=loader_generator),
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_last_gradient_accumulation_window_uses_its_actual_batch_count() -> None:
    assert [_TRAINER._accumulation_window_size(i, 5, 3) for i in range(5)] == [
        3,
        3,
        3,
        2,
        2,
    ]
    assert _TRAINER._accumulation_window_size(0, 1, 8) == 1


def test_auxiliary_boundary_mask_is_explicit_for_batch_size_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_shapes: list[tuple[int, ...]] = []
    original = _TRAINER.apply_hard_boundary_conditions

    def checked_apply(
        fields: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        observed_shapes.append(tuple(mask.shape))
        return original(fields, mask, values)

    monkeypatch.setattr(_TRAINER, "apply_hard_boundary_conditions", checked_apply)
    predicted = torch.zeros(3, 3, 4, 4, 4)
    boundary_mask = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    boundary_mask[0, 0] = True
    boundary_mask[1, 1] = True
    boundary_mask[2, 2] = True
    batch = {
        "boundary_mask": boundary_mask,
        "boundary_values": torch.ones(3, 3, 4, 4, 4),
        "solid_mask": torch.zeros(3, 4, 4, 4, dtype=torch.bool),
    }
    stats = _TRAINER.FlowNormalizationStats(
        mean=torch.zeros(3), std=torch.ones(3), count=1
    )
    args = argparse.Namespace(
        track_loss_weight=0.0,
        divergence_loss_weight=0.0,
        boundary_loss_weight=1.0,
        auxiliary_max_t_fraction=1.0,
        trajectory_substeps=1,
    )

    losses = _TRAINER._auxiliary_losses(
        predicted,
        torch.zeros(3, dtype=torch.long),
        batch,
        stats,
        4,
        args,
    )

    assert observed_shapes == [(3, 1, 4, 4, 4)]
    assert losses["boundary"].item() == pytest.approx(1.0)


def test_resume_is_exact_and_replaces_stale_best_with_valid_best(
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "case.npz"
    manifest_path = tmp_path / "manifest.json"
    _CLI_HELPERS._write_eight_cubed_case(case_path)
    _CLI_HELPERS._write_smoke_manifest(manifest_path, case_path)
    parser = _TRAINER._build_parser()

    def arguments(output: Path, epochs: int, resume: Path | None = None):
        values = [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output),
            "--epochs",
            str(epochs),
            "--batch-size",
            "1",
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
        if resume is not None:
            values.extend(("--resume", str(resume)))
        return parser.parse_args(values)

    uninterrupted_dir = tmp_path / "uninterrupted"
    stage_one_dir = tmp_path / "stage_one"
    resumed_dir = tmp_path / "resumed"
    _TRAINER.train(arguments(uninterrupted_dir, 2))
    _TRAINER.train(arguments(stage_one_dir, 1))
    stale_best = resumed_dir / "best.pt"
    resumed_dir.mkdir()
    stale_best.write_bytes(b"stale")
    returned_best = _TRAINER.train(
        arguments(resumed_dir, 2, stage_one_dir / "latest.pt")
    )

    assert returned_best == stale_best
    assert stale_best.stat().st_size > len(b"stale")
    uninterrupted = torch.load(
        uninterrupted_dir / "latest.pt", map_location="cpu", weights_only=False
    )
    resumed = torch.load(
        resumed_dir / "latest.pt", map_location="cpu", weights_only=False
    )
    assert resumed["epoch"] == 2
    assert torch.equal(
        uninterrupted["training_state"]["torch_rng_state"],
        resumed["training_state"]["torch_rng_state"],
    )
    assert torch.equal(
        uninterrupted["training_state"]["dataloader_generator_state"],
        resumed["training_state"]["dataloader_generator_state"],
    )
    for state_name in (
        "encoder_state",
        "unet_state",
        "ema_encoder_state",
        "ema_unet_state",
    ):
        assert uninterrupted[state_name].keys() == resumed[state_name].keys()
        for name in uninterrupted[state_name]:
            assert torch.equal(
                uninterrupted[state_name][name], resumed[state_name][name]
            ), f"{state_name}.{name} differs after resume"
    metric_keys = [
        key
        for key in uninterrupted["history"][-1]
        if key != "elapsed_seconds"
    ]
    assert {key: uninterrupted["history"][-1][key] for key in metric_keys} == {
        key: resumed["history"][-1][key] for key in metric_keys
    }
    expected_best = torch.load(uninterrupted_dir / "best.pt", map_location="cpu", weights_only=False)
    resumed_best = torch.load(resumed_dir / "best.pt", map_location="cpu", weights_only=False)
    assert expected_best["epoch"] == resumed_best["epoch"]
    assert expected_best["best_validation_total_loss"] == resumed_best["best_validation_total_loss"]


def test_resume_keeps_better_historical_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_path, manifest_path = tmp_path / "case.npz", tmp_path / "manifest.json"
    _CLI_HELPERS._write_eight_cubed_case(case_path)
    _CLI_HELPERS._write_smoke_manifest(manifest_path, case_path)
    losses = iter([1.0, 2.0])
    def validation(*_args, **_kwargs):
        value = next(losses)
        return {"total": value, "diffusion": value, "track": 0.0, "divergence": 0.0, "boundary": 0.0}
    monkeypatch.setattr(_TRAINER, "_run_validation", validation)
    parser = _TRAINER._build_parser()
    common = ["--manifest", str(manifest_path), "--device", "cpu", "--no-amp",
              "--diffusion-steps", "4", "--base-channels", "8", "--condition-dim", "32",
              "--time-embedding-dim", "16"]
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    _TRAINER.train(parser.parse_args([*common, "--epochs", "1", "--output-dir", str(first)]))
    _TRAINER.train(parser.parse_args([*common, "--epochs", "2", "--output-dir", str(resumed),
                                     "--resume", str(first / "latest.pt")]))
    best = torch.load(resumed / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 1 and latest["epoch"] == 2
    assert best["best_validation_total_loss"] == latest["best_validation_total_loss"] == 1.0


def test_resume_at_completed_target_finalizes_missing_artifacts(
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "case.npz"
    manifest_path = tmp_path / "manifest.json"
    _CLI_HELPERS._write_eight_cubed_case(case_path)
    _CLI_HELPERS._write_smoke_manifest(manifest_path, case_path)
    parser = _TRAINER._build_parser()
    output_dir = tmp_path / "interrupted_after_latest"
    values = [
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "1",
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
    _TRAINER.train(parser.parse_args(values))
    (output_dir / "training_summary.json").unlink()
    (output_dir / "best.pt").unlink()

    resumed = parser.parse_args(
        [*values, "--resume", str(output_dir / "latest.pt")]
    )
    returned = _TRAINER.train(resumed)

    assert returned == output_dir / "best.pt"
    assert returned.is_file()
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["epochs"] == 1


def test_completed_resume_without_historical_best_updates_checkpoint_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_path, manifest_path = tmp_path / "case.npz", tmp_path / "manifest.json"
    _CLI_HELPERS._write_eight_cubed_case(case_path)
    _CLI_HELPERS._write_smoke_manifest(manifest_path, case_path)
    validation_values = iter([(1.0, 0.75), (2.0, 1.5)])

    def validation(*_args, **_kwargs):
        total, diffusion = next(validation_values)
        return {"total": total, "diffusion": diffusion, "track": 0.0,
                "divergence": 0.0, "boundary": total - diffusion}

    monkeypatch.setattr(_TRAINER, "_run_validation", validation)
    parser = _TRAINER._build_parser()
    common = ["--manifest", str(manifest_path), "--device", "cpu", "--no-amp",
              "--diffusion-steps", "4", "--base-channels", "8", "--condition-dim", "32",
              "--time-embedding-dim", "16", "--epochs", "2"]
    original_dir = tmp_path / "original"
    _TRAINER.train(parser.parse_args([*common, "--output-dir", str(original_dir)]))
    original_latest = torch.load(original_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert original_latest["best_validation_total_loss"] == 1.0
    assert original_latest["history"][-1]["validation_total"] == 2.0

    # Simulate moving only latest.pt: the earlier, better epoch's actual weights
    # are unavailable beside the resume source, despite their score being saved.
    isolated = tmp_path / "latest_only"
    isolated.mkdir()
    isolated_latest = isolated / "latest.pt"
    torch.save(original_latest, isolated_latest)
    assert _TRAINER._historical_best(original_latest, isolated_latest) is None
    finalized = tmp_path / "finalized"
    returned = _TRAINER.train(parser.parse_args([
        *common, "--output-dir", str(finalized), "--resume", str(isolated_latest),
    ]))
    assert returned == finalized / "best.pt"
    summary = json.loads((finalized / "training_summary.json").read_text())
    assert summary["best_validation_total_loss"] == 2.0
    assert summary["best_validation_diffusion_loss"] == 1.5
    for filename in ("best.pt", "latest.pt"):
        path = finalized / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 2
        assert payload["best_validation_total_loss"] == summary["best_validation_total_loss"]
        assert payload["best_validation_diffusion_loss"] == summary["best_validation_diffusion_loss"]
        assert payload["history"] == original_latest["history"]
        for group in ("encoder_state", "unet_state", "ema_encoder_state", "ema_unet_state"):
            for key, expected in original_latest[group].items():
                torch.testing.assert_close(payload[group][key], expected, atol=0, rtol=0)
        # Recognition succeeds directly from the payload, not by finding a
        # different sibling's weights. No extra epoch was needed to repair it.
        assert _TRAINER._historical_best(payload, path) is payload
