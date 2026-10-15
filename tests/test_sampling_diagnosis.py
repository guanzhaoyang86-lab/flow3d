"""DDIM observation must preserve sampling; diagnostics must preserve provenance."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch

from flow_observation.diffusion import GaussianDiffusion


_SHAPES = [(2, 7), (2, 3, 2, 3, 4)]
_ROOT = Path(__file__).resolve().parents[1]
_DIAGNOSTIC_SCRIPT = _ROOT / "scripts" / "diagnose_diffusion_sampling.py"


def _conditioned_denoiser(
    noisy: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor | None
) -> torch.Tensor:
    del timesteps
    return noisy.tanh() * 0.07 + (0.0 if condition is None else condition)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("eta", [0.0, 1.0])
@pytest.mark.parametrize("use_generator", [False, True])
def test_ddim_trace_preserves_exact_output_and_rng_even_when_observer_mutates(
    shape: tuple[int, ...], eta: float, use_generator: bool
) -> None:
    diffusion = GaussianDiffusion(9, clip_x0=2.0).to(dtype=torch.float64)
    condition = torch.tensor(0.13, dtype=torch.float64)
    records: list[dict[str, Any]] = []

    def observer(record: dict[str, Any]) -> None:
        assert not torch.is_grad_enabled()
        records.append(record)
        for value in record.values():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                assert value.grad_fn is None
                # Callback-owned copies cannot affect an update or later steps.
                value.fill_(float("nan"))

    def sample(callback: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        torch.manual_seed(891)
        generator = torch.Generator().manual_seed(172) if use_generator else None
        result = diffusion.ddim_sample(
            _conditioned_denoiser,
            shape,
            condition,
            sampling_steps=4,
            eta=eta,
            guidance_scale=1.5,
            unconditional_condition=torch.zeros_like(condition),
            generator=generator,
            trace_callback=callback,
        )
        return (
            result,
            torch.random.get_rng_state(),
            None if generator is None else generator.get_state(),
        )

    expected, expected_global_rng, expected_generator_rng = sample(None)
    actual, actual_global_rng, actual_generator_rng = sample(observer)

    assert len(records) == 4
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(actual_global_rng, expected_global_rng)
    if use_generator:
        assert torch.equal(actual_generator_rng, expected_generator_rng)


@pytest.mark.parametrize("shape", _SHAPES)
def test_ddim_trace_exposes_raw_clipped_and_guided_values_and_final_step(
    shape: tuple[int, ...],
) -> None:
    diffusion = GaussianDiffusion(7, clip_x0=0.25).to(dtype=torch.float64)
    initial = torch.linspace(-3.0, 3.0, torch.tensor(shape).prod().item()).reshape(shape)
    initial = initial.to(dtype=torch.float64)
    condition = initial.new_tensor(0.2)
    records: list[dict[str, Any]] = []
    strength = 0.05

    sample = diffusion.ddim_sample(
        _conditioned_denoiser,
        shape,
        condition,
        sampling_steps=4,
        eta=0,
        guidance_scale=1.5,
        unconditional_condition=torch.zeros_like(condition),
        guidance_loss=lambda clean, _timesteps: (clean - 0.1).square().mean(),
        guidance_strength=strength,
        projection=lambda clean: clean * 0.5,
        initial_noise=initial,
        trace_callback=records.append,
    )

    assert [record["step"] for record in records] == [0, 1, 2, 3]
    assert [record["timestep"] for record in records] == [6, 4, 2, 0]
    assert [record["previous_timestep"] for record in records] == [4, 2, 0, -1]
    expected_keys = {
        "step", "timestep", "previous_timestep", "sample", "predicted_noise",
        "predicted_x0_raw", "predicted_x0", "clean_for_update", "updated_sample",
    }
    for index, record in enumerate(records):
        assert set(record) == expected_keys
        for value in record.values():
            if isinstance(value, torch.Tensor):
                assert value.shape == shape
                assert not value.requires_grad
                assert value.grad_fn is None
        timestep = record["timestep"]
        expected_epsilon = record["sample"].tanh() * 0.07 + 1.5 * condition
        torch.testing.assert_close(record["predicted_noise"], expected_epsilon)
        expected_raw = (
            record["sample"]
            - diffusion.sqrt_one_minus_alpha_bars[timestep] * expected_epsilon
        ) / diffusion.sqrt_alpha_bars[timestep]
        torch.testing.assert_close(record["predicted_x0_raw"], expected_raw)
        expected_clipped = expected_raw.clamp(-0.25, 0.25)
        torch.testing.assert_close(record["predicted_x0"], expected_clipped)
        expected_guided = (
            expected_clipped
            - strength * 2.0 * (expected_clipped - 0.1) / expected_clipped.numel()
        ) * 0.5
        torch.testing.assert_close(record["clean_for_update"], expected_guided)
        if index:
            torch.testing.assert_close(
                record["sample"], records[index - 1]["updated_sample"], atol=0, rtol=0
            )
    assert (records[0]["predicted_x0_raw"].abs() > 0.25).any()
    torch.testing.assert_close(records[0]["sample"], initial, atol=0, rtol=0)
    torch.testing.assert_close(records[-1]["updated_sample"], sample, atol=0, rtol=0)
    torch.testing.assert_close(records[-1]["clean_for_update"], sample, atol=0, rtol=0)


def test_ddim_trace_callback_errors_propagate_without_silently_completing() -> None:
    steps: list[int] = []

    def observer(record: dict[str, Any]) -> None:
        steps.append(record["step"])
        if record["step"] == 1:
            raise RuntimeError("diagnostic writer failed")

    with pytest.raises(RuntimeError, match="diagnostic writer failed"):
        GaussianDiffusion(5).ddim_sample(
            _conditioned_denoiser, (1, 4), None, sampling_steps=4,
            initial_noise=torch.ones(1, 4), trace_callback=observer,
        )
    assert steps == [0, 1]


@pytest.fixture(scope="module")
def diagnostic_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Tiny CPU artifacts exercise all supported checkpoint representations."""
    directory = tmp_path_factory.mktemp("diagnostic_checkpoints")
    helper_spec = importlib.util.spec_from_file_location(
        "_sampling_diagnosis_cli_helpers",
        Path(__file__).with_name("test_sparse_diffusion_cli.py"),
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)
    splits: dict[str, list[dict[str, str]]] = {
        "train": [], "validation": [], "test": [],
    }
    for index, split in enumerate(("train", "validation", "validation", "test")):
        case_id = f"synthetic-{split}-{index}"
        case_path = directory / f"{case_id}.npz"
        helpers._write_eight_cubed_case(case_path)
        with np.load(case_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        # Distinct fixtures cannot accidentally make swapped conditions equal.
        arrays["flow_field"] *= 1.0 + 0.1 * index
        arrays["trajectories_2d"] += 0.05 * index
        metadata = json.loads(str(arrays["metadata"].item()))
        metadata["case_id"] = case_id
        metadata["scope"] = "Synthetic unit test; no scientific result"
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(case_path, **arrays)
        splits[split].append({
            "case_id": case_id,
            "flow_group_id": f"distinct-group-{index}",
            "path": case_path.name,
        })
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({
        "format_version": 1, "mode": "scientific", "splits": splits,
    }, indent=2), encoding="utf-8")

    checkpoints: dict[str, Path] = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setenv("MKL_NUM_THREADS", "1")
        prepared = directory / "tensor_prepared"
        helpers._run_cli([
            str(_ROOT / "scripts/prepare_tensor_space.py"),
            "--manifest", str(manifest), "--output-dir", str(prepared),
            "--ranks", "2", "--hooi-iterations", "1", "--device", "cpu",
        ])
        for architecture in ("unet3d", "dit3d", "tensor-dit"):
            training_dir = directory / architecture
            arguments = [
                str(_ROOT / "scripts/train_sparse_track_diffusion.py"),
                "--manifest", str(manifest), "--output-dir", str(training_dir),
                "--architecture", architecture, "--epochs", "1",
                "--num-particles", "2", "--observations-per-flow", "1",
                "--batch-size", "1", "--num-workers", "0", "--diffusion-steps", "4",
                "--base-channels", "8", "--condition-dim", "16",
                "--time-embedding-dim", "16", "--dit-hidden-dim", "24",
                "--dit-depth", "1", "--dit-heads", "3", "--patch-size", "4",
                "--condition-dropout", "0.1", "--device", "cpu", "--no-amp",
                "--save-every", "1", "--seed", "31",
            ]
            if architecture == "tensor-dit":
                arguments.extend(["--tensor-artifact", str(prepared / "rank_2.pt")])
            helpers._run_cli(arguments)
            checkpoints[architecture] = training_dir / "best.pt"

    # Diagnosis must use the saved normalization/codec and validation data only.
    # These still satisfy manifest path checks, but cannot be opened as NPZ.
    for split in ("train", "test"):
        for entry in splits[split]:
            (directory / entry["path"]).write_bytes(b"must not load during diagnosis")
    (prepared / "rank_2.pt").rename(prepared / "rank_2.hidden")
    return {
        "directory": directory,
        "manifest": manifest,
        "checkpoints": checkpoints,
        "validation_case_ids": {entry["case_id"] for entry in splits["validation"]},
    }


def _diagnostic_cli(
    fixture: dict[str, Any], output: Path, checkpoints: list[Path], *, indices: str = "0,1"
) -> subprocess.CompletedProcess[str]:
    arguments = [
        sys.executable, str(_DIAGNOSTIC_SCRIPT),
        "--manifest", str(fixture["manifest"]), "--output-dir", str(output),
        "--indices", indices, "--sampling-steps", "2", "--seed", "47",
        "--num-probe-particles", "2", "--trajectory-substeps", "1",
        "--device", "cpu", "--threads", "1",
    ]
    for checkpoint in checkpoints:
        arguments.extend(["--checkpoint", str(checkpoint)])
    return subprocess.run(
        arguments, cwd=_ROOT, capture_output=True, text=True, timeout=180, check=False,
    )


def test_diagnosis_cli_covers_validation_models_traces_and_refuses_overwrite(
    diagnostic_checkpoints: dict[str, Any], tmp_path: Path,
) -> None:
    fixture = diagnostic_checkpoints
    checkpoints = list(fixture["checkpoints"].values())
    hashes_before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    output = tmp_path / "diagnosis"
    result = _diagnostic_cli(fixture, output, checkpoints)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostic_only"] is True
    assert summary["scientific_result"] is False
    assert summary["split"] == "validation"
    assert summary["status"] == "completed"
    assert summary["failures"] == []
    assert summary["manifest_sha256"] == hashlib.sha256(fixture["manifest"].read_bytes()).hexdigest()
    assert "commit" in summary["git"]
    assert len(summary["checkpoints"]) == 3
    assert {entry["architecture"] for entry in summary["checkpoints"]} == {
        "unet3d", "dit3d", "tensor-dit",
    }
    expected_variants = {
        "correct_cfg1p5_eta1", "zero_condition_cfg1p5_eta1",
        "other_case_condition_cfg1p5_eta1", "correct_cfg1_eta1", "correct_cfg1p5_eta0",
    }
    for entry in summary["checkpoints"]:
        assert entry["status"] == "completed"
        assert entry["checkpoint_sha256"] == hashes_before[Path(entry["checkpoint"])]
        assert entry["model_config"]
        assert entry["train_config"]
        assert entry["diffusion"]["num_steps"] == 4
        assert entry["data_config"]["num_particles"] == 2
        assert entry["training_provenance"]["manifest_mode"] == "scientific"
        assert {case["case_id"] for case in entry["cases"]} == fixture["validation_case_ids"]
        for case in entry["cases"]:
            assert case["ground_truth_replay"]
            assert case["teacher_forced"]
            assert case["other_condition"]["case_id"] != case["case_id"]
            assert case["other_condition"]["flow_group_id"] != case["flow_group_id"]
            assert case["other_condition"]["case_id"] in fixture["validation_case_ids"]
            assert len(case["initial_noise_sha256"]) == 64
            assert case["sampling_seed"] == 47 + case["index"]
            assert not np.intersect1d(case["particle_indices"], case["probe_particle_indices"]).size
            for teacher_row in case["teacher_forced"]:
                assert teacher_row["oracle_x0_raw_mse"] < 1e-6
                assert set(teacher_row["conditions"]) == {"correct", "zero", "other_case"}
                for condition in teacher_row["conditions"].values():
                    assert np.isfinite(condition["noise_mse_cfg1"])
                    assert np.isfinite(condition["noise_mse_cfg1p5"])
            if entry["architecture"] == "tensor-dit":
                assert "fresh codec.encode(validation flow)" == case["teacher_forced_target"]
                assert case["codec_validation_reconstruction"]
            assert {variant["name"] for variant in case["variants"]} == expected_variants
            initial_summaries = []
            for variant in case["variants"]:
                assert variant["status"] == "completed"
                trace = [json.loads(line) for line in Path(variant["trace_file"]).read_text(encoding="utf-8").splitlines()]
                initial_summaries.append(trace[0]["tensors"]["sample"])
                assert set(trace[0]["tensors"]) == {
                    "sample", "predicted_noise", "predicted_x0_raw", "predicted_x0",
                    "clean_for_update", "updated_sample",
                }
                assert "physical_norm_ratio_unknown_interior" in trace[0]["physical"]["predicted_x0_raw"]
                if entry["architecture"] == "tensor-dit":
                    factors = trace[0]["tucker"]["predicted_x0_raw"]["factors"]
                    assert len(factors) == 3
                    assert all(factor["orthogonality_residual_frobenius"] >= 0 for factor in factors)
                metrics = variant["metrics"]
                for metric in ("field_relative_l2_unknown_interior", "field_cosine_unknown_interior"):
                    assert np.isfinite(metrics[metric])
            assert all(initial == initial_summaries[0] for initial in initial_summaries)

    traces = list(output.rglob("trace.jsonl"))
    assert len(traces) == 3 * 2 * 5
    for trace_path in traces:
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[-1]["previous_timestep"] == -1
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints} == hashes_before
    files_before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    repeated = _diagnostic_cli(fixture, output, checkpoints)
    assert repeated.returncode != 0
    assert "exist" in (repeated.stdout + repeated.stderr).lower()
    assert {path: path.read_bytes() for path in output.rglob("*") if path.is_file()} == files_before


def test_diagnosis_cli_retains_failed_checkpoint_and_runs_later_models(
    diagnostic_checkpoints: dict[str, Any], tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    valid = diagnostic_checkpoints["checkpoints"]["unet3d"]
    output = tmp_path / "with_failure"
    result = _diagnostic_cli(diagnostic_checkpoints, output, [corrupt, valid])
    assert result.returncode != 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    entries = summary["checkpoints"]
    assert len(entries) == 2
    assert Path(entries[0]["checkpoint"]) == corrupt
    assert entries[0]["status"] == "failed"
    assert entries[0]["error"]
    assert Path(entries[1]["checkpoint"]) == valid
    assert entries[1]["status"] == "completed"
    assert len(entries[1]["cases"]) == 2


def test_diagnosis_cli_marks_unavailable_other_flow_condition_as_skipped(
    diagnostic_checkpoints: dict[str, Any], tmp_path: Path,
) -> None:
    original = diagnostic_checkpoints["manifest"]
    manifest = json.loads(original.read_text(encoding="utf-8"))
    manifest["splits"]["validation"] = manifest["splits"]["validation"][:1]
    for entries in manifest["splits"].values():
        for entry in entries:
            entry["path"] = str(original.parent / entry["path"])
    single_case = tmp_path / "single_validation_case.json"
    single_case.write_text(json.dumps(manifest), encoding="utf-8")
    fixture = {**diagnostic_checkpoints, "manifest": single_case}
    output = tmp_path / "single_validation_diagnosis"
    result = _diagnostic_cli(
        fixture, output, [fixture["checkpoints"]["unet3d"]], indices="0",
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    case = summary["checkpoints"][0]["cases"][0]
    assert "unavailable" in case["other_condition"]
    variants = {variant["name"]: variant for variant in case["variants"]}
    other = variants.pop("other_case_condition_cfg1p5_eta1")
    assert other["status"] == "skipped"
    assert other["reason"]
    assert all(variant["status"] == "completed" for variant in variants.values())
