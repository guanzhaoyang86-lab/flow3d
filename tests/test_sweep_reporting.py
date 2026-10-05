"""Reporting and short-job recovery must preserve completed scientific cases."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sparse_particle_sweep.py"
_SPEC = importlib.util.spec_from_file_location("sweep_reporting_under_test", _SCRIPT)
assert _SPEC and _SPEC.loader
_SWEEP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SWEEP)


def _arguments(tmp_path: Path, *extra: str):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"data_config": {"num_particles": 2}}, checkpoint)
    return _SWEEP._build_parser().parse_args([
        "--manifest", str(manifest), "--checkpoint", f"2={checkpoint}",
        "--particle-counts", "2", "--num-test-cases", "3",
        "--num-samples", "1", "--sampling-steps", "1", "--device", "cpu",
        "--output-dir", str(tmp_path / "evaluation"), *extra,
    ])


def _write_posterior(command: list[str]) -> None:
    output = Path(command[command.index("--output") + 1])
    index = int(command[command.index("--index") + 1])
    metadata = {
        "architecture": "unet3d", "training_seed": 31,
        "trained_particles": 2, "sampled_particles": 2, "split": "test",
        "method": "2-particle conditional 3D flow diffusion",
        "scientific_result": True, "scientific_result_reasons": [],
        "checkpoint_has_training_provenance": True,
        "evaluation_manifest_provenance": {"manifest_mode": "scientific"},
        "case_id": f"case-{index}",
        "sampling_steps": int(command[command.index("--sampling-steps") + 1]),
    }
    np.savez_compressed(
        output, metrics=np.asarray(json.dumps({"num_observed_particles": 2, "score": 0.5})),
        metadata=np.asarray(json.dumps(metadata)),
    )


def test_budget_stops_at_completed_boundary_and_resume_eta_uses_new_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    args = _arguments(tmp_path, "--max-runtime-seconds", "35")
    now = [0.0]
    called = []

    def fake_run(command, **kwargs):
        del kwargs
        called.append(int(command[command.index("--index") + 1]))
        _write_posterior(command)
        now[0] += 20.0
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    summary = _SWEEP.run_sweep(args)
    assert called == [0]
    assert summary["completed_runs"] == 1
    assert summary["complete"] is False
    assert summary["runtime_budget_exhausted"] is True
    assert "current-run mean" in summary["partial_reason"]
    assert json.loads((args.output_dir / "summary.json").read_text())["complete"] is False
    assert "rough_remaining_minutes=0.7" in capsys.readouterr().out

    # An old record's duration must never enter this invocation's ETA estimate.
    runs = args.output_dir / "runs.jsonl"
    record = json.loads(runs.read_text())
    record["elapsed_seconds"] = 999999.0
    runs.write_text(json.dumps(record) + "\n", encoding="utf-8")
    args.resume = True
    args.max_runtime_seconds = 50.0
    summary = _SWEEP.run_sweep(args)
    assert called == [0, 1, 2]
    assert summary["complete"] is True
    progress = capsys.readouterr().out
    assert "rough_remaining_minutes=0.3 (based on 1 new cases" in progress
    assert "excludes queue time" in progress


def test_budget_cli_exit_is_distinct_from_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--manifest", "unused", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(_SWEEP, "run_sweep", lambda args: {
        "warning": None, "complete": False, "runtime_budget_exhausted": True,
    })
    with pytest.raises(SystemExit) as error:
        _SWEEP.main()
    assert error.value.code == 75


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_budget_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        _arguments(tmp_path, "--max-runtime-seconds", value)


def test_example_figure_failure_preserves_posterior_and_resume_repairs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path, "--figures", "--num-test-cases", "1")
    sampler_calls = []
    figure_commands = []
    fail_figures = [True]

    def fake_run(command, **kwargs):
        del kwargs
        if "case" in command:
            figure_commands.append(command)
            return subprocess.CompletedProcess(command, 2 if fail_figures[0] else 0)
        sampler_calls.append(command)
        _write_posterior(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="completed sampling is preserved"):
        _SWEEP.run_sweep(args)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["complete"] is False
    assert summary["completed_runs"] == 1
    assert "example figures failed" in summary["partial_reason"]
    assert json.loads((args.output_dir / "runs.jsonl").read_text())["status"] == "completed"
    fail_figures[0] = False
    args.resume = True
    summary = _SWEEP.run_sweep(args)
    assert summary["complete"] is True
    assert summary["figures_requested"] is True
    assert len(sampler_calls) == 1
    assert len(figure_commands) == 2
    command = figure_commands[-1]
    assert command[command.index("--architecture") + 1] == "unet3d"
    assert command[command.index("--training-seed") + 1] == "31"
    assert command[command.index("--test-case-index") + 1] == "0"
    assert Path(command[command.index("--output-dir") + 1]).name == "N002_case0000"


def test_resume_retries_corrupt_archive_and_ignores_only_torn_last_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path, "--num-test-cases", "1")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        _write_posterior(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    _SWEEP.run_sweep(args)
    output = args.output_dir / "N002" / "case_0000_seed_47.npz"
    output.write_bytes(b"torn NPZ")
    runs = args.output_dir / "runs.jsonl"
    with runs.open("a", encoding="utf-8") as stream:
        stream.write('{"run_id":')
    args.resume = True
    summary = _SWEEP.run_sweep(args)
    assert len(calls) == 2
    assert summary["complete"] is True
    assert len(runs.read_text().splitlines()) == 1
    runs.write_text('{"bad":\n' + runs.read_text(), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON.*line 1"):
        _SWEEP.run_sweep(args)


def test_budget_cannot_preempt_one_long_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _arguments(tmp_path, "--max-runtime-seconds", "10")
    now = [0.0]
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        _write_posterior(command)
        now[0] += 100.0
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    summary = _SWEEP.run_sweep(args)
    assert len(calls) == 1
    assert summary["completed_runs"] == 1
    assert summary["runtime_budget_exhausted"] is True


@pytest.mark.parametrize("change", ["steps", "checkpoint", "manifest"])
def test_resume_protocol_rejects_changed_scientific_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    args = _arguments(tmp_path, "--num-test-cases", "1")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        _write_posterior(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    _SWEEP.run_sweep(args)
    protocol_before = (args.output_dir / "protocol.json").read_bytes()
    records_before = (args.output_dir / "runs.jsonl").read_bytes()
    args.resume = True
    if change == "steps":
        args.sampling_steps += 1
    elif change == "checkpoint":
        torch.save({"data_config": {"num_particles": 2}, "different_weights": 42}, args.checkpoint[0][1])
    else:
        args.manifest.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="resume protocol differs"):
        _SWEEP.run_sweep(args)
    assert len(calls) == 1
    assert (args.output_dir / "protocol.json").read_bytes() == protocol_before
    assert (args.output_dir / "runs.jsonl").read_bytes() == records_before


def test_legacy_resume_retries_cases_with_changed_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    args = _arguments(tmp_path, "--num-test-cases", "1")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        _write_posterior(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    _SWEEP.run_sweep(args)
    (args.output_dir / "protocol.json").unlink()
    args.resume = True
    args.sampling_steps += 1
    summary = _SWEEP.run_sweep(args)
    assert summary["complete"] is True
    assert len(calls) == 2
    assert "Legacy resume" in capsys.readouterr().out
    assert len((args.output_dir / "runs.jsonl").read_text().splitlines()) == 1


def test_resume_retries_archive_with_conflicting_sampling_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path, "--num-test-cases", "1")
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        _write_posterior(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_SWEEP.subprocess, "run", fake_run)
    _SWEEP.run_sweep(args)
    output = args.output_dir / "N002" / "case_0000_seed_47.npz"
    with np.load(output, allow_pickle=False) as archive:
        metrics = np.array(archive["metrics"], copy=True)
        metadata = json.loads(str(archive["metadata"].item()))
    metadata["sampling_steps"] = 999
    np.savez_compressed(output, metrics=metrics, metadata=np.asarray(json.dumps(metadata)))
    args.resume = True
    summary = _SWEEP.run_sweep(args)
    assert summary["complete"] is True
    assert len(calls) == 2
