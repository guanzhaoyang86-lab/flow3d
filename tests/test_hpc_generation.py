"""Protect real data from unsafe execution, incompatible resume and partial writes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_hpc_generation as generation
import generate_taichi_lbm3d_cavity_dataset as cavity


def test_generation_rejects_login_node_before_writing(tmp_path):
    env = os.environ.copy()
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_hpc_generation.py"), "pilot",
         "--storage-root", str(tmp_path)], env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires a Slurm job" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changed", ["code_commit", "generation_parameters", "packages", "gpu"])
def test_resume_rejects_changed_data_provenance(tmp_path, changed):
    configuration = {"code_commit": "a" * 40, "generation_parameters": ["--num-cases", "1000"],
                     "packages": {"taichi": "1.7.4"}, "gpu": "NVIDIA A100"}
    (tmp_path / "generation.json").write_text(json.dumps(configuration))
    generation.check_resume(tmp_path, configuration)
    with pytest.raises(ValueError, match="same code"):
        generation.check_resume(tmp_path, {**configuration, changed: "changed"})
    (tmp_path / "COMPLETE.json").write_text("{}")
    with pytest.raises(ValueError, match="already complete"):
        generation.check_resume(tmp_path, configuration)


@pytest.mark.parametrize("backend,commit,value", [
    ("cpu", generation.UPSTREAM_COMMIT, 0.1),
    ("cuda", "unknown", 0.1),
    ("cuda", generation.UPSTREAM_COMMIT, float("nan")),
])
def test_validation_rejects_wrong_solver_or_nonfinite_data(tmp_path, backend, commit, value):
    np.savez_compressed(tmp_path / "case_00000.npz",
                        metadata=np.asarray(json.dumps({"backend": backend, "upstream_commit": commit})),
                        flow_field=np.full((1, 3, 8, 8, 8), value))
    with pytest.raises(ValueError):
        generation.validate_archives(tmp_path, 1)


def test_interrupted_archive_write_preserves_previous_data(tmp_path, monkeypatch):
    output = tmp_path / "case.npz"
    payload = {"flow_field": torch.ones(1, 3, 8, 8, 8), "metadata": "{}"}
    cavity._save_npz(payload, output)
    original = output.read_bytes()

    def interrupted(stream, **kwargs):
        stream.write(b"partial archive")
        raise OSError("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(cavity.np, "savez_compressed", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            cavity._save_npz(payload, output)
    assert output.read_bytes() == original
    payload["flow_field"].mul_(2)
    cavity._save_npz(payload, output)
    with np.load(output, allow_pickle=False) as archive:
        assert np.all(archive["flow_field"] == 2)
    assert not output.with_suffix(".npz.tmp").exists()
