"""Exercise the real model through the HPC wrapper, without requiring a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_hpc_diffusion.py"


def invoke(tmp_path, *args, slurm_job=None):
    environment = os.environ.copy()
    environment.pop("SLURM_JOB_ID", None)
    if slurm_job is not None:
        environment["SLURM_JOB_ID"] = slurm_job
    return subprocess.run(
        [sys.executable, str(SCRIPT), "smoke", "--storage-root", str(tmp_path), *args],
        env=environment, cwd=ROOT, capture_output=True, text=True, timeout=180,
    )


def test_real_diffusion_wrapper_cpu_smoke(tmp_path):
    result = invoke(tmp_path, "--device", "cpu")
    assert result.returncode == 0, result.stdout + result.stderr
    runs = list((tmp_path / "results" / "experiments").iterdir())
    assert len(runs) == 1
    run = runs[0]
    summary = json.loads((run / "results.json").read_text())
    assert summary["status"] == "completed"
    assert summary["scientific_result"] is False
    assert summary["training"]["device"] == "cpu"
    assert summary["sampling_metadata"]["scientific_result"] is False
    assert summary["elapsed_seconds"] > 0
    assert (Path(summary["checkpoint_dir"]) / "best.pt").is_file()
    assert (Path(summary["checkpoint_dir"]) / "latest.pt").is_file()
    configuration = json.loads((run / "config.yaml").read_text())
    assert configuration["parameters"][0]["seed"] == 31
    assert configuration["parameters"][0]["base_channels"] == 8
    assert len((run / "command.txt").read_text().splitlines()) == 2
    assert "epoch" in (run / "log.txt").read_text().lower()
    with np.load(run / "posterior.npz", allow_pickle=False) as archive:
        assert archive["posterior_samples"].shape == (1, 3, 8, 8, 8)
        assert np.isfinite(archive["posterior_samples"]).all()


@pytest.mark.parametrize("args,job,message", [
    ([], None, "GPU execution requires a Slurm job"),
    (["--device", "cpu"], "123", "CPU execution is only for a local smoke test"),
    (["--device", "cpu", "--", "--epochs", "999"], None, "smoke uses fixed tiny data"),
])
def test_wrapper_rejects_unsafe_execution_before_output(tmp_path, args, job, message):
    result = invoke(tmp_path, *args, slurm_job=job)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "results").exists()
