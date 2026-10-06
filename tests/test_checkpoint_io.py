"""HPC checkpoint metadata must load unchanged on a different host OS."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import torch

from flow_observation.checkpoint_io import load_trusted_checkpoint


class _ForeignPathReference:
    """Emit the other OS's actual pathlib reducer without constructing it."""

    def __reduce__(self):
        constructor = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
        value = "/work/hdd/project/data" if os.name == "nt" else "D:/project/data"
        return constructor, (value,)


@pytest.mark.parametrize("legacy", [False, True])
def test_foreign_paths_and_tensors_load_without_mutating_pathlib_or_file(tmp_path, legacy):
    path = tmp_path / "checkpoint.pt"
    local_path = tmp_path / "native-data"
    original_classes = (pathlib.PosixPath, pathlib.WindowsPath)
    torch.save(
        {
            "train_config": {"manifest": _ForeignPathReference(), "nested": [local_path]},
            "data_config": {"num_particles": 2},
            "weights": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        },
        path,
        _use_new_zipfile_serialization=not legacy,
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(NotImplementedError):
        torch.load(path, map_location="cpu", weights_only=False)

    loaded = load_trusted_checkpoint(path)
    foreign = loaded["train_config"]["manifest"]
    expected_type = PurePosixPath if os.name == "nt" else PureWindowsPath
    assert type(foreign) is expected_type
    assert foreign.as_posix() == ("/work/hdd/project/data" if os.name == "nt" else "D:/project/data")
    assert loaded["train_config"]["nested"] == [local_path]
    assert isinstance(loaded["train_config"]["nested"][0], Path)
    torch.testing.assert_close(loaded["weights"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert loaded["weights"].device.type == "cpu"
    assert (pathlib.PosixPath, pathlib.WindowsPath) == original_classes
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_sweep_accepts_particle_count_from_hpc_checkpoint(tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_sparse_particle_sweep.py"
    spec = importlib.util.spec_from_file_location("portable_sweep_test", source)
    assert spec and spec.loader
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    checkpoint = tmp_path / "best.pt"
    torch.save({"train_config": {"manifest": _ForeignPathReference()},
                "data_config": {"num_particles": 12}}, checkpoint)
    assert sweep._checkpoint_particle_count(checkpoint) == 12
