"""Tests for the Taichi-LBM3D boundary without requiring Taichi itself."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from flow_observation.cfd import NPZCFDAdapter
from flow_observation.taichi_lbm3d import (
    make_lid_driven_cavity_geometry,
    taichi_solid_mask_to_canonical,
    taichi_velocity_to_canonical,
)


def _tagged_raw_velocity(nx: int, ny: int, nz: int) -> torch.Tensor:
    x, y, z = torch.meshgrid(
        torch.arange(nx),
        torch.arange(ny),
        torch.arange(nz),
        indexing="ij",
    )
    spatial_tag = 100.0 * x + 10.0 * y + z
    return torch.stack(
        [spatial_tag + 1000.0 * component for component in range(3)],
        dim=-1,
    )


def test_taichi_velocity_conversion_preserves_components_and_xyz_indices() -> None:
    raw = _tagged_raw_velocity(nx=4, ny=5, nz=6)
    canonical = taichi_velocity_to_canonical(raw)

    assert canonical.shape == (1, 3, 6, 5, 4)
    for component in range(3):
        for x, y, z in ((0, 0, 0), (3, 4, 5), (2, 1, 4)):
            expected = 1000.0 * component + 100.0 * x + 10.0 * y + z
            assert canonical[0, component, z, y, x].item() == expected


def test_taichi_velocity_conversion_accepts_a_time_axis() -> None:
    first = _tagged_raw_velocity(nx=4, ny=5, nz=6)
    raw = torch.stack((first, first + 0.5), dim=0)
    canonical = taichi_velocity_to_canonical(raw)

    assert canonical.shape == (2, 3, 6, 5, 4)
    torch.testing.assert_close(canonical[1], canonical[0] + 0.5)


@pytest.mark.parametrize(
    "raw",
    [
        torch.zeros((4, 5, 6)),
        torch.zeros((4, 5, 6, 2)),
        torch.zeros((4, 5, 6, 3), dtype=torch.int64),
    ],
)
def test_taichi_velocity_conversion_rejects_invalid_inputs(
    raw: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        taichi_velocity_to_canonical(raw)


def test_solid_mask_conversion_uses_zyx_storage() -> None:
    raw = torch.zeros((4, 5, 6), dtype=torch.int8)
    raw[3, 2, 5] = 1
    canonical = taichi_solid_mask_to_canonical(raw)

    assert canonical.shape == (6, 5, 4)
    assert canonical.dtype == torch.bool
    assert canonical[5, 2, 3]
    assert canonical.sum().item() == 1


def test_cavity_geometry_has_five_walls_and_an_open_moving_lid() -> None:
    nx, ny, nz = 8, 9, 10
    geometry = make_lid_driven_cavity_geometry(nx, ny, nz)

    assert geometry.shape == (nx, ny, nz)
    assert torch.all(geometry[0] == 1)
    assert torch.all(geometry[:, 0] == 1)
    assert torch.all(geometry[:, -1] == 1)
    assert torch.all(geometry[:, :, 0] == 1)
    assert torch.all(geometry[:, :, -1] == 1)
    assert torch.all(geometry[-1, 1:-1, 1:-1] == 0)
    assert torch.all(geometry[1:-1, 1:-1, 1:-1] == 0)


def test_npz_adapter_prefers_velocity_times_over_observation_times(tmp_path) -> None:
    path = tmp_path / "lbm_snapshot.npz"
    np.savez(
        path,
        flow_field=np.zeros((1, 3, 4, 5, 6), dtype=np.float32),
        velocity_times=np.array([7.0], dtype=np.float32),
        observation_times=np.linspace(0.0, 2.0, 5, dtype=np.float32),
        domain_bounds=np.array(
            ((0.0, 5.0), (0.0, 4.0), (0.0, 3.0)),
            dtype=np.float32,
        ),
        metadata=np.asarray(json.dumps({"boundary_mode": "terminate"})),
    )

    snapshots = NPZCFDAdapter(path).load()

    assert snapshots.velocity_field.shape == (1, 3, 4, 5, 6)
    torch.testing.assert_close(snapshots.snapshot_times, torch.tensor([7.0]))
    assert snapshots.boundary_mode == "terminate"
