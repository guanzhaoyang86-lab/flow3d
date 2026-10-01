from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from flow_observation.sparse_dataset import derive_physical_flow_group_id


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_taichi_lbm3d_flow_collection.py"
)
_SPEC = importlib.util.spec_from_file_location("flow_collection_integrity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_COLLECTION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COLLECTION
_SPEC.loader.exec_module(_COLLECTION)


def _arguments(tmp_path: Path, *extra: str):
    return _COLLECTION._build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--sampling",
            "grid",
            "--num-cases",
            "4",
            "--lid-speeds",
            "0.05000000000004",
            "--upstream-nius",
            "0.16667000000004",
            "--warmup-steps",
            "600,900",
            "--seeds",
            "7,17",
            *extra,
        ]
    )


def test_group_identity_excludes_snapshot_and_particle_seed_and_uses_command_precision(
    tmp_path: Path,
) -> None:
    args = _arguments(tmp_path)
    cases = _COLLECTION.plan_cases(args)
    assert len({case.flow_group_id for case in cases}) == 1
    assert {case.warmup_steps for case in cases} == {600, 900}
    assert {case.seed for case in cases} == {7, 17}

    rounded_lid = float(format(0.05000000000004, ".12g"))
    rounded_niu = float(format(0.16667000000004, ".12g"))
    assert cases[0].flow_group_id == _COLLECTION._flow_group_id(
        rounded_lid, rounded_niu, args.grid_size
    )
    metadata = {
        "field": "taichi_lbm3d_lid_driven_cavity",
        "grid_size_xyz": [args.grid_size] * 3,
        "lid_face": "x=max",
        "lid_velocity_xyz": [0.0, 0.0, rounded_lid],
        "upstream_niu_parameter": rounded_niu,
        # These must not affect the physical run identity.
        "warmup_steps": 123456,
        "snapshot_iteration": 123456,
        "seed": 999,
    }
    assert cases[0].flow_group_id == derive_physical_flow_group_id(metadata)


def test_single_case_command_receives_case_and_group_identity(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    case = _COLLECTION.plan_cases(args)[0]
    command = _COLLECTION.build_generator_command(args, case, tmp_path / "case.npz")
    assert command[command.index("--case-id") + 1] == case.case_id
    assert command[command.index("--flow-group-id") + 1] == case.flow_group_id
    assert command[command.index("--lid-speed") + 1] == format(case.lid_speed, ".12g")
    assert command[command.index("--upstream-niu") + 1] == format(
        case.upstream_niu, ".12g"
    )


def test_dry_run_does_not_overwrite_existing_collection_without_overwrite(
    tmp_path: Path,
) -> None:
    collection_path = tmp_path / "collection.json"
    original = {"real_collection": True}
    collection_path.write_text(json.dumps(original), encoding="utf-8")
    args = _arguments(tmp_path, "--dry-run")

    with pytest.raises(ValueError, match="dry-run refuses to overwrite"):
        _COLLECTION.run_collection(args)
    assert json.loads(collection_path.read_text(encoding="utf-8")) == original


def test_existing_archive_is_reused_only_after_exact_configuration_check(
    tmp_path: Path,
) -> None:
    args = _arguments(tmp_path)
    case = _COLLECTION.plan_cases(args)[0]
    archive = tmp_path / f"{case.case_id}.npz"
    metadata = {
        "case_id": case.case_id,
        "flow_group_id": case.flow_group_id,
        "grid_size_xyz": [args.grid_size] * 3,
        "warmup_steps": case.warmup_steps,
        "snapshot_iteration": case.warmup_steps,
        "seed": case.seed,
        "max_num_particles": args.num_particles,
        "num_observation_times": args.num_observation_times,
        "projection_names": args.projections.split(","),
        "integration_substeps": args.integration_substeps,
        "candidate_multiplier": args.candidate_multiplier,
        "particle_selection_mode": args.particle_selection,
        "upstream_niu_parameter": case.upstream_niu,
        "duration": args.duration,
        "particle_margin": args.particle_margin,
        "minimum_path_length": args.minimum_path_length,
        "lid_velocity_xyz": [0.0, 0.0, case.lid_speed],
    }
    np.savez_compressed(archive, metadata=np.asarray(json.dumps(metadata)))
    _COLLECTION._validate_existing_archive(archive, args, case)

    metadata["seed"] = case.seed + 1
    np.savez_compressed(archive, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="expected"):
        _COLLECTION._validate_existing_archive(archive, args, case)
