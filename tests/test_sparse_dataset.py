from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from flow_observation.sparse_dataset import (
    SparseFlowDataset,
    compute_train_flow_stats,
    denormalize_flow,
    derive_physical_flow_group_id,
    inspect_sparse_flow_case,
    normalize_flow,
    validate_sparse_flow_manifest,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_sparse_flow_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_sparse_flow_manifest", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BUILDER
_SPEC.loader.exec_module(_BUILDER)


def _write_case(
    path: Path,
    *,
    case_offset: float,
    num_particles: int = 8,
    case_id: str | None = None,
    upstream_niu: float | None = None,
) -> None:
    depth, height, width = 4, 5, 6
    num_views, num_times = 3, 4
    z, y, x = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    flow = np.stack(
        (
            case_offset + 0.1 * x,
            2.0 * case_offset + 0.1 * y,
            -case_offset + 0.1 * z,
        ),
        axis=0,
    )[None].astype(np.float32)
    tracks = np.zeros(
        (1, num_views, num_particles, num_times, 2), dtype=np.float32
    )
    for view in range(num_views):
        for particle in range(num_particles):
            tracks[0, view, particle, :, 0] = particle
            tracks[0, view, particle, :, 1] = view + np.arange(num_times) / 10.0
    projection = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    solid = np.zeros((depth, height, width), dtype=bool)
    solid[:, :, 0] = True
    metadata = {
        "case_id": case_id or path.stem,
        "field": "taichi_lbm3d_lid_driven_cavity",
        "grid_size_xyz": [width, height, depth],
        "upstream_niu_parameter": (
            upstream_niu
            if upstream_niu is not None
            else 0.1 + sum(path.stem.encode("utf-8")) * 1e-6
        ),
        "lid_face": "x=max",
        "lid_velocity_xyz": [0.0, 0.0, 0.05],
    }
    metadata["flow_group_id"] = derive_physical_flow_group_id(metadata)
    np.savez_compressed(
        path,
        flow_field=flow,
        trajectories_2d=tracks,
        projection_matrix=projection,
        observation_mask=np.ones(tracks.shape[:-1], dtype=bool),
        observation_times=np.linspace(0.0, 1.0, num_times, dtype=np.float32),
        domain_bounds=np.asarray(
            ((0.0, width - 1), (0.0, height - 1), (0.0, depth - 1)),
            dtype=np.float32,
        ),
        solid_mask=solid,
        particle_counts=np.asarray([num_particles], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata)),
        # Deliberately non-loadable with allow_pickle=False. The sparse dataset
        # must never touch hidden depth while constructing its conditions.
        initial_hidden_depth=np.asarray([[object()]], dtype=object),
    )


def _write_scientific_manifest(path: Path, cases: list[Path]) -> Path:
    assert len(cases) == 3
    entries = []
    for case in cases:
        metadata = inspect_sparse_flow_case(case)["metadata"]
        assert isinstance(metadata, dict)
        entries.append(
            {
                "case_id": metadata["case_id"],
                "flow_group_id": metadata["flow_group_id"],
                "path": case.name,
            }
        )
    manifest = {
        "format_version": 1,
        "mode": "scientific",
        "allow_overlap_for_smoke_test": False,
        "splits": {
            "train": [entries[0]],
            "validation": [entries[1]],
            "test": [entries[2]],
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _three_cases(tmp_path: Path) -> tuple[list[Path], Path]:
    cases = [tmp_path / f"case-{index}.npz" for index in range(3)]
    case_ids = ["train-case", "validation-case", "test-case"]
    for index, path in enumerate(cases):
        _write_case(path, case_offset=float(index + 1), case_id=case_ids[index])
    manifest = _write_scientific_manifest(tmp_path / "manifest.json", cases)
    return cases, manifest


def test_dataset_returns_exactly_two_physical_particles_without_hidden_depth(
    tmp_path: Path,
) -> None:
    _, manifest = _three_cases(tmp_path)
    dataset = SparseFlowDataset(manifest, "train", observations_per_flow=2, seed=11)
    item = dataset[0]

    assert len(dataset) == 2
    assert item["flow"].shape == (3, 4, 5, 6)
    assert item["tracks"].shape == (3, 2, 4, 2)
    assert item["mask"].shape == (3, 2, 4)
    assert item["projection"].shape == (3, 2, 3)
    assert item["particle_indices"].shape == (2,)
    assert torch.unique(item["particle_indices"]).numel() == 2
    assert item["case_id"] == "train-case"
    assert item["flow_group_id"].startswith("flow_")
    assert item["boundary_mask"].shape == (4, 5, 6)
    assert item["boundary_values"].shape == (3, 4, 5, 6)
    lid = item["boundary_mask"][:, :, -1] & ~item["solid_mask"][:, :, -1]
    torch.testing.assert_close(
        item["boundary_values"][2, :, :, -1][lid],
        torch.full_like(item["boundary_values"][2, :, :, -1][lid], 0.05),
    )


def test_pairing_and_particle_order_are_deterministic_per_epoch(tmp_path: Path) -> None:
    _, manifest = _three_cases(tmp_path)
    first = SparseFlowDataset(
        manifest, "train", observations_per_flow=3, seed=23
    )
    second = SparseFlowDataset(
        manifest, "train", observations_per_flow=3, seed=23
    )

    epoch_zero = [first[index]["particle_indices"].clone() for index in range(3)]
    for index in range(3):
        torch.testing.assert_close(
            first[index]["particle_indices"], second[index]["particle_indices"]
        )
        selected = first[index]
        # The first stored coordinate was tagged with its physical particle ID,
        # so this also checks that tracks and the ordered indices stay aligned.
        observed_ids = selected["tracks"][0, :, 0, 0].to(torch.long)
        torch.testing.assert_close(observed_ids, selected["particle_indices"])

    first.set_epoch(1)
    second.set_epoch(1)
    epoch_one = [first[index]["particle_indices"].clone() for index in range(3)]
    for left, right in zip(epoch_one, [second[i]["particle_indices"] for i in range(3)]):
        torch.testing.assert_close(left, right)
    assert any(not torch.equal(left, right) for left, right in zip(epoch_zero, epoch_one))


def test_observed_and_probe_particles_use_disjoint_ends_of_one_permutation(
    tmp_path: Path,
) -> None:
    _, manifest = _three_cases(tmp_path)
    dataset = SparseFlowDataset(
        manifest,
        "train",
        num_particles=2,
        num_probe_particles=3,
        observations_per_flow=1,
        seed=91,
    )
    item = dataset[0]
    observed = item["particle_indices"]
    probes = item["probe_particle_indices"]
    assert torch.is_tensor(observed) and torch.is_tensor(probes)
    assert probes.shape == (3,)
    assert not set(observed.tolist()) & set(probes.tolist())
    assert item["probe_tracks"].shape == (3, 3, 4, 2)
    assert item["probe_mask"].shape == (3, 3, 4)

    with pytest.raises(ValueError, match=r"observed \+ probe"):
        SparseFlowDataset(
            manifest, "train", num_particles=7, num_probe_particles=2
        )[0]


def test_manifest_rejects_case_id_and_path_leakage(tmp_path: Path) -> None:
    shared = {"case_id": "same", "flow_group_id": "flow-a", "path": "case.npz"}
    leaked_id = {
        "mode": "scientific",
        "splits": {
            "train": [shared],
            "validation": [{"case_id": "same", "flow_group_id": "flow-b", "path": "other.npz"}],
            "test": [{"case_id": "third", "flow_group_id": "flow-c", "path": "third.npz"}],
        },
    }
    with pytest.raises(ValueError, match="split leakage"):
        validate_sparse_flow_manifest(leaked_id, base_dir=tmp_path, require_files=False)

    leaked_path = {
        "mode": "scientific",
        "splits": {
            "train": [shared],
            "validation": [{"case_id": "different", "flow_group_id": "flow-b", "path": "case.npz"}],
            "test": [{"case_id": "third", "flow_group_id": "flow-c", "path": "third.npz"}],
        },
    }
    with pytest.raises(ValueError, match="split leakage"):
        validate_sparse_flow_manifest(
            leaked_path, base_dir=tmp_path, require_files=False
        )


def test_manifest_rejects_flow_group_leakage_but_allows_grouped_train_cases(
    tmp_path: Path,
) -> None:
    leaked_group = {
        "mode": "scientific",
        "splits": {
            "train": [
                {"case_id": "particles-a", "flow_group_id": "flow-a", "path": "a.npz"}
            ],
            "validation": [
                {"case_id": "particles-b", "flow_group_id": "flow-a", "path": "b.npz"}
            ],
            "test": [
                {"case_id": "particles-c", "flow_group_id": "flow-c", "path": "c.npz"}
            ],
        },
    }
    with pytest.raises(ValueError, match="flow groups"):
        validate_sparse_flow_manifest(
            leaked_group, base_dir=tmp_path, require_files=False
        )

    grouped_train = {
        "mode": "scientific",
        "splits": {
            "train": [
                {"case_id": "particles-a", "flow_group_id": "flow-a", "path": "a.npz"},
                {"case_id": "particles-b", "flow_group_id": "flow-a", "path": "b.npz"},
            ],
            "validation": [
                {"case_id": "particles-c", "flow_group_id": "flow-c", "path": "c.npz"}
            ],
            "test": [
                {"case_id": "particles-d", "flow_group_id": "flow-d", "path": "d.npz"}
            ],
        },
    }
    normalized = validate_sparse_flow_manifest(
        grouped_train, base_dir=tmp_path, require_files=False
    )
    assert [case.flow_group_id for case in normalized["train"]] == [
        "flow-a",
        "flow-a",
    ]


def test_train_only_component_stats_and_normalization_round_trip(tmp_path: Path) -> None:
    cases, manifest = _three_cases(tmp_path)
    # Make held-out flows extreme. Train statistics must remain unchanged.
    _write_case(cases[1], case_offset=1000.0, case_id="validation-case")
    _write_case(cases[2], case_offset=-1000.0, case_id="test-case")
    stats = compute_train_flow_stats(manifest)
    assert stats.mean.abs().max() < 10.0
    assert stats.std.shape == (3,)
    assert torch.all(stats.std > 0)

    flow = torch.randn((2, 3, 4, 5, 6), dtype=torch.float64)
    recovered = denormalize_flow(normalize_flow(flow, stats), stats)
    torch.testing.assert_close(recovered, flow, atol=1e-12, rtol=1e-12)


def test_single_case_requires_and_records_explicit_smoke_overlap(tmp_path: Path) -> None:
    case = tmp_path / "only-case.npz"
    output = tmp_path / "manifest.json"
    _write_case(case, case_offset=1.0)

    with pytest.raises(ValueError, match="at least three independent flow groups"):
        _BUILDER.build_manifest([case], output)
    manifest = _BUILDER.build_manifest(
        [case], output, allow_overlap_for_smoke_test=True
    )
    assert manifest["mode"] == "smoke_test"
    assert manifest["allow_overlap_for_smoke_test"] is True
    output.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = SparseFlowDataset(output, "test", num_particles=2)
    assert dataset.is_smoke_test
    assert dataset[0]["tracks"].shape[1] == 2


def test_builder_keeps_collection_flow_groups_in_one_split(tmp_path: Path) -> None:
    case_dir = tmp_path / "collection"
    case_dir.mkdir()
    cases = [case_dir / f"case-{index}.npz" for index in range(4)]
    nius = [0.12, 0.12, 0.15, 0.18]
    for index, (case, niu) in enumerate(zip(cases, nius, strict=True)):
        _write_case(case, case_offset=float(index + 1), upstream_niu=niu)
    group_ids = [
        inspect_sparse_flow_case(case)["metadata"]["flow_group_id"]  # type: ignore[index]
        for case in cases
    ]
    collection = {
        "format_version": 1,
        "cases": [
            {
                "case_id": case.stem,
                "flow_group_id": group_id,
                "output": str(case.resolve()),
            }
            for case, group_id in zip(cases, group_ids)
        ],
    }
    (case_dir / "collection.json").write_text(
        json.dumps(collection), encoding="utf-8"
    )

    manifest = _BUILDER.build_manifest(
        cases, tmp_path / "manifest.json", seed=7
    )
    assert manifest["mode"] == "scientific"
    assert manifest["split_unit"] == "flow_group_id"
    locations: dict[str, set[str]] = {}
    entries_by_group: dict[str, list[dict[str, str]]] = {}
    for split, entries in manifest["splits"].items():
        for entry in entries:
            locations.setdefault(entry["flow_group_id"], set()).add(split)
            entries_by_group.setdefault(entry["flow_group_id"], []).append(entry)
    assert all(len(splits) == 1 for splits in locations.values())
    assert len(entries_by_group[group_ids[0]]) == 2


def test_builder_without_collection_derives_each_physical_group_from_metadata(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "ordinary"
    case_dir.mkdir()
    cases = [case_dir / f"standalone-{index}.npz" for index in range(3)]
    for index, case in enumerate(cases):
        _write_case(case, case_offset=float(index + 1))

    manifest = _BUILDER.build_manifest(cases, tmp_path / "manifest.json")
    entries = [
        entry
        for split_entries in manifest["splits"].values()
        for entry in split_entries
    ]
    assert len({entry["flow_group_id"] for entry in entries}) == 3
    expected = {
        inspect_sparse_flow_case(case)["metadata"]["flow_group_id"]  # type: ignore[index]
        for case in cases
    }
    assert {entry["flow_group_id"] for entry in entries} == expected


def _rewrite_case_array(path: Path, key: str, value: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            name: np.array(archive[name], copy=True)
            for name in (
                "flow_field",
                "trajectories_2d",
                "projection_matrix",
                "observation_mask",
                "observation_times",
                "domain_bounds",
                "solid_mask",
                "particle_counts",
                "metadata",
            )
        }
    payload[key] = value
    np.savez_compressed(path, **payload)


@pytest.mark.parametrize(
    ("key", "replacement", "message"),
    [
        (
            "observation_times",
            np.asarray([0.0, 0.5, 0.5, 1.0], dtype=np.float32),
            "strictly increasing",
        ),
        (
            "particle_counts",
            np.asarray([8.0], dtype=np.float32),
            "must contain integers",
        ),
        (
            "solid_mask",
            np.full((4, 5, 6), 2, dtype=np.int8),
            "must contain only 0/1",
        ),
        (
            "observation_mask",
            np.zeros((1, 3, 8, 4), dtype=bool),
            "entirely unobserved",
        ),
    ],
)
def test_case_validation_rejects_corrupt_integrity_fields(
    tmp_path: Path, key: str, replacement: np.ndarray, message: str
) -> None:
    case = tmp_path / "corrupt.npz"
    _write_case(case, case_offset=1.0)
    _rewrite_case_array(case, key, replacement)
    with pytest.raises(ValueError, match=message):
        inspect_sparse_flow_case(case)


def test_builder_rejects_incompatible_case_observation_geometry(tmp_path: Path) -> None:
    cases = [tmp_path / f"geometry-{index}.npz" for index in range(3)]
    for index, case in enumerate(cases):
        _write_case(case, case_offset=float(index + 1))
    _rewrite_case_array(
        cases[1],
        "observation_times",
        np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="incompatible observation_times"):
        _BUILDER.build_manifest(cases, tmp_path / "manifest.json")


def test_builder_cross_checks_collection_npz_and_derived_group(tmp_path: Path) -> None:
    case_dir = tmp_path / "mismatch"
    case_dir.mkdir()
    cases = [case_dir / f"case-{index}.npz" for index in range(3)]
    for index, case in enumerate(cases):
        _write_case(case, case_offset=float(index + 1))
    collection = {
        "cases": [
            {
                "case_id": case.stem,
                "flow_group_id": (
                    "flow_deliberately_wrong"
                    if index == 0
                    else inspect_sparse_flow_case(case)["metadata"][  # type: ignore[index]
                        "flow_group_id"
                    ]
                ),
                "output": str(case),
            }
            for index, case in enumerate(cases)
        ]
    }
    (case_dir / "collection.json").write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees between collection"):
        _BUILDER.build_manifest(cases, tmp_path / "manifest.json")


def test_missing_physical_identity_requires_explicit_smoke_mode(tmp_path: Path) -> None:
    case = tmp_path / "legacy.npz"
    _write_case(case, case_offset=1.0)
    with np.load(case, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
    for key in ("field", "grid_size_xyz", "upstream_niu_parameter", "flow_group_id"):
        metadata.pop(key, None)
    _rewrite_case_array(case, "metadata", np.asarray(json.dumps(metadata)))

    with pytest.raises(ValueError, match="lacks complete physical metadata"):
        _BUILDER.build_manifest([case], tmp_path / "manifest.json")
    manifest = _BUILDER.build_manifest(
        [case],
        tmp_path / "manifest.json",
        allow_overlap_for_smoke_test=True,
    )
    assert manifest["mode"] == "smoke_test"


def test_scientific_manifest_requires_explicit_flow_group() -> None:
    manifest = {
        "mode": "scientific",
        "splits": {
            "train": [{"case_id": "a", "path": "a.npz"}],
            "validation": [{"case_id": "b", "path": "b.npz"}],
            "test": [{"case_id": "c", "path": "c.npz"}],
        },
    }
    with pytest.raises(ValueError, match="must explicitly provide flow_group_id"):
        validate_sparse_flow_manifest(manifest, require_files=False)
