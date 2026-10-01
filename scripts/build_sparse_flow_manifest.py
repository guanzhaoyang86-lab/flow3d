#!/usr/bin/env python
"""Build case-disjoint train/validation/test manifests for sparse flow data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.sparse_dataset import (  # noqa: E402
    derive_physical_flow_group_id,
    inspect_sparse_flow_case,
    validate_sparse_flow_manifest,
)


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError("fraction must lie strictly between 0 and 1")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a physical-flow-group sparse-flow manifest. Scientific "
            "manifests require at least three independent flow_group_id values "
            "and never overlap groups across splits."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        help="One or more case NPZ files. The flag is repeatable.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        help="One or more directories containing case NPZ files. Repeatable.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob used inside each --input-dir (default: *.npz).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --input-dir recursively.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=_fraction, default=0.8)
    parser.add_argument("--validation-fraction", type=_fraction, default=0.1)
    parser.add_argument(
        "--allow-overlap-for-smoke-test",
        action="store_true",
        help=(
            "Allow fewer than three flow groups by explicitly reusing groups across "
            "splits. The resulting manifest is marked smoke_test and is not a "
            "valid scientific evaluation."
        ),
    )
    return parser


def discover_case_paths(
    inputs: Sequence[Path],
    input_dirs: Sequence[Path],
    *,
    pattern: str = "*.npz",
    recursive: bool = False,
) -> list[Path]:
    """Resolve explicit files and directory matches without duplicates."""

    candidates = [Path(path).expanduser() for path in inputs]
    for directory in input_dirs:
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {root}")
        matches = root.rglob(pattern) if recursive else root.glob(pattern)
        candidates.extend(sorted(matches))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input case does not exist: {path}")
        if path.suffix.lower() != ".npz":
            raise ValueError(f"input case must be an NPZ archive: {path}")
        if path not in seen:
            seen.add(path)
            resolved.append(path)
    if not resolved:
        raise ValueError("provide at least one --input or --input-dir case")
    return resolved


def _split_counts(
    group_count: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0.0:
        raise ValueError("train and validation fractions must sum to less than 1")
    if group_count < 3:
        raise ValueError(
            "scientific mode requires at least three independent flow groups"
        )
    weights = (train_fraction, validation_fraction, test_fraction)
    remaining = group_count - 3
    exact = [remaining * weight for weight in weights]
    extras = [int(value) for value in exact]
    leftover = remaining - sum(extras)
    order = sorted(
        range(3), key=lambda index: exact[index] - extras[index], reverse=True
    )
    for index in order[:leftover]:
        extras[index] += 1
    return tuple(1 + value for value in extras)  # type: ignore[return-value]


def _collection_records(paths: Sequence[Path]) -> dict[Path, dict[str, Any]]:
    """Map requested archives to their complete collection records."""

    requested = set(paths)
    mapping: dict[Path, dict[str, Any]] = {}
    for directory in sorted({path.parent for path in paths}, key=str):
        collection_path = directory / "collection.json"
        if not collection_path.is_file():
            continue
        with collection_path.open("r", encoding="utf-8") as handle:
            collection = json.load(handle)
        if not isinstance(collection, dict) or not isinstance(
            collection.get("cases"), list
        ):
            raise ValueError(f"invalid collection metadata: {collection_path}")
        for record in collection["cases"]:
            if not isinstance(record, dict):
                raise ValueError(
                    f"collection cases must be objects: {collection_path}"
                )
            raw_output = record.get("output")
            group_id = record.get("flow_group_id")
            case_id = record.get("case_id")
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise ValueError(
                    f"collection case output must be a path: {collection_path}"
                )
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    f"collection case flow_group_id is missing: {collection_path}"
                )
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(
                    f"collection case case_id is missing: {collection_path}"
                )
            recorded = Path(raw_output).expanduser()
            if not recorded.is_absolute():
                recorded = collection_path.parent / recorded
            # The sibling candidate keeps the grouping valid if a complete
            # collection directory was moved after generation while filenames
            # and collection.json stayed together.
            candidates = {
                recorded.resolve(),
                (collection_path.parent / recorded.name).resolve(),
            }
            for candidate in candidates & requested:
                previous = mapping.get(candidate)
                normalized_record = dict(record)
                normalized_record["case_id"] = case_id.strip()
                normalized_record["flow_group_id"] = group_id.strip()
                if previous is not None and (
                    previous["case_id"] != normalized_record["case_id"]
                    or previous["flow_group_id"] != normalized_record["flow_group_id"]
                ):
                    raise ValueError(
                        f"conflicting collection identities for {candidate}"
                    )
                mapping[candidate] = normalized_record
    return mapping


def _compatible_array(
    left: object, right: object, *, name: str, left_path: Path, right_path: Path
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.allclose(
        left_array, right_array, rtol=0.0, atol=1e-7
    ):
        raise ValueError(
            f"case {right_path} has incompatible {name} relative to {left_path}"
        )


def _validate_case_compatibility(
    paths: Sequence[Path], inspections: Sequence[dict[str, object]]
) -> None:
    reference_path = paths[0]
    reference = inspections[0]
    reference_flow = tuple(reference["flow_shape"])  # type: ignore[arg-type]
    reference_tracks = tuple(reference["tracks_shape"])  # type: ignore[arg-type]
    for path, inspection in zip(paths[1:], inspections[1:], strict=True):
        flow_shape = tuple(inspection["flow_shape"])  # type: ignore[arg-type]
        tracks_shape = tuple(inspection["tracks_shape"])  # type: ignore[arg-type]
        if flow_shape != reference_flow:
            raise ValueError(
                f"case {path} flow shape {flow_shape} is incompatible with "
                f"{reference_path} shape {reference_flow}"
            )
        if tracks_shape[1] != reference_tracks[1] or tracks_shape[3] != reference_tracks[3]:
            raise ValueError(
                f"case {path} view/time dimensions are incompatible with {reference_path}"
            )
        for name in ("projection_matrix", "observation_times", "domain_bounds"):
            _compatible_array(
                reference[name],
                inspection[name],
                name=name,
                left_path=reference_path,
                right_path=path,
            )


def _resolve_identity(
    path: Path,
    metadata: dict[str, Any],
    collection: dict[str, Any] | None,
    *,
    allow_unsafe_smoke: bool,
) -> tuple[str, str, bool]:
    """Resolve and cross-check collection, archive, and derived identities."""

    metadata_case = metadata.get("case_id")
    if metadata_case is not None and (
        not isinstance(metadata_case, str) or not metadata_case.strip()
    ):
        raise ValueError(f"case {path} has an invalid metadata case_id")
    collection_case = collection.get("case_id") if collection else None
    if metadata_case is not None and collection_case is not None:
        if metadata_case.strip() != collection_case:
            raise ValueError(
                f"case {path} case_id disagrees between collection and NPZ metadata"
            )
    unsafe = False
    case_id = (
        collection_case
        or (metadata_case.strip() if isinstance(metadata_case, str) else None)
    )
    if case_id is None:
        if not allow_unsafe_smoke:
            raise ValueError(
                f"case {path} has no case_id identity; only explicit smoke mode may "
                "fall back to a filename"
            )
        case_id = path.stem
        unsafe = True

    metadata_group = metadata.get("flow_group_id")
    if metadata_group is not None and (
        not isinstance(metadata_group, str) or not metadata_group.strip()
    ):
        raise ValueError(f"case {path} has an invalid metadata flow_group_id")
    collection_group = collection.get("flow_group_id") if collection else None
    try:
        derived_group = derive_physical_flow_group_id(metadata)
    except (TypeError, ValueError) as error:
        if not allow_unsafe_smoke:
            raise ValueError(
                f"case {path} lacks complete physical metadata for a safe "
                f"flow_group_id: {error}"
            ) from error
        derived_group = None
        unsafe = True

    candidates = [
        value
        for value in (
            collection_group,
            metadata_group.strip() if isinstance(metadata_group, str) else None,
            derived_group,
        )
        if value is not None
    ]
    if len(set(candidates)) > 1:
        raise ValueError(
            f"case {path} flow_group_id disagrees between collection, NPZ metadata, "
            "and physical metadata"
        )
    if candidates:
        group_id = candidates[0]
    elif allow_unsafe_smoke:
        group_id = case_id
        unsafe = True
    else:  # pragma: no cover - guarded by derived metadata failure above
        raise ValueError(f"case {path} has no flow identity")
    return str(case_id), str(group_id), unsafe


def build_manifest(
    case_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    seed: int = 0,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    allow_overlap_for_smoke_test: bool = False,
) -> dict[str, object]:
    """Build and validate a manifest without writing it."""

    paths = [Path(path).expanduser().resolve() for path in case_paths]
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")
    if len(set(paths)) != len(paths):
        raise ValueError("case_paths contains duplicate files")
    if not paths:
        raise ValueError("at least one case is required")
    inspections = [inspect_sparse_flow_case(path) for path in paths]
    _validate_case_compatibility(paths, inspections)
    metadata = [inspection["metadata"] for inspection in inspections]
    assert all(isinstance(value, dict) for value in metadata)
    collection_records = _collection_records(paths)
    case_ids: list[str] = []
    flow_group_ids: list[str] = []
    unsafe_identity = False
    for path, values in zip(paths, metadata, strict=True):
        case_id, group_id, unsafe = _resolve_identity(
            path,
            values,  # type: ignore[arg-type]
            collection_records.get(path),
            allow_unsafe_smoke=allow_overlap_for_smoke_test,
        )
        case_ids.append(case_id)
        flow_group_ids.append(group_id)
        unsafe_identity |= unsafe
    if len(set(case_ids)) != len(case_ids):
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        raise ValueError("duplicate case_id identities: " + ", ".join(duplicates))

    output = Path(output_path).expanduser().resolve()
    entries_by_group: dict[str, list[dict[str, str]]] = {}
    for index in range(len(paths)):
        relative = Path(os.path.relpath(paths[index], output.parent)).as_posix()
        group_id = flow_group_ids[index]
        entries_by_group.setdefault(group_id, []).append(
            {
                "case_id": case_ids[index],
                "flow_group_id": group_id,
                "path": relative,
            }
        )
    group_order = list(entries_by_group)
    random.Random(int(seed)).shuffle(group_order)

    if unsafe_identity:
        if not allow_overlap_for_smoke_test:  # pragma: no cover - resolved above
            raise ValueError("unsafe identities require explicit smoke mode")
        all_entries = [
            entry for group in group_order for entry in entries_by_group[group]
        ]
        train = list(all_entries)
        validation = list(all_entries)
        test = list(all_entries)
        mode = "smoke_test"
        allow_overlap = True
    elif len(group_order) < 3:
        if not allow_overlap_for_smoke_test:
            raise ValueError(
                "scientific mode requires at least three independent flow groups; "
                "use --allow-overlap-for-smoke-test only for an integration smoke test"
            )
        train = list(entries_by_group[group_order[0]])
        validation = list(
            entries_by_group[
                group_order[1] if len(group_order) > 1 else group_order[0]
            ]
        )
        test = list(entries_by_group[group_order[0]])
        mode = "smoke_test"
        allow_overlap = True
    else:
        train_count, validation_count, _ = _split_counts(
            len(group_order), train_fraction, validation_fraction
        )
        train_groups = group_order[:train_count]
        validation_groups = group_order[
            train_count : train_count + validation_count
        ]
        test_groups = group_order[train_count + validation_count :]
        train = [entry for group in train_groups for entry in entries_by_group[group]]
        validation = [
            entry for group in validation_groups for entry in entries_by_group[group]
        ]
        test = [entry for group in test_groups for entry in entries_by_group[group]]
        mode = "scientific"
        allow_overlap = False

    manifest: dict[str, object] = {
        "format_version": 1,
        "dataset_kind": "sparse_projected_particle_flow",
        "mode": mode,
        "allow_overlap_for_smoke_test": allow_overlap,
        "split_unit": "flow_group_id",
        "seed": int(seed),
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
    }
    validate_sparse_flow_manifest(
        manifest, base_dir=output.parent, require_files=True
    )
    return manifest


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        paths = discover_case_paths(
            args.input,
            args.input_dir,
            pattern=args.pattern,
            recursive=args.recursive,
        )
        manifest = build_manifest(
            paths,
            args.output,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            allow_overlap_for_smoke_test=args.allow_overlap_for_smoke_test,
        )
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    counts = {
        name: len(manifest["splits"][name])  # type: ignore[index]
        for name in ("train", "validation", "test")
    }
    print(f"Wrote {output}")
    print(f"  mode: {manifest['mode']}")
    print(
        "  cases: "
        f"train={counts['train']}, validation={counts['validation']}, "
        f"test={counts['test']}"
    )


if __name__ == "__main__":
    main()
