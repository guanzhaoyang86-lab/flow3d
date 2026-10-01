#!/usr/bin/env python
"""Plan or generate a collection of independent Taichi-LBM3D cavity cases.

This script deliberately drives the existing, validated single-case generator
in subprocesses.  Each command writes one NPZ archive, which keeps the current
``flow_field [T,3,D,H,W]`` convention intact and isolates Taichi runtime state
between cases.

The varied physical inputs are lid speed, the value passed to upstream
``set_viscosity()``, and the snapshot warm-up iteration.  The lid direction
remains ``+z`` on the ``x=max`` face and the upstream initial state remains the
solver's fixed zero-velocity initialization.  The seed controls particle
sampling; it does not make a new flow when all physical parameters are equal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_GENERATOR = _REPOSITORY_ROOT / "scripts" / "generate_taichi_lbm3d_cavity_dataset.py"


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _csv_floats(value: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a comma-separated float list") from error
    if not parsed or any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("all listed values must be finite and positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("listed values must not repeat")
    return parsed


def _csv_positive_integers(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a comma-separated integer list") from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all listed values must be positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("listed values must not repeat")
    return parsed


def _csv_integers(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a comma-separated integer list") from error
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one integer")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("listed values must not repeat")
    return parsed


def _float_range(values: list[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range requires exactly two values")
    lower, upper = (float(value) for value in values)
    if not all(math.isfinite(value) and value > 0.0 for value in (lower, upper)):
        raise argparse.ArgumentTypeError("range values must be finite and positive")
    if upper < lower:
        raise argparse.ArgumentTypeError("range upper bound must be >= lower bound")
    return lower, upper


def _integer_range(values: list[str], *, positive: bool) -> tuple[int, int]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range requires exactly two values")
    lower, upper = (int(value) for value in values)
    minimum = 1 if positive else 0
    if lower < minimum or upper < lower:
        qualifier = "positive " if positive else "non-negative "
        raise argparse.ArgumentTypeError(
            f"range must contain increasing {qualifier}integers"
        )
    return lower, upper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate separate Taichi-LBM3D cavity archives for conditional "
            "diffusion training, or only write the exact command plan."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling", choices=("grid", "random"), default="grid")
    parser.add_argument("--num-cases", type=_positive_integer, default=32)
    parser.add_argument(
        "--lid-speeds",
        type=_csv_floats,
        default=_csv_floats("0.03,0.04,0.05,0.06"),
        help="Grid-mode +z lid speeds.",
    )
    parser.add_argument(
        "--upstream-nius",
        type=_csv_floats,
        default=_csv_floats("0.12,0.15,0.18,0.21"),
        help="Grid-mode values passed unchanged to upstream set_viscosity().",
    )
    parser.add_argument(
        "--warmup-steps",
        type=_csv_positive_integers,
        default=_csv_positive_integers("600,800,1000,1200"),
        help="Grid-mode snapshot iterations.",
    )
    parser.add_argument(
        "--seeds",
        type=_csv_integers,
        default=_csv_integers("7,17,27,37"),
        help="Grid-mode particle-sampling seeds.",
    )
    parser.add_argument(
        "--lid-speed-range",
        nargs=2,
        default=(0.03, 0.06),
        metavar=("MIN", "MAX"),
        help="Random-mode positive range.",
    )
    parser.add_argument(
        "--upstream-niu-range",
        nargs=2,
        default=(0.12, 0.21),
        metavar=("MIN", "MAX"),
        help="Random-mode positive range.",
    )
    parser.add_argument(
        "--warmup-step-range",
        nargs=2,
        default=(600, 1200),
        metavar=("MIN", "MAX"),
        help="Random-mode inclusive integer range.",
    )
    parser.add_argument(
        "--seed-range",
        nargs=2,
        default=(0, 2_147_483_647),
        metavar=("MIN", "MAX"),
        help="Random-mode inclusive particle-seed range.",
    )
    parser.add_argument("--planner-seed", type=int, default=20260917)

    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--generator", type=Path, default=_DEFAULT_GENERATOR)
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=_REPOSITORY_ROOT / "third_party" / "taichi_LBM3D",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--grid-size", type=_positive_integer, default=32)
    parser.add_argument("--num-particles", type=_positive_integer, default=2)
    parser.add_argument("--num-observation-times", type=_positive_integer, default=20)
    parser.add_argument("--duration", type=_positive_float, default=120.0)
    parser.add_argument("--projections", default="xz,xy,yz")
    parser.add_argument("--integration-substeps", type=_positive_integer, default=4)
    parser.add_argument("--particle-margin", type=_nonnegative_float, default=2.0)
    parser.add_argument("--candidate-multiplier", type=_positive_integer, default=32)
    parser.add_argument("--minimum-path-length", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--particle-selection",
        choices=("random", "diverse"),
        default="random",
        help="Random is the unbiased default for extreme sparse-particle studies.",
    )
    parser.add_argument("--progress-interval", type=_positive_integer, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    flow_group_id: str
    lid_speed: float
    upstream_niu: float
    warmup_steps: int
    seed: int


def _canonical_command_float(value: float) -> str:
    number = float(format(float(value), ".12g"))
    if not math.isfinite(number):
        raise ValueError("physical command values must be finite")
    return format(number, ".12g")


def _flow_group_id(
    lid_speed: float, upstream_niu: float, grid_size: int = 32
) -> str:
    # Warm-up/snapshot iteration and particle seed are intentionally excluded:
    # they select a snapshot/observation from the same deterministic run.
    encoded = json.dumps(
        {
            "field": "taichi_lbm3d_lid_driven_cavity",
            "grid_size_xyz": [int(grid_size)] * 3,
            "lid_face": "x=max",
            "lid_velocity_xyz": ["0", "0", _canonical_command_float(lid_speed)],
            "upstream_niu_parameter": _canonical_command_float(upstream_niu),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "flow_" + hashlib.sha256(encoded).hexdigest()[:12]


def _validated_random_ranges(args: argparse.Namespace) -> tuple[
    tuple[float, float], tuple[float, float], tuple[int, int], tuple[int, int]
]:
    lid_range = _float_range([str(value) for value in args.lid_speed_range])
    niu_range = _float_range([str(value) for value in args.upstream_niu_range])
    warmup_range = _integer_range(
        [str(value) for value in args.warmup_step_range], positive=True
    )
    seed_range = _integer_range(
        [str(value) for value in args.seed_range], positive=False
    )
    if seed_range[1] - seed_range[0] + 1 < args.num_cases:
        raise ValueError("--seed-range contains fewer unique seeds than --num-cases")
    return lid_range, niu_range, warmup_range, seed_range


def plan_cases(args: argparse.Namespace) -> list[CaseSpec]:
    """Create deterministic case specifications from parsed CLI arguments."""

    raw: list[tuple[float, float, int, int]]
    if args.sampling == "grid":
        combinations = list(
            itertools.product(
                args.lid_speeds,
                args.upstream_nius,
                args.warmup_steps,
                args.seeds,
            )
        )
        if args.num_cases > len(combinations):
            raise ValueError(
                f"grid contains {len(combinations)} combinations, fewer than "
                f"--num-cases={args.num_cases}"
            )
        # A seeded shuffle avoids always taking only the smallest physical
        # values when a subset of a larger grid is requested.
        generator = random.Random(args.planner_seed)
        generator.shuffle(combinations)
        raw = combinations[: args.num_cases]
    else:
        lid_range, niu_range, warmup_range, seed_range = _validated_random_ranges(args)
        generator = random.Random(args.planner_seed)
        seeds = generator.sample(
            range(seed_range[0], seed_range[1] + 1), args.num_cases
        )
        raw = [
            (
                generator.uniform(*lid_range),
                generator.uniform(*niu_range),
                generator.randint(*warmup_range),
                seeds[index],
            )
            for index in range(args.num_cases)
        ]

    result = []
    for index, (lid_speed, niu, warmup, seed) in enumerate(raw):
        command_lid_speed = float(_canonical_command_float(lid_speed))
        command_niu = float(_canonical_command_float(niu))
        result.append(
            CaseSpec(
                case_id=f"case_{index:05d}",
                flow_group_id=_flow_group_id(
                    command_lid_speed, command_niu, args.grid_size
                ),
                lid_speed=command_lid_speed,
                upstream_niu=command_niu,
                warmup_steps=int(warmup),
                seed=int(seed),
            )
        )
    return result


def build_generator_command(
    args: argparse.Namespace, case: CaseSpec, output_path: Path
) -> list[str]:
    """Build the exact single-case generator invocation for one case."""

    return [
        str(args.python),
        str(args.generator),
        "--upstream-repo",
        str(args.upstream_repo),
        "--backend",
        args.backend,
        "--grid-size",
        str(args.grid_size),
        "--warmup-steps",
        str(case.warmup_steps),
        "--lid-speed",
        format(case.lid_speed, ".12g"),
        "--upstream-niu",
        format(case.upstream_niu, ".12g"),
        "--num-particles",
        str(args.num_particles),
        "--num-observation-times",
        str(args.num_observation_times),
        "--duration",
        format(args.duration, ".12g"),
        "--projections",
        args.projections,
        "--integration-substeps",
        str(args.integration_substeps),
        "--particle-margin",
        format(args.particle_margin, ".12g"),
        "--candidate-multiplier",
        str(args.candidate_multiplier),
        "--minimum-path-length",
        format(args.minimum_path_length, ".12g"),
        "--particle-selection",
        args.particle_selection,
        "--seed",
        str(case.seed),
        "--case-id",
        case.case_id,
        "--flow-group-id",
        case.flow_group_id,
        "--progress-interval",
        str(args.progress_interval),
        "--output",
        str(output_path),
    ]


def _read_archive_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata" not in archive.files:
                raise ValueError(f"existing archive has no metadata: {path}")
            raw = np.asarray(archive["metadata"])
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot validate existing archive {path}: {error}") from error
    if raw.shape != ():
        raise ValueError(f"existing archive metadata is not scalar JSON: {path}")
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"existing archive metadata is not a JSON string: {path}")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"existing archive metadata is not an object: {path}")
    return decoded


def _validate_existing_archive(
    path: Path, args: argparse.Namespace, case: CaseSpec
) -> None:
    """Refuse to reuse an archive whose recorded command differs from the plan."""

    metadata = _read_archive_metadata(path)
    expected_exact: dict[str, object] = {
        "case_id": case.case_id,
        "flow_group_id": case.flow_group_id,
        "grid_size_xyz": [args.grid_size] * 3,
        "warmup_steps": case.warmup_steps,
        "snapshot_iteration": case.warmup_steps,
        "seed": case.seed,
        "max_num_particles": args.num_particles,
        "num_observation_times": args.num_observation_times,
        "projection_names": [
            name.strip() for name in args.projections.split(",") if name.strip()
        ],
        "integration_substeps": args.integration_substeps,
        "candidate_multiplier": args.candidate_multiplier,
        "particle_selection_mode": args.particle_selection,
    }
    for key, expected in expected_exact.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"existing archive {path} has {key}={metadata.get(key)!r}, "
                f"expected {expected!r}; use --overwrite to regenerate it"
            )
    expected_floats = {
        "upstream_niu_parameter": case.upstream_niu,
        "duration": args.duration,
        "particle_margin": args.particle_margin,
        "minimum_path_length": args.minimum_path_length,
    }
    for key, expected in expected_floats.items():
        recorded = metadata.get(key)
        try:
            matches = _canonical_command_float(
                float(recorded)
            ) == _canonical_command_float(expected)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                f"existing archive {path} has {key}={recorded!r}, expected "
                f"{_canonical_command_float(expected)!r}; use --overwrite to regenerate it"
            )
    lid_velocity = metadata.get("lid_velocity_xyz")
    if (
        not isinstance(lid_velocity, list)
        or len(lid_velocity) != 3
        or [_canonical_command_float(float(value)) for value in lid_velocity]
        != ["0", "0", _canonical_command_float(case.lid_speed)]
    ):
        raise ValueError(
            f"existing archive {path} has incompatible lid_velocity_xyz; "
            "use --overwrite to regenerate it"
        )


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_command_log(path: Path, case_records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in case_records:
            stream.write(
                json.dumps(
                    {
                        "case_id": record["case_id"],
                        "flow_group_id": record["flow_group_id"],
                        "output": record["output"],
                        "command": record["command"],
                        "command_line": record["command_line"],
                        "status": record["status"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    """Write a reproducible plan and optionally execute every case command."""

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = plan_cases(args)
    if args.dry_run and not args.overwrite:
        protected = [
            output_dir / "collection.json",
            output_dir / "collection.partial.json",
            output_dir / "commands.jsonl",
            *(output_dir / f"{case.case_id}.npz" for case in cases),
        ]
        existing = [path for path in protected if path.exists()]
        if existing:
            raise ValueError(
                "dry-run refuses to overwrite an existing collection or archive; "
                "choose an empty --output-dir or explicitly pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    case_records: list[dict[str, Any]] = []

    for case in cases:
        output_path = output_dir / f"{case.case_id}.npz"
        command = build_generator_command(args, case, output_path)
        if args.dry_run:
            status = "planned"
        elif output_path.exists() and not args.overwrite:
            _validate_existing_archive(output_path, args, case)
            status = "skipped_existing"
        else:
            subprocess.run(command, cwd=_REPOSITORY_ROOT, check=True)
            status = "generated"
        case_records.append(
            {
                **asdict(case),
                "output": str(output_path),
                "command": command,
                "command_line": subprocess.list2cmdline(command),
                "status": status,
            }
        )

        # Keep a useful partial manifest if a later subprocess fails.
        partial = {
            "format_version": 1,
            "complete": len(case_records) == len(cases),
            "dry_run": bool(args.dry_run),
            "cases": case_records,
        }
        _write_json(output_dir / "collection.partial.json", partial)

    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "dry_run": bool(args.dry_run),
        "sampling": args.sampling,
        "planner_seed": args.planner_seed,
        "num_cases": len(case_records),
        "single_case_generator": str(Path(args.generator).expanduser().resolve()),
        "case_archive_contract": "one independent generator invocation per NPZ",
        "split_policy": (
            "flow_group_id is the indivisible train/validation/test atom; never "
            "place observations sharing a flow_group_id in different splits"
        ),
        "fixed_solver_facts": {
            "geometry": "lid-driven cavity",
            "lid_face": "x=max",
            "lid_direction": "+z (not varied)",
            "initial_state": "upstream fixed zero-velocity initialization (not varied)",
            "frozen_snapshot": True,
        },
        "parameter_notes": {
            "upstream_niu": (
                "passed unchanged to upstream set_viscosity(); no independently "
                "validated Reynolds-number interpretation is claimed"
            ),
            "warmup_steps": (
                "snapshot iteration only; excluded from flow_group_id because multiple "
                "snapshots belong to one physical simulation run"
            ),
            "seed": (
                "controls particle sampling only; equal physical parameters produce "
                "the same deterministic flow and therefore share flow_group_id"
            ),
        },
        "fixed_observation_arguments": {
            "grid_size": args.grid_size,
            "num_particles": args.num_particles,
            "num_observation_times": args.num_observation_times,
            "duration": args.duration,
            "projections": args.projections,
            "integration_substeps": args.integration_substeps,
            "particle_margin": args.particle_margin,
            "candidate_multiplier": args.candidate_multiplier,
            "minimum_path_length": args.minimum_path_length,
            "particle_selection": args.particle_selection,
        },
        "cases": case_records,
    }
    _write_json(output_dir / "collection.json", metadata)
    _write_command_log(output_dir / "commands.jsonl", case_records)
    partial_path = output_dir / "collection.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    return metadata


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        metadata = run_collection(args)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        parser.error(str(error))
    action = "Planned" if args.dry_run else "Processed"
    print(f"{action} {metadata['num_cases']} cases in {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'collection.json'}")
    print(f"Commands: {args.output_dir / 'commands.jsonl'}")


if __name__ == "__main__":
    main()
