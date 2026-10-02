#!/usr/bin/env python
"""Prepare immutable, train-fitted aligned Tucker caches and validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from flow_observation.tensor_space import (  # noqa: E402
    atomic_write_new, build_tensor_space_artifact, save_tensor_space_artifact, sha256_file,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="4,8,12,16", help="Comma-separated spatial ranks; use 2 for tiny smoke data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hooi-iterations", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        ranks = [int(value) for value in args.ranks.split(",")]
    except ValueError:
        parser.error("--ranks must be comma-separated integers")
    if not ranks or len(set(ranks)) != len(ranks) or any(rank < 1 for rank in ranks):
        parser.error("--ranks must contain distinct positive integers")
    if args.hooi_iterations < 0:
        parser.error("--hooi-iterations must be non-negative")
    output_dir = args.output_dir.expanduser().resolve()
    targets = [output_dir / f"rank_{rank}.pt" for rank in ranks] + [output_dir / "report.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preparation outputs already exist; select a new output directory")
    report = {"manifest": str(args.manifest.expanduser().resolve()), "ranks": {},
              "rank_selection_split": "validation", "test_metrics_computed": False}
    for rank in ranks:
        started = time.perf_counter()
        print(f"Preparing spatial rank {rank} on {args.device}", flush=True)
        artifact = build_tensor_space_artifact(args.manifest, rank, device=args.device,
                                               hooi_iterations=args.hooi_iterations)
        path = output_dir / f"rank_{rank}.pt"
        save_tensor_space_artifact(artifact, path)
        rank_report = dict(artifact["reconstruction_report"])
        rank_report.update({"artifact": str(path), "artifact_sha256": sha256_file(path),
                            "elapsed_seconds": time.perf_counter() - started})
        report["ranks"][str(rank)] = rank_report
        print(f"Wrote {path}; validation mean relative L2 = "
              f"{rank_report['validation_mean_relative_l2_fluid']:.6g}", flush=True)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write_new(output_dir / "report.json", lambda handle: handle.write(encoded))
    print(f"Validation rank report: {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
