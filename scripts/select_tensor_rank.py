#!/usr/bin/env python
"""Select the smallest Tucker rank meeting fixed validation error limits.

This standard-library-only helper reads preparation reports without loading
tensors or touching a GPU. It does not verify artifact bytes; the HPC runner
pins and verifies the successful preparation record, report and artifact hashes.
The limits are pilot engineering gates, not claims of scientific accuracy.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _validation_only(report: Mapping, name: str) -> None:
    if report.get("rank_selection_split") != "validation":
        raise ValueError(f"{name} must use the validation split for rank selection")
    if report.get("test_metrics_computed") is not False:
        raise ValueError(f"{name} must explicitly exclude test metrics")


def select_rank(report: Mapping, *, mean_limit: float = 0.05,
                max_limit: float = 0.20) -> dict:
    """Return deterministic, JSON-serializable selection evidence.

    Every candidate must be a valid preparation report. Invalid or non-finite
    data fail closed, even when another rank would pass. If no candidate meets
    both limits, raise ValueError instead of silently relaxing either limit.
    """
    mean_limit = _number(mean_limit, "mean_limit", positive=True)
    max_limit = _number(max_limit, "max_limit", positive=True)
    if mean_limit > max_limit:
        raise ValueError("mean_limit must not exceed max_limit")
    if not isinstance(report, Mapping):
        raise ValueError("rank report must be an object")
    _validation_only(report, "rank report")
    entries = report.get("ranks")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("rank report must contain non-empty ranks")

    candidates = []
    for key, entry in entries.items():
        if not isinstance(key, str) or re.fullmatch(r"[1-9]\d*", key) is None:
            raise ValueError("rank keys must be canonical positive integers")
        rank = int(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"rank {rank} report must be an object")
        if type(entry.get("rank")) is not int or entry["rank"] != rank:
            raise ValueError(f"rank {rank} disagrees with its report")
        _validation_only(entry, f"rank {rank}")
        mean = _number(entry.get("validation_mean_relative_l2_fluid"), f"rank {rank} validation mean")
        maximum = _number(entry.get("validation_max_relative_l2_fluid"), f"rank {rank} validation maximum")
        if mean > maximum and not math.isclose(mean, maximum, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"rank {rank} validation mean exceeds its maximum")
        path, sha256 = entry.get("artifact"), entry.get("artifact_sha256")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"rank {rank} has no artifact path")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(f"rank {rank} has an invalid artifact SHA256")
        candidates.append({
            "rank": rank,
            "validation_mean_relative_l2_fluid": mean,
            "validation_max_relative_l2_fluid": maximum,
            "passed": mean <= mean_limit and maximum <= max_limit,
            "artifact": {"path": path, "sha256": sha256},
        })
    candidates.sort(key=lambda candidate: candidate["rank"])
    passing = [candidate for candidate in candidates if candidate["passed"]]
    if not passing:
        errors = ", ".join(
            f"r={candidate['rank']}: mean={candidate['validation_mean_relative_l2_fluid']:.6g}, "
            f"max={candidate['validation_max_relative_l2_fluid']:.6g}"
            for candidate in candidates
        )
        raise ValueError(f"No candidate rank meets validation limits mean<={mean_limit:g}, "
                         f"max<={max_limit:g}; {errors}. Review reconstruction reports; "
                         "limits were not relaxed automatically.")
    selected = passing[0]
    return {
        "selected_rank": selected["rank"],
        "artifact": selected["artifact"],
        "policy": {"strategy": "smallest_rank_meeting_validation_limits",
                   "mean_limit": mean_limit, "max_limit": max_limit},
        "rank_selection_split": "validation",
        "test_metrics_computed": False,
        "validation_metrics": {
            "mean_relative_l2_fluid": selected["validation_mean_relative_l2_fluid"],
            "max_relative_l2_fluid": selected["validation_max_relative_l2_fluid"],
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mean-limit", type=float, default=0.05)
    parser.add_argument("--max-limit", type=float, default=0.20)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        selected = select_rank(report, mean_limit=args.mean_limit, max_limit=args.max_limit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(selected, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
