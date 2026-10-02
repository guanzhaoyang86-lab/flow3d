from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_tensor_rank.py"
_SPEC = importlib.util.spec_from_file_location("select_tensor_rank_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
selector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(selector)


def _report():
    # Deliberately unsorted: ordering must be numerical, not input order.
    entries = [(16, 0.001, 0.01), (4, 0.10, 0.30), (12, 0.01, 0.03), (8, 0.03, 0.10)]
    return {
        "rank_selection_split": "validation", "test_metrics_computed": False,
        "manifest": "/work/dataset/manifest.json",
        "ranks": {
            str(rank): {"rank": rank, "rank_selection_split": "validation",
                        "test_metrics_computed": False,
                        "validation_mean_relative_l2_fluid": mean,
                        "validation_max_relative_l2_fluid": maximum,
                        "artifact": f"/work/tensor/rank_{rank}.pt",
                        "artifact_sha256": f"{rank:064x}"}
            for rank, mean, maximum in entries
        },
    }


def test_selects_smallest_passing_rank_and_preserves_input():
    report = _report()
    original = copy.deepcopy(report)
    selected = selector.select_rank(report)
    assert selected["selected_rank"] == 8
    assert selected["artifact"] == {"path": "/work/tensor/rank_8.pt", "sha256": f"{8:064x}"}
    assert [item["rank"] for item in selected["candidates"]] == [4, 8, 12, 16]
    assert [item["passed"] for item in selected["candidates"]] == [False, True, True, True]
    assert selected["policy"]["mean_limit"] == 0.05
    assert selected["policy"]["max_limit"] == 0.20
    assert report == original
    assert selector.select_rank(report) == selected
    assert json.loads(json.dumps(selected, allow_nan=False)) == selected


def test_thresholds_are_inclusive_and_both_required():
    report = _report()
    report["ranks"]["4"]["validation_mean_relative_l2_fluid"] = 0.05
    report["ranks"]["4"]["validation_max_relative_l2_fluid"] = 0.20
    assert selector.select_rank(report)["selected_rank"] == 4
    report["ranks"]["4"]["validation_max_relative_l2_fluid"] = 0.20001
    assert selector.select_rank(report)["selected_rank"] == 8
    report["ranks"]["4"]["validation_max_relative_l2_fluid"] = 0.20
    report["ranks"]["4"]["validation_mean_relative_l2_fluid"] = 0.05001
    assert selector.select_rank(report)["selected_rank"] == 8


def test_custom_limits_select_a_different_rank():
    assert selector.select_rank(_report(), mean_limit=0.02, max_limit=0.04)["selected_rank"] == 12


def test_no_candidate_fails_without_relaxing_policy():
    with pytest.raises(ValueError, match="No candidate rank meets validation limits"):
        selector.select_rank(_report(), mean_limit=0.0001, max_limit=0.001)


@pytest.mark.parametrize("limits", [
    {"mean_limit": 0}, {"mean_limit": -1}, {"mean_limit": float("nan")},
    {"mean_limit": float("inf")}, {"mean_limit": True}, {"mean_limit": "0.05"},
    {"max_limit": 0}, {"max_limit": -1}, {"max_limit": float("nan")},
    {"max_limit": float("inf")}, {"max_limit": True}, {"mean_limit": 0.3, "max_limit": 0.2},
])
def test_invalid_limits_fail(limits):
    with pytest.raises(ValueError):
        selector.select_rank(_report(), **limits)


@pytest.mark.parametrize("field,value", [
    ("validation_mean_relative_l2_fluid", float("nan")),
    ("validation_max_relative_l2_fluid", float("inf")),
    ("validation_mean_relative_l2_fluid", -0.1),
    ("validation_mean_relative_l2_fluid", True),
    ("validation_mean_relative_l2_fluid", "0.01"),
    ("validation_mean_relative_l2_fluid", 0.02),  # Above this rank's maximum.
    ("artifact", ""), ("artifact_sha256", "no-hash"), ("rank", 12),
    ("rank_selection_split", "test"), ("test_metrics_computed", True),
])
def test_invalid_unselected_candidate_fails_closed(field, value):
    report = _report()
    report["ranks"]["16"][field] = value
    with pytest.raises(ValueError):
        selector.select_rank(report)


@pytest.mark.parametrize("field,value", [
    ("rank_selection_split", "test"), ("rank_selection_split", "train"),
    ("test_metrics_computed", True), ("test_metrics_computed", 0),
    ("ranks", {}), ("ranks", []),
])
def test_invalid_top_level_report_fails(field, value):
    report = _report()
    report[field] = value
    with pytest.raises(ValueError):
        selector.select_rank(report)


def test_noncanonical_rank_key_is_rejected():
    report = _report()
    report["ranks"]["08"] = report["ranks"].pop("8")
    with pytest.raises(ValueError, match="canonical positive integers"):
        selector.select_rank(report)


def test_numerical_mean_roundoff_is_accepted():
    report = _report()
    report["ranks"]["4"]["validation_mean_relative_l2_fluid"] = 0.010000000000000002
    report["ranks"]["4"]["validation_max_relative_l2_fluid"] = 0.01
    assert selector.select_rank(report)["selected_rank"] == 4


def test_cli_reads_existing_schema_and_emits_selection(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    selector.main(["--report", str(path)])
    assert json.loads(capsys.readouterr().out)["selected_rank"] == 8


def test_cli_missing_report_exits_with_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        selector.main(["--report", str(tmp_path / "missing.json")])
    assert error.value.code == 2
