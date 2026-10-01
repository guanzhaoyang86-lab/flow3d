from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "visualize_projected_trajectories.py"
)
_SPEC = importlib.util.spec_from_file_location("_trajectory_visualizer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
viz = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = viz
_SPEC.loader.exec_module(viz)


def _minimal_data() -> dict[str, object]:
    samples, views, particles, times = 1, 3, 4, 3
    return {
        "flow_field": np.zeros((1, 3, 3, 3, 3), dtype=np.float32),
        "trajectories_3d": np.zeros((samples, particles, times, 3), dtype=np.float32),
        "trajectories_2d": np.zeros(
            (samples, views, particles, times, 2), dtype=np.float32
        ),
        "projection_matrix": np.zeros((views, 2, 3), dtype=np.float32),
        "observation_mask": np.ones((samples, views, particles, times), dtype=bool),
        "observation_times": np.arange(times, dtype=np.float32),
        "domain_bounds": np.asarray(((0.0, 2.0),) * 3, dtype=np.float32),
        "metadata": {
            "projection_names": ["xz", "xy", "yz"],
            "preferred_field_plane": "xz",
            "boundary_mode": "terminate",
        },
        "particle_counts": np.asarray([particles], dtype=np.int64),
    }


def test_stored_candidate_replay_maps_views_and_uses_test_indices(
    tmp_path: Path,
) -> None:
    data = _minimal_data()
    predictions = np.zeros((1, 2, 4, 3, 2), dtype=np.float32)
    predictions[:, 0] = 20.0
    predictions[:, 1] = 10.0
    validity = np.ones((1, 4, 3), dtype=bool)
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        predicted_trajectories_2d=predictions,
        prediction_validity=validity,
        test_indices=np.asarray([1, 3], dtype=np.int64),
        selected_view_indices=np.asarray([2, 0], dtype=np.int64),
    )

    stored = viz._load_stored_candidate_replay(path, data)
    assert stored is not None
    indices, replay, replay_validity = viz._stored_candidate_view(stored, 0, 0)

    np.testing.assert_array_equal(indices, [1, 3])
    np.testing.assert_array_equal(replay, predictions[0, 1, [1, 3]])
    np.testing.assert_array_equal(replay_validity, validity[0, [1, 3]])


def test_diagnostics_uses_saved_held_out_tracks_without_candidate_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = _minimal_data()
    stored = viz._StoredCandidateReplay(
        trajectories_2d=np.zeros((1, 1, 4, 3, 2), dtype=np.float32),
        validity=np.ones((1, 4, 3), dtype=bool),
        test_indices=np.asarray([1, 3], dtype=np.int64),
        selected_view_indices=np.asarray([0], dtype=np.int64),
    )
    no_op_names = (
        "_plot_velocity_field",
        "_plot_velocity_streamlines",
        "_plot_velocity_field_slices",
        "_plot_velocity_field_3d",
        "_plot_velocity_field_comparison",
        "_plot_3d_trajectories",
        "_plot_projected_trajectories",
    )
    for name in no_op_names:
        monkeypatch.setattr(viz, name, lambda *args, **kwargs: None)

    replay_calls: list[torch.Tensor] = []

    def fake_replay(
        replay_data: dict[str, object],
        sample: int,
        view: int,
        field: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        replay_calls.append(field)
        return (
            np.arange(4),
            np.zeros((4, 3, 2), dtype=np.float32),
            np.ones((4, 3), dtype=bool),
        )

    plotted: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(viz, "_replay", fake_replay)
    monkeypatch.setattr(
        viz,
        "_plot_replay_comparison",
        lambda *args, **kwargs: plotted.append((args, kwargs)),
    )

    viz.create_diagnostics(
        data,
        sample=0,
        view=0,
        max_particles=24,
        output=tmp_path,
        candidate_field=torch.zeros((1, 3, 3, 3, 3)),
        candidate_label="Coarse-grid baseline",
        stored_candidate_replay=stored,
    )

    assert len(replay_calls) == 1
    assert len(plotted) == 2
    candidate_args, candidate_kwargs = plotted[1]
    np.testing.assert_array_equal(candidate_args[3], [1, 3])
    np.testing.assert_array_equal(candidate_args[4], [1, 3])
    assert (
        candidate_kwargs["title_override"]
        == "Held-out test tracks: observed vs. Coarse-grid baseline"
    )


def test_fallback_replay_title_declares_stored_depth_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = _minimal_data()
    no_op_names = (
        "_plot_velocity_field",
        "_plot_velocity_streamlines",
        "_plot_velocity_field_slices",
        "_plot_velocity_field_3d",
        "_plot_velocity_field_comparison",
        "_plot_3d_trajectories",
        "_plot_projected_trajectories",
    )
    for name in no_op_names:
        monkeypatch.setattr(viz, name, lambda *args, **kwargs: None)

    monkeypatch.setattr(
        viz,
        "_replay",
        lambda *args, **kwargs: (
            np.arange(4),
            np.zeros((4, 3, 2), dtype=np.float32),
            np.ones((4, 3), dtype=bool),
        ),
    )
    plotted: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        viz,
        "_plot_replay_comparison",
        lambda *args, **kwargs: plotted.append((args, kwargs)),
    )

    viz.create_diagnostics(
        data,
        sample=0,
        view=0,
        max_particles=24,
        output=tmp_path,
        candidate_field=torch.zeros((1, 3, 3, 3, 3)),
        candidate_label="Legacy candidate",
    )

    assert (
        plotted[1][1]["title_override"]
        == "Stored-depth diagnostic: observed vs. Legacy candidate"
    )


def test_comparison_relative_l2_excludes_solids_and_prescribed_lid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = np.ones((1, 3, 3, 3, 3), dtype=np.float32)
    candidate = reference.copy()
    solid = np.zeros((3, 3, 3), dtype=bool)
    solid[0] = True
    candidate[:, :, 0] += 100.0
    candidate[:, :, :, :, -1] += 100.0
    candidate[0, 0, 1, 1, 1] += 1.0
    data = {
        "flow_field": reference,
        "observation_times": np.asarray([0.0], dtype=np.float32),
        "domain_bounds": np.asarray(((0.0, 2.0),) * 3, dtype=np.float32),
        "solid_mask": solid,
        "metadata": {"boundary_mode": "terminate", "lid_face": "x=max"},
    }

    assert viz._interior_relative_l2(data, reference, candidate) == pytest.approx(
        1.0 / 6.0
    )

    titles: list[str] = []

    def capture_title(fig: object, path: Path) -> None:
        titles.append(fig._suptitle.get_text())
        viz.plt.close(fig)

    monkeypatch.setattr(viz, "_save_figure", capture_title)
    viz._plot_velocity_field_comparison(
        data,
        reference,
        candidate,
        tmp_path,
        0,
        candidate_label="Coarse-grid baseline",
        field_plane="xy",
    )
    assert "interior relative L2 = 1.667e-01" in titles[0]
    assert "global relative L2" not in titles[0]
