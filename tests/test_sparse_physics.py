from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from flow_observation.advection import advect_particles
from flow_observation.multiview import triangulate_initial_positions
from flow_observation.recovery import project_trajectory_views
from flow_observation.sparse_physics import (
    apply_hard_boundary_conditions,
    batched_divergence_mse,
    batched_multiview_trajectory_consistency,
    batched_spatial_smoothness_mse,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_COLLECTION_SCRIPT = (
    _REPOSITORY_ROOT / "scripts" / "generate_taichi_lbm3d_flow_collection.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_flow_collection_generator", _COLLECTION_SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_COLLECTION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COLLECTION
_SPEC.loader.exec_module(_COLLECTION)


def _cameras() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # xz
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # xy
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # yz
        ],
        dtype=torch.float64,
    )


def _synthetic_observations(
    fields: torch.Tensor,
    initial_positions: torch.Tensor,
    times: torch.Tensor,
    cameras: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    projected = []
    for index in range(fields.shape[0]):
        trajectories = advect_particles(
            fields[index : index + 1],
            initial_positions[index : index + 1],
            times,
            integrator="rk4",
            num_substeps=2,
            domain_bounds=bounds,
            boundary_mode="terminate",
        )
        projected.append(project_trajectory_views(trajectories, cameras))
    return torch.cat(projected, dim=0)


def test_hard_boundary_overwrite_is_exact_and_blocks_fixed_gradients() -> None:
    torch.manual_seed(4)
    fields = torch.randn((2, 3, 4, 5, 6), dtype=torch.float64, requires_grad=True)
    fixed_mask = torch.zeros((4, 5, 6), dtype=torch.bool)
    fixed_mask[0] = True
    fixed_mask[:, :, -1] = True
    fixed_values = torch.zeros((3, 4, 5, 6), dtype=torch.float64)
    fixed_values[0] = 0.25
    fixed_values[1] = -0.5
    fixed_values[2] = 0.75

    constrained = apply_hard_boundary_conditions(fields, fixed_mask, fixed_values)
    expanded_mask = fixed_mask[None, None].expand_as(constrained)
    expanded_values = fixed_values[None].expand_as(constrained)

    assert torch.equal(constrained[expanded_mask], expanded_values[expanded_mask])
    assert torch.equal(constrained[~expanded_mask], fields[~expanded_mask])

    constrained.sum().backward()
    assert fields.grad is not None
    assert torch.equal(fields.grad[expanded_mask], torch.zeros_like(fields.grad[expanded_mask]))
    assert torch.equal(fields.grad[~expanded_mask], torch.ones_like(fields.grad[~expanded_mask]))


def test_rank_four_boundary_mask_is_rejected_when_batch_is_three() -> None:
    fields = torch.zeros((3, 3, 4, 4, 4), dtype=torch.float64)
    ambiguous = torch.zeros((3, 4, 4, 4), dtype=torch.bool)
    values = torch.zeros((3, 3, 4, 4, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="ambiguous when B == 3"):
        apply_hard_boundary_conditions(fields, ambiguous, values)

    explicit_per_sample = apply_hard_boundary_conditions(
        fields, ambiguous[:, None], values
    )
    explicit_per_component = apply_hard_boundary_conditions(
        fields, ambiguous[None], values
    )
    assert explicit_per_sample.shape == fields.shape
    assert explicit_per_component.shape == fields.shape


def test_batched_physics_wrappers_match_independent_fields() -> None:
    torch.manual_seed(3)
    fields = torch.randn((2, 3, 5, 5, 5), dtype=torch.float64)
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=torch.float64)

    smooth_batch = batched_spatial_smoothness_mse(fields, reduction="none")
    div_batch = batched_divergence_mse(fields, bounds, reduction="none")
    smooth_single = torch.stack(
        [batched_spatial_smoothness_mse(fields[i : i + 1]) for i in range(2)]
    )
    div_single = torch.stack(
        [batched_divergence_mse(fields[i : i + 1], bounds) for i in range(2)]
    )

    assert torch.allclose(smooth_batch, smooth_single)
    assert torch.allclose(div_batch, div_single)


def test_batched_replay_matches_per_sample_replay_for_two_particles() -> None:
    fields = torch.zeros((2, 3, 5, 5, 5), dtype=torch.float64)
    fields[0, 0] = 0.10
    fields[1, 2] = -0.08
    initial = torch.tensor(
        [
            [[0.20, 0.35, 0.40], [0.55, 0.60, 0.65]],
            [[0.30, 0.25, 0.70], [0.65, 0.55, 0.45]],
        ],
        dtype=torch.float64,
    )
    times = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.float64)
    cameras = _cameras()
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=torch.float64)
    observed = _synthetic_observations(fields, initial, times, cameras, bounds)

    batch_loss, batch_prediction, batch_validity = (
        batched_multiview_trajectory_consistency(
            fields,
            observed,
            cameras,
            times,
            bounds,
            num_substeps=2,
        )
    )
    single_results = [
        batched_multiview_trajectory_consistency(
            fields[index : index + 1],
            observed[index : index + 1],
            cameras,
            times,
            bounds,
            num_substeps=2,
        )
        for index in range(2)
    ]
    expected_prediction = torch.cat([result[1] for result in single_results], dim=0)
    expected_validity = torch.cat([result[2] for result in single_results], dim=0)
    expected_loss = torch.stack([result[0] for result in single_results]).mean()

    assert observed.shape[2] == 2
    assert torch.allclose(batch_prediction, expected_prediction, atol=1e-12, rtol=0.0)
    assert torch.equal(batch_validity, expected_validity)
    assert torch.allclose(batch_loss, expected_loss, atol=1e-14, rtol=0.0)
    assert torch.allclose(batch_prediction, observed, atol=1e-12, rtol=0.0)


def test_replay_triangulates_each_particle_from_valid_time_zero_views() -> None:
    fields = torch.zeros((1, 3, 5, 5, 5), dtype=torch.float64)
    fields[:, 0] = 0.05
    initial = torch.tensor(
        [[[0.20, 0.35, 0.40], [0.55, 0.60, 0.65]]], dtype=torch.float64
    )
    times = torch.tensor([0.0, 0.2, 0.4], dtype=torch.float64)
    cameras = _cameras()
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=torch.float64)
    observed = _synthetic_observations(fields, initial, times, cameras, bounds)
    corrupted = observed.clone()
    mask = torch.ones(observed.shape[:-1], dtype=torch.bool)
    # Each particle drops a different view, but its two remaining views still
    # jointly span x/y/z.  Corrupting the invalid pixels catches accidental use.
    mask[0, 2, 0, 0] = False
    corrupted[0, 2, 0, 0] = 999.0
    mask[0, 0, 1, 0] = False
    corrupted[0, 0, 1, 0] = -999.0

    recovered = triangulate_initial_positions(corrupted, cameras, mask)
    assert torch.allclose(recovered, initial, atol=1e-12, rtol=0.0)
    loss, predicted, validity = batched_multiview_trajectory_consistency(
        fields,
        corrupted,
        cameras,
        times,
        bounds,
        observation_mask=mask,
        num_substeps=2,
    )
    assert bool(validity.all())
    assert torch.allclose(predicted[..., 1:, :], observed[..., 1:, :], atol=1e-12)
    assert float(loss) < 1e-24


def test_triangulation_rejects_particle_with_rank_deficient_valid_views() -> None:
    observations = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float64)
    mask = torch.ones(observations.shape[:-1], dtype=torch.bool)
    mask[0, 1:, 1, 0] = False

    with pytest.raises(ValueError, match="batch 0, particle 1; got rank 2"):
        triangulate_initial_positions(observations, _cameras(), mask)


def test_trajectory_guidance_gradient_is_finite_and_nonzero() -> None:
    reference = torch.zeros((1, 3, 5, 5, 5), dtype=torch.float64)
    reference[:, 0] = 0.16
    initial = torch.tensor(
        [[[0.25, 0.30, 0.35], [0.50, 0.55, 0.60]]], dtype=torch.float64
    )
    times = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)
    cameras = _cameras()
    bounds = torch.tensor(((0.0, 1.0),) * 3, dtype=torch.float64)
    observed = _synthetic_observations(reference, initial, times, cameras, bounds)
    candidate = torch.zeros_like(reference, requires_grad=True)

    loss, _, validity = batched_multiview_trajectory_consistency(
        candidate,
        observed,
        cameras,
        times,
        bounds,
        num_substeps=2,
    )
    loss.backward()

    assert bool(validity.all())
    assert candidate.grad is not None
    assert bool(torch.isfinite(candidate.grad).all())
    assert float(candidate.grad.abs().sum()) > 0.0
    assert float(loss.detach()) > 0.0


def test_collection_dry_run_records_exact_single_case_command(tmp_path: Path) -> None:
    parser = _COLLECTION._build_parser()
    output_dir = tmp_path / "collection"
    args = parser.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--sampling",
            "grid",
            "--num-cases",
            "1",
            "--lid-speeds",
            "0.041",
            "--upstream-nius",
            "0.155",
            "--warmup-steps",
            "777",
            "--seeds",
            "123",
            "--num-particles",
            "2",
            "--dry-run",
        ]
    )
    metadata = _COLLECTION.run_collection(args)

    assert metadata["dry_run"] is True
    assert metadata["num_cases"] == 1
    record = metadata["cases"][0]
    command = record["command"]
    expected_pairs = {
        "--lid-speed": "0.041",
        "--upstream-niu": "0.155",
        "--warmup-steps": "777",
        "--seed": "123",
        "--num-particles": "2",
    }
    for option, expected in expected_pairs.items():
        assert command[command.index(option) + 1] == expected

    assert record["status"] == "planned"
    assert not list(output_dir.glob("*.npz"))
    assert (output_dir / "collection.json").is_file()
    assert (output_dir / "commands.jsonl").is_file()
    stored = json.loads((output_dir / "collection.json").read_text(encoding="utf-8"))
    assert "flow_group_id is the indivisible" in stored["split_policy"]
    assert stored["fixed_solver_facts"]["lid_direction"] == "+z (not varied)"
    assert stored["fixed_solver_facts"]["initial_state"].endswith("(not varied)")


def test_equal_physics_with_different_seeds_share_split_group(tmp_path: Path) -> None:
    parser = _COLLECTION._build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--sampling",
            "grid",
            "--num-cases",
            "2",
            "--lid-speeds",
            "0.05",
            "--upstream-nius",
            "0.16667",
            "--warmup-steps",
            "1000",
            "--seeds",
            "7,17",
            "--dry-run",
        ]
    )
    cases = _COLLECTION.plan_cases(args)

    assert len(cases) == 2
    assert cases[0].seed != cases[1].seed
    assert cases[0].flow_group_id == cases[1].flow_group_id
