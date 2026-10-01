"""Differentiable observations of projected Lagrangian trajectories."""

from .advection import advect_particles
from .cfd import (
    CFDVelocityAdapter,
    NPZCFDAdapter,
    VelocitySnapshotData,
    load_velocity_snapshots,
)
from .diffusion import GaussianDiffusion
from .interpolation import sample_velocity
from .likelihood import projected_trajectory_nll
from .multiview import triangulate_initial_positions
from .projection import lift_projected_initial_positions, project_trajectories
from .recovery import (
    CoarseVelocityField,
    divergence_mse,
    masked_trajectory_mse,
    project_trajectory_views,
    spatial_smoothness_mse,
)
from .sparse_dataset import (
    FlowNormalizationStats,
    SparseFlowDataset,
    compute_train_flow_stats,
    denormalize_flow,
    normalize_flow,
)
from .sparse_physics import (
    apply_hard_boundary_conditions,
    batched_divergence_mse,
    batched_multiview_replay,
    batched_trajectory_consistency,
)
from .taichi_lbm3d import (
    make_lid_driven_cavity_geometry,
    taichi_solid_mask_to_canonical,
    taichi_velocity_to_canonical,
)

__all__ = [
    "advect_particles",
    "CFDVelocityAdapter",
    "GaussianDiffusion",
    "FlowNormalizationStats",
    "make_lid_driven_cavity_geometry",
    "lift_projected_initial_positions",
    "load_velocity_snapshots",
    "NPZCFDAdapter",
    "SparseFlowDataset",
    "apply_hard_boundary_conditions",
    "batched_divergence_mse",
    "batched_multiview_replay",
    "batched_trajectory_consistency",
    "compute_train_flow_stats",
    "denormalize_flow",
    "normalize_flow",
    "project_trajectories",
    "projected_trajectory_nll",
    "sample_velocity",
    "taichi_solid_mask_to_canonical",
    "taichi_velocity_to_canonical",
    "VelocitySnapshotData",
]
