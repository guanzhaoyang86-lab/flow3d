# Taichi-LBM3D projected-trajectory pipeline

## GitHub 与 NCSA 工作流（中文）

本仓库包含完整科研源码（根目录的 `src/`、`scripts/`、`tests/`）和
[中文 HPC 工作流模板](project/README.md)。你当前的 DeltaAI 分配请使用
[DeltaAI 真实模型部署说明](docs/deltaai.md)：已配置 ARM/GH200、`biup-dtai-gh`、
`ghx4` 和站点 PyTorch 2.10.0，通过 `diffusion-*` 作业连接根目录的真实科研模型。
`project/src/` 的原始小模型仍作为独立工作流示例。
数据尚未上传时，可使用 [Delta 生成数据、DeltaAI 训练](docs/delta-generation.md)：
在 Delta 的 A100 上运行固定版本 Taichi 求解器，直接写入两系统共享的 `/work/hdd`。
ACCESS 项目为 `PHY260443`，实际 Slurm 计费账户和存储路径须按站点分配配置。

在 HPC 首次克隆整个仓库后，从 `project/` 运行工作流模板：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/guanzhaoyang86-lab/flow3d.git project_code
cd project_code/project
source configs/deltaai.env.example
bash scripts/submit.sh diffusion-smoke
```

仓库已公开，HPC 使用 HTTPS 克隆和拉取无需 GitHub 令牌。提交脚本会同步整个仓库并保存固定
提交的源码快照，支持 `project/` 子目录布局。`diffusion-smoke` 使用微型合成数据运行
真实训练和推理；`diffusion-train` / `diffusion-inference` 接受服务器上已有的数据。
首次正式训练前必须通过真实 Slurm GPU 小测试；本地 CPU 验证不能替代这一步。
数据、权重、日志、渲染产物、临时文件、虚拟环境和第三方检出目录均不进入 Git。
外部 Taichi-LBM3D 依赖按下文固定提交安装。

This repository uses [Taichi-LBM3D](https://github.com/yjhp1016/taichi_LBM3D) as its only user-facing 3D flow source.

```text
Taichi-LBM3D velocity field
        -> 3D particle advection
        -> 2D camera projections
        -> trajectory likelihood and diagnostics
```

The differentiable PyTorch core provides trilinear velocity sampling, Euler/RK4 particle advection, camera projection, synchronized-view triangulation, and masked trajectory losses. A regularized `8^3` coarse-grid baseline now estimates a steady candidate field from projected tracks. It is an inverse-pipeline baseline, not the project's proposed method and not a claim of unique dense recovery from a single view.

## Setup

Python 3.10 or newer is required. The tested upstream revision is `fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c`.

```powershell
git clone https://github.com/yjhp1016/taichi_LBM3D.git third_party\taichi_LBM3D
git -C third_party\taichi_LBM3D checkout fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c
python -m venv --system-site-packages .venv-lbm3d
.\.venv-lbm3d\Scripts\python.exe -m pip install -e ".[test,lbm3d]"
```

Taichi remains isolated in the optional environment; importing the differentiable observation package does not import Taichi.

## Generate the 3D cavity example

Run a `32^3` lid-driven cavity calculation, advect 128 particles, and store XZ/XY/YZ projections:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\generate_taichi_lbm3d_cavity_dataset.py --grid-size 32 --warmup-steps 1000 --duration 120 --candidate-multiplier 16 --minimum-path-length 0.5 --output outputs\lbm3d_cavity_dataset.npz
```

Render the reference field and trajectories:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\visualize_projected_trajectories.py --input outputs\lbm3d_cavity_dataset.npz --view xz --output-dir outputs\lbm3d_cavity_reference_visualizations
```

With no `--candidate-input`, the visualizer produces only:

- `velocity_field.png`
- `velocity_field_streamlines.png`
- `velocity_field_slices.png`
- `velocity_field_3d.png`
- `trajectories_3d.png`
- `projected_trajectories.png`
- `ground_truth_replay.png`

The recovery baseline below supplies a candidate field with the same `[T,3,D,H,W]` shape. Any future inverse method can use the same `--candidate-input` contract; the visualizer then adds recovered-field comparison and replay figures.

## Run the coarse-grid inverse baseline

Estimate an `8^3` steady velocity grid from the synchronized XZ/XY/YZ tracks, with a projected-motion-stratified 80/24/24 particle split:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\recover_taichi_lbm3d_coarse_grid.py --input outputs\lbm3d_cavity_dataset.npz --device cpu
```

The optimizer triangulates time-zero positions from the selected 2D views. It deliberately does not load `trajectories_3d` or `initial_hidden_depth`; the Taichi reference field is used only after the candidate is fixed, for evaluation. The objective combines projected-track error, spatial smoothness, and divergence regularization while enforcing the known cavity walls and moving lid.

Render the result:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\visualize_projected_trajectories.py --input outputs\lbm3d_cavity_dataset.npz --candidate-input outputs\lbm3d_cavity_baseline_recovery.npz --candidate-label "Coarse-grid baseline" --view xz --field-plane xz --output-dir outputs\lbm3d_cavity_baseline_visualizations
```

The checked run used 80 training, 24 validation, and 24 held-out particles from the same cavity snapshot. Its test projected-coordinate RMSE is `0.173` lattice cells, versus `0.612` for a zero field and `0.394` for the known-boundary-only field. On the unknown interior, field relative L2 is `0.301` and velocity cosine similarity is `0.954`. Including the prescribed moving lid would improve those figures to `0.227` and `0.974`, so the interior-only values are the primary recovery metrics. The weak y component remains difficult (`relative L2 = 1.313`), and the selected checkpoint is the final 150-step budget rather than a demonstrated convergence plateau.

The recovery archive is `outputs/lbm3d_cavity_baseline_recovery.npz`. The diagnostics directory adds `optimization_convergence.png`, `recovered_field_comparison.png`, and `recovered_field_replay.png` to the reference figures.

## Two-particle conditional diffusion

The proposed sparse-observation path is implemented separately from the
coarse-grid baseline:

```text
many independent Taichi-LBM3D flow cases
        -> two projected particle tracks per training example
        -> permutation-invariant track encoder
        -> conditional 3D U-Net diffusion
        -> multiple 3D flow samples, posterior mean, and uncertainty
```

The diffusion variable is the complete normalized velocity volume
`[B,3,D,H,W]`. Two particle tracks are conditioning data, not the object being
diffused. At test time the ground-truth field remains hidden and is used only
for post-hoc metrics. The initial controlled experiment keeps 20 observation
times, three synchronized orthographic views, and zero observation noise so
particle count is the only changed variable.

Generate independent case archives. For the complete particle-count ablation,
store a shared pool of at least 160 random valid particles per flow: at most
128 are observations and 32 remain disjoint held-out probes. This does **not**
give 160 particles to the model; `--num-particles 2` in the training command
below still exposes exactly two tracks. Random placement avoids biasing the
sparse experiment toward unusually long or spatially diverse trajectories:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\generate_taichi_lbm3d_flow_collection.py `
  --output-dir outputs\lbm3d_multiflow `
  --sampling random --num-cases 1000 `
  --num-particles 160 --particle-selection random
```

For a two-particle-only pilot, the stored pool may be reduced to two, but that
archive cannot later support the 4--128 sweep or held-out probes. Every NPZ is
one generator invocation. Both the NPZ and `collection.json` record a
`flow_group_id`; files sharing one physical simulation but differing only in
snapshot iteration or particle seed remain in the same split. Existing
archives are configuration-checked before reuse. Build and validate the split
manifest with:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\build_sparse_flow_manifest.py `
  --input-dir outputs\lbm3d_multiflow `
  --output outputs\lbm3d_multiflow\manifest.json
```

Train the professor-requested two-particle model:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\train_sparse_track_diffusion.py `
  --manifest outputs\lbm3d_multiflow\manifest.json `
  --num-particles 2 --batch-size 1 `
  --output-dir outputs\two_particle_diffusion
```

The first stage optimizes DDPM epsilon loss. After a stable denoiser checkpoint
exists, low-noise fine-tuning can enable differentiable replay and physics:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\train_sparse_track_diffusion.py `
  --manifest outputs\lbm3d_multiflow\manifest.json `
  --resume outputs\two_particle_diffusion\best.pt `
  --epochs 150 `
  --num-particles 2 `
  --track-loss-weight 0.1 `
  --divergence-loss-weight 0.01 `
  --boundary-loss-weight 0.01 `
  --output-dir outputs\two_particle_diffusion_physics
```

Generate 16 posterior fields for one unseen test flow:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\sample_sparse_track_diffusion.py `
  --checkpoint outputs\two_particle_diffusion\best.pt `
  --manifest outputs\lbm3d_multiflow\manifest.json `
  --num-particles 2 --num-probe-particles 32 `
  --num-samples 16 --sampling-steps 50 `
  --output outputs\two_particle_posterior.npz
```

After training one checkpoint for each observation count, run the controlled
`N={2,4,8,32,64,128}` comparison with:

```powershell
.\.venv-lbm3d\Scripts\python.exe scripts\run_sparse_particle_sweep.py `
  --manifest outputs\lbm3d_multiflow\manifest.json `
  --checkpoint 2=outputs\diffusion_n2\best.pt `
  --checkpoint 4=outputs\diffusion_n4\best.pt `
  --checkpoint 8=outputs\diffusion_n8\best.pt `
  --checkpoint 32=outputs\diffusion_n32\best.pt `
  --checkpoint 64=outputs\diffusion_n64\best.pt `
  --checkpoint 128=outputs\diffusion_n128\best.pt `
  --num-test-cases 100 --num-probe-particles 32 --num-samples 16 `
  --output-dir outputs\particle_count_sweep
```

Using a single fixed-count checkpoint at other particle counts is allowed only
through the explicit `--allow-untrained-count-extrapolation` smoke-test flag;
the resulting summary is marked non-scientific.

`SparseFlowDataset` never reads stored hidden depth or reference 3D
trajectories as model inputs. It returns only projected tracks, masks, known
cameras/times/bounds, known cavity boundary values, and the complete field
target during training. A one-case manifest may be created only with
`--allow-overlap-for-smoke-test`; its checkpoints and samples are explicitly
marked non-scientific.

Checkpoints store the manifest hash, manifest mode, and all split flow-group
IDs. Sampling is marked scientific only when the checkpoint came from a
scientific manifest, the requested split is held-out test, the particle count
matches training, and the provenance checks agree. Legacy and smoke
checkpoints are always labeled non-scientific.

The current `.venv-lbm3d` PyTorch build does not contain RTX 5090 `sm_120`
kernels. CPU is sufficient for the included smoke checks, but full `32^3`
training requires a PyTorch build that supports the visible GPU architecture.

## Data convention

Physical coordinates and velocity components are always ordered `(x,y,z)`. Stored velocity grids use `[T,3,D,H,W]`, where `(D,H,W)=(z,y,x)`.

| Quantity | Shape |
| --- | --- |
| velocity field | `[T_v,3,D,H,W]` |
| velocity times | `[T_v]` |
| domain bounds | `[3,2]` |
| 3D trajectories | `[S,N,T_obs,3]` |
| projection matrices | `[V,2,3]` |
| projected trajectories | `[S,V,N,T_obs,2]` |
| hidden initial depth | `[S,V,N]` |
| observation mask | `[S,V,N,T_obs]` |

The upstream solver stores velocity as `[x,y,z,component]`; `taichi_velocity_to_canonical()` performs and tests the exact conversion. The moving lid is the `x=max` face and moves in `+z`, so XZ is the primary circulation plane.

## Package layout

- `src/flow_observation/taichi_lbm3d.py`: Taichi velocity/mask conversion and cavity geometry.
- `src/flow_observation/cfd.py`: validated CFD snapshot data and NPZ loading.
- `src/flow_observation/interpolation.py`: differentiable space/time velocity sampling.
- `src/flow_observation/advection.py`: differentiable Euler and RK4 integration.
- `src/flow_observation/projection.py`: camera projection and hidden-coordinate lifting.
- `src/flow_observation/multiview.py`: synchronized-view initial-position triangulation.
- `src/flow_observation/recovery.py`: bounded coarse fields and recovery losses.
- `src/flow_observation/likelihood.py`: projected-trajectory likelihood.
- `scripts/generate_taichi_lbm3d_cavity_dataset.py`: Taichi solve, particle advection, projection, and archive generation.
- `scripts/visualize_projected_trajectories.py`: flow-field and trajectory diagnostics.
- `scripts/recover_taichi_lbm3d_coarse_grid.py`: regularized steady inverse baseline.
- `tests/test_observation_pipeline.py`: numerical, boundary, batching, likelihood, and gradient checks.
- `tests/test_taichi_lbm3d_adapter.py`: axis, mask, geometry, and snapshot-time checks.
- `tests/test_recovery.py`: multi-view geometry, regularizers, boundaries, and inverse-gradient checks.

## Validation and limitations

Run the complete test suite with:

```powershell
.\.venv-lbm3d\Scripts\python.exe -m pytest -q
```

The saved cavity archive is a three-component numerical CFD reference snapshot. Ground-truth replay uses the same reference field and therefore checks the forward implementation; it is not a recovered field. The 1000-step run is not asserted to be steady-state converged; inspect `checkpoint_relative_change` before making a convergence claim.

The value passed to upstream `set_viscosity()` is recorded as `upstream_niu_parameter`; no independently validated physical-viscosity or Reynolds-number interpretation is claimed.

The coarse-grid result is a controlled multi-view numerical baseline. Sparse tracks constrain the regions they traverse; unobserved volume is determined mainly by low-resolution parameterization and regularization. It is not yet the project's own method, a time-varying recovery, a calibrated perspective-camera experiment, or a real-data result.

## Upstream citation

If this solver is used in research, cite the Taichi-LBM3D paper listed in the [upstream repository](https://github.com/yjhp1016/taichi_LBM3D).
