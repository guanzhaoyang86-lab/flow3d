# Delta A100：两小时时限的正式训练矩阵

本页在 Delta 的 `dt-login...`（x86_64）执行。每个任务申请一张 A100，使用
`biup-delta-gpu` / `gpuA100x4`，8 个 CPU、32 GB 主机内存、2 小时运行上限，
最多 16 个任务并发。实际启动时间和并发数由 Slurm 决定。

正式训练仍为三种模型 × 七种粒子数 × 三个种子，共 63 个任务，每个 100 epochs。
时限从任务实际运行开始计算，2 小时不是预计完成时间。每轮原子保存 `latest.pt`；
到时未完成会被 Slurm 停止，需要显式续训。测试集评估单独提交。

## 环境和数据

复用之前在 Delta 成功生成数据、训练基线的环境：
`~/envs/flow3d-lbm3d`（Python 3.11、PyTorch 2.5.1/CUDA 12.4）。
加载站点 `miniforge3-python` 后激活该环境，不安装或修改站点 base 环境。
如果该环境尚不存在，先按 [Delta 数据生成说明](delta-generation.md) 配置环境。
本次训练不需要重新生成数据、运行 Taichi 或重新做 Tucker 分解。

原始 manifest 和 Tucker 缓存位于两集群共享的 `/work/hdd/biup/$USER`。
缓存通过 `map_location="cpu"` 加载，再搬到作业分配的 GPU；加载时验证原始
数据文件和 manifest 的 SHA256。训练计划、运行环境记录会注明执行集群和 GPU。

## 从已有 GH200 计划复用所选秩，提交 A100 新批次

以下代码使用已提交的 GH200 批次 `3299429` 对应计划，读取并校验它选定的
缓存文件，不重新选秩。此操作创建新的 A100 计划和结果目录，不修改 GH200 计划。
其他项目请替换 `GH_PLAN` 和配置中的数据 manifest 路径。
`matrix-followup` 的作业依赖仅支持同一集群；跨集群使用下面的已完成缓存复用方式。

```bash
(
set -euo pipefail
cd ~/projects/project_code
git pull --ff-only
cd project
source configs/delta-matrix.env.example
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$FLOW3D_CONDA_ENV"
python -c 'import platform, torch; print(platform.machine(), platform.python_version(), torch.__version__, torch.version.cuda); assert platform.machine() == "x86_64"'

GH_PLAN=/work/hdd/biup/zguan2/results/matrices/2026-10-03_074635_matrix-train_51493784/plan.json
TENSOR_ARTIFACT=$(python ../scripts/run_hpc_matrix.py inspect --plan "$GH_PLAN" --field tensor_artifact)
FLOW3D_ARRAY_TASKS=all bash scripts/submit.sh matrix-train \
  --phase full --epochs 100 --tensor-artifact "$TENSOR_ARTIFACT"
)
```

提交脚本再次拉取 GitHub，冻结新的只读代码快照，打印计划路径及作业编号。
保存打印出来的 A100 `plan.json`，恢复训练和评估要使用这份计划。
`FLOW3D_TRAIN_TIME` 控制矩阵时限；通用变量 `FLOW3D_TIME` 不控制矩阵。
代码中的模型、粒子数、随机种子和训练配置保持一致；实际浮点结果和速度可能随
硬件/PyTorch 版本不同而变化，报告时分别记录，不能混合比较两种 GPU 的耗时。

**A100 新批次与 GH200 原批次不会自动去重或互相取消。** 如果两边运行相同任务，
会分别消耗两份分配额度。确定保留哪一边后，再按实际状态处理重复的待运行任务；
不要对正在产出结果的整个批次盲目取消。

```bash
squeue -r -u "$USER"
squeue --start -r -u "$USER"
# 将下面的编号替换为实际 A100 数组编号和运行中的任务编号
tail -n 60 /work/hdd/biup/$USER/logs/ARRAY_JOB_ID_TASK_ID.out
tail -n 60 /work/hdd/biup/$USER/logs/ARRAY_JOB_ID_TASK_ID.err
```

## 超时续训

先用 `sacct -X -j 数组编号 --format=JobID,State,ExitCode,Elapsed` 确认具体任务已
超时或失败。只对需要恢复的任务编号调用 `matrix-resume`，不要重新提交整个
已完成的矩阵。以下是恢复编号 0、5 的示例：

```bash
source configs/delta-matrix.env.example
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$FLOW3D_CONDA_ENV"
export A100_PLAN=/work/hdd/biup/zguan2/results/matrices/实际A100计划/plan.json
FLOW3D_ARRAY_TASKS=0,5 bash scripts/submit.sh matrix-resume --plan "$A100_PLAN"
```

续训使用原始快照和原始计划，从最后一个完整 epoch 继续到总计 100 epochs。
已经完成的任务禁止覆盖；锁防止同一任务同时续跑。GH200 旧计划默认归属
DeltaAI，不能借 `matrix-resume` 静默迁移到 A100。

## 评估

所有 63 个训练任务成功完成后，在已激活的 Delta 环境里提交：

```bash
bash scripts/submit.sh matrix-evaluate --training-plan "$A100_PLAN"
```

评估默认上限 8 小时，可用 `FLOW3D_EVALUATE_TIME` 单独修改；1000 步扩散训练和
多案例后验采样的耗时不同，2 小时训练上限不意味着完整评估也能在 2 小时内结束。
评估协议和汇总方式见 [实验矩阵说明](diffats-experiments.md)。
