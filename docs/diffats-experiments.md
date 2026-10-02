# GH200 上的完整三维与 DiffATS 实验

现有 1000 个流场可以复用，不重新运行 Taichi。本流程在 DeltaAI 的 ARM/GH200 节点上使用站点 PyTorch，所有压缩准备、训练和评估均通过 Slurm。登录节点只同步代码、生成任务计划及提交作业。

## 实验矩阵

三种架构：现有 `unet3d`、直接处理完整三维场的 `dit3d`、对齐 Tucker 表示的 `tensor-dit`。后两者的默认 Transformer 宽度 192、深度 4、注意力头 6；完整三维 DiT 的 patch 大小为 4。不同架构的参数量与耗时应实际报告，不能把架构变化的收益都归结为降维。

| 阶段 | 架构数 | 观测粒子数 | 种子 | 总训练任务 | 每任务 epochs |
|---|---:|---|---|---:|---:|
| 试跑 pilot | 3 | 2、24、96 | 31 | 9 | 10 |
| 正式 full | 3 | 2、4、6、12、24、48、96 | 31、32、33 | 63 | 100 |

每个任务独立训练、独立保存权重，使用一张 GH200。正式训练从头开始，方便统一比较。默认 batch size 1、AdamW 学习率 0.0002、weight decay 0.0001、1000 个扩散时间步；其余实际默认值保存在训练输出。pilot 用于确认全部分支正常运行并测量资源消耗，不将其 10 epoch 模型作为正式 100 epoch 结果。

三个空间方向的对齐因子形状各为 `32×r`，速度分量轴保留在 `3×r×r×r` 核心中。总表示大小为 `3r³+96r`；秩不是粗网格分辨率。选择锚点、归一化等统计只使用训练划分，按验证集重建质量选秩，测试集留待最终评估。秩不同的正式矩阵应分别新建计划，不能中途替换 artifact。

默认并发上限 4 是本工作流的提交设置，不是站点保证或账户额度。资源申请上限：准备 2 小时、单训练任务 8 小时、单评估任务 8 小时；这不是 GH200 实测耗时，可通过环境变量调整，`ghx4` 最长 48 小时。

## 登录与准备

在本地 Windows 终端输入：

```bash
ssh zguan2@dtai-login.delta.ncsa.illinois.edu
```

下列命令全部在服务器的 `gh-login...` 窗口执行：

```bash
cd ~/projects/project_code
git pull --ff-only
cd project
module load python/miniforge3_pytorch/2.10.0
conda activate base
source configs/deltaai-matrix.env.example
```

配置已经填好实际 Slurm 账户 `biup-dtai-gh`、分区 `ghx4` 和原数据 manifest：

```text
/work/hdd/biup/zguan2/datasets/2026-10-01_115342_lbm3d-full_22594936_7966ccef/manifest.json
```

先检查计划，再提交压缩准备：

```bash
bash scripts/submit.sh tensor-prepare --ranks 4,8,12,16 --dry-run
bash scripts/submit.sh tensor-prepare --ranks 4,8,12,16
```

`--dry-run` 只创建计划和显示 sbatch 命令，不提交任务、不会运行 GPU 代码；正式提交去掉该参数。正式提交会 `git pull --ff-only`，保存完整仓库只读快照，整个数组共用同一提交。未推送的修改、网络同步失败都会阻止新任务提交。

终端打印 `计划：.../plan.json` 和 Slurm JOB_ID。压缩输出目录记录在该计划的 `prepare_output` 字段；完成后里面有 `report.json` 与 `rank_4.pt`、`rank_8.pt`、`rank_12.pt`、`rank_16.pt`。查看报告的验证误差、各速度分量及物理重建指标后，再选一个秩；不要默认认为更高压缩比更好。锚点重叠与近似重建误差需要一起检查。

## 当前预处理自动衔接 9 组试跑

已经提交的 `tensor-prepare` 只有一个计算任务，它依次准备各秩的缓存，不包含 63 组训练。可以现在另提交一个有依赖的试跑数组：预处理成功后自动选秩，再开始 9 个独立训练子任务，无需重新运行预处理，也无需保持 SSH 在线。正式 63 组训练仍需要检查试跑结果后手动提交。

先在 DeltaAI 的 `gh-login...` 窗口更新代码并加载配置；`uname -m` 应为 `aarch64`：

```bash
cd ~/projects/project_code &&
git pull --ff-only &&
cd project &&
module load python/miniforge3_pytorch/2.10.0 &&
conda activate base &&
source configs/deltaai-matrix.env.example
```

找到当前预处理提交时打印的完整 `计划：.../plan.json` 路径和 Slurm 数组主编号，替换下方两个示例值。不要使用 `--dry-run` 产生的预处理计划；数组编号应为 `1234567` 这种主编号，不带 `_0` 子任务后缀。

```bash
export PREPARE_PLAN=/work/hdd/biup/zguan2/results/matrices/实际预处理计划/plan.json
export PREPARE_JOB_ID=1234567

bash scripts/submit.sh matrix-followup \
  --prepare-plan "$PREPARE_PLAN" --after-job "$PREPARE_JOB_ID" --dry-run

bash scripts/submit.sh matrix-followup \
  --prepare-plan "$PREPARE_PLAN" --after-job "$PREPARE_JOB_ID"
```

正式命令仅执行一次，并保存它打印的新训练计划路径和数组编号。该命令不会修改已有的预处理作业。预处理尚未完成时，新数组设置 `afterok` 依赖：提交后就进入队列等待，预处理成功退出后才满足依赖条件，是否立即获得 GH200 仍由 Slurm 调度决定。若预处理已完成，且成功记录、报告和文件校验均通过，则直接提交试跑，不再依赖可能已从 Slurm 缓存中清除的旧作业。

自动选秩采用以下固定策略：在准备报告包含的候选秩中，选出满足**验证集平均相对 L2 误差不超过 0.05，最大相对 L2 误差不超过 0.20**的最小秩。没有任何候选满足时停止，不自动放宽阈值。两项阈值只是用于试跑的可调整工程门槛，不证明科研精度达标；仍需检查各速度分量、锚点重叠和物理重建指标。只根据训练／验证信息选择，测试集不参与选秩。

需要改变门槛时，在上述命令中明确增加 `--rank-mean-limit` 和 `--rank-max-limit`，例如 `--rank-mean-limit 0.03 --rank-max-limit 0.15`。策略写入不可变的训练计划；首次开始时会记录 `rank_selection.json`，包含所选秩以及预处理成功记录、报告和 artifact 的 SHA256。所有子任务校验并使用同一个选择，后续恢复不能更改策略或替换 artifact。

自动试跑固定为三个架构 × 粒子数 2、24、96 × 种子 31，共 9 个训练任务，默认每组 10 epochs。每个子任务申请一张 GH200，默认最多同时运行 4 个，每个任务限时 8 小时。因此可以只看到一个数组主编号；用下面的命令展开查看各子任务，`0` 至 `8` 是九个独立实验：

```bash
squeue -r -u "$USER"
```

预处理失败或被取消时，后续数组不会开始训练。脚本设置 `--kill-on-invalid-dep=yes`，请求 Slurm 在依赖已不可能满足时自动取消后续数组；检查状态时也可能看到 `DependencyNeverSatisfied`。修复预处理后，需要针对新的成功预处理提交衔接计划。若只有试跑子任务失败，应使用本页的 `matrix-resume --plan "$TRAINING_PLAN"` 恢复原试跑计划，并通过 `FLOW3D_ARRAY_TASKS` 指定失败编号；不要再次运行 `matrix-followup`，否则会创建另一批独立试跑。

## 试跑与正式训练

若已提交上一节的自动衔接，跳过下面两条手动 `--phase pilot` 命令，避免重复训练。自动试跑完成后，用实际自动试跑计划路径替换下面示例，读取其已选定的 artifact，再检查结果并执行本节的 `--phase full` 命令：

```bash
export TRAINING_PLAN=/work/hdd/biup/zguan2/results/matrices/实际自动试跑计划/plan.json
TENSOR_ARTIFACT=$(python -c 'import json, pathlib, sys; p = pathlib.Path(sys.argv[1]).parent / "rank_selection.json"; print(json.loads(p.read_text(encoding="utf-8"))["artifact"]["path"])' "$TRAINING_PLAN") &&
export TENSOR_ARTIFACT
```

把准备阶段实际输出目录填入变量。以下路径中的 `实际准备目录` 必须替换；`rank_8.pt` 只是示例，不是已经选定的秩。

```bash
export TENSOR_ARTIFACT=/work/hdd/biup/zguan2/datasets/tensor_space/实际准备目录/rank_8.pt
bash scripts/submit.sh matrix-train --phase pilot --tensor-artifact "$TENSOR_ARTIFACT" --dry-run
bash scripts/submit.sh matrix-train --phase pilot --tensor-artifact "$TENSOR_ARTIFACT"
```

9 个试跑任务完成并检查损失、有限值、显存和耗时后，提交正式矩阵：

```bash
export FLOW3D_ARRAY_CONCURRENCY=4
export FLOW3D_TRAIN_TIME=08:00:00
bash scripts/submit.sh matrix-train --phase full --tensor-artifact "$TENSOR_ARTIFACT" --dry-run
bash scripts/submit.sh matrix-train --phase full --tensor-artifact "$TENSOR_ARTIFACT"
```

任务编号按架构、粒子数、种子依次排列；完整映射写在计划 `tasks` 中。63 个训练任务是一个秩的全矩阵。不自动提交第二个秩，也不自动消耗资源做评估。

## 查看状态与结果

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES%60
tail -n 60 /work/hdd/biup/zguan2/logs/JOB_ID_0.out
tail -n 60 /work/hdd/biup/zguan2/logs/JOB_ID_0.err
```

`JOB_ID` 替换为数组编号，`0` 替换为具体 task ID。排队时 `PD` 表示等待。断开 SSH 后 Slurm 作业继续执行。

目录安排：

```text
/work/hdd/biup/zguan2/
  datasets/tensor_space/<准备计划>/        report.json 和各秩 artifact
  results/matrices/<训练计划>/plan.json   全矩阵配置、文件哈希、统一代码提交
  results/matrices/<训练计划>/tasks/      各任务 results.json、各次尝试的环境与日志
  checkpoints/<训练计划>/<架构_N_种子>/   latest.pt、best.pt、训练记录
  logs/<数组编号>_<任务编号>.out          Slurm 标准输出
  logs/<数组编号>_<任务编号>.err          Slurm 标准错误
```

`plan.json` 保存 manifest 内容与 SHA256，以及手动选定的 tensor artifact 的 SHA256，任务启动前再次核对。自动衔接计划先固定预处理计划和选秩策略，artifact 尚未生成时不预填其 SHA256；成功准备后的选择及 artifact SHA256 固定在同目录的 `rank_selection.json` 中，所有子任务再次核对。修改数据 manifest、替换 artifact 或使用别的代码快照会被拒绝。底层训练器继续校验科学数据划分及 checkpoint 兼容性。

## 评估

全部训练任务完成后，将正式训练的实际计划路径填入：

```bash
export TRAINING_PLAN=/work/hdd/biup/zguan2/results/matrices/实际训练计划/plan.json
bash scripts/submit.sh matrix-evaluate --training-plan "$TRAINING_PLAN" --dry-run
bash scripts/submit.sh matrix-evaluate --training-plan "$TRAINING_PLAN"
```

每个训练模型对应一个评估任务，默认评估 manifest 的所有测试样本（当前 101 个），64 个未观测探针粒子、16 个后验样本、50 步 DDIM、eta 1、CFG 1.5。每个测试案例使用 `47+case_index` 的观测/采样种子。观测粒子数最大 96，与 64 个探针合计为数据现有的 160 个粒子，探针不会作为条件输入。

完整 63 模型评估的后验场及参考场、均值、方差数组合计约 48 GB（未压缩，不含权重和其他记录）。先用 pilot 的评估耗时估算正式矩阵；评估耗时可能明显超过训练，8 小时仍只是每任务上限。

所有架构均解码至原始 `3×32×32×32` 流场后评价，矩阵统一使用 `--boundary-projection final`，仅对最终解码场施加已知边界；tensor latent 的中间步骤不能按体素覆盖边界。每个任务在 `evaluation/summary.json` 和 `summary.csv` 输出全测试集汇总，`evaluation/runs.jsonl` 保留单案例记录。潜变量训练 loss 与全网格训练 loss 不能直接比较。时间和显存应在同一种 GH200 上实测。

评估提交会冻结当前最新 GitHub 代码，并记录训练代码提交；需要比较多批矩阵时，保持评估实现相同。若科研代码更新改变指标定义，应重评估相关全部模型。

评估完成后，可在登录节点汇总小型 JSON/CSV（不运行 GPU）：

```bash
export EVALUATION_PLAN=/work/hdd/biup/zguan2/results/matrices/实际评估计划/plan.json
python ../scripts/run_hpc_matrix.py summarize --plan "$EVALUATION_PLAN"
```

结果位于评估计划同级 `summary/`：`per_seed.csv` 保留模型×粒子数×训练种子的测试均值与测试样本标准差；`aggregate.csv` 对各训练种子的测试均值再次求平均与样本标准差（跨种子 ddof=1，单种子时留空）；`per_case.jsonl` 保留逐案例原始记录；`summary.json` 包含完整任务状态。缺失、失败或科学性检查未通过的任务明确列出，不会被默默当作成功或填成 0。未全部完成时 `complete` 为 false。

## 明确恢复失败任务

恢复只允许原计划、原代码快照、原训练配置。先查明失败原因，再列出需要恢复的 task ID，避免重复提交已完成任务：

```bash
export FLOW3D_ARRAY_TASKS=3,7
export FLOW3D_TRAIN_TIME=16:00:00
bash scripts/submit.sh matrix-resume --plan "$TRAINING_PLAN" --dry-run
bash scripts/submit.sh matrix-resume --plan "$TRAINING_PLAN"
unset FLOW3D_ARRAY_TASKS
```

有 `latest.pt` 时从已完成的 epoch 恢复，并保留原 `best.pt` 对应的历史最佳模型；没有 checkpoint 且输出为空时重试原任务；完成任务禁止覆盖。评估中断也可用其评估计划 `matrix-resume`，已完成案例会复用。压缩准备失败则重新提交准备任务，使用新输出目录。锁防止同一任务同时运行。快照必须保留到全部任务及重试完成。

需要改变粒子数、秩、模型、随机种子或 epochs 时，提交新的计划；不要将它们作为原任务恢复。没有自动跨粒子数使用 checkpoint 的流程。

参考：[DiffATS 论文](https://arxiv.org/html/2605.09275v1)、[DiffATS 作者代码](https://github.com/JinhuaLyu/DiffATS)、[DiT 官方实现](https://github.com/facebookresearch/DiT)、[Slurm 数组](https://slurm.schedmd.com/job_array.html)。此处是面向投影粒子轨迹与三分量稳态流场的适配，不是论文原 PDE 条件输入的直接复现。
