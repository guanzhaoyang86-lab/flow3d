# GH200 全测试集评估与组会图片

在 DeltaAI 的 `gh-login...` 登录节点提交，GPU 计算由 Slurm 分配的 GH200 执行。
评估使用现有训练权重和数据，不重新训练、不取消原训练批次，也不等待尚未完成的训练任务。
代码通过 GitHub 更新；图片、指标和权重保存在 `/work/hdd/biup/zguan2`。

## 本次评估范围和时限

`--training-tasks completed` 在提交时固定一份已完成训练任务的选择，并验证其训练结果、
来源和 checkpoint。提交后才完成的训练任务不会自动加入这份评估计划。
例如当时只完成了训练任务 0–11，本次就是 UNet3D、2/4/6/12 个粒子、每组 3 个训练种子，
共 12 个评估任务，而不是把全部 63 个训练任务当作已经完成。

每个选中的模型都计划评估完整的 101 个测试流场，协议如下：

| 项目 | 设置 |
| --- | --- |
| GPU | 每任务 1 张 GH200 |
| 单任务 Slurm 上限 | 1 小时 30 分钟 |
| 数组并发上限 | 4，实际由调度器决定 |
| 测试流场 | 原 manifest 中全部 101 个 test cases |
| 每案例后验样本 | 16 |
| 采样步数 | 50 |
| 未观测探针粒子 | 64 |
| 基础采样种子 | 47，各案例使用相同可复现的规则 |
| DDIM eta / CFG scale | 1.0 / 1.5 |
| 边界投影 | `final` |

目前没有这套完整评估的 GH200 实测耗时，不能保证每任务 90 分钟跑完。
扩散采样耗时不能用训练 100 轮的耗时直接推算；排队时间也不计入 Slurm 运行时限。
脚本会根据已完成案例给出运行进度和剩余耗时估计，首次估计受初始化开销影响较大。

## 提交一次，保存评估计划路径

还未登录时，在本地终端执行：

```bash
ssh zguan2@dtai-login.delta.ncsa.illinois.edu
```

以下整段在服务器 `gh-login...` 窗口执行一次：

```bash
(
set -euo pipefail
cd ~/projects/project_code
git pull --ff-only
cd project
source configs/deltaai-evaluation.env.example
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

TRAIN_PLAN=/work/hdd/biup/zguan2/results/matrices/2026-10-03_074635_matrix-train_51493784/plan.json
FLOW3D_ARRAY_TASKS=all bash scripts/submit.sh matrix-evaluate \
  --training-plan "$TRAIN_PLAN" \
  --training-tasks completed
)
```

保存输出里的“计划：……/plan.json”和 `Submitted batch job` 编号。
这是新的评估计划，后续查结果和续跑都使用它，不要误用上面的训练计划。
评估数组编号从 0 重新连续排列；计划另行记录它对应的原训练任务编号。
代码快照与评估配置在提交时固定，后续 GitHub 更新不改变已提交作业。

文档提供提交命令，不代表已经从本地替你提交了作业。只有服务器返回实际作业编号后，
才能确认提交成功。SSH 密码或 Duo 交互需要由用户本人完成。

## 排队、进度和第一张图片

将 `评估数组编号` 替换为刚返回的数字：

```bash
squeue --start -r -j 评估数组编号
sacct -X -j 评估数组编号 --format=JobID%20,State%20,ExitCode,Elapsed
tail -n 60 /work/hdd/biup/zguan2/logs/评估数组编号_0.out
tail -n 60 /work/hdd/biup/zguan2/logs/评估数组编号_0.err
```

每个模型的第一个测试案例（test index 0）成功后，会用 CPU 自动生成图片，
无需等该模型全部 101 个案例结束。位置相对于该任务的记录目录：

```text
tasks/评估任务目录/
    results.json
    evaluation/
        runs.jsonl
        summary.json
        summary.csv
        figures/
            Nxxx_case0000/       PNG 和 PDF 图片
    report/                     完整任务结束后的组会汇总
```

首案例图可用于展示真实场、重建场及误差，但它只代表这个固定案例，
不能替代全测试集统计。部分运行也会保留已经产生的案例文件和进度记录。

## 生成组会报告

先在同一服务器窗口设置实际评估计划路径：

```bash
cd ~/projects/project_code/project
source configs/deltaai-evaluation.env.example
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
export EVAL_PLAN=/work/hdd/biup/zguan2/results/matrices/替换为实际评估计划目录/plan.json
```

完整任务结束后会自动生成任务报告；需要跨模型、粒子数、种子汇总时执行：

```bash
python ../scripts/visualize_matrix_evaluation.py report \
  --plan "$EVAL_PLAN" \
  --output-dir "$(dirname "$EVAL_PLAN")/report"
```

报告可以在评估只完成一部分时生成，内容会注明当前覆盖范围，不将缺失结果补零或伪造。
产物包括可放入组会幻灯片的 PNG/PDF、JSON/CSV 指标和中文 `meeting_report.md`，
不生成 PowerPoint 文件。

报告目录包含 `summary.json`、`per_seed.csv`、`particle_error_curves.png/.pdf`，
以及 `cases/` 下各模型固定首案例的 `flow_slices`、`trajectory_projections`、
`uncertainty` PNG/PDF。小图会复制到报告目录，下载整个 `report/` 即可离线查看，
无需依赖服务器上的 posterior NPZ。

汇报时区分两种覆盖率：本次选择的评估任务完成了多少，以及原始 63 项训练矩阵
有多少已纳入本次评估。即使本次 12/12 全部完成，也不能称为原始 63 项实验全部完成。
还应记录每个模型已完成的测试案例数和训练种子数。

完整评估模型的指标才进入正式统计曲线；未完成案例的图片是进度预览。
多种子曲线中的误差条表示训练种子之间的样本标准差，不是单案例不确定度。
流场相对 L2、轨迹 RMSE 和训练期间的噪声预测验证损失是不同指标，不能相互替代。

## 预算用完或超时后的续跑

90 分钟 Slurm 上限对应约 85 分钟的评估预算，预留 5 分钟用于环境启动、
结果记录和图片。评估会在案例之间检查预算，预算不足时保留已完成案例并提前退出，
退出码为 75，结果标为未完成。Slurm 可能显示 `FAILED` / `75:0`；这类有明确
`time_budget_exhausted` 记录的退出表示需续跑，不等同于模型计算报错。

单个案例仍可能跨过剩余预算并撞上 Slurm 硬时限，此时可显示 `TIMEOUT`。
强制结束还可能使磁盘上的最后状态停在 `running`，应结合 `sacct` 判断实际作业状态。
不要只看 `results.json` 或队列中是否消失。

先确认对应子任务已结束，查看日志和 `evaluation/summary.json`，然后仅续跑未完成的
评估编号。以下假设需要续跑的是评估任务 0、5，按实际结果修改：

```bash
cd ~/projects/project_code/project
source configs/deltaai-evaluation.env.example
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
FLOW3D_ARRAY_TASKS=0,5 bash scripts/submit.sh matrix-resume --plan "$EVAL_PLAN"
```

续跑复用原评估计划、原代码快照和采样协议，跳过验证通过的完整案例；
未完成案例重新计算。不要再次执行 `matrix-evaluate --training-tasks completed`
来恢复旧评估，那会创建新目录并重复计算。也不要把仍在运行或已经完成的任务加入续跑。

## 取回图片和报告

指标和图片属于实验结果，可以单独下载；源码仍通过 GitHub 同步。
先用报告命令生成跨任务汇总，再在本地 Windows 终端打开 SFTP：

```text
mkdir D:\2D\deliverables
sftp zguan2@dtai-login.delta.ncsa.illinois.edu
```

进入 `sftp>` 后，把下面路径中的计划目录替换为实际评估计划目录：

```text
lcd D:/2D/deliverables
get -r /work/hdd/biup/zguan2/results/matrices/实际评估计划目录/report gh200_meeting_report
bye
```

也可以单独下载某一任务的 `evaluation/figures`，给每个模型指定不同本地目录名，
避免多个同名 `figures` 目录相互覆盖。无需下载大量 posterior NPZ 或全部训练权重
就能查看已生成的图片和汇总指标。如果 SSH/SFTP 网络仍不稳定，服务器结果仍保留，
恢复连接后再下载即可。
