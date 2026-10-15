# GH200 小规模扩散采样诊断

在 DeltaAI 的 `gh-login...` 登录节点、仓库根目录执行以下整段。
代码必须先经 GitHub 同步；提交器检查干净的 Git 工作区并冻结完整提交。
不需要另建环境或手动加载模块，作业会加载站点 ARM PyTorch 2.10.0 的 `base` 环境。

```bash
(set -euo pipefail; git pull --ff-only; bash project/scripts/submit_diagnosis.sh)
```

这条命令只提交一个独立作业：1 张 GH200、8 CPU、64G 内存，默认时限 30 分钟。
不创建数组，不继承旧作业依赖或 hold，也不重训模型。输入为
`/work/hdd/biup/$USER` 下的原 manifest
`datasets/2026-10-01_115342_lbm3d-full_22594936_7966ccef/manifest.json`，
以及 `checkpoints/2026-10-03_074635_matrix-train_51493784/` 下
`dit3d_N002_s31/best.pt` 和 `tensor-dit_N002_s31/best.pt`。
固定 validation indices `0,1,2`，50 个采样步，种子 47，16 个未观测探针粒子。
此诊断用于定位采样故障，不能代替全测试集评估。

每个模型、每个案例做以下五组对照，复用同一初始噪声；前三组还复用同一组随机更新噪声：

| 条件 | CFG | eta | 用途 |
|---|---:|---:|---|
| 正确轨迹 | 1.5 | 1 | 原评估参数对照 |
| 全零条件向量 | 1.5 | 1 | 检查条件是否影响结果 |
| 另一验证流场的轨迹 | 1.5 | 1 | 检查正确条件与错误条件是否有区别 |
| 正确轨迹 | 1 | 1 | 单独检查 CFG 影响 |
| 正确轨迹 | 1.5 | 0 | 单独检查随机更新影响 |

不改变 checkpoint 的训练噪声日程、归一化、EMA 权重或截断设置。另用真实验证场加噪，
在 t=999、979、500、0 检查噪声预测；这些真值输入仅用于明确标注的诊断，不能当作生成结果。
每组只生成一个样本，不能据此宣布某个参数组合在完整测试集更优。

`diagnosis/summary.json` 记录最终指标、模型参数、权重哈希、环境和运行状态。
各 `checkpoint_*/validation_*/` 内有 `sampling_trace.png`；各对照子目录内有
`trace.jsonl` 和 `result.json`，失败时保留 `failure.json`。
Tucker trace 同时记录核心、三个因子的幅度与正交残差，以及解码后的速度场幅度。
“预测场内部范数超过真值 10 倍”仅为定位用的描述性标记，不能单独证明模型故障；
噪声状态也会记录解码幅度，但不会按这个阈值判为异常。

提交前可在本地或服务器检查命令，dry-run 不要求数据、模块、GPU 或 Slurm，也不写目录：

```bash
bash project/scripts/submit_diagnosis.sh --dry-run
```

脚本支持 `--manifest PATH`、重复的 `--checkpoint PATH`、`--indices 0,1,2`、
`--sampling-steps 50`、`--seed 47`、`--num-probe-particles 16` 和 `--time 00:30:00`。
指定 checkpoint 会替换两个默认权重；例如只测一个模型可传一次。
如确需更长时限可用 `--time 01:00:00`，最多 1 小时。时限是申请上限，
并非已实测的完成时间保证。全部输入文件只读，输出总是写入新建的唯一目录。

记录提交输出中的作业号和结果路径。每次运行目录为：

```text
/work/hdd/biup/$USER/results/sampling-diagnosis/时间戳.随机串/
    job_id.txt                 实际 Slurm 作业号
    commit.txt                 冻结的源码提交
    submit_command.txt         实际提交命令
    diagnostic_command.txt     诊断参数和输入路径
    作业号.out / 作业号.err      日志
    exit_code.txt              作业正常退出时的退出码
    diagnosis/                 诊断结果和逐步 trace
```

用实际作业号查询：

```bash
squeue -j 作业号
sacct -j 作业号 --format=JobID,State,ExitCode,Elapsed
```

已完成的诊断记录会逐项保存；超时或强制终止后应结合 `sacct` 判断完成状态，
不能因存在图片或 trace 就称整个诊断成功。`exit_code.txt` 在强制终止时可能缺失。
只有 payload 启动前的已知 interconnect 错误会在原 allocation 内最多额外重试 3 次；
运行中的模型错误不会重跑。启动记录位于存储根目录的 `logs/launch-作业号-*/`。

本地修改和 dry-run 均不会提交服务器作业；只有服务器返回实际 `Submitted batch job` 后才算提交成功。
