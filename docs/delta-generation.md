# Delta 生成流场数据并接着训练（中文）

本流程适用于 zguan2 于 2026-10-01 确认的环境。代码在本地修改并经 GitHub 同步，
数据生成和训练都通过 Slurm。数据直接写入 HPC，不依赖从国内上传约 455 MB 的数据包。
可以让 Delta 同一个 A100 作业在生成完成后自动训练，也可单独在 DeltaAI 训练。

| 用途 | 系统与架构 | Slurm 账户 | 分区 / GPU |
| --- | --- | --- | --- |
| Taichi-LBM3D 数据生成 | Delta / x86_64 | biup-delta-gpu | gpuA100x4 / 1 × A100 |
| diffusion 训练 | DeltaAI / aarch64 | biup-dtai-gh | ghx4 / 1 × GH200 120GB |

ACCESS 项目编号仍是 `PHY260443`，实际 `--account` 使用表中的站点计费账户。
`gpuA100x4` 是每节点有四张卡的分区名称；脚本只申请其中一张。

[NCSA 文档](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html)
确认 `/work/hdd` 和 `/work/nvme` 在 Delta、DeltaAI 间共享，但两个系统的 HOME 分开。
已验证数据目录为 `/work/hdd/biup/zguan2/datasets`。环境和代码分别在各自系统 HOME 安装。
如果提示符是 `gh-login...`，当前位于 DeltaAI，无法使用 Delta HOME 中已安装的
生成环境。回到 `dt-login...` 的 Delta 窗口即可；不需要在 DeltaAI 重装求解器。

## 1. 在 Delta 登录节点下载代码、准备独立环境

使用 ACCESS 门户提供的 Delta 登录地址；官方推荐地址见
[Delta 登录说明](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)。
下面的命令全部在 `zguan2@dt-login...` 的服务器终端执行，不在 Windows CMD 执行。

首次在 Delta 克隆：

```bash
mkdir -p ~/projects
git clone https://github.com/guanzhaoyang86-lab/flow3d.git ~/projects/project_code
cd ~/projects/project_code
```

如果 Delta 已有这个目录，则改为：

```bash
cd ~/projects/project_code
git pull --ff-only
```

安装环境和固定版本的上游求解器：

```bash
bash project/scripts/setup_delta_generation.sh
```

脚本加载 `miniforge3-python`，在 `~/envs/flow3d-lbm3d` 创建 Python 3.11 环境，
安装 PyTorch 2.5.1 CUDA 12.4、Taichi 1.7.4，以及 `requirements-generation.txt` 固定的依赖。
它只导入包检查版本，不在登录节点启动模拟，也不修改站点 base 环境。
如果下载中断，可以重跑该脚本；遇到 module 加载错误，先执行 `module spider miniforge3-python`
核对站点加载要求。出现其他错误时停在该步骤，保留错误输出。

选择 Python 3.11 的依据是 [Taichi 1.7.4 发布文件](https://pypi.org/project/taichi/1.7.4/#files)：
提供 Linux x86_64 的 Python 3.10–3.12 wheel，没有 Python 3.13 或 Linux aarch64 wheel。
PyTorch 安装使用[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)中的 cu124 索引。
实际 GPU 可用性由下一步 Slurm 作业验证。

上游仓库为 `https://github.com/yjhp1016/taichi_LBM3D`，固定提交
`fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c`，保存在 `~/projects/taichi_LBM3D`。
保留该检出目录和环境，不在作业运行期间修改它们。

## 2. 先提交 3 个真实流场样本

安装成功后：

```bash
cd ~/projects/project_code/project
source configs/delta-generation.env.example
bash scripts/submit.sh generate-pilot
```

申请 1 GPU、8 CPU、32 GB 内存，最长 30 分钟。提交脚本先 `git pull --ff-only`，
冻结完整仓库快照，再提交作业；GitHub 不可达时停止提交，计算节点不安装依赖或访问网络。

将下面的 `JOB_ID` 换成 `Submitted batch job ...` 返回的数字：

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
tail -n 100 /work/hdd/biup/$USER/logs/JOB_ID.out
tail -n 100 /work/hdd/biup/$USER/logs/JOB_ID.err
```

成功应同时满足：Slurm `COMPLETED / 0:0`；日志显示 Taichi CUDA 后端；
最后出现 `Completed. Manifest: .../manifest.json`；同目录有 `COMPLETE.json`。
该试运行生成真实 CFD 样本，但三个样本只用于验证流程，不足以评估模型泛化或论文结果。

数据目录形如：

```text
/work/hdd/biup/zguan2/datasets/<UTC时间>_lbm3d-pilot_<作业号>_<随机标识>/
  case_00000.npz ... case_00002.npz
  collection.json
  generation.json
  manifest.json
  SHA256SUMS
  COMPLETE.json
  attempts/<作业号>_<标识>/
    environment.json
    pip-freeze.txt
    command.txt
    log.txt
    results.json
```

## 3. 验证成功后，生成 1000 个样本并自动训练

```bash
cd ~/projects/project_code/project
source configs/delta-generation.env.example
export FLOW3D_TIME=48:00:00
bash scripts/submit.sh generate-full --train-epochs 1
```

同一 Slurm 作业会顺序完成：1000 个真实样本生成 → 数据校验和划分 → A100 上训练 1 个 epoch。
训练使用刚生成的 manifest，batch size 1、每个样本抽取 2 个观测粒子、随机种子 31，
其他参数沿用真实 diffusion 训练入口的默认值。160 是数据中保存的粒子数，2 是本轮模型输入粒子数。
首轮 1 epoch 用于验证完整数据链路；需要更多轮次时修改 `--train-epochs`。
只生成数据则省略该参数。失败的数据生成不会进入训练，也不需要在作业中重新申请 GPU。

48 小时是生成和训练合计的最长时限，不是预计耗时；先用 pilot 的实际耗时估算。作业成功提交后，关闭 SSH
不影响 Slurm 继续执行。取消使用 `scancel JOB_ID`。
每次全新提交生成独立目录，不覆盖已有数据。三个 pilot 样本使用独立随机计划，不拼入正式集。

参数与现有本地集合保持一致：随机计划种子 20260918；32³ 网格；lid speed 0.03–0.06；
上游 `niu` 参数 0.12–0.21；预热 600–1200 步；160 粒子；20 个观测时刻；
时长 120；三个投影视图 xz/xy/yz；积分子步数 4。
`niu` 原样传入上游求解器，不声称已独立验证其雷诺数解释。

流程检查所有 NPZ 的有限数值、上游版本和 CUDA 后端，再由已有 manifest 工具按独立
物理流场分组划分 train/validation/test（1000 个独立组为 798/101/101），防止跨集合泄漏。
保存实际数据的 SHA256 校验和及数据版本。CPU 与 CUDA 计算可能有浮点差异，
这批数据是新的服务器版本，不保证与本地数据包逐字节相同。
生成通过只证明数据契约与流程检查通过；物理收敛和科研适用性仍需研究者评估。

自动训练时，日志会先出现 `Data generation validated. Starting follow-up training`，
最后出现 `Pipeline completed: generation and training`。还应确认 `sacct` 为 `COMPLETED / 0:0`。
数据目录的 `COMPLETE.json` 只表示生成阶段完成，整个流程状态在
`attempts/<作业号>_<标识>/pipeline.json`。
训练的完整实验记录位于 `/work/hdd/biup/$USER/results/experiments/`，
权重位于 `/work/hdd/biup/$USER/checkpoints/`，具体路径会打印在同一作业日志中。

若生成成功但训练失败，数据仍然保留。可用 `diffusion-train --manifest 实际路径 -- ...`
单独重试训练，无需重新生成 1000 个样本。

## 4. 超时或失败后的续跑

先用 `sacct` 确认原作业已结束，检查失败原因。未生成 `COMPLETE.json` 的目录可在
同一代码提交、包版本、GPU 型号和参数下恢复；下面把路径换成日志中的实际 Dataset 路径：

```bash
source configs/delta-generation.env.example
export FLOW3D_TIME=48:00:00
bash scripts/submit.sh generate-full --resume /work/hdd/biup/$USER/datasets/实际未完成目录 --train-epochs 1
```

pilot 续跑使用 `generate-pilot --resume ...`。
已完成样本经过参数校验后跳过；新增样本采用临时文件加原子替换，避免作业中断留下
半个 NPZ。并发写同一目录会被文件锁拒绝。代码或依赖改变时应创建新集合。
被强制终止时 `results.json` 可能停留在 running，实际作业状态以 `sacct` 为准。

## 5. 可选：在 DeltaAI 读取共享数据训练

需要换到 GH200 训练时，记下成功生成的数据 manifest 绝对路径。在 DeltaAI 的 SSH 终端执行：

```bash
cd ~/projects/project_code
git pull --ff-only
cd project
source configs/deltaai.env.example
test -f /work/hdd/biup/$USER/datasets/实际完整数据目录/COMPLETE.json
bash scripts/submit.sh diffusion-train \
  --manifest /work/hdd/biup/$USER/datasets/实际完整数据目录/manifest.json \
  -- --epochs 1 --batch-size 1 --num-workers 0 --seed 31
```

`test` 返回非零时先检查路径和生成状态，不提交训练。
训练用 DeltaAI 自己的 PyTorch 2.10.0 模块，不能使用 Delta HOME 中的 x86_64 环境。
后续实验记录、checkpoint 和日志位置见 [DeltaAI 说明](deltaai.md)。
