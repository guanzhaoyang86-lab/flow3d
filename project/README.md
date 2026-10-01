# 从稀疏投影粒子轨迹生成重建三维流场

**Generative Reconstruction of Three-Dimensional Flow Fields from Sparse Projected Particle Trajectories**

本项目提供中文开发与计算工作流：在中国的本地电脑开发、调试和提交代码；GitHub 是唯一代码同步中心；NCSA ACCESS 的 Delta / DeltaAI 负责通过 Slurm 执行 GPU 实验。ACCESS 项目编号为 `PHY260443`。

**当前 zguan2 的 DeltaAI 实际分配请先读 [真实 diffusion 模型部署说明](../docs/deltaai.md)。**
服务器没有数据时，使用 [Delta 数据生成流程](../docs/delta-generation.md)：
`generate-pilot` 在 A100 上生成 3 个真实样本，验证后用 `generate-full` 生成 1000 个样本。
从完整仓库的 `project/` 目录执行 `source configs/deltaai.env.example` 后，
用 `bash scripts/submit.sh diffusion-smoke` 首测；`diffusion-train` / `diffusion-inference`
已连接根目录真实科研代码，支持单 GPU。下文 `train` / `inference` 是通用小模型模板的入口。

当前 `src/` 是可运行的合成数据训练与推理示例，用于验证环境、作业提交和实验记录，**尚未实现论文级生成式三维流场重建模型，也未包含真实数据集**。接入科学模型时，替换 `models/` 和 `datasets/`，保留运行记录与 Slurm 接口。

```text
本地电脑：编辑代码 → CPU 小测试 → git commit → git push
                                                ↓
                                           GitHub 仓库
                                                ↓
NCSA 登录节点：SSH → git pull --ff-only → 固定代码快照 → sbatch
                                                            ↓
                               Delta A100 / DeltaAI H100：训练与推理
                                                            ↓
                                         HPC 工作存储：权重、日志、结果
```

## 1. 项目结构

```text
project/
├── README.md
├── requirements.txt             # 本地完整 Python 依赖
├── requirements-common.txt      # 不主动安装 PyTorch 的通用依赖
├── environment.yml              # Python 3.11 的可移植环境规格
├── .gitignore
├── .gitattributes               # 保持 Linux 脚本使用 LF 换行
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── models/
│   ├── datasets/
│   └── utils/
├── configs/
│   ├── train.yaml
│   ├── model.yaml
│   └── hpc.env.example          # 集群配置模板，个人副本放在仓库外
├── scripts/
│   ├── common.sh
│   ├── prepare_run.sh          # 同步 GitHub 并生成提交快照
│   ├── submit.sh               # 推荐的 Slurm 提交入口
│   ├── runtime.sh              # 运行环境与单节点 GPU 启动
│   ├── train.slurm
│   └── inference.slurm
├── tests/                      # CPU 与提交逻辑的验证代码
├── data/                       # 本地占位，Git 忽略
├── checkpoints/                # 权重目录占位，Git 忽略
├── logs/                       # 本地日志，Git 忽略
└── experiments/                # 本地实验记录；运行产物不提交
```

Git 不保存空目录。克隆后可执行 `mkdir -p data checkpoints logs experiments`；本地 PowerShell 可执行 `New-Item -ItemType Directory -Force data,checkpoints,logs,experiments`。本地实验运行会按需创建自己的 `experiments/` 子目录。

仓库只提交源代码、配置、文档和环境文件。`.gitignore` 排除数据、权重、运行日志、实验产物、`__pycache__/`、`*.pyc`、`.vscode/`、`.idea/`、`*.pt`、`*.pth`、`*.ckpt`。添加 `.gitignore` 不会自动取消已经跟踪的文件；首次提交前用 `git status --short` 检查暂存区。

## 2. NCSA 配置需要区分的事项

以下内容于 **2026-10-01** 根据 NCSA 官方文档核对；实际分配、账户、模块和存储权限以登录后查询为准。

| 项目 | Delta GPU | DeltaAI |
| --- | --- | --- |
| 目标 GPU | A100 | GH200 节点中的 H100 GPU |
| 常用分区 | `gpuA100x4` | `ghx4` |
| CPU 架构 | x86_64 | ARM / aarch64 |
| 本模板 GPU 请求 | `--gpus-per-node=1` | `--gpus-per-node=1` |
| 示例最长墙钟时间 | `48:00:00` | `48:00:00` |
| 运行数据路径 | 已授权的 `/work/hdd/<本地项目代码>/$USER` 等 | `/work/hdd/<本地项目代码>/$USER` 或获批的 `/work/nvme/...` |

分区和时限来自 [Delta 作业说明](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html) 与 [DeltaAI 作业说明](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/running-jobs.html)。DeltaAI 使用 Grace ARM CPU 与 H100 GPU 组成的 GH200，因此不能直接复制本地 Windows 或 Delta x86_64 的 Python 环境；见 [DeltaAI 架构](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html)。

`PHY260443` 是 ACCESS 项目编号。本模板保留它作为默认 Slurm 账户，但 NCSA 的实际计费账户和存储目录代码可能与 ACCESS 编号不同。登录后执行 `accounts`，找到 `PHY260443` 对应资源的账户，并设置 `FLOW3D_ACCOUNT`；不要自行猜测账户后缀。见 [DeltaAI 账户说明](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/job-accounting.html)。

`/scratch/$USER` 是原始需求中的抽象示例，不能当作两个集群的统一真实路径。Delta 文档目前将 `/work/hdd` 列为工作存储；已有 `$SCRATCH` 或历史 `/scratch/<项目代码>/$USER` 路径应先确认可用。**DeltaAI 不提供 `/scratch` 文件系统。** 本项目使用显式设置的 `FLOW3D_ROOT` 实现相同的工作存储布局，详见 [Delta 存储](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html) 与 [DeltaAI 存储](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/data-mgmt.html)。

## 3. 本地开发与 GitHub 初始化

本目录支持两种布局：单独作为仓库，或位于完整科研仓库的 `project/` 子目录。
本次发布的 `guanzhaoyang86-lab/flow3d` 使用第二种布局：克隆整个仓库后，先
`cd project_code/project` 再执行本文模板命令；日常 `git pull` 会同步整个仓库。
本文后续示例中的 `~/projects/project_code` 是独立模板的示例路径，在本仓库中
应相应使用 `~/projects/project_code/project`。快照保留完整科研仓库，并从快照内
的 `project/` 启动作业；历史复现的 `FLOW3D_CODE_DIR` 也应包含末尾的 `/project`。

已有本仓库的用户无需再次 `git init`。以下初始化命令只用于将模板单独发布为另一个仓库；
如需这样做，先复制到独立目录以避免创建嵌套仓库。`<repository_url>` 应替换为实际地址。

```bash
cd /path/to/project
git init -b main
git add .
git status --short
git commit -m "Initialize flow3d local-to-HPC workflow"
git remote add origin <repository_url>
git push -u origin main
```

本地可使用 Miniforge / Conda 创建 Python 3.11 环境：

```bash
conda create -n flow3d python=3.11
conda activate flow3d
pip install -r requirements.txt
python src/train.py --config configs/train.yaml --device cpu --smoke-test
```

也可选择 `conda env create -f environment.yml`，然后 `conda activate flow3d`。两种方式选择一种即可。`environment.yml` 包含 Python 3.11、PyTorch、NumPy、SciPy、Matplotlib、pandas、tqdm 和 PyYAML；它描述依赖范围，不是跨架构、跨 CUDA 平台的二进制锁文件。

日常开发流程：

```bash
git pull --ff-only

# 在本地修改 src/、configs/ 和文档
python src/train.py --config configs/train.yaml --device cpu --smoke-test

git add .
git status --short
git commit -m "update model"
git push
```

CPU 小测试会生成独立实验目录；结束后终端显示保存位置。推理时使用该次训练实际生成的权重：

```bash
python src/inference.py --config configs/train.yaml --device cpu \
  --checkpoint /absolute/path/to/last.pt --smoke-test
```

离线回归验证可执行 `python -m unittest discover -s tests -p "test_*.py"`。提交测试使用临时 Git 仓库与模拟 Slurm，不连接 NCSA；Windows 需安装 Git for Windows 的 Bash。它不能替代真实集群上的 GPU 小测试。

HPC 不承担编辑代码、解决合并冲突或创建开发提交的工作。配置修改也在本地提交到 GitHub，再由 HPC 拉取。私钥、令牌、密码、个人集群配置均不放入仓库。

## 4. 首次登录 HPC 与目录准备

登录节点按 ACCESS 门户给出的信息填写，不在代码中固定主机名：

```bash
ssh username@<NCSA_LOGIN_NODE>
```

登录后首次克隆：

```bash
mkdir -p ~/projects
cd ~/projects
git clone <repository_url> project_code
cd project_code
git remote -v
git branch -vv
accounts
```

目录按用途分开：

```text
HOME
~/projects/project_code/           # GitHub 仓库工作副本
~/projects/flow3d_runs/             # 每次提交保存的只读源码快照
~/.config/flow3d/hpc.env            # 本人的集群参数
~/.conda/envs/...                  # 个人环境，小型文件

FLOW3D_ROOT                        # 真实分配的工作存储，不是 HOME
├── datasets/
├── checkpoints/
├── logs/
└── results/
    └── experiments/
```

例如，若实际分配的工作目录是 `/work/hdd/abcd/$USER`，则把它设置为 `FLOW3D_ROOT`；其中 `abcd` 必须替换为查询得到的本地项目代码。不要直接创建 `/scratch/$USER` 来绕过权限或路径错误。HOME 只放代码、环境和小文件，实验读写留在工作存储；DeltaAI 不在 HOME 建立指向 `/work` 的符号链接。

将集群配置放在仓库外，避免 HPC 工作副本出现本地修改：

```bash
mkdir -p ~/.config/flow3d
cp configs/hpc.env.example ~/.config/flow3d/hpc.env
```

填写其中的 `FLOW3D_ROOT`、账户和环境信息。个人站点参数不属于科学代码开发；可在 HPC 调整这个仓库外文件。典型 Delta 参数如下：

```bash
export FLOW3D_CLUSTER=delta
export FLOW3D_ACCOUNT=PHY260443
export FLOW3D_ROOT="/work/hdd/<本地项目代码>/$USER"
export FLOW3D_PARTITION=gpuA100x4
export FLOW3D_GPUS=1
export FLOW3D_TIME=48:00:00
export FLOW3D_CPUS=8
export FLOW3D_MEM=32G
export FLOW3D_CONDA_ENV=flow3d
```

DeltaAI 至少替换以下参数，并使用为该机型安装的环境：

```bash
export FLOW3D_CLUSTER=deltaai
export FLOW3D_ROOT="/work/hdd/<本地项目代码>/$USER"
export FLOW3D_PARTITION=ghx4
```

`FLOW3D_ACCOUNT=PHY260443` 仅在 `accounts` 确认它是有效的 Slurm 账户时直接使用，否则填写对应的 NCSA 账户。配置文件只是可信的本地 Bash 文件，使用前先检查内容：

```bash
source ~/.config/flow3d/hpc.env
mkdir -p "$FLOW3D_ROOT"/{datasets,checkpoints,logs,results}
test -w "$FLOW3D_ROOT"
```

Slurm 在启动脚本之前就打开输出文件，所以日志目录必须在 `sbatch` 前存在。推荐的 `submit.sh` 会负责这一步。

## 5. HPC Python / CUDA 环境

本地 CPU 环境和两类 HPC 环境分别安装。`module load cuda` 本身不会把 CPU 版 PyTorch 变成 GPU 版，也不保证与任意 PyTorch wheel 兼容。

### Delta A100

先查询站点模块，再创建 Python 3.11 环境。NCSA 的 Red Hat 9 升级说明列出的驱动为 570.148.08、基础 CUDA 为 12.8，并提供 `miniforge3-python` 模块；下面据此给出 PyTorch 2.8.0 / CUDA 12.8 的安装示例。实际使用前仍需确认模块可用，见 [Delta 软件更新说明](https://docs.ncsa.illinois.edu/systems/delta/en/latest/whats_new.html)。

```bash
module spider miniforge3-python
module load miniforge3-python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --prefix "$HOME/.conda/envs/flow3d" python=3.11 pip
conda activate "$HOME/.conda/envs/flow3d"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-common.txt
python -m pip check
```

该 Python 3.11 / Linux x86_64 包可在 [PyTorch 官方 CUDA 12.8 索引](https://download.pytorch.org/whl/cu128/torch/) 核对。如站点环境变化，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择兼容组合。不要照搬本地 Windows 包。GPU 可用性通过后面的 Slurm 小测试验证，登录节点不做 GPU 计算。

对应个人配置中设置：

```bash
export FLOW3D_MODULES="miniforge3-python"
export FLOW3D_CONDA_ENV="$HOME/.conda/envs/flow3d"
export FLOW3D_CONDA_SH=""
```

查询到可用的具体模块版本后，把不带版本的 `miniforge3-python` 改为该版本，以免站点默认版本切换影响环境。

### DeltaAI H100 / GH200

DeltaAI 使用 aarch64。若保持本项目的 Python 3.11 要求，可以用站点模块提供 Conda，再在 HOME 创建独立环境。以下示例采用 NCSA 文档推荐的 CUDA 13.0 稳定包索引，并固定已核实提供 Python 3.11 / aarch64 wheel 的 `torch==2.10.0`：

```bash
module spider python/miniforge3_pytorch
module load python/miniforge3_pytorch/2.10.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --prefix "$HOME/.conda/envs/flow3d-deltaai" python=3.11 pip
conda activate "$HOME/.conda/envs/flow3d-deltaai"
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-common.txt
python -m pip check
python -c "import platform, torch; print(platform.machine(), platform.python_version(), torch.__version__, torch.version.cuda)"
```

这里的模块版本只用于初始化站点 Conda；自建环境拥有独立 PyTorch。实际安装前确认该版本仍可加载，并在 Slurm 小测试中验证 GPU。配方依据是 [NCSA PyTorch 自建环境说明](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html) 和 [PyTorch 官方 CUDA 13.0 wheel 索引](https://download.pytorch.org/whl/cu130/torch/)。

对应个人配置设置为：

```bash
export FLOW3D_MODULES="python/miniforge3_pytorch/2.10.0"
export FLOW3D_CONDA_ENV="$HOME/.conda/envs/flow3d-deltaai"
export FLOW3D_CONDA_SH=""
```

另一种选择是直接使用站点已经验证的 PyTorch 模块环境。官方当前列出的 2.10 / 2.11 / 2.12 模块使用 Python 3.12；选择它们意味着明确接受 HPC Python 版本与本地 3.11 不同。不要把克隆站点环境误称为 Python 3.11，也不要直接修改共享模块环境；环境定制参见 [DeltaAI 软件说明](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html)。

`runtime.sh` 在作业中先加载 `FLOW3D_MODULES`，再初始化 Conda 并激活 `FLOW3D_CONDA_ENV`。若 Conda 不在 `PATH`，设置 `FLOW3D_CONDA_SH` 为本人真实的 `conda.sh` 绝对路径。正式作业中不安装依赖。保留模块的确切版本和环境包清单，以支持复现实验。

## 6. 每次实验：同步、固定源码、提交

推荐在 HPC 登录节点执行：

```bash
cd ~/projects/project_code
source ~/.config/flow3d/hpc.env
git pull --ff-only
bash scripts/submit.sh train --config configs/train.yaml --smoke-test
```

小测试成功后提交完整配置：

```bash
bash scripts/submit.sh train --config configs/train.yaml
```

`submit.sh` 自身还会调用 `prepare_run.sh` 拉取，因此显式 `git pull` 只是便于人工检查。提交流程会：

1. 确认当前工作副本干净、跟踪 GitHub 上游分支，没有未推送的本地提交。
2. 使用 `git pull --ff-only` 同步；网络错误或版本分叉时停止，不提交旧代码。
3. 把该提交导出到 HOME 下唯一的只读源码快照，记录完整 Git SHA。
4. 预先创建工作存储目录，将绝对日志路径和资源参数传给 `sbatch`。
5. 作业从这个快照启动，产物写入 `FLOW3D_ROOT`。

这里的“始终运行 GitHub 最新版本”指**提交时成功同步的上游分支版本**。正在排队或运行的作业固定使用当时的提交；后续 `git pull` 不会改变它的源码。计算节点不需要访问 GitHub，也不在启动时拉取代码。通过 `FLOW3D_SNAPSHOT_ROOT` 可更改快照目录，保留待运行及仍需复现的快照。

在网络不稳定时，先保证本地 `git push` 成功，再登录 HPC 提交。`sbatch` 返回作业编号后，SSH 断开通常不会结束已提交作业。如果连接在返回编号前断开，先检查 `squeue`、`sacct` 和日志，避免重复提交同一个实验。

### 修改 GPU 数量、分区与时限

```bash
# Delta A100：单 GPU、两小时
FLOW3D_GPUS=1 FLOW3D_TIME=02:00:00 \
  bash scripts/submit.sh train --config configs/train.yaml

# 在 DeltaAI 登录节点、已加载 DeltaAI 个人配置：四 GPU、十二小时
FLOW3D_CLUSTER=deltaai FLOW3D_PARTITION=ghx4 \
FLOW3D_GPUS=4 FLOW3D_TIME=12:00:00 \
  bash scripts/submit.sh train --config configs/train.yaml
```

改变资源变量不会切换远程集群；必须登录对应的 Delta / DeltaAI。当前示例支持单节点多 GPU，使用 `torchrun` 启动 DDP，每个 GPU 一个进程；不提供多节点训练配置。GPU 数量增加时还需根据模型调整总 batch size、CPU 和内存。

GPU 型号通常由分区选择。以下是用户要求的 Slurm 语法示例，但只有站点 `sinfo` 显示对应 GRES 类型名时才能使用：

```bash
#SBATCH --gres=gpu:a100:1
# 或
#SBATCH --gres=gpu:h100:1
```

DeltaAI 的 H100 属于 GH200 平台，不能假定 Slurm 类型一定叫 `h100`。本模板默认使用指定分区加 `--gpus-per-node`，避免硬编码错误的 GPU 类型。只有确认类型名后，才可设置 `FLOW3D_GPU_TYPE=a100` 或 `h100`；包装脚本会生成 `--gpus-per-node=类型:数量`。不要同时叠加互相冲突的 `--gres` 和 `--gpus-per-node` 请求。可用 `sinfo -o '%P %G %l'` 查看分区、GPU 资源与时限。

### 直接使用 Slurm 模板

`scripts/train.slurm` 与 `scripts/inference.slurm` 均可用于 `sbatch`。先准备快照，再显式覆盖站点账户、分区和输出路径：

```bash
cd ~/projects/project_code
source ~/.config/flow3d/hpc.env
FLOW3D_CODE_DIR="$(bash scripts/prepare_run.sh)" || exit 1
export FLOW3D_CODE_DIR
cd "$FLOW3D_CODE_DIR"
mkdir -p "$FLOW3D_ROOT/logs"
sbatch --account="$FLOW3D_ACCOUNT" \
  --partition="$FLOW3D_PARTITION" \
  --gpus-per-node="$FLOW3D_GPUS" \
  --time="$FLOW3D_TIME" \
  --cpus-per-task="$FLOW3D_CPUS" --mem="$FLOW3D_MEM" \
  --output="$FLOW3D_ROOT/logs/%j.out" \
  --error="$FLOW3D_ROOT/logs/%j.err" \
  scripts/train.slurm --config configs/train.yaml
```

其中最基本的提交语法是 `sbatch scripts/train.slurm`。只有模板中的账户、分区、日志目录和环境均适用于当前站点时才可省略覆盖参数。模板保留的 `/scratch/%u/logs/%j.out` 是需求中的格式示例，不是 DeltaAI 的有效默认目录。`#SBATCH` 不展开 `$USER` 等 shell 变量；`%u`、`%j` 是 Slurm 自己的替换符。推荐始终使用 `submit.sh` 自动传入真实路径。

### 推理

```bash
cd ~/projects/project_code
source ~/.config/flow3d/hpc.env
bash scripts/submit.sh inference --config configs/train.yaml \
  --checkpoint "$FLOW3D_ROOT/checkpoints/<训练实验编号>/last.pt"
```

推理模板只使用一个 GPU；若前面设置了四 GPU，应先 `export FLOW3D_GPUS=1`。训练和推理的 CUDA 模式均要求 Slurm 作业环境；不会在登录节点直接执行 GPU 任务。

## 7. 作业管理与查看结果

```bash
# 当前队列
squeue -u "$USER"

# 全部可见历史 / 指定作业详细状态
sacct
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES

# 取消指定作业
scancel JOB_ID

# 查看输出与错误日志
cat "$FLOW3D_ROOT/logs/JOB_ID.out"
cat "$FLOW3D_ROOT/logs/JOB_ID.err"
tail -f "$FLOW3D_ROOT/logs/JOB_ID.out"
```

只有当自己的真实存储路径确实是 `/scratch/$USER` 时，等价命令才是 `cat /scratch/$USER/logs/JOB_ID.out`。DeltaAI 使用配置的 `/work/...` 路径。

在 SSH 会话中查看结果：

```bash
ls "$FLOW3D_ROOT/results/experiments"
cat "$FLOW3D_ROOT/results/experiments/<实验编号>/results.json"
cat "$FLOW3D_ROOT/results/experiments/<实验编号>/log.txt"
ls "$FLOW3D_ROOT/checkpoints/<实验编号>"
```

GitHub 只同步代码、配置、文档和环境文件，所以这里不通过 Git 提交权重、数据、原始日志或整个实验目录，也不额外引入 `scp` / `rsync` 同步通道。大文件继续保存在 HPC 工作存储；可以阅读小型指标后，在本地写成研究笔记并作为文档提交到 GitHub。若以后需要把完整检查点下载到本地，需要先另行确定大文件传输和长期保存策略；当前工作流不包含这种传输。

真实数据集应已存在于授权的 HPC 工作存储，或由另一个已批准的数据准备流程生成。本模板不会从 GitHub 获取真实训练数据，也不会伪装成已经完成数据上传。工作存储不等于长期备份。

## 8. 实验记录与复现

本地实验写入项目的 `experiments/`。HPC 实验写入外部工作存储，避免运行时写入 HOME 的源码快照：

```text
$FLOW3D_ROOT/
├── checkpoints/
│   └── <实验编号>/
│       └── last.pt
├── logs/
│   ├── <SLURM_JOB_ID>.out
│   └── <SLURM_JOB_ID>.err
└── results/
    └── experiments/
        └── 2026-XX-XX_<实验名称>_<唯一标识>/
            ├── config.yaml
            ├── command.txt
            ├── results.json
            ├── checkpoint/    # 指向同一工作存储中的权重目录
            └── log.txt
```

每次运行保存解析后的实验配置、执行命令、模型参数、数据集版本、随机种子、GPU 名称、训练用时、Git 提交、Slurm 作业信息与软件环境记录。`environment.txt` 保存 `pip freeze` 清单；版本、资源进程数和提交命令保存在 `results.json`，站点模块列表保存在 Slurm 日志。推理额外生成 `predictions.npz`，并记录输入检查点的 SHA-256。唯一标识防止同名实验覆盖。失败的运行应结合 `log.txt`、Slurm stderr 和 `sacct` 判断；不能把有目录或有检查点等同于训练成功。

保存的 `config.yaml` 内联模型参数，可直接作为下次运行的 `--config`。本地权重位于 `experiments/<实验编号>/checkpoint/last.pt`；HPC 的 `checkpoint/` 是同一工作存储内的符号链接，若系统不允许创建链接，则以 `checkpoint_location.txt` 和 `results.json` 中的实际路径为准。

新增真实数据集时，在配置中维护不可变的数据版本，最好使用数据清单与校验和；同一个“版本名”不能对应后来被覆盖的数据。仅固定随机种子不足以保证 A100、H100、不同 CUDA / PyTorch 版本之间逐位一致。

复现需要同时保留：

- 完整 Git 提交与对应源码快照。
- 本次保存的配置、数据版本以及原始数据。
- PyTorch / CUDA / Python 版本、模块版本与依赖清单。
- 所需检查点、资源参数和执行命令。

日常 `submit.sh` 始终同步 GitHub 最新提交。复现历史提交属于明确的历史版本运行，应使用保存的只读快照，按上面的“直接使用 Slurm 模板”方法提交，并将 `--config` 指向那次保存的 `config.yaml`。例如：

```bash
source ~/.config/flow3d/hpc.env
export FLOW3D_CODE_DIR="$HOME/projects/flow3d_runs/<历史快照目录>"
cd "$FLOW3D_CODE_DIR"

sbatch --account="$FLOW3D_ACCOUNT" --partition="$FLOW3D_PARTITION" \
  --gpus-per-node="$FLOW3D_GPUS" --time="$FLOW3D_TIME" \
  --cpus-per-task="$FLOW3D_CPUS" --mem="$FLOW3D_MEM" \
  --output="$FLOW3D_ROOT/logs/%j.out" --error="$FLOW3D_ROOT/logs/%j.err" \
  scripts/train.slurm \
  --config "$FLOW3D_ROOT/results/experiments/<历史实验编号>/config.yaml"
```

不在可变的工作副本上直接 `git checkout` 后继续排队。历史复现不声称运行“最新版本”；它主动选择已记录的旧版本，并生成新的实验目录。当前示例的检查点用于推理，不将“读取权重”宣称为完整断点续训；完整续训还需要实现并恢复优化器、调度器、随机数与采样器状态。

## 9. 常见问题

| 现象 | 处理 |
| --- | --- |
| `Invalid account` / `Invalid account or account/partition combination` | 执行 `accounts`，核对 `PHY260443` 对应资源的本地账户和分区权限。 |
| 日志打不开、作业立即失败 | 检查 `FLOW3D_ROOT` 实际挂载与写权限，并在提交前建立 `logs/`。 |
| `git pull --ff-only` 失败 | 在本地处理分叉并推送；HPC 保持干净。网络失败时重新同步后提交。 |
| 拒绝提交：工作树有修改 | 把科学代码修改移回本地开发流程；个人集群配置放在仓库外。 |
| `conda: command not found` | 按站点配置加载模块或设置真实的 `conda.sh` 路径，再激活个人环境。 |
| DeltaAI 包不兼容 / PyTorch 没有 CUDA | 使用 ARM 架构的站点 GPU 软件栈；不要复制 Windows/x86 环境。 |
| 请求 H100 GRES 被拒绝 | 使用 `ghx4` 与通用 GPU 数量请求；以 `sinfo` 和官方文档中的名称为准。 |
| SSH 断开但不确定是否提交成功 | 先查 `squeue` / `sacct`；已经被 Slurm 接受的作业独立运行。 |

部署完成的判断顺序：本地 CPU 小测试通过 → GitHub 推送成功 → HPC 配置与环境就绪 → Slurm GPU 小测试完成 → 检查 `results.json`、检查点及 `sacct` 成功状态 → 提交正式实验。

## 10. 本地验证

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

发布前共 6 项测试通过：训练生成检查点、保存配置重放并得到相同 CPU 训练损失、检查点推理，以及提交链路中的最新提交同步、固定快照、参数传递和失败拒绝。额外验证了模板位于完整科研仓库的 `project/` 子目录时，仍能保留完整仓库快照并从正确子目录启动。提交流程使用本地 Git 仓库及模拟 `sbatch`，不会连接 HPC 或消耗配额；测试产物被 Git 忽略。

本次实测环境为 Windows、Python 3.12.4、PyTorch 2.5.1；已检查 Bash 语法与 Git 忽略规则。交付环境规格仍为要求的 Python 3.11，但尚未实测新建的 3.11 环境、真实 Slurm、A100/H100 或多 GPU。GitHub 发布不代表 HPC 已部署；尚未登录 NCSA 或提交真实 GPU 作业。首次部署请先完成文档中的 Slurm 小测试。
