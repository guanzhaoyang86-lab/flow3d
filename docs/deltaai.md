# DeltaAI：真实 diffusion 模型的环境验证与作业提交

本页针对 zguan2 已确认的分配（2026-10-01），从完整 `flow3d` 仓库执行。
代码在本地开发，通过 GitHub 仓库同步；当前仓库公开，HTTPS 拉取不需要令牌。服务器仅负责 Slurm 计算。

| 项目 | 已确认值 |
| --- | --- |
| ACCESS 项目 | PHY260443 |
| Slurm 计费账户 | biup-dtai-gh |
| 分区 | ghx4 |
| CPU / GPU | aarch64 / NVIDIA GH200 120GB |
| HDD 工作存储 | /work/hdd/biup/zguan2 |
| NVMe 工作存储 | /work/nvme/biup/zguan2 |
| 首测模块 | python/miniforge3_pytorch/2.10.0 |
| 首测申请 | 1 GPU、16 CPU、32G 内存、20 分钟 |

登录主机使用 ACCESS 门户提供、且你已经成功登录的 SSH 地址。

## 1. 登录节点只检查环境和提交任务

```bash
module load python/miniforge3_pytorch/2.10.0
conda activate base
python -c "import platform, torch, numpy; print('arch:', platform.machine()); print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('NumPy:', numpy.__version__)"
```

[NCSA PyTorch 官方说明](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/python/pytorch.html)
将该模块列为 PyTorch 2.10.0、CUDA 12.9、Python 3.12。实际值以命令输出和作业记录为准。
本仓库要求 Python >=3.10；原通用模板的 Python 3.11 环境仍可用于本地，HPC 首测采用站点预配置的 Python 3.12。
只使用站点 base，不向其执行 pip/conda 安装。以上命令仅导入包和查询版本，不启动 GPU 计算。
登录节点上 CUDA 是否可见不作为验收标准；GPU 检查由后续 Slurm 作业完成。

## 2. 同步 GitHub 仓库

首次克隆：

```bash
mkdir -p ~/projects
git clone https://github.com/guanzhaoyang86-lab/flow3d.git ~/projects/project_code
cd ~/projects/project_code/project
```

已经克隆时：

```bash
cd ~/projects/project_code
git pull --ff-only
cd project
```

这是私有仓库。HTTPS 认证使用你的 GitHub 用户名和访问令牌，或使用已配置的 SSH 密钥。
不使用 ACCESS 密码登录 GitHub，不把令牌写进命令、远端 URL 或仓库文件。

## 3. 先提交真实模型的小测试

在 `~/projects/project_code/project` 目录执行：

```bash
source configs/deltaai.env.example
bash scripts/submit.sh diffusion-smoke
```

提交脚本会创建工作目录，执行 `git pull --ff-only`，确认没有本地改动或未推送提交，
保存整个仓库的只读快照，然后提交作业。网络失败时不提交旧代码。
配置文件已填写实际账户和路径，无需编辑 Slurm 模板。小测试最长 20 分钟，通常实际运行更短。

小测试执行根目录的真实科研入口：

1. 检查 PyTorch 是否可见且能使用唯一分配的 GPU。
2. 在工作存储生成一个 `8³` 合成流场与稀疏轨迹，明确标记为 `smoke_test`。
3. 用 `scripts/train_sparse_track_diffusion.py` 训练 1 个 epoch，保存 checkpoint。
4. 用 `scripts/sample_sparse_track_diffusion.py` 载入 checkpoint 并推理。
5. 检查输出是否有限，记录参数、环境、Git 提交、数据 manifest 和耗时。

**这些合成数据仅用于环境验证，不构成科研结果。** 该步骤不需要 Taichi 或正式数据集。
真实模型目前只支持单 GPU；申请多 GPU 会在提交前被拒绝，避免启动互不协作的重复训练。

查询作业（将 JOB_ID 替换为提交返回的数字）：

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
tail -n 100 /work/hdd/biup/$USER/logs/JOB_ID.out
tail -n 100 /work/hdd/biup/$USER/logs/JOB_ID.err
```

验收条件：Slurm 为 `COMPLETED`、退出码为 `0:0`；实验 `results.json` 的 `status`
为 `completed`；训练设备为 `cuda`；存在 `best.pt`、`latest.pt` 和 `posterior.npz`。
成功后再扩大模型或训练时长。取消任务使用 `scancel JOB_ID`。

## 4. 正式数据与训练

先准备服务器可读的 NPZ 数据和 manifest。数据、checkpoint 不放入 GitHub。
数据可按 [Delta 生成数据流程](delta-generation.md) 直接生成到共享工作目录，
代码仍只经 GitHub 同步，无需从本地上传数据。该流程也支持在 Delta 的 A100 上自动接着训练；
需要使用 GH200 时再按本页提交 DeltaAI 作业。
不得直接引用 Windows 的 `D:\...` 路径；推荐 manifest 内使用相对该 manifest 的 POSIX 相对路径。
科研 manifest 必须按物理流场分组划分 train/validation/test，不能复用小测试的重叠数据划分。

Taichi 是可选的数据生成依赖，核心训练和推理不导入 Taichi。
[Taichi 1.7.4 发布文件](https://pypi.org/project/taichi/1.7.4/#files)没有 Linux aarch64 wheel，
因此不要在 DeltaAI 直接照搬本地的 `pip install -e '.[lbm3d]'`。
仓库中的 Windows 三种子数据生成总控脚本也不用于本页的 Linux 训练流程。

正式数据已放在指定目录且小测试通过后，可以执行下例。`manifest.json` 必须真实存在，
第一轮建议 1 个 epoch、1 个 GPU、20 分钟，确认数据链路后再按需要增加预算：

```bash
cd ~/projects/project_code/project
source configs/deltaai.env.example
bash scripts/submit.sh diffusion-train \
  --manifest /work/hdd/biup/$USER/datasets/manifest.json \
  -- --epochs 1 --batch-size 1 --num-workers 0 --seed 31
```

`--` 后面的参数交给真实训练脚本，模型大小、粒子数、损失项均可在那里设置。
设备固定为 CUDA，输出路径由作业生成，避免覆盖旧实验。增加时长可先执行
`export FLOW3D_TIME=04:00:00`；分区允许的最大时间以 `sinfo` 为准。

恢复训练使用 `-- --resume /work/hdd/biup/$USER/checkpoints/<RUN_ID>/latest.pt --epochs 100`，
同时保留原来的模型、数据和训练参数。`--epochs` 是最终总轮数。
恢复作业写入新的实验目录，原 checkpoint 保留。

推理：

```bash
bash scripts/submit.sh diffusion-inference \
  --manifest /work/hdd/biup/$USER/datasets/manifest.json \
  --checkpoint /work/hdd/biup/$USER/checkpoints/<RUN_ID>/best.pt \
  -- --num-samples 4 --sampling-steps 50 --seed 47
```

## 5. 存储与复现

```text
~/projects/project_code/                    GitHub 工作副本
~/projects/flow3d_runs/<COMMIT>.<SUFFIX>/     作业只读代码快照
/work/hdd/biup/zguan2/
├── datasets/                              正式数据
├── checkpoints/<RUN_ID>/                  best.pt / latest.pt / training_summary.json
├── logs/<JOB_ID>.out 和 <JOB_ID>.err         Slurm 日志
└── results/experiments/<RUN_ID>/
    ├── config.yaml                        完整解析参数，JSON 格式的合法 YAML
    ├── command.txt                        训练/推理的等价命令
    ├── environment.json                   Git、Python、PyTorch、CUDA、GPU、Slurm、模块
    ├── dataset.json                       manifest 内容、SHA256、解析后的数据路径
    ├── results.json                       状态、耗时、训练摘要/推理指标
    ├── log.txt                            子进程输出
    └── posterior.npz                      小测试/推理输出
```

保持原始数据不可变，使用 manifest 的版本字段/校验值管理数据版本；manifest 哈希本身不能检测
NPZ 内容被原地覆盖。复现实验应保存对应数据版本、代码快照、环境和完整参数。
已排队的作业运行固定快照，后续 `git pull` 不影响它。需要精确复跑历史版本时，
先恢复 `environment.json` 中记录的模块和集群配置，把 `FLOW3D_CODE_DIR` 设置为其中提交命令的
`--chdir` 快照路径，设置 `FLOW3D_COMMIT` 为该快照的 `.flow3d-commit` 内容，然后运行记录的
`sbatch` 命令。保留旧快照和原数据。默认提交入口总是同步当前 GitHub 分支，用于提交新实验。
相同随机种子不保证跨硬件/库版本位级一致。

硬性取消或超时可能使 `results.json` 停留在 `running`；最终作业状态以 `sacct` 为准。
结果首先在服务器上述目录查看，不会自动推送到 GitHub；如需本地完整结果，另行确定符合约束的取回方式。

本地仅验证 wrapper 的 CPU 路径：

```powershell
python scripts/run_hpc_diffusion.py smoke --device cpu --storage-root tmp/hpc-smoke
```

本地通过不代表 ARM/GH200 已通过验收，必须完成第 3 步的真实 Slurm 作业。
