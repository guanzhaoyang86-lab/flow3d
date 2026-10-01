#!/bin/bash -l
#SBATCH --job-name=flow3d_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --no-requeue

# 账户、分区、输出路径和快照由 submit.sh 提供。
set -euo pipefail
[[ -n "${FLOW3D_CODE_DIR:-}" ]] || { echo '请通过 scripts/submit.sh 提交' >&2; exit 1; }
source "$FLOW3D_CODE_DIR/scripts/runtime.sh"
[[ "$FLOW3D_GPUS" = 1 && "$FLOW3D_VISIBLE_GPUS" = 1 ]] || flow3d_die '真实 diffusion 入口仅支持单 GPU'
# 当前真实科研模型没有 DDP；只启动一个进程。
srun --ntasks=1 --gpu-bind=none python ../scripts/run_hpc_diffusion.py "$@"
