#!/bin/bash -l
#SBATCH --job-name=flow3d_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue

set -euo pipefail
[[ -n "${FLOW3D_CODE_DIR:-}" ]] || { echo '请通过 scripts/submit.sh 提交' >&2; exit 1; }
source "$FLOW3D_CODE_DIR/scripts/runtime.sh"
[[ "$FLOW3D_CLUSTER" = deltaai && "$FLOW3D_GPUS" = 1 && "$FLOW3D_VISIBLE_GPUS" = 1 ]] || flow3d_die '矩阵任务使用 DeltaAI 单 GPU'
srun --ntasks=1 --gpu-bind=none python ../scripts/run_hpc_matrix.py run "$@"
