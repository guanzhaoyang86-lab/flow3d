#!/bin/bash -l
#SBATCH --job-name=flow3d_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --no-requeue

set -euo pipefail
SECONDS=0
[[ -n "${FLOW3D_CODE_DIR:-}" ]] || { echo '请通过 scripts/submit.sh 提交' >&2; exit 1; }
source "$FLOW3D_CODE_DIR/scripts/runtime.sh"
[[ "$FLOW3D_GPUS" = 1 && "$FLOW3D_VISIBLE_GPUS" = 1 ]] || flow3d_die '矩阵任务必须使用单 GPU'
# The scientific runner stays in the plan's original snapshot on resume.
# The separately frozen launcher may include newer Slurm startup fixes.
launcher_dir="${FLOW3D_LAUNCHER_DIR:-$FLOW3D_CODE_DIR}"
printf 'launcher_commit=%s\n' "${FLOW3D_LAUNCHER_COMMIT:-$FLOW3D_COMMIT}"
if [[ -n "${FLOW3D_EVALUATE_BUDGET_SECONDS:-}" ]]; then
    export FLOW3D_EVALUATE_BUDGET_SECONDS=$((FLOW3D_EVALUATE_BUDGET_SECONDS - SECONDS))
    (( FLOW3D_EVALUATE_BUDGET_SECONDS > 0 )) || exit 75
fi
python "$launcher_dir/scripts/retry_srun.py" -- \
    python "$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py" run "$@"
