#!/bin/bash
# 由 Slurm 模板 source；不执行 git pull 或安装依赖。
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'GPU 程序只能通过 Slurm 启动\n' >&2; exit 1; }
export FLOW3D_CODE_DIR="${FLOW3D_CODE_DIR:-${SLURM_SUBMIT_DIR:?}}"
cd -- "$FLOW3D_CODE_DIR"
source "$FLOW3D_CODE_DIR/scripts/common.sh"
[[ -r .flow3d-commit ]] || flow3d_die '请先运行 prepare_run.sh 或通过 submit.sh 创建提交快照'
snapshot_commit=$(cat .flow3d-commit)
[[ "$snapshot_commit" =~ ^[0-9a-f]{40,64}$ ]] || flow3d_die '无效的快照提交标识'
[[ -z "${FLOW3D_COMMIT:-}" || "$FLOW3D_COMMIT" = "$snapshot_commit" ]] || flow3d_die '快照与预期提交不一致'
export FLOW3D_COMMIT="$snapshot_commit"
flow3d_settings
[[ "${SLURM_JOB_NUM_NODES:-1}" = 1 ]] || flow3d_die '当前模板仅支持单节点多 GPU'
if [[ -n "${FLOW3D_MODULES:-}" ]]; then
    command -v module >/dev/null || flow3d_die '没有 module 命令，请检查登录 shell'
    read -r -a flow3d_modules <<< "$FLOW3D_MODULES"
    module load "${flow3d_modules[@]}"
fi
if [[ -n "${FLOW3D_CONDA_SH:-}" ]]; then
    source "$FLOW3D_CONDA_SH"
else
    command -v conda >/dev/null || flow3d_die '请设置 FLOW3D_CONDA_SH 或加载站点 Python 模块'
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$FLOW3D_CONDA_ENV"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$FLOW3D_ROOT/logs/.matplotlib-${SLURM_JOB_ID}"
printf 'commit=%s job=%s cluster=%s host=%s\n' "$FLOW3D_COMMIT" "$SLURM_JOB_ID" "$FLOW3D_CLUSTER" "$(hostname)"
python -c 'import platform, torch; print(platform.machine(), platform.python_version(), torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable: verify architecture / PyTorch / allocation"; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
gpu_count=$(python -c 'import torch; print(torch.cuda.device_count())')
export FLOW3D_VISIBLE_GPUS="$gpu_count"
if command -v module >/dev/null; then module list 2>&1; fi
