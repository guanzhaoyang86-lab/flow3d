#!/bin/bash
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
mode="${1:-train}"
if (( $# )); then shift; fi
[[ "$mode" = train || "$mode" = inference ]] || flow3d_die '用法: bash scripts/submit.sh train|inference [Python 参数]'
flow3d_settings
if [[ "$mode" = inference && "$FLOW3D_GPUS" != 1 ]]; then
    flow3d_die '推理模板只使用单 GPU，请设置 FLOW3D_GPUS=1'
fi
command -v sbatch >/dev/null || flow3d_die '请在 NCSA 登录节点提交作业'
export FLOW3D_CODE_DIR
FLOW3D_CODE_DIR=$(bash "$code_dir/scripts/prepare_run.sh")
export FLOW3D_COMMIT
FLOW3D_COMMIT=$(cat "$FLOW3D_CODE_DIR/.flow3d-commit")
options=(--account="$FLOW3D_ACCOUNT" --partition="$FLOW3D_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$FLOW3D_CPUS" --mem="$FLOW3D_MEM"
    --time="$FLOW3D_TIME" --output="$FLOW3D_ROOT/logs/%j.out"
    --error="$FLOW3D_ROOT/logs/%j.err" --chdir="$FLOW3D_CODE_DIR" --export=ALL)
# 只使用一种 GPU 请求语法，避免 gres 和 gpus-per-node 冲突。
gpu_request="$FLOW3D_GPUS"
if [[ -n "${FLOW3D_GPU_TYPE:-}" ]]; then
    gpu_request="$FLOW3D_GPU_TYPE:$FLOW3D_GPUS"
fi
options+=(--gpus-per-node="$gpu_request")
if [[ -n "${FLOW3D_CONSTRAINT:-}" ]]; then options+=(--constraint="$FLOW3D_CONSTRAINT"); fi
printf -v FLOW3D_SUBMIT_COMMAND '%q ' sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/$mode.slurm" "$@"
export FLOW3D_SUBMIT_COMMAND
sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/$mode.slurm" "$@"
