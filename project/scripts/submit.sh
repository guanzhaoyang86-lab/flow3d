#!/bin/bash
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
mode="${1:-train}"
if (( $# )); then shift; fi
case "$mode" in
    tensor-prepare|matrix-train|matrix-evaluate|matrix-resume|matrix-followup)
        exec bash "$code_dir/scripts/submit_matrix.sh" "$mode" "$@"
        ;;
    train|inference) template="$mode"; run_args=("$@") ;;
    diffusion-smoke|diffusion-train|diffusion-inference)
        template=diffusion
        run_args=("${mode#diffusion-}" "$@")
        ;;
    generate-pilot|generate-full)
        template=generate
        run_args=("${mode#generate-}" "$@")
        ;;
    *) flow3d_die '用法: bash scripts/submit.sh train|inference|diffusion-smoke|diffusion-train|diffusion-inference|generate-pilot|generate-full [Python 参数]' ;;
esac
flow3d_settings
if [[ "$mode" = inference && "$FLOW3D_GPUS" != 1 ]]; then
    flow3d_die '推理模板只使用单 GPU，请设置 FLOW3D_GPUS=1'
fi
if [[ "$template" = diffusion && "$FLOW3D_GPUS" != 1 ]]; then
    flow3d_die '真实 diffusion 入口目前只支持单 GPU，请设置 FLOW3D_GPUS=1'
fi
if [[ "$template" = generate ]]; then
    [[ "$FLOW3D_CLUSTER" = delta && "$FLOW3D_GPUS" = 1 ]] || flow3d_die '数据生成使用 Delta，且 FLOW3D_GPUS=1'
    [[ "$(uname -m)" = x86_64 ]] || flow3d_die '当前主机不是 x86_64。请返回 Delta（dt-login...）提交数据生成；生成环境位于 Delta 自己的 HOME。'
    [[ -f "${FLOW3D_UPSTREAM_REPO:-}/Single_phase/LBM_3D_SinglePhase_Solver.py" ]] || flow3d_die '请先运行 setup_delta_generation.sh 准备上游求解器'
fi
command -v sbatch >/dev/null || flow3d_die '请在 NCSA 登录节点提交作业'
export FLOW3D_CODE_DIR
FLOW3D_CODE_DIR=$(bash "$code_dir/scripts/prepare_run.sh")
export FLOW3D_COMMIT
FLOW3D_COMMIT=$(cat "$FLOW3D_CODE_DIR/.flow3d-commit")
if [[ "$template" = diffusion && ! -f "$FLOW3D_CODE_DIR/../scripts/run_hpc_diffusion.py" ]]; then
    flow3d_die 'diffusion 作业需要包含根目录 scripts/ 的完整科研仓库'
fi
if [[ "$template" = generate && ! -f "$FLOW3D_CODE_DIR/../scripts/run_hpc_generation.py" ]]; then
    flow3d_die '数据生成需要包含根目录 scripts/ 的完整科研仓库'
fi
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
printf -v FLOW3D_SUBMIT_COMMAND '%q ' sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/$template.slurm" "${run_args[@]}"
export FLOW3D_SUBMIT_COMMAND
sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/$template.slurm" "${run_args[@]}"
