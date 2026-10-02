#!/bin/bash
# 整个数组只冻结一次 GitHub 提交；数组任务绝不访问 GitHub。
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
mode="${1:?缺少矩阵模式}"; shift
dry_run=0
for argument in "$@"; do [[ "$argument" != --dry-run ]] || dry_run=1; done
flow3d_settings
[[ "$FLOW3D_CLUSTER" = deltaai && "$FLOW3D_GPUS" = 1 ]] || flow3d_die '请 source configs/deltaai-matrix.env.example；每个任务使用一张 GH200'
concurrency="${FLOW3D_ARRAY_CONCURRENCY:-4}"
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'FLOW3D_ARRAY_CONCURRENCY 必须为正整数'
command -v python >/dev/null || flow3d_die '请先加载站点 Python 模块'
if (( ! dry_run )); then
    [[ "$(uname -m)" = aarch64 ]] || flow3d_die '请登录 DeltaAI（gh-login...，aarch64）提交 GH200 任务'
    command -v sbatch >/dev/null || flow3d_die '请在 NCSA 登录节点提交'
fi
if [[ "$mode" = matrix-resume ]]; then
    # 恢复原计划使用原快照，配置与代码均不可改变。
    plan=$(python "$code_dir/../scripts/run_hpc_matrix.py" resume "$@")
    FLOW3D_CODE_DIR=$(python "$code_dir/../scripts/run_hpc_matrix.py" inspect --plan "$plan" --field code_dir)
    FLOW3D_COMMIT=$(python "$code_dir/../scripts/run_hpc_matrix.py" inspect --plan "$plan" --field commit)
else
    if (( dry_run )); then
        FLOW3D_CODE_DIR="$code_dir"
        FLOW3D_COMMIT=$(cd -- "$code_dir" && git rev-parse HEAD)
    else
        FLOW3D_CODE_DIR=$(bash "$code_dir/scripts/prepare_run.sh")
        FLOW3D_COMMIT=$(cat "$FLOW3D_CODE_DIR/.flow3d-commit")
    fi
    plan=$(python "$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py" plan "$mode" "$@" --code-dir "$FLOW3D_CODE_DIR" --commit "$FLOW3D_COMMIT")
fi
export FLOW3D_CODE_DIR FLOW3D_COMMIT
runner="$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py"
export FLOW3D_PLAN_SHA256
FLOW3D_PLAN_SHA256=$(python "$runner" inspect --plan "$plan" --field sha256)
count=$(python "$runner" inspect --plan "$plan" --field count)
actual_mode=$(python "$runner" inspect --plan "$plan" --field mode)
case "$actual_mode" in
    tensor-prepare) time_limit="${FLOW3D_PREPARE_TIME:-02:00:00}" ;;
    matrix-train) time_limit="${FLOW3D_TRAIN_TIME:-08:00:00}" ;;
    matrix-evaluate) time_limit="${FLOW3D_EVALUATE_TIME:-08:00:00}" ;;
    *) flow3d_die '无效的计划模式' ;;
esac
python "$runner" check-time "$time_limit"
array=$(python "$runner" array --plan "$plan" --tasks "${FLOW3D_ARRAY_TASKS:-all}")
options=(--job-name="flow3d_${actual_mode#matrix-}" --account="$FLOW3D_ACCOUNT" --partition="$FLOW3D_PARTITION"
    --nodes=1 --ntasks=1 --cpus-per-task="$FLOW3D_CPUS" --mem="$FLOW3D_MEM"
    --gpus-per-node=1 --time="$time_limit" --array="$array%$concurrency"
    --output="$FLOW3D_ROOT/logs/%A_%a.out" --error="$FLOW3D_ROOT/logs/%A_%a.err"
    --chdir="$FLOW3D_CODE_DIR" --export=ALL)
run_args=(--plan "$plan")
[[ "$mode" != matrix-resume ]] || run_args+=(--resume)
if [[ "$mode" = matrix-followup ]]; then
    dependency=$(python "$runner" dependency --plan "$plan")
    if [[ -n "$dependency" ]]; then
        options+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
        printf '等待预处理作业 %s 成功后自动进入 9 组试跑。\n' "$dependency"
    else
        printf '预处理成功记录已验证，直接排队 9 组试跑。\n'
    fi
fi
printf -v FLOW3D_SUBMIT_COMMAND '%q ' sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/matrix.slurm" "${run_args[@]}"
export FLOW3D_SUBMIT_COMMAND
printf '计划：%s\n模式：%s；总任务：%s；本次数组：%s；并发上限：%s\n' "$plan" "$actual_mode" "$count" "$array" "$concurrency"
printf '%s\n' "$FLOW3D_SUBMIT_COMMAND"
if (( dry_run )); then
    printf 'dry-run：未提交 Slurm。新计划未冻结代码，不可直接运行；正式提交请去掉 --dry-run。\n'
else
    sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/matrix.slurm" "${run_args[@]}"
fi
