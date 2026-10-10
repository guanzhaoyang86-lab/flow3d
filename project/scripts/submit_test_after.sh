#!/bin/bash
# Submit one evaluation array after explicitly named training arrays succeed.
# Build its immutable evaluation plan once, after all checkpoints are ready.
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
training_plan=''
dependency=''
dry_run=0
while (( $# )); do
    case "$1" in
        --training-plan)
            (( $# >= 2 )) || flow3d_die '--training-plan 缺少路径'
            training_plan="$2"; shift 2 ;;
        --dependency)
            (( $# >= 2 )) || flow3d_die '--dependency 缺少作业号'
            dependency="$2"; shift 2
            [[ "$dependency" =~ ^[1-9][0-9]*(:[1-9][0-9]*)*$ ]] || flow3d_die '--dependency 必须为作业号或冒号分隔的作业号'
            ;;
        --dry-run) dry_run=1; shift ;;
        *) flow3d_die "未知参数：$1" ;;
    esac
done
[[ -n "$training_plan" && -r "$training_plan" ]] || flow3d_die '请用 --training-plan 指定可读取的训练计划'
training_plan="$(cd -- "$(dirname -- "$training_plan")" && pwd -P)/$(basename -- "$training_plan")"
flow3d_settings
[[ "$FLOW3D_GPUS" = 1 ]] || flow3d_die '矩阵测试必须使用单 GPU'
concurrency="${FLOW3D_ARRAY_CONCURRENCY:-16}"
time_limit="${FLOW3D_EVALUATE_TIME:-01:30:00}"
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'FLOW3D_ARRAY_CONCURRENCY 必须为正整数'
export FLOW3D_LAUNCH_RETRIES="${FLOW3D_LAUNCH_RETRIES:-3}"
export FLOW3D_LAUNCH_RETRY_DELAY="${FLOW3D_LAUNCH_RETRY_DELAY:-30}"
[[ "$FLOW3D_LAUNCH_RETRIES" =~ ^[0-3]$ ]] || flow3d_die 'FLOW3D_LAUNCH_RETRIES 必须为 0 到 3'
[[ "$FLOW3D_LAUNCH_RETRY_DELAY" =~ ^[0-9]+$ && "$FLOW3D_LAUNCH_RETRY_DELAY" -le 300 ]] || flow3d_die 'FLOW3D_LAUNCH_RETRY_DELAY 必须为 0 到 300 秒'
case "$FLOW3D_CLUSTER" in
    deltaai) expected_arch=aarch64 ;;
    delta)
        expected_arch=x86_64
        [[ "$FLOW3D_PARTITION" = gpuA100x4 || "$FLOW3D_PARTITION" = gpuA100x8 ]] || flow3d_die 'Delta 矩阵目前只支持 A100 分区'
        ;;
    *) flow3d_die '矩阵测试只支持 DeltaAI/GH200 或 Delta/A100' ;;
esac
command -v python >/dev/null || flow3d_die '请先加载站点 Python 模块'
if (( dry_run )); then
    FLOW3D_CODE_DIR="$code_dir"
    FLOW3D_COMMIT=$(cd -- "$code_dir" && git rev-parse HEAD)
else
    [[ "$(uname -m)" = "$expected_arch" ]] || flow3d_die '当前登录节点架构与 FLOW3D_CLUSTER 不符'
    command -v sbatch >/dev/null || flow3d_die '请在 NCSA 登录节点提交'
    command -v flock >/dev/null || flow3d_die '需要 flock 以安全地共享测试计划'
    FLOW3D_CODE_DIR=$(bash "$code_dir/scripts/prepare_run.sh")
    FLOW3D_COMMIT=$(cat "$FLOW3D_CODE_DIR/.flow3d-commit")
fi
export FLOW3D_CODE_DIR FLOW3D_COMMIT
export FLOW3D_LAUNCHER_DIR="$FLOW3D_CODE_DIR" FLOW3D_LAUNCHER_COMMIT="$FLOW3D_COMMIT"
export FLOW3D_TRAINING_PLAN="$training_plan"
runner="$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py"
[[ "$(python "$runner" inspect --plan "$training_plan" --field mode)" = matrix-train ]] || flow3d_die '必须提供 matrix-train 计划'
export FLOW3D_TRAINING_PLAN_SHA256 FLOW3D_EVAL_TASK_COUNT FLOW3D_EVALUATE_BUDGET_SECONDS
FLOW3D_TRAINING_PLAN_SHA256=$(python "$runner" inspect --plan "$training_plan" --field sha256)
FLOW3D_EVAL_TASK_COUNT=$(python "$runner" inspect --plan "$training_plan" --field count)
[[ "$FLOW3D_EVAL_TASK_COUNT" =~ ^[1-9][0-9]*$ ]] || flow3d_die '训练计划没有可测试任务'
FLOW3D_EVALUATE_BUDGET_SECONDS=$(python "$runner" time-budget "$time_limit")
export SLURM_EXPORT_ENV=ALL
# Ambient sbatch defaults must not accidentally hold or redirect this array.
unset SBATCH_DEPENDENCY
options=(--parsable --job-name=flow3d_test_all --account="$FLOW3D_ACCOUNT"
    --partition="$FLOW3D_PARTITION" --nodes=1 --ntasks=1
    --cpus-per-task="$FLOW3D_CPUS" --mem="$FLOW3D_MEM" --gpus-per-node=1
    --time="$time_limit" --array="0-$((FLOW3D_EVAL_TASK_COUNT - 1))%$concurrency"
    --no-requeue --chdir="$FLOW3D_CODE_DIR" --export=ALL
    --output="$FLOW3D_ROOT/logs/%A_%a.out" --error="$FLOW3D_ROOT/logs/%A_%a.err")
[[ "${FLOW3D_SUBMIT_HOLD:-0}" != 1 ]] || options+=(--hold)
if [[ -n "$dependency" ]]; then
    options+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
fi
if (( dry_run )); then
    printf '训练计划：%s；全量测试：%s 组；并发上限：%s\n' "$training_plan" "$FLOW3D_EVAL_TASK_COUNT" "$concurrency"
    printf '%q ' sbatch "${options[@]}" '<提交时生成的 evaluate.slurm>'
    printf '\ndry-run：未冻结代码、未创建测试计划、未提交 Slurm；正式提交会检查所有训练结果。\n'
    exit 0
fi
export FLOW3D_EVAL_COORD
FLOW3D_EVAL_COORD=$(mktemp -d "$FLOW3D_ROOT/results/evaluate-all.XXXXXXXX")
cat > "$FLOW3D_EVAL_COORD/ensure_plan.sh" <<'PLAN'
#!/bin/bash
set -euo pipefail
coord="$FLOW3D_EVAL_COORD"
runner="$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py"
exec 9> "$coord/plan.lock"
flock -x 9
if [[ -f "$coord/plan.failed" ]]; then
    cat "$coord/plan.err" >&2
    echo '全量测试计划验证失败；请修复训练或计划后重新提交测试。' >&2
    exit 1
fi
touch "$coord/plan.err"
trap 'touch "$coord/plan.failed"; cat "$coord/plan.err" >&2' ERR
[[ "$(python "$runner" inspect --plan "$FLOW3D_TRAINING_PLAN" --field sha256)" = "$FLOW3D_TRAINING_PLAN_SHA256" ]] || {
    echo '提交后的训练计划发生变化，拒绝运行测试。' >> "$coord/plan.err"
    false
}
if [[ ! -s "$coord/eval_plan.txt" ]]; then
    python "$runner" plan matrix-evaluate \
        --training-plan "$FLOW3D_TRAINING_PLAN" --training-tasks all \
        --storage-root "$FLOW3D_ROOT" --code-dir "$FLOW3D_CODE_DIR" \
        --commit "$FLOW3D_COMMIT" --cluster "$FLOW3D_CLUSTER" \
        > "$coord/eval_plan.tmp" 2> "$coord/plan.err"
    plan=$(cat "$coord/eval_plan.tmp")
    [[ "$(python "$runner" inspect --plan "$plan" --field count)" = "$FLOW3D_EVAL_TASK_COUNT" ]]
    python "$runner" inspect --plan "$plan" --field sha256 > "$coord/eval_plan.sha256"
    mv -- "$coord/eval_plan.tmp" "$coord/eval_plan.txt"
fi
plan=$(cat "$coord/eval_plan.txt")
[[ "$(python "$runner" inspect --plan "$plan" --field sha256)" = "$(cat "$coord/eval_plan.sha256")" ]]
printf '%s\n' "$plan"
PLAN
cat > "$FLOW3D_EVAL_COORD/evaluate.slurm" <<'SLURM'
#!/bin/bash -l
set -eo pipefail
SECONDS=0
source "$FLOW3D_LAUNCHER_DIR/scripts/runtime.sh"
[[ "$FLOW3D_GPUS" = 1 && "$FLOW3D_VISIBLE_GPUS" = 1 ]]
plan=$(bash "$FLOW3D_EVAL_COORD/ensure_plan.sh")
export FLOW3D_PLAN_SHA256
FLOW3D_PLAN_SHA256=$(cat "$FLOW3D_EVAL_COORD/eval_plan.sha256")
export FLOW3D_EVALUATE_BUDGET_SECONDS=$((FLOW3D_EVALUATE_BUDGET_SECONDS - SECONDS))
(( FLOW3D_EVALUATE_BUDGET_SECONDS > 0 ))
printf '测试计划：%s\n' "$plan"
python "$FLOW3D_LAUNCHER_DIR/scripts/retry_srun.py" -- \
    python "$FLOW3D_CODE_DIR/../scripts/run_hpc_matrix.py" run --plan "$plan"
SLURM
chmod a-w "$FLOW3D_EVAL_COORD/ensure_plan.sh" "$FLOW3D_EVAL_COORD/evaluate.slurm"
if [[ -n "$dependency" ]]; then
    printf '全量测试等待这些训练作业全部成功：%s\n' "$dependency"
else
    # No live dependency: validate all completed training records before paying
    # for any GPU allocation. Never silently omit missing or failed models.
    bash "$FLOW3D_EVAL_COORD/ensure_plan.sh"
fi
printf -v FLOW3D_SUBMIT_COMMAND '%q ' sbatch "${options[@]}" "$FLOW3D_EVAL_COORD/evaluate.slurm"
export FLOW3D_SUBMIT_COMMAND
printf '训练计划：%s；全量测试：%s 组；互连启动失败最多额外重试 %s 次。\n' "$training_plan" "$FLOW3D_EVAL_TASK_COUNT" "$FLOW3D_LAUNCH_RETRIES"
printf '提交记录：%s\n测试计划路径文件：%s/eval_plan.txt\n' "$FLOW3D_EVAL_COORD" "$FLOW3D_EVAL_COORD"
jobid=$(sbatch "${options[@]}" "$FLOW3D_EVAL_COORD/evaluate.slurm")
jobid=${jobid%%;*}
[[ "$jobid" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'sbatch 未返回有效作业号，请检查队列后再提交'
printf '%s\n' "$jobid" > "$FLOW3D_EVAL_COORD/job_id.txt"
printf 'Submitted batch job %s\n' "$jobid"
