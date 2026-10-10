#!/bin/bash
# Replace wholly pending training/test arrays with the current GitHub launcher.
# Already running or completed array elements require a separate resume workflow.
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
training_plan=''; training_job=''; test_job=''
while (( $# )); do
    (( $# >= 2 )) || flow3d_die "参数缺少值：$1"
    case "$1" in
        --training-plan) training_plan="$2" ;;
        --training-job) training_job="$2" ;;
        --test-job) test_job="$2" ;;
        *) flow3d_die "未知参数：$1" ;;
    esac
    shift 2
done
[[ -r "$training_plan" ]] || flow3d_die '请提供可读取的 --training-plan'
[[ "$training_job" =~ ^[1-9][0-9]*$ && "$test_job" =~ ^[1-9][0-9]*$ && "$training_job" != "$test_job" ]] || flow3d_die '请提供两个不同的正整数作业号'
for tool in squeue scontrol scancel python; do
    command -v "$tool" >/dev/null || flow3d_die "缺少命令：$tool"
done

pending_indices() {
    local job="$1" rows job_id state
    rows=$(squeue -h -r -j "$job" -o '%i|%T') || return
    [[ -n "$rows" ]] || flow3d_die "作业 $job 已不在队列，停止替换"
    while IFS='|' read -r job_id state; do
        job_id="${job_id//[[:space:]]/}"; state="${state//[[:space:]]/}"
        [[ "$job_id" =~ ^${job}_([0-9]+)$ ]] || flow3d_die "作业 $job 的数组编号异常：$job_id"
        [[ "$state" = PENDING ]] || flow3d_die "$job_id 状态为 $state；仅允许替换全部 PENDING 的数组"
        printf '%s\n' "${BASH_REMATCH[1]}"
    done <<< "$rows"
}

training_ids=$(pending_indices "$training_job" | sort -n)
test_ids=$(pending_indices "$test_job" | sort -n)
count=$(python -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["mode"] == "matrix-train"; print(len(p["tasks"]))' "$training_plan")
[[ "$count" =~ ^[1-9][0-9]*$ ]] || flow3d_die '训练计划没有任务'
expected_ids=$(seq 0 "$((count - 1))")
[[ "$test_ids" = "$expected_ids" ]] || flow3d_die '测试数组不是完整待运行数组；为保护已有结果，停止替换'
while read -r index; do
    (( index < count )) || flow3d_die '训练任务编号超出计划范围'
done <<< "$training_ids"
[[ "$(printf '%s\n' "$training_ids" | sort -nu)" = "$training_ids" ]] || flow3d_die '训练数组包含重复编号'

flow3d_settings
record=$(mktemp -d "$FLOW3D_ROOT/logs/replace-pending.XXXXXXXX")
new_training=''; new_test=''; finished=0
read_job_id() {
    local ids
    [[ -f "$1" ]] || return 1
    ids=$(sed 's/\r$//' "$1" | awk '/^Submitted batch job [1-9][0-9]*$/ {print $4}')
    [[ "$ids" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s\n' "$ids"
}
on_exit() {
    local status=$?
    if (( ! finished )); then
        new_training="${new_training:-$(read_job_id "$record/training.log" || true)}"
        new_test="${new_test:-$(read_job_id "$record/test.log" || true)}"
        printf '\n替换未完成，未自动取消或释放剩余作业。\n旧训练=%s；旧测试=%s；新训练=%s；新测试=%s\n提交日志：%s\n请先用 squeue 检查这些编号，再决定恢复或取消暂缓的任务。\n' \
            "$training_job" "$test_job" "${new_training:-未获取}" "${new_test:-未获取}" "$record" >&2
    fi
    return "$status"
}
trap on_exit EXIT

scontrol hold "$test_job"
scontrol hold "$training_job"
[[ "$(pending_indices "$training_job" | sort -n)" = "$training_ids" && "$(pending_indices "$test_job" | sort -n)" = "$test_ids" ]] || flow3d_die '暂缓期间旧任务状态变化，停止替换'

export FLOW3D_SUBMIT_HOLD=1
export FLOW3D_ARRAY_TASKS
FLOW3D_ARRAY_TASKS=$(printf '%s\n' "$training_ids" | paste -sd, -)
bash "$code_dir/scripts/submit.sh" matrix-resume --plan "$training_plan" | tee "$record/training.log"
new_training=$(read_job_id "$record/training.log") || flow3d_die '无法确认新训练作业号；请检查提交日志和队列'
bash "$code_dir/scripts/submit_test_after.sh" --training-plan "$training_plan" --dependency "$new_training" | tee "$record/test.log"
new_test=$(read_job_id "$record/test.log") || flow3d_die '无法确认新测试作业号；请检查提交日志和队列'
[[ "$new_training" != "$new_test" && "$new_training" != "$training_job" && "$new_training" != "$test_job" && "$new_test" != "$training_job" && "$new_test" != "$test_job" ]] || flow3d_die '提交返回重复作业号，停止替换'

# The second check plus --state=PENDING closes the hold/start race: never cancel
# a running old job, and never release new duplicates while old jobs remain.
[[ "$(pending_indices "$training_job" | sort -n)" = "$training_ids" && "$(pending_indices "$test_job" | sort -n)" = "$test_ids" ]] || flow3d_die '提交期间旧任务状态变化，新任务保持暂缓'
scancel --state=PENDING "$test_job" "$training_job"
remaining=$(squeue -h -r -u "$USER" -o '%i|%T')
while IFS='|' read -r job_id state; do
    job_id="${job_id//[[:space:]]/}"
    [[ "$job_id" != "$training_job" && "$job_id" != "$training_job"_* && "$job_id" != "$test_job" && "$job_id" != "$test_job"_* ]] || flow3d_die '旧作业仍在队列，新任务保持暂缓；请检查队列'
done <<< "$remaining"
scontrol release "$new_test"
scontrol release "$new_training"
finished=1
printf '\n替换完成：训练=%s；全量测试=%s（等待新训练全部成功）。\n新任务重新排队；记录：%s\n' "$new_training" "$new_test" "$record"
