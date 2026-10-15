#!/bin/bash
# One independent GH200 sampling diagnostic; no training/evaluation dependencies.
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"

usage() {
    cat <<'HELP'
Usage: bash project/scripts/submit_diagnosis.sh [options]
  --dry-run                 Show the job without reading data, freezing or submitting
  --manifest PATH           Existing dataset manifest
  --checkpoint PATH         Existing checkpoint; repeat to replace the two defaults
  --indices 0,1,2            Fixed validation case indices
  --sampling-steps 50        Reverse diffusion steps
  --seed 47                 Sampling seed
  --num-probe-particles 16   Unobserved probe particles
  --time 00:30:00            Slurm wall time, at most 01:00:00
  --help                    Show this help

Run on a DeltaAI ARM login node. The job loads the existing site PyTorch module
and base environment. FLOW3D_ROOT and FLOW3D_SNAPSHOT_ROOT may override storage.
HELP
}

# Deliberately independent of any previously sourced training/array settings.
export FLOW3D_CLUSTER=deltaai FLOW3D_ACCOUNT=biup-dtai-gh FLOW3D_PARTITION=ghx4
export FLOW3D_GPUS=1 FLOW3D_CPUS=8 FLOW3D_MEM=64G
export FLOW3D_MODULES=python/miniforge3_pytorch/2.10.0 FLOW3D_CONDA_ENV=base
export FLOW3D_CONDA_SH='' FLOW3D_GPU_TYPE='' FLOW3D_CONSTRAINT=''
diagnosis_user="${USER:-${USERNAME:-$(id -un)}}"
export FLOW3D_ROOT="${FLOW3D_ROOT:-/work/hdd/biup/$diagnosis_user}"
export FLOW3D_LAUNCH_RETRIES="${FLOW3D_LAUNCH_RETRIES:-3}"
export FLOW3D_LAUNCH_RETRY_DELAY="${FLOW3D_LAUNCH_RETRY_DELAY:-30}"
unset FLOW3D_EVALUATE_BUDGET_SECONDS FLOW3D_PLAN_SHA256
manifest="$FLOW3D_ROOT/datasets/2026-10-01_115342_lbm3d-full_22594936_7966ccef/manifest.json"
checkpoint_root="$FLOW3D_ROOT/checkpoints/2026-10-03_074635_matrix-train_51493784"
checkpoints=()
indices=0,1,2
sampling_steps=50
seed=47
probe_particles=16
time_limit=00:30:00
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --manifest|--checkpoint|--indices|--sampling-steps|--seed|--num-probe-particles|--time)
            (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || flow3d_die "$1 缺少值"
            case "$1" in
                --manifest) manifest="$2" ;;
                --checkpoint) checkpoints+=("$2") ;;
                --indices) indices="$2" ;;
                --sampling-steps) sampling_steps="$2" ;;
                --seed) seed="$2" ;;
                --num-probe-particles) probe_particles="$2" ;;
                --time) time_limit="$2" ;;
            esac
            shift 2 ;;
        *) flow3d_die "未知参数：$1；使用 --help 查看参数" ;;
    esac
done
if (( ${#checkpoints[@]} == 0 )); then
    checkpoints=("$checkpoint_root/dit3d_N002_s31/best.pt" "$checkpoint_root/tensor-dit_N002_s31/best.pt")
fi
[[ "$FLOW3D_ROOT" = /* ]] || flow3d_die 'FLOW3D_ROOT 必须为绝对路径'
[[ "$indices" =~ ^[0-9]+(,[0-9]+)*$ ]] || flow3d_die '--indices 必须为逗号分隔的非负整数'
[[ "$sampling_steps" =~ ^[1-9][0-9]*$ ]] || flow3d_die '--sampling-steps 必须为正整数'
[[ "$seed" =~ ^[0-9]+$ ]] || flow3d_die '--seed 必须为非负整数'
[[ "$probe_particles" =~ ^[1-9][0-9]*$ ]] || flow3d_die '--num-probe-particles 必须为正整数'
[[ "$FLOW3D_LAUNCH_RETRIES" =~ ^[0-3]$ ]] || flow3d_die 'FLOW3D_LAUNCH_RETRIES 必须为 0 到 3'
[[ "$FLOW3D_LAUNCH_RETRY_DELAY" =~ ^[0-9]{1,3}$ ]] && (( 10#$FLOW3D_LAUNCH_RETRY_DELAY <= 300 )) || flow3d_die 'FLOW3D_LAUNCH_RETRY_DELAY 必须为 0 到 300 秒'
[[ "$time_limit" =~ ^([0-9]{1,2}):([0-5][0-9]):([0-5][0-9])$ ]] || flow3d_die '--time 格式必须为 HH:MM:SS'
time_seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
(( time_seconds > 0 && time_seconds <= 3600 )) || flow3d_die '诊断时间必须大于零且不超过 01:00:00'
export FLOW3D_TIME="$time_limit"

if (( dry_run )); then
    # Do not call flow3d_settings/prepare_run: even directories stay untouched.
    FLOW3D_CODE_DIR='<frozen-repository>/project'
    run_dir="$FLOW3D_ROOT/results/sampling-diagnosis/<new-unique-run>"
else
    [[ "$(uname -m)" = aarch64 ]] || flow3d_die '请在 DeltaAI/GH200 ARM 登录节点提交'
    command -v sbatch >/dev/null || flow3d_die '请在 NCSA 登录节点提交（缺少 sbatch）'
    [[ -f "$manifest" && -r "$manifest" ]] || flow3d_die "manifest 不可读：$manifest"
    manifest="$(cd -- "$(dirname -- "$manifest")" && pwd -P)/$(basename -- "$manifest")"
    for i in "${!checkpoints[@]}"; do
        checkpoint="${checkpoints[$i]}"
        [[ -f "$checkpoint" && -r "$checkpoint" ]] || flow3d_die "checkpoint 不可读：$checkpoint"
        checkpoints[$i]="$(cd -- "$(dirname -- "$checkpoint")" && pwd -P)/$(basename -- "$checkpoint")"
    done
    FLOW3D_CODE_DIR=$(bash "$code_dir/scripts/prepare_run.sh")
    FLOW3D_COMMIT=$(cat "$FLOW3D_CODE_DIR/.flow3d-commit")
    [[ -r "$FLOW3D_CODE_DIR/../scripts/diagnose_diffusion_sampling.py" && -r "$FLOW3D_CODE_DIR/scripts/diagnose.slurm" ]] || flow3d_die 'Git 快照缺少诊断代码；请先提交并同步完整诊断改动'
    mkdir -p -- "$FLOW3D_ROOT/results/sampling-diagnosis"
    run_dir=$(mktemp -d "$FLOW3D_ROOT/results/sampling-diagnosis/$(date -u +%Y%m%d_%H%M%S).XXXXXXXX")
    export FLOW3D_COMMIT FLOW3D_CODE_DIR
    export FLOW3D_LAUNCHER_DIR="$FLOW3D_CODE_DIR" FLOW3D_LAUNCHER_COMMIT="$FLOW3D_COMMIT"
    export FLOW3D_DIAGNOSIS_RUN_DIR="$run_dir"
fi

arguments=(--manifest "$manifest" --output-dir "$run_dir/diagnosis"
    --indices "$indices" --device cuda --sampling-steps "$sampling_steps"
    --seed "$seed" --num-probe-particles "$probe_particles")
for checkpoint in "${checkpoints[@]}"; do arguments+=(--checkpoint "$checkpoint"); done
# Clear all ambient sbatch defaults, including dependencies, arrays and holds.
# This script runs in its own Bash process, so the caller's environment is intact.
for variable in "${!SBATCH_@}"; do unset "$variable"; done
export SLURM_EXPORT_ENV=ALL
options=(--parsable --job-name=flow3d_diagnose --account="$FLOW3D_ACCOUNT"
    --partition="$FLOW3D_PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=8
    --mem=64G --gpus-per-node=1 --time="$time_limit" --no-requeue
    --chdir="$FLOW3D_CODE_DIR" --export=ALL
    --output="$run_dir/%j.out" --error="$run_dir/%j.err")
printf '单作业诊断：%s 个 checkpoint；validation indices=%s；1 GH200；时限=%s\n' "${#checkpoints[@]}" "$indices" "$time_limit"
printf '结果目录：%s/diagnosis\n' "$run_dir"
if (( dry_run )); then
    printf '%q ' sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/diagnose.slurm" "${arguments[@]}"
    printf '\ndry-run：未读取数据或权重、未联网、未冻结代码、未创建目录、未提交 Slurm。\n'
    exit 0
fi
printf '%s\n' "$FLOW3D_COMMIT" > "$run_dir/commit.txt"
printf '%q ' python "$FLOW3D_CODE_DIR/../scripts/diagnose_diffusion_sampling.py" "${arguments[@]}" > "$run_dir/diagnostic_command.txt"
printf '\n' >> "$run_dir/diagnostic_command.txt"
printf -v FLOW3D_SUBMIT_COMMAND '%q ' sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/diagnose.slurm" "${arguments[@]}"
export FLOW3D_SUBMIT_COMMAND
printf '%s\n' "$FLOW3D_SUBMIT_COMMAND" > "$run_dir/submit_command.txt"
jobid=$(sbatch "${options[@]}" "$FLOW3D_CODE_DIR/scripts/diagnose.slurm" "${arguments[@]}")
jobid=${jobid%%;*}
[[ "$jobid" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'sbatch 未返回有效作业号，请检查队列后再提交'
printf '%s\n' "$jobid" > "$run_dir/job_id.txt"
printf '日志：%s/%s.out 和 %s/%s.err\n' "$run_dir" "$jobid" "$run_dir" "$jobid"
printf 'Submitted batch job %s\n' "$jobid"
