#!/bin/bash
# 公共配置；所有路径在提交前确定，Slurm 指令不会展开 shell 变量。
flow3d_die() { printf 'flow3d: %s\n' "$*" >&2; exit 1; }

flow3d_settings() {
    export FLOW3D_CLUSTER="${FLOW3D_CLUSTER:-delta}"
    export FLOW3D_ACCOUNT="${FLOW3D_ACCOUNT:-PHY260443}"
    case "$FLOW3D_CLUSTER" in
        delta) export FLOW3D_PARTITION="${FLOW3D_PARTITION:-gpuA100x4}" ;;
        deltaai) export FLOW3D_PARTITION="${FLOW3D_PARTITION:-ghx4}" ;;
        *) flow3d_die 'FLOW3D_CLUSTER 必须是 delta 或 deltaai' ;;
    esac
    export FLOW3D_GPUS="${FLOW3D_GPUS:-1}"
    export FLOW3D_TIME="${FLOW3D_TIME:-48:00:00}"
    export FLOW3D_CPUS="${FLOW3D_CPUS:-8}"
    export FLOW3D_MEM="${FLOW3D_MEM:-32G}"
    export FLOW3D_CONDA_ENV="${FLOW3D_CONDA_ENV:-flow3d}"
    [[ "$FLOW3D_GPUS" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'GPU 数量必须是正整数'
    [[ "$FLOW3D_CPUS" =~ ^[1-9][0-9]*$ ]] || flow3d_die 'CPU 数量必须是正整数'
    [[ -n "${FLOW3D_ROOT:-}" && "$FLOW3D_ROOT" = /* ]] || flow3d_die '请设置绝对路径 FLOW3D_ROOT'
    [[ "$FLOW3D_ROOT" != *'<'* && "$FLOW3D_ROOT" != *'>'* ]] || flow3d_die '请替换 FLOW3D_ROOT 中的占位符'
    case "$FLOW3D_CLUSTER:$FLOW3D_ROOT" in
        delta:/scratch/*|delta:/work/*|deltaai:/work/*) ;;
        *) flow3d_die 'Delta 存储应为 /scratch 或 /work；DeltaAI 应为 /work（没有 /scratch）' ;;
    esac
    [[ "$FLOW3D_ACCOUNT" != *'<'* ]] || flow3d_die '请用 accounts 核对实际 Slurm 账户'
    [[ "$FLOW3D_PARTITION" != *'<'* ]] || flow3d_die '请用 sinfo 核对分区'
    mkdir -p -- "$FLOW3D_ROOT"/{datasets,checkpoints,logs,results/experiments}
    [[ -w "$FLOW3D_ROOT/logs" ]] || flow3d_die '日志目录不可写'
    export FLOW3D_ROOT
}
