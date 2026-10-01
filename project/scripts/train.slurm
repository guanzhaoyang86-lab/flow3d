#!/bin/bash -l
#SBATCH --job-name=flow3d_training
#SBATCH --account=PHY260443
#SBATCH --time=48:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/%u/logs/%j.out
#SBATCH --error=/scratch/%u/logs/%j.err
#SBATCH --no-requeue

# 推荐通过 submit.sh 覆盖账户、存储路径、分区、GPU 数和时限。
# 如站点支持，可将 --gpus-per-node 替换为 --gres=gpu:a100:1
# 或 --gres=gpu:h100:1。不要同时保留两种请求。
set -euo pipefail
source "${FLOW3D_CODE_DIR:-${SLURM_SUBMIT_DIR:?}}/scripts/runtime.sh"
srun --ntasks=1 --gpu-bind=none \
    python -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$FLOW3D_VISIBLE_GPUS" src/train.py \
    --config configs/train.yaml "$@" --device cuda
