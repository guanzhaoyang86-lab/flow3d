#!/bin/bash -l
#SBATCH --job-name=flow3d_inference
#SBATCH --account=PHY260443
#SBATCH --time=01:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/%u/logs/%j.out
#SBATCH --error=/scratch/%u/logs/%j.err
#SBATCH --no-requeue

set -euo pipefail
source "${FLOW3D_CODE_DIR:-${SLURM_SUBMIT_DIR:?}}/scripts/runtime.sh"
[[ "$FLOW3D_VISIBLE_GPUS" = 1 ]] || flow3d_die '推理模板仅支持单 GPU'
srun --ntasks=1 --gpu-bind=none python src/inference.py \
    --config configs/train.yaml "$@" --device cuda
