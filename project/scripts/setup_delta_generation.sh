#!/bin/bash -l
# 登录节点只下载代码并安装用户环境；这里不启动模拟或训练。
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/configs/delta-generation.env.example"
[[ $(uname -m) = x86_64 ]] || { echo '请在 Delta x86_64 登录节点运行此脚本' >&2; exit 1; }
command -v module >/dev/null || { echo '缺少 module，请在 Delta 登录 shell 中运行' >&2; exit 1; }
module load "$FLOW3D_MODULES"
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p -- "$HOME/envs" "$HOME/projects"
if [[ ! -d "$FLOW3D_CONDA_ENV/conda-meta" ]]; then
    conda create -y --override-channels -c conda-forge -p "$FLOW3D_CONDA_ENV" python=3.11 pip
fi
conda activate "$FLOW3D_CONDA_ENV"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Environment must use Python 3.11"'
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$code_dir/requirements-generation.txt"
python -m pip check

upstream_commit=fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c
if [[ ! -e "$FLOW3D_UPSTREAM_REPO" ]]; then
    git clone https://github.com/yjhp1016/taichi_LBM3D.git "$FLOW3D_UPSTREAM_REPO"
fi
[[ -z $(git -C "$FLOW3D_UPSTREAM_REPO" status --porcelain) ]] || { echo '上游求解器存在改动，请先检查' >&2; exit 1; }
git -C "$FLOW3D_UPSTREAM_REPO" checkout --detach "$upstream_commit"
python -c 'import platform, torch, taichi; print("Architecture:", platform.machine()); print("Python:", platform.python_version()); print("PyTorch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("Taichi:", taichi.__version__)'
printf '\n环境准备完成。下一步在 project/ 执行：\nsource configs/delta-generation.env.example\nbash scripts/submit.sh generate-pilot\n'
