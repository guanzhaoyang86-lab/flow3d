#!/bin/bash
set -euo pipefail
code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$code_dir/scripts/common.sh"
flow3d_settings
cd -- "$code_dir"
repo_root=$(git rev-parse --show-toplevel)
repo_root=$(cd -- "$repo_root" && pwd -P)
code_prefix=$(git rev-parse --show-prefix)
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || flow3d_die 'HPC 工作区有改动，请在本地开发并经 GitHub 同步'
branch=$(git symbolic-ref --quiet --short HEAD) || flow3d_die '提交新实验必须位于跟踪 GitHub 的分支'
remote=$(git config --get "branch.$branch.remote") || flow3d_die '分支没有远端'
remote_url=$(git remote get-url "$remote")
[[ "$remote_url" =~ ^https://github\.com/|^git@github\.com:|^ssh://git@github\.com/ ]] || flow3d_die '同步远端必须是 github.com（HTTPS 或 SSH）'
git rev-parse --verify '@{upstream}' >/dev/null || flow3d_die '请设置 GitHub upstream 分支'
# 网络失败立即中止；不以旧版本提交，不在 GPU 作业中访问网络。
git pull --ff-only >&2
commit=$(git rev-parse HEAD)
[[ "$commit" = "$(git rev-parse '@{upstream}')" ]] || flow3d_die '本地领先或偏离 GitHub，拒绝提交未同步代码'
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || flow3d_die '同步后工作区不干净'
snapshot_root="${FLOW3D_SNAPSHOT_ROOT:-$HOME/projects/flow3d_runs}"
mkdir -p -- "$snapshot_root"
snapshot_root=$(cd -- "$snapshot_root" && pwd -P)
[[ "$snapshot_root/" != "$repo_root/"* ]] || flow3d_die '快照目录必须位于源码仓库外'
snapshot=$(mktemp -d "$snapshot_root/${commit:0:12}.XXXXXXXX")
(cd -- "$repo_root" && git archive "$commit") | tar -x -C "$snapshot"
# 独立模板和完整科研仓库中的 project/ 子目录均可使用；始终保存整个仓库。
snapshot_code="${snapshot}/${code_prefix}"
snapshot_code="${snapshot_code%/}"
printf '%s\n' "$commit" > "$snapshot_code/.flow3d-commit"
chmod -R a-w "$snapshot"
printf '已冻结 GitHub 提交 %s\n' "$commit" >&2
printf '%s\n' "$snapshot_code"
