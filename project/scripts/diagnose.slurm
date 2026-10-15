#!/bin/bash -l
# Resources are fixed by submit_diagnosis.sh; run through that wrapper.
set -euo pipefail
[[ -n "${FLOW3D_CODE_DIR:-}" && -n "${FLOW3D_DIAGNOSIS_RUN_DIR:-}" ]] || {
    printf '请通过 project/scripts/submit_diagnosis.sh 提交诊断\n' >&2
    exit 1
}
[[ "${FLOW3D_CLUSTER:-}" = deltaai && "$(uname -m)" = aarch64 ]] || {
    printf '诊断必须运行在 DeltaAI/GH200 ARM 节点\n' >&2
    exit 1
}
trap 'status=$?; printf "%s\n" "$status" > "$FLOW3D_DIAGNOSIS_RUN_DIR/exit_code.txt"' EXIT
source "$FLOW3D_CODE_DIR/scripts/runtime.sh"
[[ "$FLOW3D_GPUS" = 1 && "$FLOW3D_VISIBLE_GPUS" = 1 ]] || flow3d_die '诊断必须使用单 GPU'
printf 'Sampling diagnosis: commit=%s output=%s/diagnosis\n' "$FLOW3D_COMMIT" "$FLOW3D_DIAGNOSIS_RUN_DIR"
python "$FLOW3D_CODE_DIR/scripts/retry_srun.py" -- \
    python "$FLOW3D_CODE_DIR/../scripts/diagnose_diffusion_sampling.py" "$@"
