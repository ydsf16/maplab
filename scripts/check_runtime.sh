#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"

missing=0
check_file() { [[ -f "$2" ]] || { echo "missing $1: $2" >&2; missing=1; }; }
check_dir() { [[ -d "$2" ]] || { echo "missing $1: $2" >&2; missing=1; }; }
check_dir "Maplab runtime" "${MAPLAB_RUNTIME_ROOT}/workspace/devel"
check_file "SuperPoint ONNX" "${SUPERPOINT_ONNX_MODEL}"
check_file "LightGlue ONNX" "${LIGHTGLUE_MATCHER_ONNX_MODEL}"
check_file "SALAD checkpoint" "${PHONE_AI_SALAD_CHECKPOINT}"
check_file "DINOv2 checkpoint" "${PHONE_AI_DINOV2_CHECKPOINT}"
check_dir "DA3 model" "${PHONE_AI_DA3_MODEL}"
check_file "Mosaic3D checkpoint" "${PHONE_AI_MOSAIC3D_ROOT}/models/sc+ar+sc++.ckpt"
[[ "${missing}" -eq 0 ]] || exit 2
"${PHONE_AI_DA3_PYTHON}" -c 'import cv2, onnxruntime, torch; print("Phone AI runtime OK")'
