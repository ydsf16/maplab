#!/usr/bin/env bash
# Package model weights and their required source trees for offline deployment.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    -h|--help)
      echo "usage: bash scripts/package_phoneai_assets.sh --output <phone-ai-assets.tar.zst>"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${output}" ]] || { echo "--output is required" >&2; exit 2; }

check_file() { [[ -f "$2" ]] || { echo "missing $1: $2" >&2; exit 2; }; }
check_dir() { [[ -d "$2" ]] || { echo "missing $1: $2" >&2; exit 2; }; }
check_file "SuperPoint ONNX" "${SUPERPOINT_ONNX_MODEL}"
check_file "LightGlue ONNX" "${LIGHTGLUE_MATCHER_ONNX_MODEL}"
check_dir "SALAD repo" "${PHONE_AI_SALAD_REPO}"
check_file "SALAD checkpoint" "${PHONE_AI_SALAD_CHECKPOINT}"
check_dir "DINOv2 repo" "${PHONE_AI_DINOV2_REPO}"
check_file "DINOv2 checkpoint" "${PHONE_AI_DINOV2_CHECKPOINT}"
check_dir "DA3 repo" "${PHONE_AI_DA3_REPO}"
check_dir "DA3 model" "${PHONE_AI_DA3_MODEL}"
check_dir "Mosaic3D assets" "${PHONE_AI_MOSAIC3D_ROOT}"
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }

stage="$(mktemp -d)"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
mkdir -p "${stage}/models/lightglue" "${stage}/models/salad" \
  "${stage}/models/dinov2" "${stage}/models/da3"
ln -s "${SUPERPOINT_ONNX_MODEL}" "${stage}/models/lightglue/superpoint_2048.onnx"
ln -s "${LIGHTGLUE_MATCHER_ONNX_MODEL}" "${stage}/models/lightglue/superpoint_lightglue.onnx"
ln -s "${PHONE_AI_SALAD_REPO}" "${stage}/models/salad/repo"
ln -s "${PHONE_AI_SALAD_CHECKPOINT}" "${stage}/models/salad/dino_salad.ckpt"
ln -s "${PHONE_AI_DINOV2_REPO}" "${stage}/models/dinov2/repo"
ln -s "${PHONE_AI_DINOV2_CHECKPOINT}" "${stage}/models/dinov2/dinov2_vitb14_pretrain.pth"
ln -s "${PHONE_AI_DA3_REPO}" "${stage}/models/da3/repo"
ln -s "${PHONE_AI_DA3_MODEL}" "${stage}/models/da3/DA3-GIANT-1.1"
ln -s "${PHONE_AI_MOSAIC3D_ROOT}" "${stage}/models/mosaic3d"
mkdir -p "$(dirname "${output}")"
tar --dereference --zstd -cf "${output}" -C "${stage}" models
echo "created ${output}"
