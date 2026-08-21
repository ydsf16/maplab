#!/usr/bin/env bash
# Shared runtime layout. Source this file from Phone AI entry points.

_phone_ai_repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_phone_ai_repo_dir}/.phoneai.env" ]]; then
  # shellcheck disable=SC1091
  source "${_phone_ai_repo_dir}/.phoneai.env"
fi

export PHONE_AI_HOME="${PHONE_AI_HOME:-${_phone_ai_repo_dir}}"
_phone_ai_default_data_dir="${HOME}/data"
if [[ -d "${HOME}/autodl-tmp" ]]; then
  _phone_ai_default_data_dir="${HOME}/autodl-tmp/phone-ai-data"
fi
_phone_ai_system_runtime="${HOME}/phone-ai-runtime"
_phone_ai_default_runtime_dir="${_phone_ai_default_data_dir}/runtime"
_phone_ai_default_models_dir="${_phone_ai_default_data_dir}/models"
_phone_ai_system_image=0
if [[ -d "${_phone_ai_system_runtime}/maplab-focal" ]]; then
  _phone_ai_default_runtime_dir="${_phone_ai_system_runtime}"
  _phone_ai_default_models_dir="${_phone_ai_system_runtime}/models"
  _phone_ai_system_image=1
fi
export PHONE_AI_DATA_DIR="${PHONE_AI_DATA_DIR:-${_phone_ai_default_data_dir}}"
export PHONE_AI_RUNTIME_DIR="${PHONE_AI_RUNTIME_DIR:-${_phone_ai_default_runtime_dir}}"
export PHONE_AI_MODELS_DIR="${PHONE_AI_MODELS_DIR:-${_phone_ai_default_models_dir}}"
export PHONE_AI_RECORDINGS_DIR="${PHONE_AI_RECORDINGS_DIR:-${PHONE_AI_DATA_DIR}/recordings}"

export MAPLAB_RUNTIME_ROOT="${MAPLAB_RUNTIME_ROOT:-${PHONE_AI_RUNTIME_DIR}/maplab-focal}"
export MAPLAB_RUNTIME_WORKSPACE="${MAPLAB_RUNTIME_WORKSPACE:-/workspace}"
export MAPLAB_IMPORTER_BINARY="${MAPLAB_IMPORTER_BINARY:-${MAPLAB_RUNTIME_WORKSPACE}/devel/lib/sensor_recorder_importer/sensor_recorder_to_vimap}"
if [[ "${_phone_ai_system_image}" -eq 1 ]]; then
  export LIGHTGLUE_PYTHON="${LIGHTGLUE_PYTHON:-${PHONE_AI_RUNTIME_DIR}/da3/venv/bin/python}"
  export SALAD_PYTHON="${SALAD_PYTHON:-${HOME}/miniconda3/bin/python}"
  # The AutoDL image ships Rerun with the system interpreter.  Do not inherit
  # a stale RERUN_PYTHON from an interactive shell (often a Conda Python).
  export RERUN_PYTHON="${PHONE_AI_RERUN_PYTHON:-/usr/bin/python3}"
  export SUPERPOINT_ONNX_MODEL="${SUPERPOINT_ONNX_MODEL:-${PHONE_AI_RUNTIME_DIR}/third_party/LightGlue-ONNX-v1/weights/superpoint_2048.onnx}"
  export LIGHTGLUE_MATCHER_ONNX_MODEL="${LIGHTGLUE_MATCHER_ONNX_MODEL:-${PHONE_AI_RUNTIME_DIR}/third_party/LightGlue-ONNX-v1/weights/superpoint_lightglue.onnx}"
  export PHONE_AI_SALAD_REPO="${PHONE_AI_SALAD_REPO:-${PHONE_AI_RUNTIME_DIR}/third_party/salad}"
  export PHONE_AI_SALAD_CHECKPOINT="${PHONE_AI_SALAD_CHECKPOINT:-${PHONE_AI_RUNTIME_DIR}/third_party/salad/dino_salad.ckpt}"
  export PHONE_AI_DINOV2_REPO="${PHONE_AI_DINOV2_REPO:-${PHONE_AI_RUNTIME_DIR}/third_party/dinov2}"
  export PHONE_AI_DINOV2_CHECKPOINT="${PHONE_AI_DINOV2_CHECKPOINT:-${PHONE_AI_RUNTIME_DIR}/third_party/dinov2/dinov2_vitb14_pretrain.pth}"
  export TORCH_HOME="${TORCH_HOME:-${PHONE_AI_RUNTIME_DIR}/torch-hub}"
  export PHONE_AI_DA3_REPO="${PHONE_AI_DA3_REPO:-${PHONE_AI_RUNTIME_DIR}/da3/repo}"
  export PHONE_AI_DA3_MODEL="${PHONE_AI_DA3_MODEL:-${PHONE_AI_RUNTIME_DIR}/da3/models/DA3-GIANT-1.1}"
  export PHONE_AI_DA3_PYTHON="${PHONE_AI_DA3_PYTHON:-${PHONE_AI_RUNTIME_DIR}/da3/venv/bin/python}"
  export PHONE_AI_MOSAIC3D_ROOT="${PHONE_AI_MOSAIC3D_ROOT:-${PHONE_AI_RUNTIME_DIR}/mosaic3d}"
else
  export LIGHTGLUE_PYTHON="${LIGHTGLUE_PYTHON:-${PHONE_AI_RUNTIME_DIR}/venvs/lightglue/bin/python}"
  # The portable AutoDL bundle stores the LightGlue/DA3 venv on the data disk
  # and uses the preinstalled Miniconda interpreter for SALAD.
  if [[ -x "${HOME}/miniconda3/bin/python" ]]; then
    _phone_ai_salad_python_default="${HOME}/miniconda3/bin/python"
  else
    _phone_ai_salad_python_default="${PHONE_AI_RUNTIME_DIR}/venvs/phone-ai/bin/python"
  fi
  export SALAD_PYTHON="${SALAD_PYTHON:-${_phone_ai_salad_python_default}}"
  export RERUN_PYTHON="${PHONE_AI_RERUN_PYTHON:-/usr/bin/python3}"
  export SUPERPOINT_ONNX_MODEL="${SUPERPOINT_ONNX_MODEL:-${PHONE_AI_MODELS_DIR}/lightglue/superpoint_2048.onnx}"
  export LIGHTGLUE_MATCHER_ONNX_MODEL="${LIGHTGLUE_MATCHER_ONNX_MODEL:-${PHONE_AI_MODELS_DIR}/lightglue/superpoint_lightglue.onnx}"
  export PHONE_AI_SALAD_REPO="${PHONE_AI_SALAD_REPO:-${PHONE_AI_MODELS_DIR}/salad/repo}"
  export PHONE_AI_SALAD_CHECKPOINT="${PHONE_AI_SALAD_CHECKPOINT:-${PHONE_AI_MODELS_DIR}/salad/dino_salad.ckpt}"
  export PHONE_AI_DINOV2_REPO="${PHONE_AI_DINOV2_REPO:-${PHONE_AI_MODELS_DIR}/dinov2/repo}"
  export PHONE_AI_DINOV2_CHECKPOINT="${PHONE_AI_DINOV2_CHECKPOINT:-${PHONE_AI_MODELS_DIR}/dinov2/dinov2_vitb14_pretrain.pth}"
  export TORCH_HOME="${TORCH_HOME:-${PHONE_AI_MODELS_DIR}/torch-hub}"
  export PHONE_AI_DA3_REPO="${PHONE_AI_DA3_REPO:-${PHONE_AI_MODELS_DIR}/da3/repo}"
  export PHONE_AI_DA3_MODEL="${PHONE_AI_DA3_MODEL:-${PHONE_AI_MODELS_DIR}/da3/DA3-GIANT-1.1}"
  export PHONE_AI_DA3_PYTHON="${PHONE_AI_DA3_PYTHON:-${LIGHTGLUE_PYTHON}}"
  export PHONE_AI_MOSAIC3D_ROOT="${PHONE_AI_MOSAIC3D_ROOT:-${PHONE_AI_MODELS_DIR}/mosaic3d}"
fi

unset _phone_ai_repo_dir _phone_ai_default_data_dir _phone_ai_system_runtime _phone_ai_default_runtime_dir _phone_ai_default_models_dir _phone_ai_system_image _phone_ai_salad_python_default
