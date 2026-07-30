#!/usr/bin/env bash
# Prepare a fresh AutoDL Ubuntu GPU instance for Phone AI.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${HOME}/data"
if [[ -d "${HOME}/autodl-tmp" ]]; then data_dir="${HOME}/autodl-tmp/phone-ai-data"; fi
prefix="${PHONE_AI_RUNTIME_DIR:-${data_dir}/runtime}"
models_dir="${PHONE_AI_MODELS_DIR:-${data_dir}/models}"
runtime_archive="${PHONE_AI_MAPLAB_RUNTIME_ARCHIVE:-}"
assets_archive="${PHONE_AI_ASSETS_ARCHIVE:-}"
python_runtime_archive="${PHONE_AI_PYTHON_RUNTIME_ARCHIVE:-}"
skip_models=0
remove_local_archives=0
downloaded_archives=()

usage() {
  cat <<EOF
usage: bash scripts/bootstrap_autodl.sh [options]

  --prefix <dir>                  Runtime directory (default: ${prefix})
  --models-dir <dir>              Downloaded model assets (default: ${models_dir})
  --maplab-runtime-archive <file-or-url>
                                  Portable Maplab/ROS Noetic runtime archive.
  --assets-archive <file-or-url> Licensed Phone AI model/source asset archive.
  --python-runtime-archive <file-or-url>
                                  Validated CUDA Python runtime for offline setup.
  --remove-local-archives         Remove supplied local archives after extraction.
  --skip-models                   Only prepare dependencies and runtime layout.

The Maplab runtime is intentionally a separately versioned release asset.
Model weights are not bundled because their upstream licenses vary.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) prefix="$2"; shift 2 ;;
    --models-dir) models_dir="$2"; shift 2 ;;
    --maplab-runtime-archive) runtime_archive="$2"; shift 2 ;;
    --assets-archive) assets_archive="$2"; shift 2 ;;
    --python-runtime-archive) python_runtime_archive="$2"; shift 2 ;;
    --remove-local-archives) remove_local_archives=1; shift ;;
    --skip-models) skip_models=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(id -u)" -eq 0 ]] || { echo "run as root on an AutoDL instance" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "an NVIDIA GPU instance is required" >&2; exit 2; }
command -v apt-get >/dev/null || { echo "Ubuntu/Debian apt-get is required" >&2; exit 2; }

runtime_dir="${prefix}"
venv_dir="${runtime_dir}/venvs/phone-ai"
mkdir -p "${runtime_dir}" "${models_dir}" "${data_dir}/tmp"
export TMPDIR="${TMPDIR:-${data_dir}/tmp}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git proot python3 python3-venv python3-pip \
  build-essential cmake pkg-config libgl1 libglib2.0-0 ffmpeg zstd

if [[ -n "${runtime_archive}" && ! -d "${runtime_dir}/maplab-focal" ]]; then
  archive="${runtime_dir}/maplab-runtime.tar.zst"
  if [[ "${runtime_archive}" =~ ^https?:// ]]; then
    curl -L --fail --retry 3 "${runtime_archive}" -o "${archive}"
    downloaded_archives+=("${archive}")
  else
    archive="${runtime_archive}"
    [[ "${remove_local_archives}" -eq 1 ]] && downloaded_archives+=("${archive}")
  fi
  mkdir -p "${runtime_dir}/maplab-focal"
  tar --use-compress-program=unzstd -xf "${archive}" -C "${runtime_dir}/maplab-focal" --strip-components=1
fi

if [[ ! -d "${runtime_dir}/maplab-focal" ]]; then
  cat >&2 <<EOF
Maplab runtime is missing. Re-run with --maplab-runtime-archive <file-or-url>.
The archive is a separately published Phone AI release asset because it contains
the ROS Noetic Proot runtime and compiled Maplab tools.
EOF
  exit 3
fi

if [[ -n "${assets_archive}" ]]; then
  archive="${runtime_dir}/phone-ai-assets.tar.zst"
  if [[ "${assets_archive}" =~ ^https?:// ]]; then
    curl -L --fail --retry 3 "${assets_archive}" -o "${archive}"
    downloaded_archives+=("${archive}")
  else
    archive="${assets_archive}"
    [[ "${remove_local_archives}" -eq 1 ]] && downloaded_archives+=("${archive}")
  fi
  tar --use-compress-program=unzstd -xf "${archive}" -C "${models_dir}" --strip-components=1
fi

bundled_python=0
if [[ -n "${python_runtime_archive}" ]]; then
  archive="${runtime_dir}/phone-ai-python-runtime.tar.zst"
  if [[ "${python_runtime_archive}" =~ ^https?:// ]]; then
    curl -L --fail --retry 3 "${python_runtime_archive}" -o "${archive}"
    downloaded_archives+=("${archive}")
  else
    archive="${python_runtime_archive}"
    [[ "${remove_local_archives}" -eq 1 ]] && downloaded_archives+=("${archive}")
  fi
  tar --use-compress-program=unzstd -xf "${archive}" -C /
  bundled_python=1
fi

for archive in "${downloaded_archives[@]}"; do
  rm -f -- "${archive}"
done

if [[ "${bundled_python}" -eq 0 ]]; then
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    python3 -m venv "${venv_dir}"
  fi
  pip_index="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  "${venv_dir}/bin/python" -m pip install --index-url "${pip_index}" --upgrade pip wheel
  "${venv_dir}/bin/python" -m pip install --index-url "${pip_index}" \
    numpy scipy opencv-python-headless onnxruntime-gpu faiss-cpu \
    rerun-sdk pyyaml pillow tqdm open-clip-torch timm transformers jaxtyping \
    spconv-cu126 pytorch-lightning torchmetrics pytorch-metric-learning
  "${venv_dir}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision
fi

install -m 0644 "${repo_dir}/configs/runtime.env.example" "${repo_dir}/.phoneai.env"
sed -i \
  -e "s|^PHONE_AI_RUNTIME_DIR=.*|PHONE_AI_RUNTIME_DIR=${runtime_dir}|" \
  -e "s|^PHONE_AI_MODELS_DIR=.*|PHONE_AI_MODELS_DIR=${models_dir}|" \
  "${repo_dir}/.phoneai.env"
if [[ "${bundled_python}" -eq 1 ]]; then
  cat >> "${repo_dir}/.phoneai.env" <<'EOF'
SALAD_PYTHON=/root/miniconda3/bin/python
LIGHTGLUE_PYTHON=/root/autodl-tmp/da3/venv/bin/python
PHONE_AI_DA3_PYTHON=/root/autodl-tmp/da3/venv/bin/python
RERUN_PYTHON=python3
EOF
fi

# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"
[[ -x "${PHONE_AI_DA3_PYTHON}" ]] || { echo "DA3 Python path was not created" >&2; exit 2; }
[[ -d "${MAPLAB_RUNTIME_ROOT}/workspace/devel" ]] || { echo "invalid Maplab runtime: ${MAPLAB_RUNTIME_ROOT}" >&2; exit 2; }

if [[ "${skip_models}" -eq 0 && -z "${assets_archive}" ]]; then
  cat <<EOF
Runtime is ready. Place the licensed model assets at:
  ${SUPERPOINT_ONNX_MODEL}
  ${LIGHTGLUE_MATCHER_ONNX_MODEL}
  ${PHONE_AI_SALAD_CHECKPOINT}
  ${PHONE_AI_DINOV2_CHECKPOINT}
  ${PHONE_AI_DA3_MODEL}
  ${PHONE_AI_MOSAIC3D_ROOT}/models/sc+ar+sc++.ckpt
Then run: bash scripts/check_runtime.sh
EOF
fi
