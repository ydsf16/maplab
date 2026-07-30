#!/usr/bin/env bash
# Package the validated CUDA Python environment for fully offline AutoDL setup.
set -euo pipefail
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    -h|--help)
      echo "usage: bash scripts/package_python_runtime.sh --output <phone-ai-python-runtime.tar.zst>"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${output}" ]] || { echo "--output is required" >&2; exit 2; }
[[ -x /root/miniconda3/bin/python ]] || { echo "missing /root/miniconda3" >&2; exit 2; }
[[ -x /root/autodl-tmp/da3/venv/bin/python ]] || { echo "missing DA3 virtual environment" >&2; exit 2; }
site_packages=/usr/local/lib/python3.10/dist-packages
[[ -d "${site_packages}/rerun_sdk" ]] || { echo "missing system Rerun SDK" >&2; exit 2; }
[[ -d "${site_packages}/rerun_bindings" ]] || { echo "missing Rerun bindings" >&2; exit 2; }
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }
mkdir -p "$(dirname "${output}")"
# Rerun's Python package is split across its SDK, native bindings and wheel
# dependencies.  Keep the host Python lightweight, but make Rerun functional
# after restoring this archive on a clean AutoDL image.
tar --zstd -cf "${output}" -C / \
  root/miniconda3 root/autodl-tmp/da3/venv \
  usr/local/lib/python3.10/dist-packages/rerun_sdk \
  usr/local/lib/python3.10/dist-packages/rerun_bindings \
  usr/local/lib/python3.10/dist-packages/rerun_sdk.pth \
  usr/local/lib/python3.10/dist-packages/rerun_sdk-*.dist-info \
  usr/local/lib/python3.10/dist-packages/attrs \
  usr/local/lib/python3.10/dist-packages/attr \
  usr/local/lib/python3.10/dist-packages/attrs-*.dist-info \
  usr/local/lib/python3.10/dist-packages/numpy \
  usr/local/lib/python3.10/dist-packages/numpy.libs \
  usr/local/lib/python3.10/dist-packages/numpy-*.dist-info \
  usr/local/lib/python3.10/dist-packages/PIL \
  usr/local/lib/python3.10/dist-packages/pillow.libs \
  usr/local/lib/python3.10/dist-packages/pillow-*.dist-info \
  usr/local/lib/python3.10/dist-packages/psutil \
  usr/local/lib/python3.10/dist-packages/psutil-*.dist-info \
  usr/local/lib/python3.10/dist-packages/pyarrow \
  usr/local/lib/python3.10/dist-packages/pyarrow-*.dist-info \
  usr/local/lib/python3.10/dist-packages/typing_extensions.py \
  usr/local/lib/python3.10/dist-packages/typing_extensions-*.dist-info \
  root/.local/lib/python3.10/site-packages/yaml \
  root/.local/lib/python3.10/site-packages/pyyaml-*.dist-info
echo "created ${output}"
