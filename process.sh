#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "superpoint-lightglue" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-features/run_superpoint_lightglue.sh" "$@"
fi

if [[ "${1:-}" == "loop-closure" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-loop-closure/run_loop_closure.sh" "$@"
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "error: ${python_bin} is required (Ubuntu: apt-get install -y python3)" >&2
  exit 127
fi

exec "${python_bin}" \
  "${repo_dir}/tools/sensor-recorder-pipeline/pipeline.py" "$@"
