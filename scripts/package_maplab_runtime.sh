#!/usr/bin/env bash
# Create the portable ROS Noetic + compiled Maplab runtime consumed by bootstrap_autodl.sh.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"
output=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    -h|--help)
      echo "usage: bash scripts/package_maplab_runtime.sh --output <maplab-runtime.tar.zst>"
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${output}" ]] || { echo "--output is required" >&2; exit 2; }
[[ -d "${MAPLAB_RUNTIME_ROOT}/workspace/devel" ]] || {
  echo "invalid Maplab runtime: ${MAPLAB_RUNTIME_ROOT}" >&2; exit 2;
}
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }

mkdir -p "$(dirname "${output}")"
tar --zstd -cf "${output}" \
  --exclude='workspace/build' --exclude='workspace/logs' \
  -C "$(dirname "${MAPLAB_RUNTIME_ROOT}")" "$(basename "${MAPLAB_RUNTIME_ROOT}")"
echo "created ${output}"
