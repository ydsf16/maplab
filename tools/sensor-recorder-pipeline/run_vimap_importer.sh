#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"
runtime_root="${MAPLAB_RUNTIME_ROOT}"
runtime_workspace="${MAPLAB_RUNTIME_WORKSPACE}"
importer="${MAPLAB_IMPORTER_BINARY}"

command -v proot >/dev/null 2>&1 || {
  echo "proot is required to run the isolated Maplab importer" >&2
  exit 2
}
[[ -d "${runtime_root}" ]] || {
  echo "Maplab runtime root does not exist: ${runtime_root}" >&2
  exit 2
}

bind_arguments=()
for argument in "$@"; do
  case "${argument}" in
    --normalized_data=*|--output_map=*)
      host_path="${argument#*=}"
      bind_path="${host_path}"
      while [[ ! -e "${bind_path}" && "${bind_path}" != "/" ]]; do
        bind_path="$(dirname "${bind_path}")"
      done
      if [[ "${bind_path}" == "/" ]]; then
        echo "Unable to find an existing parent for: ${host_path}" >&2
        exit 2
      fi
      bind_arguments+=("-b" "${bind_path}:${bind_path}")
      ;;
  esac
done

exec proot -R "${runtime_root}" "${bind_arguments[@]}" \
  -w "${runtime_workspace}" /bin/bash -lc \
  'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"; status=$?; exit "${status}"' \
  bash "${importer}" "$@"
