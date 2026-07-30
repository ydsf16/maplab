#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "full" ]]; then
  shift
  data=""
  output=""
  config="${repo_dir}/configs/iphone_arkit_640.json"
  force=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data) data="$2"; shift 2 ;;
      --output) output="$2"; shift 2 ;;
      --config) config="$2"; shift 2 ;;
      --force) force=1; shift ;;
      -h|--help)
        cat <<'EOF'
usage: process.sh full --data <SensorRecorder folder> --output <result folder> [--config <camera json>] [--force]

Runs: validation/import -> SuperPoint+LightGlue -> initial VI-BA ->
SALAD/LightGlue/PnP loop closure -> PGO -> PGO-gated observation fusion -> final VI-BA preview.
The default camera configuration is configs/iphone_arkit_640.json.
EOF
        exit 0 ;;
      *) echo "unknown full-pipeline argument: $1" >&2; exit 2 ;;
    esac
  done
  [[ -n "${data}" && -n "${output}" ]] || {
    echo "full requires --data and --output" >&2; exit 2;
  }
  import_args=(single --data "${data}" --output "${output}" --config "${config}")
  stage_args=(--output "${output}")
  if [[ "${force}" -eq 1 ]]; then
    import_args+=(--force)
    stage_args+=(--force)
  fi
  pipeline_timing_file="${output}/pipeline_timing.tsv"
  full_start="$(date +%s)"
  "${python_bin}" "${repo_dir}/tools/sensor-recorder-pipeline/pipeline.py" "${import_args[@]}"
  import_end="$(date +%s)"
  "${repo_dir}/tools/sensor-recorder-features/run_superpoint_lightglue.sh" "${stage_args[@]}" --initial-vi-ba
  frontend_vi_ba_end="$(date +%s)"
  "${repo_dir}/tools/sensor-recorder-loop-closure/run_loop_closure.sh" "${stage_args[@]}"
  loop_end="$(date +%s)"
  "${repo_dir}/tools/sensor-recorder-pipeline/run_pose_export.sh" --output "${output}"
  full_end="$(date +%s)"
  printf 'stage\twall_seconds\nimport\t%s\nfrontend_and_initial_vi_ba\t%s\nloop_pgo_fusion_and_final_vi_ba\t%s\npose_export\t%s\ntotal\t%s\n' \
    "$((import_end - full_start))" "$((frontend_vi_ba_end - import_end))" \
    "$((loop_end - frontend_vi_ba_end))" "$((full_end - loop_end))" \
    "$((full_end - full_start))" > "${pipeline_timing_file}"
  echo "created ${pipeline_timing_file}"
  exit 0
fi

if [[ "${1:-}" == "superpoint-lightglue" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-features/run_superpoint_lightglue.sh" "$@"
fi

if [[ "${1:-}" == "geometry" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-geometry/run_geometry.sh" "$@"
fi

if [[ "${1:-}" == "semantics" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-semantics/run_semantics.sh" "$@"
fi

if [[ "${1:-}" == "loop-closure" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-loop-closure/run_loop_closure.sh" "$@"
fi

if [[ "${1:-}" == "export-poses" ]]; then
  shift
  exec "${repo_dir}/tools/sensor-recorder-pipeline/run_pose_export.sh" "$@"
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "error: ${python_bin} is required (Ubuntu: apt-get install -y python3)" >&2
  exit 127
fi

exec "${python_bin}" \
  "${repo_dir}/tools/sensor-recorder-pipeline/pipeline.py" "$@"
