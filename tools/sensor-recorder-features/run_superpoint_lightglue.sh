#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result=""
model="${LIGHTGLUE_ONNX_MODEL:-/root/autodl-tmp/third_party/LightGlue-ONNX-v1/weights/superpoint_2048_lightglue_end2end.onnx}"
onnx_python="${LIGHTGLUE_PYTHON:-/root/autodl-tmp/venvs/lightglue/bin/python}"
runtime_root="${MAPLAB_RUNTIME_ROOT:-/root/autodl-tmp/maplab-focal}"
runtime_workspace="${MAPLAB_RUNTIME_WORKSPACE:-/workspace}"
max_pair_gap=3
force=0
visual_ba=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) result="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --python) onnx_python="$2"; shift 2 ;;
    --max-pair-gap) max_pair_gap="$2"; shift 2 ;;
    --visual-ba) visual_ba=1; shift ;;
    --force) force=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${result}" ]] || { echo "--output is required" >&2; exit 2; }
result="$(realpath "${result}")"
[[ -d "${result}/maps/00_imported/vi_map" ]] || {
  echo "missing imported VI-Map: ${result}/maps/00_imported/vi_map" >&2
  exit 2
}
[[ -f "${model}" ]] || { echo "missing ONNX model: ${model}" >&2; exit 2; }
[[ -x "${onnx_python}" ]] || { echo "missing Python runtime: ${onnx_python}" >&2; exit 2; }
command -v proot >/dev/null || { echo "proot is required" >&2; exit 2; }

features="${result}/features/superpoint_lightglue"
if [[ "${visual_ba}" -eq 1 ]]; then
  stage="${result}/maps/02_superpoint_lightglue_visual_ba"
  rrd="${result}/rerun/$(basename "${result}")_superpoint_lightglue_visual_ba.rrd"
else
  stage="${result}/maps/01_superpoint_lightglue_no_ba"
  rrd="${result}/rerun/$(basename "${result}")_superpoint_lightglue_no_ba.rrd"
fi
if [[ -e "${stage}" && "${force}" -ne 1 ]]; then
  echo "stage already exists; pass --force to rebuild: ${stage}" >&2
  exit 2
fi
if [[ "${force}" -eq 1 ]]; then
  rm -rf -- "${features}" "${stage}"
fi
mkdir -p "${features}" "${stage}" "$(dirname "${rrd}")"

"${onnx_python}" "${repo_dir}/tools/sensor-recorder-features/superpoint_lightglue_onnx.py" \
  --normalized-data "${result}/normalized" \
  --model "${model}" \
  --output "${features}" \
  --max-pair-gap "${max_pair_gap}" \
  --provider cuda

cp -a "${result}/maps/00_imported/vi_map" "${stage}/vi_map"
ba_arguments=(--run_ba=false)
if [[ "${visual_ba}" -eq 1 ]]; then
  ba_arguments=(
    --run_ba=true --use_imu=false --optimize_biases=false
    --optimize_velocity=false --optimize_extrinsics=false
    --ba_iterations=30 --ba_feature_type=SuperPoint
    --ba_outlier_rejection_use_reprojection_error=true
    --ba_outlier_rejection_max_reprojection_error_px=3.0
    --ba_outlier_rejection_reject_every_n_iters=3
  )
fi
proot -R "${runtime_root}" -b "${result}:${result}" -w "${runtime_workspace}" \
  /bin/bash -lc \
  'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"' \
  bash "${runtime_workspace}/devel/lib/sensor_recorder_importer/sensor_recorder_brisk_ba" \
  --map="${stage}/vi_map" --report="${stage}/report.json" \
  --run_frontend=false --tracks_csv="${features}/keypoints.csv" \
  --prune_bad_landmarks=true "${ba_arguments[@]}"

mkdir -p "${stage}/export"
proot -R "${runtime_root}" -b "${result}:${result}" -w "${runtime_workspace}" \
  /bin/bash -lc \
  'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"' \
  bash "${runtime_workspace}/devel/lib/sensor_recorder_importer/sensor_recorder_vimap_export" \
  --map="${stage}/vi_map" --output="${stage}/export"

rerun_python="${RERUN_PYTHON:-python3}"
"${rerun_python}" "${repo_dir}/tools/sensor-recorder-pipeline/export_vimap_rerun.py" \
  --keyframes "${result}/normalized/keyframes.csv" \
  --report "${stage}/report.json" \
  --config "${repo_dir}/configs/iphone_arkit_640.json" \
  --images-dir "${result}/normalized/keyframe_images" \
  --optimized-dir "${stage}/export" \
  --pairs-csv "${features}/pairs.csv" \
  --output "${rrd}"
"${rerun_python}" -m rerun rrd verify "${rrd}"
echo "created ${rrd}"
