#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result=""
stage="auto"
runtime_root="${MAPLAB_RUNTIME_ROOT:-/root/autodl-tmp/maplab-focal}"
runtime_workspace="${MAPLAB_RUNTIME_WORKSPACE:-/workspace}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) result="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
usage: run_pose_export.sh --output <result> [--stage auto|initial|final]

Exports poses/imu_poses_tum.txt at all VIWLS IMU timestamps and
poses/image_poses_tum.txt at all frames.csv timestamps covered by the VIWLS IMU.
imu_poses_tum.txt contains T_M_I; image_poses_tum.txt contains T_M_C.
EOF
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${result}" ]] || { echo "--output is required" >&2; exit 2; }
[[ "${stage}" == "auto" || "${stage}" == "initial" || "${stage}" == "final" ]] || {
  echo "--stage must be auto, initial, or final" >&2; exit 2;
}
command -v proot >/dev/null || { echo "proot is required" >&2; exit 2; }
result="$(realpath "${result}")"

initial_map="${result}/maps/04_initial_vi_ba_intrinsics/vi_map"
final_map="${result}/maps/08_visual_inertial_ba_loops_preview/vi_map"
case "${stage}" in
  final) map_path="${final_map}" ;;
  initial) map_path="${initial_map}" ;;
  auto)
    if [[ -d "${final_map}" ]]; then map_path="${final_map}"; else map_path="${initial_map}"; fi ;;
esac
[[ -d "${map_path}" ]] || { echo "missing optimized VI-Map: ${map_path}" >&2; exit 2; }
[[ -f "${result}/normalized/frames.csv" ]] || { echo "missing normalized frames.csv" >&2; exit 2; }

pose_dir="${result}/poses"
rm -rf -- "${pose_dir}"
mkdir -p "${pose_dir}"
proot -R "${runtime_root}" -b "${result}:${result}" -w "${runtime_workspace}" \
  /bin/bash -lc \
  'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"' \
  bash "${runtime_workspace}/devel/lib/sensor_recorder_importer/sensor_recorder_pose_export" \
  --map="${map_path}" --image_timestamps="${result}/normalized/frames.csv" \
  --output="${pose_dir}"

stage_dir="$(dirname "${map_path}")"
rrd="${result}/rerun/$(basename "${result}")_dense_poses.rrd"
rerun_python="${RERUN_PYTHON:-python3}"
mkdir -p "$(dirname "${rrd}")"
"${rerun_python}" "${repo_dir}/tools/sensor-recorder-pipeline/export_vimap_rerun.py" \
  --keyframes "${result}/normalized/keyframes.csv" \
  --report "${stage_dir}/report.json" \
  --config "${repo_dir}/configs/iphone_arkit_640.json" \
  --images-dir "${result}/normalized/keyframe_images" \
  --optimized-dir "${stage_dir}/export" \
  --pairs-csv "${result}/features/superpoint_lightglue/pairs.csv" \
  --loops-yaml "${result}/loops/salad_lightglue_pnp/verified_loops.yaml" \
  --pgo-decisions-yaml "${result}/loops/salad_lightglue_pnp/accepted_loops_after_pgo.yaml" \
  --imu-poses-tum "${pose_dir}/imu_poses_tum.txt" \
  --image-poses-tum "${pose_dir}/image_poses_tum.txt" \
  --output "${rrd}"
"${rerun_python}" -m rerun rrd verify "${rrd}"
echo "created ${pose_dir}/imu_poses_tum.txt, ${pose_dir}/image_poses_tum.txt, and ${rrd}"
