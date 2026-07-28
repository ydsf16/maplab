#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
result=""
force=0
run_visual_ba=0
runtime_root="${MAPLAB_RUNTIME_ROOT:-/root/autodl-tmp/maplab-focal}"
runtime_workspace="${MAPLAB_RUNTIME_WORKSPACE:-/workspace}"
salad_python="${SALAD_PYTHON:-/root/miniconda3/bin/python}"
lightglue_model="${LIGHTGLUE_ONNX_MODEL:-/root/autodl-tmp/third_party/LightGlue-ONNX-v1/weights/superpoint_2048_lightglue_end2end.onnx}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) result="$2"; shift 2 ;;
    --run-visual-ba) run_visual_ba=1; shift ;;
    --force) force=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$result" ]] || { echo "--output is required" >&2; exit 2; }
result="$(realpath "$result")"
source_stage="$result/maps/04_visual_ba_intrinsics"
source_map="$source_stage/vi_map"
[[ -d "$source_map" ]] || { echo "missing $source_map" >&2; exit 2; }
[[ -x "$salad_python" && -f "$lightglue_model" ]] || { echo "SALAD or LightGlue runtime missing" >&2; exit 2; }
command -v proot >/dev/null || { echo "proot is required" >&2; exit 2; }

loops="$result/loops/salad_lightglue_pnp"
stage06="$result/maps/06_posegraph_salad_lightglue"
stage07="$result/maps/07_global_visual_ba_loops"
stage08="$result/maps/08_visual_inertial_ba_loops_preview"
if [[ "$force" -eq 1 ]]; then
  rm -rf -- "$loops" "$stage06" "$stage07" "$stage08"
elif [[ -e "$stage08" || -e "$loops" ]]; then
  echo "loop stages exist; pass --force to rebuild" >&2; exit 2
fi
mkdir -p "$loops/source_export" "$stage06" "$stage07" "$stage08" "$result/rerun"

run_native() {
  proot -R "$runtime_root" -b "$result:$result" -w "$runtime_workspace" \
    /bin/bash -lc 'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"' bash "$@"
}
run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_vimap_export" \
  --map="$source_map" --output="$loops/source_export"

TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/torch-hub}" \
  "$salad_python" "$repo_dir/tools/sensor-recorder-loop-closure/salad_lightglue_loop_closure.py" \
  --normalized-data "$result/normalized" --optimized-dir "$loops/source_export" \
  --optimized-report "$source_stage/report.json" \
  --output "$loops" --lightglue-model "$lightglue_model"

if [[ ! -s "$loops/verified_loops.yaml" || "$(grep -c 'camera_from:' "$loops/verified_loops.yaml" || true)" -eq 0 ]]; then
  echo "No verified loops; wrote diagnostics to $loops and skipped optimization."
  exit 0
fi
cp -a "$source_map" "$stage06/vi_map"
run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_posegraph_relax" \
  --map="$stage06/vi_map" --loops_yaml="$loops/verified_loops.yaml" \
  --accepted_loops_yaml="$loops/accepted_loops_after_pgo.yaml" \
  --min_switch_variable=0.8 --max_mahalanobis_squared=12.59
run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_loop_observations" \
  --map="$stage06/vi_map" --loops_yaml="$loops/accepted_loops_after_pgo.yaml" \
  --min_merge_support=1
cp "$source_stage/report.json" "$stage06/report.json"

if [[ "$run_visual_ba" -eq 1 ]]; then
  cp -a "$stage06/vi_map" "$stage07/vi_map"
  run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_brisk_ba" \
    --map="$stage07/vi_map" --report="$stage07/report.json" --run_frontend=false --run_ba=true \
    --use_imu=false --optimize_biases=false --optimize_velocity=false --optimize_intrinsics=false \
    --optimize_extrinsics=false --ba_feature_type=SuperPoint --ba_iterations=50 --prune_bad_landmarks=true
  cp -a "$stage07/vi_map" "$stage08/vi_map"
else
  rm -rf -- "$stage07"
  cp -a "$stage06/vi_map" "$stage08/vi_map"
fi
run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_brisk_ba" \
  --map="$stage08/vi_map" --report="$stage08/report.json" --run_frontend=false --run_ba=true \
  --use_imu=true --optimize_biases=true --optimize_velocity=true --optimize_intrinsics=false \
  --optimize_extrinsics=false --ba_feature_type=SuperPoint --ba_iterations=50 --prune_bad_landmarks=true

for stage in "$stage06" "$stage08"; do
  run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_vimap_export" \
    --map="$stage/vi_map" --output="$stage/export"
  if [[ -f "$stage/report.json" ]]; then
    python3 "$repo_dir/tools/sensor-recorder-pipeline/export_vimap_rerun.py" \
      --keyframes "$result/normalized/keyframes.csv" --report "$stage/report.json" \
      --config "$repo_dir/configs/iphone_arkit_640.json" --images-dir "$result/normalized/keyframe_images" \
      --optimized-dir "$stage/export" --loops-yaml "$loops/verified_loops.yaml" \
      --pgo-decisions-yaml "$loops/accepted_loops_after_pgo.yaml" \
      --output "$result/rerun/$(basename "$stage").rrd"
  fi
done
if [[ "$run_visual_ba" -eq 1 ]]; then
  run_native "$runtime_workspace/devel/lib/sensor_recorder_importer/sensor_recorder_vimap_export" \
    --map="$stage07/vi_map" --output="$stage07/export"
  python3 "$repo_dir/tools/sensor-recorder-pipeline/export_vimap_rerun.py" \
    --keyframes "$result/normalized/keyframes.csv" --report "$stage07/report.json" \
    --config "$repo_dir/configs/iphone_arkit_640.json" --images-dir "$result/normalized/keyframe_images" \
    --optimized-dir "$stage07/export" --loops-yaml "$loops/verified_loops.yaml" \
    --pgo-decisions-yaml "$loops/accepted_loops_after_pgo.yaml" \
    --output "$result/rerun/$(basename "$stage07").rrd"
fi
echo "Loop, pose graph, visual BA and VI-BA preview complete."
