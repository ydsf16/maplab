#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo_dir}/scripts/phoneai_env.sh"
output=""
sessions=()
data_root="${PHONE_AI_RECORDINGS_DIR}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --session) sessions+=("$2"); shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    *) echo "usage: $0 --output <dir> --session <single-session-result> [--session ...]" >&2; exit 2 ;;
  esac
done
[[ -n "$output" && "${#sessions[@]}" -ge 2 ]] || { echo "need output and at least two sessions" >&2; exit 2; }
output="$(realpath -m "$output")"
runtime_root="${MAPLAB_RUNTIME_ROOT}"
runtime_workspace="${MAPLAB_RUNTIME_WORKSPACE}"
native="$runtime_workspace/devel/lib/sensor_recorder_importer"
mkdir -p "$output/session_exports" "$output/21_multisession_verified_loops" "$output/22_multisession_posegraph" "$output/23_multisession_observation_fusion" "$output/24_multisession_vi_ba"

run_native() {
  proot -R "$runtime_root" -b "$output:$output" -b "$data_root:$data_root" -w "$runtime_workspace" \
    /bin/bash -lc 'source /opt/ros/noetic/setup.bash; source /workspace/devel/setup.bash; "$@"' bash "$@"
}

map_csv=""
printf 'sessions:\n' > "$output/sessions.yaml"
for result in "${sessions[@]}"; do
  result="$(realpath "$result")"
  id="$(basename "$result" _full)"
  source="$result/maps/08_visual_inertial_ba_loops_preview/vi_map"
  [[ -d "$source" ]] || source="$result/maps/04_initial_vi_ba_intrinsics/vi_map"
  export_dir="$output/session_exports/$id"
  rm -rf "$export_dir"; mkdir -p "$export_dir"
  run_native "$native/sensor_recorder_vimap_export" --map="$source" --output="$export_dir"
  printf '  - id: "%s"\n    result: "%s"\n    export: "%s"\n    raw_data: "%s"\n' "$id" "$result" "$export_dir" "$data_root/$id" >> "$output/sessions.yaml"
  map_csv+="${map_csv:+,}$source"
done

run_native "$native/sensor_recorder_multisession_merge" --maps="$map_csv" --output="$output/22_multisession_posegraph/joint_vimap"

TORCH_HOME="${TORCH_HOME}" "${SALAD_PYTHON}" \
  "$repo_dir/tools/sensor-recorder-multisession/multisession_loop_closure.py" \
  --sessions-yaml "$output/sessions.yaml" --output "$output/21_multisession_verified_loops" \
  --superpoint-model "${SUPERPOINT_ONNX_MODEL}" \
  --lightglue-matcher-model "${LIGHTGLUE_MATCHER_ONNX_MODEL}" \
  --salad-repo "${PHONE_AI_SALAD_REPO}" --salad-checkpoint "${PHONE_AI_SALAD_CHECKPOINT}" \
  --dinov2-repo "${PHONE_AI_DINOV2_REPO}" --dinov2-checkpoint "${PHONE_AI_DINOV2_CHECKPOINT}"

loops="$output/21_multisession_verified_loops/pgo_cross_session_loops.yaml"
if [[ ! -s "$loops" || "$(grep -c 'camera_from:' "$loops" || true)" -eq 0 ]]; then
  echo "No session-pair RANSAC consensus; joint map retained without PGO."; exit 0
fi
run_native "$native/sensor_recorder_posegraph_relax" --map="$output/22_multisession_posegraph/joint_vimap" --loops_yaml="$loops" --accepted_loops_yaml="$output/22_multisession_posegraph/accepted_loops_after_pgo.yaml" --min_switch_variable=0.8 --max_mahalanobis_squared=12.59
cp -a "$output/22_multisession_posegraph/joint_vimap" "$output/23_multisession_observation_fusion/joint_vimap"
run_native "$native/sensor_recorder_loop_observations" --map="$output/23_multisession_observation_fusion/joint_vimap" --loops_yaml="$output/22_multisession_posegraph/accepted_loops_after_pgo.yaml" --min_merge_support=1
cp -a "$output/23_multisession_observation_fusion/joint_vimap" "$output/24_multisession_vi_ba/joint_vimap"
run_native "$native/sensor_recorder_brisk_ba" --map="$output/24_multisession_vi_ba/joint_vimap" --report="$output/24_multisession_vi_ba/report.json" --run_frontend=false --run_ba=true --use_imu=true --optimize_biases=true --optimize_velocity=true --optimize_intrinsics=false --optimize_extrinsics=false --ba_feature_type=SuperPoint --ba_iterations=50 --ba_outlier_rejection_use_reprojection_error=true --ba_outlier_rejection_max_reprojection_error_px=5.0 --ba_outlier_rejection_reject_every_n_iters=3 --prune_bad_landmarks=true
echo "Multi-session VI-BA complete: $output"
