#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; slam=""; data=""; output=""; res=504; window_size=20; overlap=4; keep_intermediates=0
# shellcheck disable=SC1091
source "${repo}/scripts/phoneai_env.sh"
while [[ $# -gt 0 ]]; do case "$1" in --slam-output) slam="$2";shift 2;; --data) data="$2";shift 2;; --output) output="$2";shift 2;; --process-res) res="$2";shift 2;; --window-size) window_size="$2";shift 2;; --window-overlap) overlap="$2";shift 2;; --keep-intermediates) keep_intermediates=1;shift;; *) echo "Unknown option: $1" >&2; exit 2;; esac; done
[[ -n "$slam" && -n "$data" && -n "$output" ]] || exit 2
(( window_size > overlap && overlap >= 0 )) || { echo "window-size must exceed window-overlap" >&2; exit 2; }
py="${PHONE_AI_DA3_PYTHON}"; mkdir -p "$output/windows"
selection="$output/_selection"; "$py" "$repo/tools/sensor-recorder-geometry/prepare_da3_input.py" --slam-output "$slam" --data "$data" --output "$selection" --max-frames 0 --metadata-only
frame_count=$(python3 -c "import json; print(json.load(open('$selection/manifest.json'))['frames'])")
npz_args=(); image_args=(); da3_input_args=(); da3_output_args=(); start=0; window_id=0
while (( start < frame_count )); do
  end=$((start + window_size)); (( end > frame_count )) && end=$frame_count
  work="$output/windows/window_$(printf '%03d' "$window_id")"; mkdir -p "$work"
  "$py" "$repo/tools/sensor-recorder-geometry/prepare_da3_input.py" --slam-output "$slam" --data "$data" --output "$work/input" --max-frames 0 --start-index "$start" --end-index "$end"
  da3_input_args+=(--input "$work/input"); da3_output_args+=(--output "$work/da3_output")
  npz_args+=(--npz "$work/da3_output/exports/mini_npz/results.npz"); image_args+=(--images "$work/input/images")
  (( end == frame_count )) && break
  start=$((end - overlap)); window_id=$((window_id + 1))
done
"$py" "$repo/tools/sensor-recorder-geometry/run_da3_windows.py" "${da3_input_args[@]}" "${da3_output_args[@]}" --process-res "$res"
"$py" "$repo/tools/sensor-recorder-geometry/fuse_da3_tsdf.py" "${npz_args[@]}" "${image_args[@]}" --output "$output/tsdf"
camera_args=(); while IFS= read -r camera; do camera_args+=(--camera "$camera"); done < <(find "$output/windows" -path '*/input/camera_params.npz' | sort)
"${RERUN_PYTHON}" "$repo/tools/sensor-recorder-geometry/export_geometry_rerun.py" --ply "$output/tsdf/tsdf_pointcloud.ply" "${camera_args[@]}" --output "$output/rerun_geometry.rrd"
python3 - <<PY
import glob, json
from pathlib import Path
window_stats=[json.load(open(path)) for path in glob.glob('$output/windows/*/da3_output/run_stats.json')]
stats={'window_size_frames':int('$window_size'),'window_overlap_frames':int('$overlap'),'selected_frames':int('$frame_count'),'windows':len(window_stats),'integrated_depth_observations':sum(item['frames'] for item in window_stats),'process_res':int('$res'),'da3_model_load_seconds':window_stats[0]['model_load_seconds_shared'],'da3_inference_seconds':sum(item['inference_seconds'] for item in window_stats)}
Path('$output/run_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
PY
if [[ "$keep_intermediates" -eq 0 ]]; then
  rm -rf -- "$output/windows"
  rm -f -- "$output/tsdf/tsdf_mesh_raw.ply"
  echo "cleaned DA3 windows and raw mesh; pass --keep-intermediates to retain them"
fi
