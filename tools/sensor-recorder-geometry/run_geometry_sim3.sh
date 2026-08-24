#!/usr/bin/env bash
# DA3 self-pose geometry, aligned window-by-window to the VI-BA map with Sim(3).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${repo}/scripts/phoneai_env.sh"

slam=""; data=""; output=""; process_res=504; window_size=100; overlap=20; slam_stage=auto; keep_intermediates=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slam-output) slam="$2"; shift 2 ;;
    --data) data="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --process-res) process_res="$2"; shift 2 ;;
    --window-size) window_size="$2"; shift 2 ;;
    --window-overlap) overlap="$2"; shift 2 ;;
    --slam-stage) slam_stage="$2"; shift 2 ;;
    --keep-intermediates) keep_intermediates=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$slam" && -n "$data" && -n "$output" ]] || { echo "--slam-output, --data and --output are required" >&2; exit 2; }
(( window_size > overlap && overlap >= 0 )) || { echo "window-size must exceed window-overlap" >&2; exit 2; }
[[ "$slam_stage" == auto || "$slam_stage" == initial || "$slam_stage" == final ]] || { echo "slam-stage must be auto, initial, or final" >&2; exit 2; }

py="${PHONE_AI_DA3_PYTHON}"
mkdir -p "$output/windows"
selection="$output/_selection"
"$py" "$repo/tools/sensor-recorder-geometry/prepare_da3_input.py" \
  --slam-output "$slam" --data "$data" --output "$selection" --slam-stage "$slam_stage" --metadata-only
frame_count=$(python3 -c "import json; print(json.load(open('$selection/manifest.json'))['frames'])")

input_args=(); output_args=(); npz_args=(); image_args=(); camera_args=()
start=0; index=0
while (( start < frame_count )); do
  end=$((start + window_size)); (( end > frame_count )) && end=$frame_count
  window="$output/windows/window_$(printf '%03d' "$index")"
  "$py" "$repo/tools/sensor-recorder-geometry/prepare_da3_input.py" \
    --slam-output "$slam" --data "$data" --output "$window/input" --slam-stage "$slam_stage" \
    --start-index "$start" --end-index "$end"
  input_args+=(--input "$window/input")
  output_args+=(--output "$window/da3_output")
  camera_args+=(--camera "$window/input/camera_params.npz")
  (( end == frame_count )) && break
  start=$((end - overlap)); index=$((index + 1))
done

"$py" "$repo/tools/sensor-recorder-geometry/run_da3_windows_unconditioned.py" \
  "${input_args[@]}" "${output_args[@]}" --process-res "$process_res"
"$py" "$repo/tools/sensor-recorder-geometry/align_da3_segments_sim3.py" \
  --windows-root "$output/windows" --output-root "$output/sim3" --overlap "$overlap"

for window in "$output"/windows/window_*; do
  id="$(basename "$window" | sed 's/window_//')"
  npz_args+=(--npz "$output/sim3/window_${id}_global_results.npz")
  image_args+=(--images "$window/input/images")
done
"$py" "$repo/tools/sensor-recorder-geometry/fuse_da3_tsdf.py" \
  "${npz_args[@]}" "${image_args[@]}" --output "$output/tsdf"
"${RERUN_PYTHON}" "$repo/tools/sensor-recorder-geometry/export_geometry_rerun.py" \
  --ply "$output/tsdf/tsdf_pointcloud.ply" "${camera_args[@]}" --output "$output/rerun_geometry.rrd"
"${RERUN_PYTHON}" -m rerun rrd verify "$output/rerun_geometry.rrd"

python3 - <<PY
import json
from pathlib import Path
sim3 = json.loads(Path('$output/sim3/sim3_report.json').read_text())
Path('$output/run_stats.json').write_text(json.dumps({
  'method': 'DA3 self-pose/no-intrinsics per window + Umeyama Sim(3) to VI-BA + TSDF',
  'slam_stage': '$slam_stage', 'window_size_frames': int('$window_size'),
  'window_overlap_frames': int('$overlap'), 'selected_frames': int('$frame_count'),
  'windows': len(sim3), 'process_res': int('$process_res'), 'sim3': sim3,
}, indent=2) + '\\n')
PY
cp "$output/sim3/sim3_report.json" "$output/sim3_report.json"
if [[ "$keep_intermediates" -eq 0 ]]; then
  rm -rf -- "$output/windows" "$output/sim3" "$output/_selection"
  rm -f -- "$output/tsdf/tsdf_mesh_raw.ply"
fi
echo "created $output/tsdf/tsdf_mesh_clean.glb and $output/tsdf/tsdf_pointcloud.ply"
