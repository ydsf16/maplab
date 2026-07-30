#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; joint=""; output=""
while [[ $# -gt 0 ]]; do case "$1" in --joint-output) joint="$2";shift 2;;--output) output="$2";shift 2;;*) exit 2;;esac;done
[[ -n "$joint" && -n "$output" ]] || exit 2
py=/root/autodl-tmp/da3/venv/bin/python; "$py" "$repo/tools/sensor-recorder-multisession/prepare_multisession_geometry.py" --joint-output "$joint" --output "$output"
in=(); out=(); npz=(); imgs=(); while IFS= read -r d; do in+=(--input "$d"); out+=(--output "${d%/input}/da3_output"); npz+=(--npz "${d%/input}/da3_output/exports/mini_npz/results.npz"); imgs+=(--images "$d/images"); done < <(find "$output/windows" -path '*/input' -type d | sort)
"$py" "$repo/tools/sensor-recorder-geometry/run_da3_windows.py" "${in[@]}" "${out[@]}" --process-res 504
"$py" "$repo/tools/sensor-recorder-geometry/fuse_da3_tsdf.py" "${npz[@]}" "${imgs[@]}" --output "$output/tsdf"
cams=(); while IFS= read -r c; do cams+=(--camera "$c"); done < <(find "$output/windows" -path '*/input/camera_params.npz' | sort)
python3 "$repo/tools/sensor-recorder-geometry/export_geometry_rerun.py" --ply "$output/tsdf/tsdf_pointcloud.ply" "${cams[@]}" --output "$output/rerun_multisession_geometry.rrd"
python3 -m rerun rrd verify "$output/rerun_multisession_geometry.rrd"
