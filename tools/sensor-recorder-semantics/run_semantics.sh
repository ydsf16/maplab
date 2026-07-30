#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
geometry=""; output=""
while [[ $# -gt 0 ]]; do case "$1" in
  --geometry-output) geometry="$2"; shift 2;;
  --output) output="$2"; shift 2;;
  -h|--help)
    echo "usage: process.sh semantics --geometry-output <geometry folder> --output <result folder>"
    exit 0;;
  *) echo "Unknown option: $1" >&2; exit 2;;
esac; done
[[ -n "$geometry" && -n "$output" ]] || { echo "--geometry-output and --output are required" >&2; exit 2; }
py=/root/autodl-tmp/da3/venv/bin/python
assets=/root/autodl-tmp/mosaic3d
mkdir -p "$output"
PYTHONPATH="$assets/repo" "$py" "$repo/tools/sensor-recorder-semantics/mosaic3d_custom_infer.py" \
  --repo "$assets/repo" --checkpoint "$assets/models/sc+ar+sc++.ckpt" \
  --clip-config-dir "$assets/models/recap_clip" \
  --input-ply "$geometry/tsdf/tsdf_pointcloud.ply" --output-dir "$output" \
  --class-catalog "$repo/tools/sensor-recorder-semantics/semantic_ontology_compact_v1.json" \
  --profile all --world-coordinate maplab_z_up
camera_args=(); while IFS= read -r camera; do camera_args+=(--camera-npz "$camera"); done < <(find "$geometry/windows" -path '*/input/camera_params.npz' | sort)
python3 "$repo/tools/sensor-recorder-semantics/export_mosaic3d_rerun.py" \
  --semantic-npz "$output/semantic_scene.npz" --class-names "$output/class_names.json" \
  --stats "$output/semantic_stats.json" "${camera_args[@]}" --output "$output/rerun_semantic.rrd"
python3 -m rerun rrd verify "$output/rerun_semantic.rrd"
