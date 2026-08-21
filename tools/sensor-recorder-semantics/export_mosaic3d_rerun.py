#!/usr/bin/env python3
"""Export Mosaic3D semantic points with the ARKit trajectory to Rerun."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_rerun_python = "/usr/bin/python3"
if os.path.isfile(_rerun_python) and os.environ.get("PHONE_AI_RERUN_REEXEC") != "1" and os.path.realpath(sys.executable) != _rerun_python:
    os.environ["PHONE_AI_RERUN_REEXEC"] = "1"
    os.execv(_rerun_python, [_rerun_python, __file__, *sys.argv[1:]])

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-npz", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--camera-npz", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = np.load(args.semantic_npz)
    points = scene["points"].astype(np.float32)
    source_colors = scene["source_colors"].astype(np.uint8)
    semantic_colors = scene["semantic_colors"].astype(np.uint8)
    labels = scene["labels"].astype(np.int32)
    confidence = scene["confidence"].astype(np.float32)
    class_names_data = json.loads(args.class_names.read_text())
    if isinstance(class_names_data, list):
        class_names = {index: name for index, name in enumerate(class_names_data)}
    else:
        class_names = {int(class_id): name for class_id, name in class_names_data.items()}
    stats = json.loads(args.stats.read_text())

    cameras = [np.load(camera_path) for camera_path in args.camera_npz]
    c2w = np.concatenate([np.linalg.inv(camera["extrinsics"]).astype(np.float64) for camera in cameras])
    centers = c2w[:, :3, 3]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("MOSAIC3D_ARKIT_194944_SEMANTIC")
    rr.save(str(args.output))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Spatial3DView(
                name="Mosaic3D semantic world",
                origin="/world",
                contents=["/world/**"],
                background=[28, 31, 36],
            ),
            collapse_panels=True,
        )
    )

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/raw_tsdf",
        rr.Points3D(points, colors=source_colors, radii=0.004),
        static=True,
    )
    rr.log(
        "world/semantic/all",
        rr.Points3D(points, colors=semantic_colors, radii=0.005),
        static=True,
    )
    label_positions = []
    label_texts = []
    label_colors = []
    for class_id, class_name in class_names.items():
        mask = labels == class_id
        if not np.any(mask):
            continue
        entity_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", class_name).strip("_")
        rr.log(
            f"world/semantic/by_class/{class_id:03d}_{entity_name}",
            rr.Points3D(
                points[mask],
                colors=semantic_colors[mask],
                radii=0.006,
            ),
            static=True,
        )
        label_positions.append(np.median(points[mask], axis=0))
        label_texts.append(f"{class_name} [{class_id}]")
        label_colors.append(semantic_colors[mask][0])

    if label_positions:
        rr.log(
            "world/semantic/labels",
            rr.Points3D(
                np.asarray(label_positions),
                colors=np.asarray(label_colors),
                radii=rr.Radius.ui_points(5.0),
                labels=label_texts,
                show_labels=True,
            ),
            static=True,
        )

    rr.log(
        "world/trajectory/path",
        rr.LineStrips3D([centers], radii=rr.Radius.ui_points(2.0), colors=[40, 180, 255]),
        static=True,
    )
    rr.log(
        "world/trajectory/keyframes",
        rr.Points3D(centers, radii=rr.Radius.ui_points(3.0), colors=[255, 190, 60]),
        static=True,
    )
    rr.log(
        "world/trajectory/endpoints",
        rr.Points3D(
            [centers[0], centers[-1]],
            radii=rr.Radius.ui_points(7.0),
            colors=[[50, 230, 100], [255, 70, 70]],
            labels=["start", "end"],
            show_labels=True,
        ),
        static=True,
    )
    top_classes = sorted(stats["classes"], key=lambda item: item["points"], reverse=True)[:10]
    summary_lines = [
        f"model: {stats['model']}",
        f"points: {len(points):,}",
        f"voxels: {stats['voxel_count']:,} at {stats['grid_size_m']:.3f} m",
        f"peak GPU: {stats['peak_gpu_gib']:.2f} GiB",
        f"mean confidence: {float(confidence.mean()):.4f}",
        "",
        "Top semantic classes:",
    ]
    summary_lines.extend(
        f"- {item['class']}: {item['points']:,} ({100 * item['fraction']:.2f}%)"
        for item in top_classes
    )
    rr.log("metadata", rr.TextDocument("\n".join(summary_lines)), static=True)

    rr.disconnect()
    print(f"wrote {args.output} with {len(points):,} semantic points")


if __name__ == "__main__":
    main()
