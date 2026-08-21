#!/usr/bin/env python3
"""Prepare pose-conditioned DA3 inputs directly from raw ARKit poses."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=float)


def rotation_angle(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slam-output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--min-translation-m", type=float, default=0.15)
    p.add_argument("--min-rotation-deg", type=float, default=5.0)
    p.add_argument("--max-interval-s", type=float, default=1.0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=-1)
    p.add_argument("--metadata-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(a.config.read_text())
    r_ci = np.asarray(cfg["extrinsics"]["rotation"], dtype=float)
    t_ci = np.asarray(cfg["extrinsics"]["translation_m"], dtype=float)
    t_ci_h = np.eye(4); t_ci_h[:3, :3] = r_ci; t_ci_h[:3, 3] = t_ci
    t_ic_h = np.linalg.inv(t_ci_h)

    frames = list(csv.DictReader((a.slam_output / "normalized" / "frames.csv").open()))
    selected, last = [], None
    for row in frames:
        t = float(row["timestamp_ns"]) * 1e-9
        t_m_i = np.eye(4)
        t_m_i[:3, :3] = quat_wxyz_to_matrix([float(row[k]) for k in ("q_M_I_w", "q_M_I_x", "q_M_I_y", "q_M_I_z")])
        t_m_i[:3, 3] = [float(row[k]) for k in ("p_M_I_x_m", "p_M_I_y_m", "p_M_I_z_m")]
        # normalized/frames.csv is the directly transformed ARKit pose as T_M_I.
        # Undo only the configured initial T_C_I to recover raw ARKit T_M_C.
        t_m_c = t_m_i @ t_ic_h
        if last is None or np.linalg.norm(t_m_c[:3, 3] - last[1][:3, 3]) >= a.min_translation_m or rotation_angle(last[1][:3, :3], t_m_c[:3, :3]) >= a.min_rotation_deg or t - last[0] >= a.max_interval_s:
            selected.append((row, t, t_m_c)); last = (t, t_m_c)
    end = len(selected) if a.end_index < 0 else min(a.end_index, len(selected))
    selected = selected[a.start_index:end]
    if not selected:
        raise RuntimeError("Selected ARKit frame interval is empty")
    if a.metadata_only:
        (a.output / "manifest.json").write_text(json.dumps({"frames": len(selected), "selected_frame_range": [a.start_index, end], "pose_source": "raw_arkit"}, indent=2) + "\n")
        return

    cap = cv2.VideoCapture(str(a.data / "wide.mp4")); images = a.output / "images"; images.mkdir(exist_ok=True)
    intrinsics, extrinsics, rows = [], [], []
    for i, (row, t, t_m_c) in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["record_slot"])); ok, image = cap.read()
        if not ok:
            raise RuntimeError("cannot decode record_slot " + row["record_slot"])
        name = f"frame_{i:06d}.jpg"; cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sx, sy = image.shape[1] / float(row["width_px"]), image.shape[0] / float(row["height_px"])
        k = np.array([[float(row["fx_px"]) * sx, 0, float(row["cx_px"]) * sx], [0, float(row["fy_px"]) * sy, float(row["cy_px"]) * sy], [0, 0, 1]], dtype=float)
        intrinsics.append(k); extrinsics.append(np.linalg.inv(t_m_c)); rows.append({"da3_index": i, "record_slot": int(row["record_slot"]), "timestamp_ns": int(row["timestamp_ns"]), "image": f"images/{name}"})
    cap.release()
    np.savez_compressed(a.output / "camera_params.npz", intrinsics=np.asarray(intrinsics), extrinsics=np.asarray(extrinsics))
    with (a.output / "frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (a.output / "manifest.json").write_text(json.dumps({"frames": len(rows), "selected_frame_range": [a.start_index, end], "pose_source": "raw_arkit_T_M_C", "intrinsics": "recorded_arkit_intrinsics", "extrinsics": "T_C_M", "units": "meters", "camera": "OpenCV RDF"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
