#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
import yaml


def quaternion_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_keyframes(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"expected at least two keyframes in {path}")
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an imported VI-Map to Rerun.")
    parser.add_argument("--keyframes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--video", type=Path)
    image_source.add_argument("--images-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--optimized-dir",
        type=Path,
        help="CSV directory produced by sensor_recorder_vimap_export",
    )
    parser.add_argument(
        "--include-bad-landmarks",
        action="store_true",
        help="Include landmarks rejected by Maplab quality checks in the 3D view.",
    )
    parser.add_argument(
        "--loops-yaml", type=Path,
        help="Verified PnP loop candidates; rendered against the exported optimized poses.",
    )
    parser.add_argument(
        "--pgo-decisions-yaml", type=Path,
        help="Pose-graph switch and residual decisions for loop candidates.",
    )
    return parser.parse_args()


def z_colormap(points: np.ndarray) -> np.ndarray:
    """Robust Viridis-like RGB colors from landmark Z coordinates."""
    if not len(points):
        return np.empty((0, 3), dtype=np.uint8)
    low, high = np.percentile(points[:, 2], [2.0, 98.0])
    value = np.full(len(points), 0.5) if high <= low else np.clip(
        (points[:, 2] - low) / (high - low), 0.0, 1.0)
    palette = np.asarray(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float64,
    )
    location = value * (len(palette) - 1)
    lower = np.floor(location).astype(np.int32)
    upper = np.minimum(lower + 1, len(palette) - 1)
    blend = (location - lower)[:, None]
    return ((1.0 - blend) * palette[lower] + blend * palette[upper]).astype(np.uint8)


def loop_indices(edges: list[dict], timestamps: list[int]) -> list[tuple[int, int]]:
    timestamp_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    result = []
    for edge in edges:
        try:
            start = timestamp_to_index.get(int(edge["camera_from"]["timestamp_ns"]))
            end = timestamp_to_index.get(int(edge["camera_to"]["timestamp_ns"]))
        except (KeyError, TypeError, ValueError):
            continue
        if start is not None and end is not None:
            result.append((start, end))
    return result


def load_loop_indices(path: Path | None, timestamps: list[int]) -> list[tuple[int, int]]:
    if path is None or not path.is_file():
        return []
    edges = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return loop_indices(edges if isinstance(edges, list) else [], timestamps)


def load_pgo_decisions(
    path: Path | None, timestamps: list[int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict], list[dict]]:
    if path is None or not path.is_file():
        return [], [], [], []
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(document, list):
        document = {"accepted": document, "rejected": []}
    accepted = document.get("accepted", []) if isinstance(document, dict) else []
    rejected = document.get("rejected", []) if isinstance(document, dict) else []
    return (
        loop_indices(accepted, timestamps), loop_indices(rejected, timestamps),
        accepted, rejected,
    )


def extract_keyframe_images(
    video_path: Path, rows: list[dict[str, str]], output_dir: Path
) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    frame_indices = [int(row["frame_index"]) for row in rows]
    select_expression = "+".join(
        f"eq(n\\,{frame_index})" for frame_index in frame_indices
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-v", "error",
            "-i", str(video_path),
            "-vf", f"select={select_expression}",
            "-vsync", "0",
            "-q:v", "2",
            str(output_dir / "keyframe_%06d.jpg"),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}")
    images = sorted(output_dir.glob("keyframe_*.jpg"))
    if len(images) != len(rows):
        raise RuntimeError(
            f"expected {len(rows)} keyframe images, extracted {len(images)}"
        )
    return images


def load_keyframe_images(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[Path]:
    if args.images_dir is not None:
        images = sorted(args.images_dir.glob("keyframe_*.jpg"))
        if len(images) != len(rows):
            raise RuntimeError(
                f"expected {len(rows)} images in {args.images_dir}, found {len(images)}"
            )
        return images
    raise AssertionError("video images must be extracted in a temporary directory")


def main() -> None:
    args = parse_arguments()
    rows = load_keyframes(args.keyframes)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    verified = report.get("verified_counts")
    if verified is not None and len(rows) != int(verified["vertices"]):
        raise ValueError("keyframe count does not match the reloaded VI-Map report")

    calibration = report.get("calibration", {})
    final_intrinsics = calibration.get("final_intrinsics")
    if isinstance(final_intrinsics, list) and len(final_intrinsics) == 4:
        for row in rows:
            for name, value in zip(("fx_px", "fy_px", "cx_px", "cy_px"), final_intrinsics):
                row[name] = str(value)

    initial_positions = np.asarray(
        [[float(row[f"p_M_I_{axis}_m"]) for axis in "xyz"] for row in rows]
    )
    positions = initial_positions.copy()
    velocities = np.asarray(
        [[float(row[f"v_M_I_{axis}_m_s"]) for axis in "xyz"] for row in rows]
    )
    accel_biases = np.zeros((len(rows), 3), dtype=np.float64)
    gyro_biases = np.zeros((len(rows), 3), dtype=np.float64)
    rotations = np.asarray(
        [
            quaternion_matrix(
                float(row["q_M_I_w"]),
                float(row["q_M_I_x"]),
                float(row["q_M_I_y"]),
                float(row["q_M_I_z"]),
            )
            for row in rows
        ]
    )
    landmarks = np.empty((0, 3), dtype=np.float64)
    bad_landmarks = np.empty((0, 3), dtype=np.float64)
    keypoints_by_vertex: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    frame_keypoint_counts = report.get("frame_keypoint_counts", [])
    pair_match_counts = {
        int(item["current_index"]): item
        for item in report.get("pair_match_counts", [])
    }
    vertex_timestamps = [int(row["timestamp_ns"]) for row in rows]
    if args.optimized_dir is not None:
        optimized_rows = load_csv(args.optimized_dir / "vertices.csv")
        if len(optimized_rows) != len(rows):
            raise ValueError("optimized vertex count does not match keyframes")
        vertex_timestamps = [int(row["timestamp_ns"]) for row in optimized_rows]
        positions = np.asarray(
            [[float(row[f"p_{axis}_m"]) for axis in "xyz"] for row in optimized_rows]
        )
        velocities = np.asarray(
            [[float(row[f"v_{axis}_m_s"]) for axis in "xyz"] for row in optimized_rows]
        )
        accel_biases = np.asarray(
            [
                [float(row[f"accel_bias_{axis}"]) for axis in "xyz"]
                for row in optimized_rows
            ]
        )
        gyro_biases = np.asarray(
            [
                [float(row[f"gyro_bias_{axis}"]) for axis in "xyz"]
                for row in optimized_rows
            ]
        )
        rotations = np.asarray(
            [
                quaternion_matrix(
                    float(row["q_w"]), float(row["q_x"]),
                    float(row["q_y"]), float(row["q_z"]),
                )
                for row in optimized_rows
            ]
        )
        landmark_rows = load_csv(args.optimized_dir / "landmarks.csv")
        good_rows = [row for row in landmark_rows if row.get("quality") == "2"]
        bad_rows = [row for row in landmark_rows if row.get("quality") != "2"]
        landmarks = np.asarray(
            [[float(row[f"{axis}_m"]) for axis in "xyz"] for row in good_rows]
        )
        bad_landmarks = np.asarray(
            [[float(row[f"{axis}_m"]) for axis in "xyz"] for row in bad_rows]
        )
        grouped: dict[int, list[tuple[float, float, bool]]] = {}
        for row in load_csv(args.optimized_dir / "keypoints.csv"):
            grouped.setdefault(int(row["vertex_index"]), []).append(
                (float(row["u_px"]), float(row["v_px"]), row["has_landmark"] == "1")
            )
        for vertex_index, points in grouped.items():
            matched = np.asarray([[u, v] for u, v, valid in points if valid])
            unmatched = np.asarray([[u, v] for u, v, valid in points if not valid])
            keypoints_by_vertex[vertex_index] = (matched, unmatched)

    candidate_loop_indices = load_loop_indices(args.loops_yaml, vertex_timestamps)
    accepted_loop_indices, rejected_loop_indices, accepted_loop_records, rejected_loop_records = (
        load_pgo_decisions(args.pgo_decisions_yaml, vertex_timestamps)
    )

    edge_segments = np.stack([positions[:-1], positions[1:]], axis=1)
    initial_edge_segments = np.stack(
        [initial_positions[:-1], initial_positions[1:]], axis=1
    )

    final_q_C_I = calibration.get("final_q_C_I_wxyz")
    final_t_C_I = calibration.get("final_t_C_I_m")
    if (
        isinstance(final_q_C_I, list) and len(final_q_C_I) == 4
        and isinstance(final_t_C_I, list) and len(final_t_C_I) == 3
    ):
        r_camera_imu = quaternion_matrix(*map(float, final_q_C_I))
        t_camera_imu = np.asarray(final_t_C_I, dtype=np.float64)
    else:
        r_camera_imu = np.asarray(config["extrinsics"]["rotation"], dtype=np.float64)
        t_camera_imu = np.asarray(config["extrinsics"]["translation_m"], dtype=np.float64)
    r_imu_camera = r_camera_imu.T
    t_imu_camera = -r_imu_camera @ t_camera_imu

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.images_dir is not None:
        images = load_keyframe_images(args, rows)
        write_rerun(
            args, rows, report, config, initial_positions, positions, velocities,
            accel_biases, gyro_biases, rotations, initial_edge_segments,
            edge_segments, landmarks,
            bad_landmarks, keypoints_by_vertex, frame_keypoint_counts,
            pair_match_counts, candidate_loop_indices, accepted_loop_indices,
            rejected_loop_indices, accepted_loop_records, rejected_loop_records,
            r_imu_camera, t_imu_camera, images,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="vimap-rerun-images-") as temp_dir:
            images = extract_keyframe_images(args.video, rows, Path(temp_dir))
            write_rerun(
                args, rows, report, config, initial_positions, positions, velocities,
                accel_biases, gyro_biases, rotations, initial_edge_segments,
                edge_segments, landmarks,
                bad_landmarks, keypoints_by_vertex, frame_keypoint_counts,
                pair_match_counts, candidate_loop_indices, accepted_loop_indices,
                rejected_loop_indices, accepted_loop_records, rejected_loop_records,
                r_imu_camera, t_imu_camera, images,
            )

    print(
        f"wrote {args.output} with {len(rows)} vertices, "
        f"{len(edge_segments)} VIWLS edges and {len(rows)} images"
    )


def write_rerun(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    report: dict,
    config: dict,
    initial_positions: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accel_biases: np.ndarray,
    gyro_biases: np.ndarray,
    rotations: np.ndarray,
    initial_edge_segments: np.ndarray,
    edge_segments: np.ndarray,
    landmarks: np.ndarray,
    bad_landmarks: np.ndarray,
    keypoints_by_vertex: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_keypoint_counts: list[int],
    pair_match_counts: dict[int, dict],
    candidate_loop_indices: list[tuple[int, int]],
    accepted_loop_indices: list[tuple[int, int]],
    rejected_loop_indices: list[tuple[int, int]],
    accepted_loop_records: list[dict],
    rejected_loop_records: list[dict],
    r_imu_camera: np.ndarray,
    t_imu_camera: np.ndarray,
    images: list[Path],
) -> None:
    rr.init("sensor_recorder_vimap", spawn=False)
    rr.save(str(args.output))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/pose_graph/initial",
        rr.LineStrips3D(
            initial_edge_segments, colors=[130, 130, 130], radii=0.0015,
        ),
        static=True,
    )
    def log_loop_edges(path: str, indices: list[tuple[int, int]], color: list[int]) -> None:
        if not indices:
            return
        loop_segments = np.asarray([[positions[start], positions[end]] for start, end in indices])
        rr.log(
            path,
            rr.LineStrips3D(loop_segments, colors=color, radii=0.006),
            static=True,
        )
        rr.log(
            f"{path}/endpoints",
            rr.Points3D(
                np.asarray([positions[index] for edge in indices for index in edge]),
                colors=color, radii=rr.Radius.ui_points(5.0),
            ),
            static=True,
        )
    log_loop_edges("world/loop_closures/pnp_candidates", candidate_loop_indices, [150, 150, 150])
    log_loop_edges("world/loop_closures/pgo_accepted", accepted_loop_indices, [70, 230, 110])
    log_loop_edges("world/loop_closures/pgo_rejected", rejected_loop_indices, [245, 70, 70])
    rr.log(
        "world/pose_graph/vertices",
        rr.Points3D(
            positions,
            colors=[70, 170, 255],
            radii=rr.Radius.ui_points(4.0),
        ),
        static=True,
    )
    rr.log(
        "world/pose_graph/viwls_edges",
        rr.LineStrips3D(edge_segments, colors=[120, 210, 255], radii=0.002),
        static=True,
    )
    rr.log(
        "world/pose_graph/velocity",
        rr.Arrows3D(
            origins=positions,
            vectors=velocities * 0.15,
            colors=[255, 185, 55],
            radii=rr.Radius.ui_points(1.0),
        ),
        static=True,
    )
    rr.log(
        "world/pose_graph/endpoints",
        rr.Points3D(
            [positions[0], positions[-1]],
            colors=[[60, 230, 100], [255, 70, 70]],
            radii=rr.Radius.ui_points(7.0),
            labels=["start", "end"],
            show_labels=True,
        ),
        static=True,
    )
    if len(landmarks):
        rr.log(
            "world/landmarks",
            rr.Points3D(
                landmarks,
                colors=z_colormap(landmarks),
                radii=rr.Radius.ui_points(2.0),
            ),
            static=True,
        )
    if args.include_bad_landmarks and len(bad_landmarks):
        rr.log(
            "world/debug/bad_landmarks",
            rr.Points3D(
                bad_landmarks,
                colors=[255, 70, 70],
                radii=rr.Radius.ui_points(1.0),
            ),
            static=True,
        )
    rr.log(
        "world/imu/camera",
        rr.Transform3D(
            translation=t_imu_camera,
            mat3x3=r_imu_camera,
            relation=rr.TransformRelation.ParentFromChild,
        ),
        static=True,
    )
    rr.log(
        "metadata",
        rr.TextDocument(
            "\n".join(
                [
                    f"VI-Map: {report.get('output_map', 'optimized VI-Map')}",
                    f"vertices: {len(rows)}",
                    f"VIWLS edges: {len(rows) - 1}",
                    f"landmarks: {len(landmarks)}",
                    f"bad landmarks: {len(bad_landmarks)}",
                    f"raw images: {len(images)}",
                    f"detected keypoints: {sum(frame_keypoint_counts)}",
                    f"PnP loop candidates: {len(candidate_loop_indices)}",
                    f"PGO accepted loop edges: {len(accepted_loop_indices)}",
                    f"PGO rejected loop edges: {len(rejected_loop_indices)}",
                    "PGO gate: switch >= 0.8, Mahalanobis^2 <= 12.59",
                    "landmark color: robust Z height (purple low, yellow high)",
                    "pose: T_M_I, meters",
                    "camera: OpenCV RDF",
                    "extrinsic: initial T_C_I, translation zero",
                ]
            )
        ),
        static=True,
    )
    for decision, records in (("accepted", accepted_loop_records), ("rejected", rejected_loop_records)):
        for record_index, record in enumerate(records):
            switch = record.get("pgo_switch_variable", float("nan"))
            mahalanobis = record.get("pgo_mahalanobis_squared", float("nan"))
            reason = record.get("pgo_rejection_reason", "")
            rr.log(
                f"diagnostics/loop_pgo/{decision}/{record_index}",
                rr.TextDocument(
                    f"switch: {switch}\\nMahalanobis^2: {mahalanobis}\\nreason: {reason}"
                ),
                static=True,
            )

    for index, (row, position, rotation, image) in enumerate(
        zip(rows, positions, rotations, images)
    ):
        rr.set_time("keyframe", sequence=index)
        rr.set_time("source_frame", sequence=int(row["frame_index"]))
        rr.set_time("sensor_time", duration=float(row["sensor_sec"]))
        rr.log(
            "world/imu",
            rr.Transform3D(
                translation=position,
                mat3x3=rotation,
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )
        image_from_camera = np.asarray(
            [
                [float(row["fx_px"]), 0.0, float(row["cx_px"])],
                [0.0, float(row["fy_px"]), float(row["cy_px"])],
                [0.0, 0.0, 1.0],
            ]
        )
        rr.log(
            "world/imu/camera",
            rr.Pinhole(
                image_from_camera=image_from_camera,
                resolution=[int(row["width_px"]), int(row["height_px"])],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.12,
                color=[245, 245, 245],
                line_width=0.0015,
            ),
        )
        rr.log("world/imu/camera/image", rr.EncodedImage(path=image))
        if index < len(frame_keypoint_counts):
            rr.log(
                "frontend/detected_keypoints",
                rr.Scalars(float(frame_keypoint_counts[index])),
            )
        pair = pair_match_counts.get(index)
        if pair is not None:
            rr.log("frontend/pair_inliers", rr.Scalars(float(pair["inliers"])))
            rr.log("frontend/pair_outliers", rr.Scalars(float(pair["outliers"])))
        for axis_index, axis in enumerate("xyz"):
            rr.log(
                f"state/imu_bias/accelerometer/{axis}_m_s2",
                rr.Scalars(float(accel_biases[index, axis_index])),
            )
            rr.log(
                f"state/imu_bias/gyroscope/{axis}_rad_s",
                rr.Scalars(float(gyro_biases[index, axis_index])),
            )
        rr.log(
            "state/imu_bias/accelerometer/norm_m_s2",
            rr.Scalars(float(np.linalg.norm(accel_biases[index]))),
        )
        rr.log(
            "state/imu_bias/gyroscope/norm_rad_s",
            rr.Scalars(float(np.linalg.norm(gyro_biases[index]))),
        )
        if index in keypoints_by_vertex:
            matched, unmatched = keypoints_by_vertex[index]
            if len(unmatched):
                rr.log(
                    "world/imu/camera/image/keypoints/unmatched",
                    rr.Points2D(
                        unmatched,
                        colors=[130, 170, 255],
                        radii=rr.Radius.ui_points(2.0),
                    ),
                )
            if len(matched):
                rr.log(
                    "world/imu/camera/image/keypoints/landmark_observations",
                    rr.Points2D(
                        matched,
                        colors=[80, 255, 120],
                        radii=rr.Radius.ui_points(3.0),
                    ),
                )

    rr.disconnect()


if __name__ == "__main__":
    main()
