#!/usr/bin/env python3
"""Sensor Recorder Pro to maplab offline processing pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PIPELINE_VERSION = "0.3.0"
STAGES = ("validate", "normalize", "create_vimap", "export_rerun")
REQUIRED_FILES = (
    "meta.json",
    "arkit_pose.csv",
    "accelerometer.csv",
    "gyroscope.csv",
    "wide.mp4",
)


class PipelineError(RuntimeError):
    pass


@dataclass
class Context:
    data_dir: Path
    output_dir: Path
    config_path: Path
    config: Dict[str, Any]
    force: bool
    vimap_importer: str | None
    manifest: Dict[str, Any]
    input_fingerprint: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"unable to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(data_dir: Path, config_path: Path) -> Tuple[str, Dict[str, Any]]:
    digest = hashlib.sha256()
    details: Dict[str, Any] = {}
    for path in [data_dir / name for name in REQUIRED_FILES] + [config_path]:
        if not path.is_file():
            raise PipelineError(f"required file is missing: {path}")
        file_digest = sha256_file(path)
        relative_name = (
            path.name if path.parent == data_dir else f"config:{path.resolve()}"
        )
        details[relative_name] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_digest,
        }
        digest.update(relative_name.encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
    return digest.hexdigest(), details


def read_csv(path: Path, required_columns: Sequence[str]) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            data_lines = (line for line in stream if not line.startswith("#"))
            reader = csv.DictReader(data_lines)
            if reader.fieldnames is None:
                raise PipelineError(f"CSV has no header: {path}")
            missing = sorted(set(required_columns) - set(reader.fieldnames))
            if missing:
                raise PipelineError(
                    f"CSV {path} is missing columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as error:
        raise PipelineError(f"unable to read CSV {path}: {error}") from error
    if not rows:
        raise PipelineError(f"CSV has no data rows: {path}")
    return rows


def as_float(row: Dict[str, str], name: str, path: Path) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError(f"invalid numeric value for {name} in {path}") from error
    if not math.isfinite(value):
        raise PipelineError(f"non-finite numeric value for {name} in {path}")
    return value


def as_int(row: Dict[str, str], name: str, path: Path) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError(f"invalid integer value for {name} in {path}") from error


def ensure_strictly_increasing(
    rows: Sequence[Dict[str, str]], column: str, path: Path
) -> None:
    previous = as_float(rows[0], column, path)
    for row_index, row in enumerate(rows[1:], start=2):
        current = as_float(row, column, path)
        if current <= previous:
            raise PipelineError(
                f"{path}:{row_index}: {column} is not strictly increasing"
            )
        previous = current


def matrix_multiply(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def matrix_vector(a: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(a[row][k] * vector[k] for k in range(3)) for row in range(3)]


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-12:
        raise PipelineError("zero-length pose quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def matrix_to_quaternion(rotation: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def validate_recording(context: Context) -> Dict[str, Any]:
    data_dir = context.data_dir
    meta = load_json(data_dir / "meta.json")
    if meta.get("format_version") not in context.config["input"]["format_versions"]:
        raise PipelineError(
            f"unsupported recording format_version: {meta.get('format_version')}"
        )
    if meta.get("state") != "finished":
        raise PipelineError(f"recording state is not finished: {meta.get('state')}")
    if meta.get("capture_mode") != "arkit":
        raise PipelineError("only ARKit capture_mode is supported in this importer")

    pose_path = data_dir / "arkit_pose.csv"
    pose_rows = read_csv(
        pose_path,
        (
            "frame_index", "record_slot", "sensor_sec", "utc_sec",
            "tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz",
            "fx_px", "fy_px", "cx_px", "cy_px", "width_px", "height_px",
        ),
    )
    accel_path = data_dir / "accelerometer.csv"
    accel_rows = read_csv(
        accel_path, ("sensor_sec", "ax_m_s2", "ay_m_s2", "az_m_s2")
    )
    gyro_path = data_dir / "gyroscope.csv"
    gyro_rows = read_csv(
        gyro_path, ("sensor_sec", "gx_rad_s", "gy_rad_s", "gz_rad_s")
    )
    ensure_strictly_increasing(pose_rows, "sensor_sec", pose_path)
    ensure_strictly_increasing(accel_rows, "sensor_sec", accel_path)
    ensure_strictly_increasing(gyro_rows, "sensor_sec", gyro_path)

    warnings: List[str] = []
    frame_indices = [as_int(row, "frame_index", pose_path) for row in pose_rows]
    if frame_indices != list(range(frame_indices[0], frame_indices[0] + len(frame_indices))):
        warnings.append("frame_index is not continuous")
    record_slots = [as_int(row, "record_slot", pose_path) for row in pose_rows]
    if any(b <= a for a, b in zip(record_slots, record_slots[1:])):
        warnings.append("record_slot is not strictly increasing")

    quaternion_norm_errors = []
    for row in pose_rows:
        values = [as_float(row, key, pose_path) for key in ("qw", "qx", "qy", "qz")]
        quaternion_norm_errors.append(abs(math.sqrt(sum(value * value for value in values)) - 1.0))
    maximum_quaternion_norm_error = max(quaternion_norm_errors)
    if maximum_quaternion_norm_error > 1e-3:
        warnings.append("pose quaternion norm error exceeds 1e-3")

    camera_start = as_float(pose_rows[0], "sensor_sec", pose_path)
    camera_end = as_float(pose_rows[-1], "sensor_sec", pose_path)
    imu_start = max(
        as_float(accel_rows[0], "sensor_sec", accel_path),
        as_float(gyro_rows[0], "sensor_sec", gyro_path),
    )
    imu_end = min(
        as_float(accel_rows[-1], "sensor_sec", accel_path),
        as_float(gyro_rows[-1], "sensor_sec", gyro_path),
    )
    if camera_start < imu_start:
        warnings.append(
            f"camera starts {imu_start - camera_start:.6f}s before joint IMU coverage"
        )
    if camera_end > imu_end:
        warnings.append(
            f"camera ends {camera_end - imu_end:.6f}s after joint IMU coverage"
        )

    report = {
        "schema_version": 1,
        "status": "valid",
        "created_at": utc_now(),
        "input": str(data_dir),
        "format_version": meta.get("format_version"),
        "capture_mode": meta.get("capture_mode"),
        "counts": {
            "camera_poses": len(pose_rows),
            "accelerometer": len(accel_rows),
            "gyroscope": len(gyro_rows),
        },
        "time_ranges_sensor_sec": {
            "camera": [camera_start, camera_end],
            "joint_imu": [imu_start, imu_end],
        },
        "maximum_quaternion_norm_error": maximum_quaternion_norm_error,
        "warnings": warnings,
    }
    atomic_write_json(context.output_dir / "validation" / "report.json", report)
    return report


def pair_imu(
    accel_rows: Sequence[Dict[str, str]],
    gyro_rows: Sequence[Dict[str, str]],
    accel_path: Path,
    gyro_path: Path,
    t0: float,
    tolerance: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    accel_times = [as_float(row, "sensor_sec", accel_path) for row in accel_rows]
    paired: List[Dict[str, Any]] = []
    accel_index = 0
    deltas: List[float] = []
    for gyro_row in gyro_rows:
        gyro_time = as_float(gyro_row, "sensor_sec", gyro_path)
        while (
            accel_index + 1 < len(accel_times)
            and abs(accel_times[accel_index + 1] - gyro_time)
            < abs(accel_times[accel_index] - gyro_time)
        ):
            accel_index += 1
        delta = accel_times[accel_index] - gyro_time
        if abs(delta) > tolerance:
            continue
        accel_row = accel_rows[accel_index]
        deltas.append(abs(delta))
        paired.append(
            {
                "timestamp_ns": round((gyro_time - t0) * 1e9),
                "sensor_sec": f"{gyro_time:.9f}",
                "accel_sensor_sec": f"{accel_times[accel_index]:.9f}",
                "gyro_sensor_sec": f"{gyro_time:.9f}",
                "ax_m_s2": f"{as_float(accel_row, 'ax_m_s2', accel_path):.12g}",
                "ay_m_s2": f"{as_float(accel_row, 'ay_m_s2', accel_path):.12g}",
                "az_m_s2": f"{as_float(accel_row, 'az_m_s2', accel_path):.12g}",
                "gx_rad_s": f"{as_float(gyro_row, 'gx_rad_s', gyro_path):.12g}",
                "gy_rad_s": f"{as_float(gyro_row, 'gy_rad_s', gyro_path):.12g}",
                "gz_rad_s": f"{as_float(gyro_row, 'gz_rad_s', gyro_path):.12g}",
            }
        )
    if len(paired) < 2:
        raise PipelineError("fewer than two accelerometer/gyroscope pairs")
    return paired, {
        "paired_count": len(paired),
        "dropped_gyro_count": len(gyro_rows) - len(paired),
        "maximum_pair_delta_sec": max(deltas),
        "median_pair_delta_sec": statistics.median(deltas),
    }


def normalize_recording(context: Context) -> Dict[str, Any]:
    data_dir = context.data_dir
    output_dir = context.output_dir / "normalized"
    pose_path = data_dir / "arkit_pose.csv"
    accel_path = data_dir / "accelerometer.csv"
    gyro_path = data_dir / "gyroscope.csv"
    pose_rows = read_csv(
        pose_path,
        (
            "frame_index", "record_slot", "sensor_sec", "utc_sec",
            "tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz",
            "fx_px", "fy_px", "cx_px", "cy_px", "width_px", "height_px",
        ),
    )
    accel_rows = read_csv(
        accel_path, ("sensor_sec", "ax_m_s2", "ay_m_s2", "az_m_s2")
    )
    gyro_rows = read_csv(
        gyro_path, ("sensor_sec", "gx_rad_s", "gy_rad_s", "gz_rad_s")
    )

    first_times = [
        as_float(pose_rows[0], "sensor_sec", pose_path),
        as_float(accel_rows[0], "sensor_sec", accel_path),
        as_float(gyro_rows[0], "sensor_sec", gyro_path),
    ]
    t0 = min(first_times)
    config = context.config
    rotation_mw = config["world_alignment"]["rotation_target_from_source"]
    rotation_ci = config["extrinsics"]["rotation"]
    translation_ci = config["extrinsics"]["translation_m"]

    normalized_frames: List[Dict[str, Any]] = []
    previous_quaternion: Tuple[float, float, float, float] | None = None
    positions: List[List[float]] = []
    times: List[float] = []
    for row in pose_rows:
        sensor_sec = as_float(row, "sensor_sec", pose_path)
        rotation_wc = quaternion_to_matrix(
            *(as_float(row, key, pose_path) for key in ("qw", "qx", "qy", "qz"))
        )
        rotation_mi = matrix_multiply(matrix_multiply(rotation_mw, rotation_wc), rotation_ci)
        camera_position_w = [as_float(row, key, pose_path) for key in ("tx_m", "ty_m", "tz_m")]
        imu_offset_w = matrix_vector(rotation_wc, translation_ci)
        position_mi = matrix_vector(
            rotation_mw,
            [camera_position_w[index] + imu_offset_w[index] for index in range(3)],
        )
        quaternion_mi = matrix_to_quaternion(rotation_mi)
        if previous_quaternion is not None and sum(
            a * b for a, b in zip(previous_quaternion, quaternion_mi)
        ) < 0.0:
            quaternion_mi = tuple(-value for value in quaternion_mi)
        previous_quaternion = quaternion_mi
        positions.append(position_mi)
        times.append(sensor_sec)
        normalized_frames.append(
            {
                "frame_index": as_int(row, "frame_index", pose_path),
                "record_slot": as_int(row, "record_slot", pose_path),
                "timestamp_ns": round((sensor_sec - t0) * 1e9),
                "sensor_sec": f"{sensor_sec:.9f}",
                "utc_sec": f"{as_float(row, 'utc_sec', pose_path):.9f}",
                "p_M_I_x_m": f"{position_mi[0]:.12g}",
                "p_M_I_y_m": f"{position_mi[1]:.12g}",
                "p_M_I_z_m": f"{position_mi[2]:.12g}",
                "q_M_I_w": f"{quaternion_mi[0]:.12g}",
                "q_M_I_x": f"{quaternion_mi[1]:.12g}",
                "q_M_I_y": f"{quaternion_mi[2]:.12g}",
                "q_M_I_z": f"{quaternion_mi[3]:.12g}",
                "v_M_I_x_m_s": "0",
                "v_M_I_y_m_s": "0",
                "v_M_I_z_m_s": "0",
                "fx_px": row["fx_px"],
                "fy_px": row["fy_px"],
                "cx_px": row["cx_px"],
                "cy_px": row["cy_px"],
                "width_px": row["width_px"],
                "height_px": row["height_px"],
                "tracking_state": row.get("tracking_state", "unknown"),
            }
        )

    for index, frame in enumerate(normalized_frames):
        left = max(0, index - 1)
        right = min(len(normalized_frames) - 1, index + 1)
        delta_time = times[right] - times[left]
        if delta_time <= 0.0:
            raise PipelineError("invalid pose timestamps while deriving velocity")
        velocity = [
            (positions[right][axis] - positions[left][axis]) / delta_time
            for axis in range(3)
        ]
        frame["v_M_I_x_m_s"] = f"{velocity[0]:.12g}"
        frame["v_M_I_y_m_s"] = f"{velocity[1]:.12g}"
        frame["v_M_I_z_m_s"] = f"{velocity[2]:.12g}"

    frame_fields = list(normalized_frames[0].keys())
    write_csv(output_dir / "frames.csv", frame_fields, normalized_frames)

    paired_imu, pairing_report = pair_imu(
        accel_rows,
        gyro_rows,
        accel_path,
        gyro_path,
        t0,
        float(config["imu"]["pairing_tolerance_sec"]),
    )
    write_csv(output_dir / "imu.csv", list(paired_imu[0].keys()), paired_imu)

    imu_start = float(paired_imu[0]["sensor_sec"])
    imu_end = float(paired_imu[-1]["sensor_sec"])
    stride = int(config["keyframes"]["stride"])
    if stride < 1:
        raise PipelineError("keyframes.stride must be at least one")
    eligible_frames = [
        frame
        for frame in normalized_frames
        if not config["keyframes"]["require_imu_coverage"]
        or imu_start <= float(frame["sensor_sec"]) <= imu_end
    ]
    keyframes = eligible_frames[::stride]
    if eligible_frames and keyframes[-1] is not eligible_frames[-1]:
        keyframes.append(eligible_frames[-1])
    if len(keyframes) < 2:
        raise PipelineError("fewer than two keyframes after IMU coverage filtering")
    write_csv(output_dir / "keyframes.csv", frame_fields, keyframes)

    calibration_row: Dict[str, Any] = {}
    for row_index in range(3):
        for column_index in range(3):
            calibration_row[f"R_C_I_{row_index}{column_index}"] = (
                config["extrinsics"]["rotation"][row_index][column_index]
            )
    for axis, value in zip("xyz", config["extrinsics"]["translation_m"]):
        calibration_row[f"t_C_I_{axis}_m"] = value
    for name in (
        "gravity_m_s2",
        "gyro_noise_density",
        "gyro_bias_random_walk_noise_density",
        "acc_noise_density",
        "acc_bias_random_walk_noise_density",
    ):
        calibration_row[name] = config["imu"][name]
    write_csv(
        output_dir / "calibration.csv",
        list(calibration_row.keys()),
        [calibration_row],
    )

    intrinsics = {
        name: statistics.median(as_float(row, name, pose_path) for row in pose_rows)
        for name in ("fx_px", "fy_px", "cx_px", "cy_px")
    }
    canonical_metadata = {
        "schema_version": 1,
        "source": {
            "format": "sensor_recorder_pro",
            "directory": str(data_dir),
            "input_fingerprint": context.input_fingerprint,
        },
        "time": {
            "origin_sensor_sec": t0,
            "timestamp_unit": "nanoseconds",
            "camera_imu_offset_sec": config["timestamp"]["camera_imu_offset_sec"],
        },
        "frames": {
            "source_camera": "opencv_rdf",
            "source_world": "arkit_gravity_y_up",
            "map": "maplab_gravity_negative_z",
            "pose_convention": "T_M_I_parent_from_child",
        },
        "camera": {
            "model": config["camera"]["model"],
            "distortion_model": config["camera"]["distortion_model"],
            "reference_intrinsics": intrinsics,
            "width_px": as_int(pose_rows[0], "width_px", pose_path),
            "height_px": as_int(pose_rows[0], "height_px", pose_path),
        },
        "extrinsics": config["extrinsics"],
        "world_alignment": config["world_alignment"],
        "imu": config["imu"],
        "counts": {
            "frames": len(normalized_frames),
            "keyframes": len(keyframes),
            "imu_pairs": len(paired_imu),
            "frames_outside_imu_coverage": len(normalized_frames) - len(eligible_frames),
        },
        "imu_pairing": pairing_report,
    }
    atomic_write_json(output_dir / "recording.json", canonical_metadata)
    return canonical_metadata


def extract_keyframe_images(context: Context, normalized_dir: Path) -> int:
    image_config = context.config.get("images", {})
    if not image_config.get("enabled", True):
        return 0

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError("ffmpeg is required to extract VI-Map keyframe images")

    keyframes_path = normalized_dir / "keyframes.csv"
    keyframes = read_csv(keyframes_path, ("frame_index",))
    frame_indices = [as_int(row, "frame_index", keyframes_path) for row in keyframes]
    if len(set(frame_indices)) != len(frame_indices):
        raise PipelineError("keyframe frame_index values must be unique")

    images_dir = normalized_dir / "keyframe_images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)
    output_pattern = images_dir / "keyframe_%06d.jpg"
    select_expression = "+".join(
        f"eq(n\\,{frame_index})" for frame_index in frame_indices
    )
    command = [
        ffmpeg,
        "-v", "error",
        "-i", str(context.data_dir / "wide.mp4"),
        "-vf", f"select={select_expression}",
        "-vsync", "0",
        "-q:v", str(image_config.get("jpeg_quality", 2)),
        str(output_pattern),
    ]
    result = subprocess.run(command, check=False, text=True)
    if result.returncode != 0:
        raise PipelineError(
            f"ffmpeg keyframe extraction failed with exit code {result.returncode}"
        )

    extracted = sorted(images_dir.glob("keyframe_*.jpg"))
    if len(extracted) != len(keyframes):
        raise PipelineError(
            f"expected {len(keyframes)} keyframe images, extracted {len(extracted)}"
        )
    for row, image_path in zip(keyframes, extracted):
        row["image_path"] = str(image_path.relative_to(normalized_dir))
    fieldnames = list(keyframes[0].keys())
    write_csv(keyframes_path, fieldnames, keyframes)
    return len(extracted)


def resolve_vimap_importer(context: Context) -> str | None:
    candidates = []
    if context.vimap_importer:
        candidates.append(context.vimap_importer)
    environment_candidate = os.environ.get("SENSOR_RECORDER_VIMAP_IMPORTER")
    if environment_candidate:
        candidates.append(environment_candidate)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            str(repo_root / "tools" / "sensor-recorder-pipeline" / "run_vimap_importer.sh"),
            str(repo_root / "devel" / "lib" / "sensor_recorder_importer" / "sensor_recorder_to_vimap"),
            str(repo_root / "build" / "sensor_recorder_importer" / "sensor_recorder_to_vimap"),
        ]
    )
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    command = shutil.which("sensor_recorder_to_vimap")
    return command


def create_vimap(context: Context) -> Dict[str, Any]:
    normalized_dir = context.output_dir / "normalized"
    for required in (
        "recording.json",
        "calibration.csv",
        "keyframes.csv",
        "imu.csv",
    ):
        if not (normalized_dir / required).is_file():
            raise PipelineError(f"normalized input is missing: {normalized_dir / required}")

    image_count = extract_keyframe_images(context, normalized_dir)

    stage_dir = context.output_dir / "maps" / "00_imported"
    vimap_dir = stage_dir / "vi_map"
    request = {
        "schema_version": 1,
        "normalized_directory": str(normalized_dir),
        "output_directory": str(vimap_dir),
        "created_at": utc_now(),
    }
    atomic_write_json(stage_dir / "import_request.json", request)

    importer = resolve_vimap_importer(context)
    if importer is None:
        raise PipelineError(
            "native sensor_recorder_to_vimap executable was not found; "
            "the import request is ready at " + str(stage_dir / "import_request.json")
        )
    if vimap_dir.exists():
        if context.force:
            shutil.rmtree(vimap_dir)
        else:
            raise PipelineError(
                f"VI-Map output already exists: {vimap_dir}; rerun with --force"
            )
    command = [
        importer,
        f"--normalized_data={normalized_dir}",
        f"--output_map={vimap_dir}",
    ]
    result = subprocess.run(command, check=False, text=True)
    if result.returncode != 0:
        raise PipelineError(
            f"native VI-Map importer failed with exit code {result.returncode}"
        )
    if not (vimap_dir / "metadata").is_file():
        raise PipelineError("native importer did not create VI-Map metadata")
    normalized_metadata = json.loads(
        (normalized_dir / "recording.json").read_text(encoding="utf-8")
    )
    vertex_count = int(normalized_metadata["counts"]["keyframes"])
    report = {
        "schema_version": 1,
        "status": "created_and_reloaded",
        "importer": importer,
        "output_map": str(vimap_dir),
        "verified_counts": {
            "vertices": vertex_count,
            "viwls_edges": vertex_count - 1,
            "raw_image_resources": image_count,
        },
    }
    atomic_write_json(stage_dir / "report.json", report)
    return report


def export_rerun(context: Context) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    exporter = repo_root / "tools" / "sensor-recorder-pipeline" / "export_vimap_rerun.py"
    normalized_dir = context.output_dir / "normalized"
    vimap_report = context.output_dir / "maps" / "00_imported" / "report.json"
    images_dir = normalized_dir / "keyframe_images"
    for required in (exporter, normalized_dir / "keyframes.csv", vimap_report):
        if not required.is_file():
            raise PipelineError(f"Rerun export input is missing: {required}")
    if not images_dir.is_dir():
        raise PipelineError(f"Rerun keyframe image directory is missing: {images_dir}")

    rerun_python = os.environ.get("RERUN_PYTHON", sys.executable)
    dependency_check = subprocess.run(
        [rerun_python, "-c", "import rerun"], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if dependency_check.returncode != 0:
        raise PipelineError(
            f"Rerun SDK is unavailable in {rerun_python}; "
            "install it with: python3 -m pip install rerun-sdk==0.33.0"
        )

    output_dir = context.output_dir / "rerun"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{context.data_dir.name}_vimap.rrd"
    rerun_environment = os.environ.copy()
    rerun_environment.setdefault("RERUN_TELEMETRY_ENABLED", "0")
    with tempfile.TemporaryDirectory(prefix="rerun-export-", dir=str(output_dir)) as temp:
        temporary_output = Path(temp) / output_path.name
        command = [
            rerun_python, str(exporter),
            "--keyframes", str(normalized_dir / "keyframes.csv"),
            "--report", str(vimap_report),
            "--config", str(context.config_path),
            "--images-dir", str(images_dir),
            "--output", str(temporary_output),
        ]
        result = subprocess.run(
            command, check=False, text=True, env=rerun_environment
        )
        if result.returncode != 0:
            raise PipelineError(
                f"Rerun export failed with exit code {result.returncode}"
            )
        verify = subprocess.run(
            [rerun_python, "-m", "rerun", "rrd", "verify", str(temporary_output)],
            check=False, text=True, env=rerun_environment,
        )
        if verify.returncode != 0:
            raise PipelineError(
                f"Rerun verification failed with exit code {verify.returncode}"
            )
        os.replace(temporary_output, output_path)

    report = {
        "schema_version": 1,
        "status": "created_and_verified",
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "images": int(json.loads(vimap_report.read_text(encoding="utf-8"))
                      ["verified_counts"]["raw_image_resources"]),
    }
    atomic_write_json(output_dir / "report.json", report)
    return report


def stage_fingerprint(context: Context, stage: str) -> str:
    digest = hashlib.sha256()
    digest.update(PIPELINE_VERSION.encode("ascii"))
    digest.update(sha256_file(Path(__file__).resolve()).encode("ascii"))
    digest.update(context.input_fingerprint.encode("ascii"))
    digest.update(stage.encode("ascii"))
    if stage == "create_vimap":
        importer = resolve_vimap_importer(context)
        if importer is not None:
            digest.update(sha256_file(Path(importer)).encode("ascii"))
    elif stage == "export_rerun":
        exporter = Path(__file__).with_name("export_vimap_rerun.py")
        digest.update(sha256_file(exporter).encode("ascii"))
    for dependency in STAGES[: STAGES.index(stage)]:
        state = context.manifest.get("stages", {}).get(dependency, {})
        digest.update(str(state.get("fingerprint", "")).encode("ascii"))
    return digest.hexdigest()


def run_stage(context: Context, stage: str) -> None:
    fingerprint = stage_fingerprint(context, stage)
    previous = context.manifest.setdefault("stages", {}).get(stage)
    if (
        not context.force
        and previous
        and previous.get("status") == "completed"
        and previous.get("fingerprint") == fingerprint
    ):
        print(f"[skip] {stage}: unchanged")
        return

    print(f"[run]  {stage}")
    context.manifest["status"] = "running"
    context.manifest["current_stage"] = stage
    atomic_write_json(context.output_dir / "manifest.json", context.manifest)
    try:
        if stage == "validate":
            result = validate_recording(context)
        elif stage == "normalize":
            result = normalize_recording(context)
        elif stage == "create_vimap":
            result = create_vimap(context)
        elif stage == "export_rerun":
            result = export_rerun(context)
        else:
            raise AssertionError(stage)
    except BaseException as error:
        context.manifest["status"] = "failed"
        context.manifest["stages"][stage] = {
            "status": "failed",
            "fingerprint": fingerprint,
            "finished_at": utc_now(),
            "error": str(error),
        }
        atomic_write_json(context.output_dir / "manifest.json", context.manifest)
        raise
    context.manifest["stages"][stage] = {
        "status": "completed",
        "fingerprint": fingerprint,
        "finished_at": utc_now(),
        "summary": result,
    }
    atomic_write_json(context.output_dir / "manifest.json", context.manifest)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    arguments = list(argv)
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "single")

    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="process.sh")
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single", help="process one recording")
    single.add_argument("--data", required=True, type=Path)
    single.add_argument("--output", required=True, type=Path)
    single.add_argument(
        "--config", type=Path, default=repo_root / "configs" / "iphone_arkit.json"
    )
    single.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    single.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    single.add_argument("--force", action="store_true")
    single.add_argument("--vimap-importer")
    return parser.parse_args(arguments)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    data_dir = args.data.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not data_dir.is_dir():
        raise PipelineError(f"recording directory does not exist: {data_dir}")
    if args.from_stage not in STAGES or args.to_stage not in STAGES:
        raise PipelineError("invalid stage")
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise PipelineError("--from stage must not be later than --to stage")

    config = load_json(config_path)
    input_fingerprint, input_files = fingerprint_files(data_dir, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
    else:
        manifest = {
            "schema_version": 1,
            "pipeline_version": PIPELINE_VERSION,
            "created_at": utc_now(),
            "stages": {},
        }
    manifest.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "input": str(data_dir),
            "output": str(output_dir),
            "config": str(config_path),
            "input_fingerprint": input_fingerprint,
            "input_files": input_files,
        }
    )
    atomic_write_json(output_dir / "config.resolved.json", config)
    context = Context(
        data_dir=data_dir,
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        force=args.force,
        vimap_importer=args.vimap_importer,
        manifest=manifest,
        input_fingerprint=input_fingerprint,
    )
    for stage in STAGES[start : end + 1]:
        run_stage(context, stage)
    context.manifest["status"] = "completed"
    context.manifest["current_stage"] = None
    context.manifest["finished_at"] = utc_now()
    atomic_write_json(manifest_path, context.manifest)
    print(f"[done] {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
