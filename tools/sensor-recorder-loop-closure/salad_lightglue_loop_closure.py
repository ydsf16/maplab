#!/usr/bin/env python3
"""External SALAD + LightGlue + PnP loop closure for Sensor Recorder VI-Maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import itertools
import time
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
import torch
import yaml
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-data", required=True, type=Path)
    parser.add_argument("--optimized-dir", required=True, type=Path)
    parser.add_argument("--optimized-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lightglue-model", type=Path, help="Legacy fused SuperPoint+LightGlue ONNX model.")
    parser.add_argument("--superpoint-model", type=Path, help="Split SuperPoint ONNX model.")
    parser.add_argument("--lightglue-matcher-model", type=Path, help="Split LightGlue ONNX model.")
    parser.add_argument("--feature-cache", type=Path, help="Per-frame frontend SuperPoint cache directory.")
    parser.add_argument("--salad-repo", type=Path, default=Path("/root/autodl-tmp/third_party/salad"))
    parser.add_argument("--salad-checkpoint", type=Path, default=Path("/root/autodl-tmp/third_party/salad/dino_salad.ckpt"))
    parser.add_argument("--dinov2-repo", type=Path, default=Path("/root/autodl-tmp/third_party/dinov2"))
    parser.add_argument("--dinov2-checkpoint", type=Path, default=Path("/root/autodl-tmp/third_party/dinov2/dinov2_vitb14_pretrain.pth"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-pool-size", type=int, default=10,
                        help="SALAD candidates considered before adaptive similarity gating.")
    parser.add_argument("--covisibility-similarity-ratio", type=float, default=0.5)
    parser.add_argument("--min-salad-similarity", type=float, default=0.0,
                        help="Fallback SALAD threshold for frames without covisible neighbors.")
    parser.add_argument(
        "--high-recall", action="store_true",
        help="Use 20 SALAD candidates per query instead of the faster default 10.",
    )
    parser.add_argument("--temporal-exclusion-frames", type=int, default=30)
    parser.add_argument("--min-trajectory-separation-m", type=float, default=0.5)
    parser.add_argument("--max-spatial-candidate-distance-m", type=float, default=10.0,
                        help="Reject a candidate before LightGlue/PnP when current camera centers are farther apart.")
    parser.add_argument("--min-lightglue-score", type=float, default=0.15)
    parser.add_argument("--pnp-ransac-px", type=float, default=3.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)
    parser.add_argument("--max-loop-delta-translation-m", type=float, default=4.0)
    parser.add_argument("--support-window-frames", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--loop-nms-frames", type=int, default=30)
    parser.add_argument("--covisibility-min-landmarks", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--salad-batch-size", type=int, default=16)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    return rows


def rotation_from_wxyz(values: list[float]) -> np.ndarray:
    w, x, y, z = values
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw(rotation: np.ndarray) -> list[float]:
    q = np.empty(4, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[:] = [(rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s, 0.25 * s]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q[:] = [0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s,
                    (rotation[0, 2] + rotation[2, 0]) / s, (rotation[2, 1] - rotation[1, 2]) / s]
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q[:] = [(rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                    (rotation[1, 2] + rotation[2, 1]) / s, (rotation[0, 2] - rotation[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q[:] = [(rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s,
                    0.25 * s, (rotation[1, 0] - rotation[0, 1]) / s]
    q /= np.linalg.norm(q)
    return q.tolist()


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def yaml_transform(matrix: np.ndarray) -> dict[str, list[float]]:
    return {"translation_m": matrix[:3, 3].tolist(), "rotation_xyzw": quat_xyzw(matrix[:3, :3])}


def angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def salad_descriptors(paths: list[Path], device: str, salad_repo: Path, salad_checkpoint: Path,
                      dinov2_repo: Path, dinov2_checkpoint: Path, batch_size: int) -> tuple[np.ndarray, list[float]]:
    """Load SALAD fully offline, including its DINOv2 backbone weights."""
    for path in (salad_repo, salad_checkpoint, dinov2_repo, dinov2_checkpoint):
        if not path.exists():
            raise RuntimeError(f"missing offline SALAD asset: {path}")
    original_load = torch.hub.load
    original_load_state_dict = torch.hub.load_state_dict_from_url

    def local_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs["source"] = "local"
            kwargs["pretrained"] = True
            kwargs["weights"] = str(dinov2_checkpoint)
            return original_load(str(dinov2_repo), model, *args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    def local_checkpoint(url, *args, **kwargs):
        if "serizba/salad" in str(url):
            return torch.load(salad_checkpoint, map_location="cpu", weights_only=True)
        return original_load_state_dict(url, *args, **kwargs)

    torch.hub.load = local_hub_load
    torch.hub.load_state_dict_from_url = local_checkpoint
    try:
        model = original_load(str(salad_repo), "dinov2_salad", source="local", pretrained=False)
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_load_state_dict
    model.eval().to(device)
    tensors: list[torch.Tensor] = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((322, 322), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array)
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensors.append(tensor)
    output: list[np.ndarray] = []
    per_frame_ms: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch_start = time.perf_counter()
            result = model(torch.stack(tensors[start:start + batch_size]).to(device))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if isinstance(result, dict):
                result = result["global_descriptor"]
            per_frame_ms.extend([1000.0 * (time.perf_counter() - batch_start) / len(result)] * len(result))
            output.append(result.detach().float().cpu().numpy())
    descriptors = np.concatenate(output, axis=0)
    return descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True), per_frame_ms


def normalize_keypoints(points: np.ndarray, width: int, height: int) -> np.ndarray:
    size = np.array([width, height], dtype=np.float32)
    return ((points - size / 2.0) / (max(width, height) / 2.0)).astype(np.float32)


def keypoint_mapping(points: np.ndarray, exported: np.ndarray) -> np.ndarray:
    distances = np.sum((points[:, None, :] - exported[None, :, :]) ** 2, axis=2)
    indices = distances.argmin(axis=1)
    # BA/pruning can remove individual keypoints.  Keep the remaining exact map
    # association and discard only unmatched ONNX points from 2D-3D PnP.
    indices[np.sqrt(distances[np.arange(len(points)), indices]) > 1.0] = -1
    return indices


def covariance(object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera: np.ndarray) -> np.ndarray:
    projected, jacobian = cv2.projectPoints(object_points, rvec, tvec, camera, None)
    residual = image_points - projected.reshape(-1, 2)
    sigma2 = max(float(np.sum(residual * residual) / max(1, 2 * len(residual) - 6)), 0.25)
    information = jacobian[:, :6].T @ jacobian[:, :6]
    cov_rt = sigma2 * np.linalg.pinv(information)
    order = [3, 4, 5, 0, 1, 2]  # Maplab residual is translation, rotation.
    cov = cov_rt[np.ix_(order, order)]
    lower = np.array([0.02, 0.02, 0.02, math.radians(0.5), math.radians(0.5), math.radians(0.5)]) ** 2
    upper = np.array([0.5, 0.5, 0.5, math.radians(10), math.radians(10), math.radians(10)]) ** 2
    values, vectors = np.linalg.eigh((cov + cov.T) * 0.5)
    values = np.clip(values, lower.min(), upper.max())
    cov = vectors @ np.diag(values) @ vectors.T
    for index in range(6):
        cov[index, index] = float(np.clip(cov[index, index], lower[index], upper[index]))
    return cov


def main() -> int:
    args = arguments()
    split_models = args.superpoint_model is not None or args.lightglue_matcher_model is not None
    if split_models and (args.superpoint_model is None or args.lightglue_matcher_model is None):
        raise RuntimeError("--superpoint-model and --lightglue-matcher-model must be supplied together")
    if not split_models and args.lightglue_model is None:
        raise RuntimeError("supply --lightglue-model or both split ONNX model paths")
    if args.feature_cache is not None and not split_models:
        raise RuntimeError("--feature-cache requires split SuperPoint and LightGlue models")
    if args.salad_batch_size < 1:
        raise RuntimeError("--salad-batch-size must be positive")
    if args.high_recall:
        args.top_k = 20
        args.retrieval_pool_size = max(args.retrieval_pool_size, 20)
    if args.retrieval_pool_size < args.top_k:
        raise RuntimeError("--retrieval-pool-size must be at least --top-k")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.normalized_data / "keyframes.csv")
    vertices = read_csv(args.optimized_dir / "vertices.csv")
    exported_keypoints = read_csv(args.optimized_dir / "keypoints.csv")
    landmarks = read_csv(args.optimized_dir / "landmarks.csv")
    if len(rows) != len(vertices):
        raise RuntimeError("keyframe and optimized vertex counts differ")
    calibration = json.loads(args.optimized_report.read_text())["calibration"]
    T_C_I = transform(rotation_from_wxyz(calibration["final_q_C_I_wxyz"]), np.array(calibration["final_t_C_I_m"], dtype=np.float64))
    T_I_C = np.linalg.inv(T_C_I)
    T_M_C: list[np.ndarray] = []
    for vertex in vertices:
        T_M_I = transform(rotation_from_wxyz([float(vertex[key]) for key in ("q_w", "q_x", "q_y", "q_z")]), np.array([float(vertex[key]) for key in ("p_x_m", "p_y_m", "p_z_m")]))
        T_M_C.append(T_M_I @ T_I_C)
    trajectory_distance = np.concatenate((
        np.zeros(1, dtype=np.float64),
        np.cumsum(np.linalg.norm(np.diff(np.asarray([pose[:3, 3] for pose in T_M_C]), axis=0), axis=1)),
    ))
    landmark_xyz = {row["landmark_id"]: np.array([float(row[key]) for key in ("x_m", "y_m", "z_m")]) for row in landmarks}
    frame_export: dict[int, tuple[np.ndarray, list[str]]] = {}
    for frame in range(len(rows)):
        frame_rows = [row for row in exported_keypoints if int(row["vertex_index"]) == frame]
        frame_export[frame] = (np.array([[float(row["u_px"]), float(row["v_px"])] for row in frame_rows]), [row["landmark_id"] for row in frame_rows])
    observers: dict[str, set[int]] = defaultdict(set)
    for frame, (_, landmark_ids) in frame_export.items():
        for landmark_id in landmark_ids:
            if landmark_id in landmark_xyz:
                observers[landmark_id].add(frame)
    covisibility: dict[tuple[int, int], int] = defaultdict(int)
    for frames in observers.values():
        for first, second in itertools.combinations(sorted(frames), 2):
            covisibility[(first, second)] += 1
    with (args.output / "covisibility_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["first_frame", "second_frame", "shared_landmarks"])
        writer.writeheader()
        writer.writerows({"first_frame": first, "second_frame": second, "shared_landmarks": count}
                         for (first, second), count in sorted(covisibility.items())
                         if count >= args.covisibility_min_landmarks)
    image_paths = [args.normalized_data / row["image_path"] for row in rows]
    salad_start = time.perf_counter()
    descriptors, salad_per_frame_ms = salad_descriptors(
        image_paths, args.device, args.salad_repo, args.salad_checkpoint,
        args.dinov2_repo, args.dinov2_checkpoint, args.salad_batch_size)
    salad_seconds = time.perf_counter() - salad_start
    np.save(args.output / "salad_descriptors.npy", descriptors)
    covisible_neighbors: dict[int, list[int]] = defaultdict(list)
    for (first, second), count in covisibility.items():
        if count >= args.covisibility_min_landmarks:
            covisible_neighbors[first].append(second)
            covisible_neighbors[second].append(first)
    adaptive_salad_thresholds = np.full(len(rows), args.min_salad_similarity, dtype=np.float32)
    covisible_salad_reference = np.full(len(rows), np.nan, dtype=np.float32)
    for frame, neighbors in covisible_neighbors.items():
        reference = float(np.median(descriptors[frame] @ descriptors[np.asarray(neighbors)].T))
        covisible_salad_reference[frame] = reference
        adaptive_salad_thresholds[frame] = max(
            args.min_salad_similarity, args.covisibility_similarity_ratio * reference)
    # Exact inner-product FAISS index: equivalent ranking for normalized SALAD
    # descriptors today, but avoids an O(N^2) Python/Numpy retrieval path as
    # recordings become much longer.
    index = faiss.IndexFlatIP(descriptors.shape[1])
    index.add(np.ascontiguousarray(descriptors.astype(np.float32)))
    _, retrieval_indices = index.search(
        np.ascontiguousarray(descriptors.astype(np.float32)),
        min(len(rows), args.retrieval_pool_size + args.temporal_exclusion_frames + 1),
    )
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if split_models:
        matcher_session = ort.InferenceSession(str(args.lightglue_matcher_model), providers=providers)
        if args.feature_cache is None:
            extractor_session = ort.InferenceSession(str(args.superpoint_model), providers=providers)
            sessions = (extractor_session, matcher_session)
        else:
            extractor_session = None
            sessions = (matcher_session,)
    else:
        session = ort.InferenceSession(str(args.lightglue_model), providers=providers)
        sessions = (session,)
    width, height = int(rows[0]["width_px"]), int(rows[0]["height_px"])
    # Cached split-model features already contain every local input required by
    # LightGlue. Avoid decoding the full image sequence a second time.
    images = None
    if args.feature_cache is None:
        images = [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32)[None, None] / 255.0
            for path in image_paths
        ]
    feature_start = time.perf_counter()
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    if args.feature_cache is not None:
        if not args.feature_cache.is_dir():
            raise RuntimeError(f"missing frontend feature cache: {args.feature_cache}")
        for frame in range(len(rows)):
            path = args.feature_cache / f"frame_{frame:06d}.npz"
            if not path.is_file():
                raise RuntimeError(f"missing frontend feature cache entry: {path}")
            with np.load(path) as stored:
                points = stored["keypoints"].astype(np.float32)
                features = stored["descriptors"].astype(np.float32)
            cache[frame] = (points, keypoint_mapping(points, frame_export[frame][0]), features)
    elif split_models:
        assert images is not None
        for frame, image in enumerate(images):
            points, _, features = extractor_session.run(None, {"image": image})
            points = points[0].astype(np.float32)
            cache[frame] = (points, keypoint_mapping(points, frame_export[frame][0]), features[0].astype(np.float32))
    feature_seconds = time.perf_counter() - feature_start
    rejected: list[dict[str, object]] = []
    verified: list[dict[str, object]] = []
    per_frame_stats = [defaultdict(float) for _ in rows]
    seen_pairs: set[tuple[int, int]] = set()
    matcher_start = time.perf_counter()
    for query in range(len(rows)):
        candidates = [int(index) for index in retrieval_indices[query] if int(index) != query]
        kept = 0
        for candidate in candidates:
            pair = tuple(sorted((candidate, query)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            spatial = float(np.linalg.norm(T_M_C[candidate][:3, 3] - T_M_C[query][:3, 3]))
            odom_distance = float(abs(trajectory_distance[candidate] - trajectory_distance[query]))
            base = {"query_frame": query, "candidate_frame": candidate, "salad_similarity": float(descriptors[query] @ descriptors[candidate]), "spatial_distance_m": spatial, "trajectory_distance_m": odom_distance}
            per_frame_stats[query]["retrieved_candidates"] += 1
            if abs(candidate - query) <= args.temporal_exclusion_frames:
                rejected.append(base | {"reason": "temporal_neighbor"})
                continue
            if odom_distance < args.min_trajectory_separation_m:
                rejected.append(base | {"reason": "trajectory_distance_lt_threshold"})
                continue
            if spatial > args.max_spatial_candidate_distance_m:
                rejected.append(base | {"reason": "spatial_distance_gt_threshold", "max_spatial_candidate_distance_m": args.max_spatial_candidate_distance_m})
                continue
            if covisibility.get(pair, 0) >= args.covisibility_min_landmarks:
                rejected.append(base | {"reason": "covisibility_neighbor", "shared_landmarks": covisibility[pair]})
                continue
            if base["salad_similarity"] < adaptive_salad_thresholds[query]:
                rejected.append(base | {"reason": "salad_similarity_lt_adaptive_threshold", "adaptive_salad_threshold": float(adaptive_salad_thresholds[query]), "covisible_salad_reference": None if np.isnan(covisible_salad_reference[query]) else float(covisible_salad_reference[query])})
                continue
            per_frame_stats[query]["similarity_gated_candidates"] += 1
            pnp_start = time.perf_counter()
            if split_models:
                k0, ids0, desc0 = cache[candidate]
                k1, ids1, desc1 = cache[query]
                matches0, _, scores0, _ = matcher_session.run(None, {
                    "kpts0": normalize_keypoints(k0, width, height)[None],
                    "kpts1": normalize_keypoints(k1, width, height)[None],
                    "desc0": desc0[None], "desc1": desc1[None],
                })
            else:
                assert images is not None
                k0, k1, matches0, _, scores0, _ = session.run(None, {"image0": images[candidate], "image1": images[query]})
                k0, k1 = k0[0], k1[0]
                for frame, points in ((candidate, k0), (query, k1)):
                    if frame not in cache:
                        cache[frame] = (points, keypoint_mapping(points, frame_export[frame][0]), None)
                ids0 = cache[candidate][1]
                ids1 = cache[query][1]
            per_frame_stats[query]["lightglue_pnp_ms"] += 1000.0 * (time.perf_counter() - pnp_start)
            per_frame_stats[query]["lightglue_pnp_attempts"] += 1
            chosen = np.flatnonzero((matches0[0] >= 0) & (scores0[0] >= args.min_lightglue_score))
            object_points, image_points, correspondences = [], [], []
            for index0 in chosen:
                exported_index = int(ids0[index0])
                if exported_index < 0:
                    continue
                landmark_id = frame_export[candidate][1][exported_index]
                if landmark_id in landmark_xyz:
                    object_points.append(landmark_xyz[landmark_id])
                    image_points.append(k1[int(matches0[0, index0])])
                    correspondences.append({
                        "candidate_landmark_id": landmark_id,
                        "query_keypoint_index": int(ids1[int(matches0[0, index0])]),
                    })
            if len(object_points) < args.min_pnp_inliers:
                rejected.append(base | {"reason": "pnp_inliers", "pnp_inliers": 0, "lightglue_matches": int(len(chosen))})
                continue
            cells = {
                (min(3, int(point[0] * 4 / width)), min(2, int(point[1] * 3 / height)))
                for point in image_points
            }
            if len(cells) < 6:
                rejected.append(base | {"reason": "geometric_inconsistency", "lightglue_matches": int(len(chosen)), "grid_cells": len(cells)})
                continue
            camera = np.array([[float(rows[query]["fx_px"]), 0, float(rows[query]["cx_px"])], [0, float(rows[query]["fy_px"]), float(rows[query]["cy_px"])], [0, 0, 1]], dtype=np.float64)
            object_array = np.asarray(object_points, dtype=np.float64)
            image_array = np.asarray(image_points, dtype=np.float64)
            # Positional arguments keep this compatible with OpenCV 4 and 5.
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_array, image_array, camera, None, None, None, False,
                2000, args.pnp_ransac_px, 0.999, cv2.SOLVEPNP_EPNP)
            count = 0 if inliers is None else len(inliers)
            if not ok or count < args.min_pnp_inliers:
                rejected.append(base | {"reason": "pnp_inliers", "pnp_inliers": int(count), "lightglue_matches": int(len(chosen))})
                continue
            inlier_ids = inliers.reshape(-1)
            cv2.solvePnPRefineLM(object_array[inlier_ids], image_array[inlier_ids], camera, None, rvec, tvec)
            projected, _ = cv2.projectPoints(object_array[inlier_ids], rvec, tvec, camera, None)
            reprojection = np.linalg.norm(image_array[inlier_ids] - projected.reshape(-1, 2), axis=1)
            rotation, _ = cv2.Rodrigues(rvec)
            T_Cq_M = transform(rotation, tvec.reshape(3))
            T_M_Cq_pnp = np.linalg.inv(T_Cq_M)
            delta = T_M_Cq_pnp @ np.linalg.inv(T_M_C[query])
            delta_translation_m = float(np.linalg.norm(delta[:3, 3]))
            if delta_translation_m > args.max_loop_delta_translation_m:
                rejected.append(base | {"reason": "loop_delta_translation_gt_threshold", "delta_translation_m": delta_translation_m, "max_loop_delta_translation_m": args.max_loop_delta_translation_m, "pnp_inliers": int(count)})
                continue
            T_from_to = np.linalg.inv(T_M_C[candidate]) @ T_M_Cq_pnp
            disagreement = np.linalg.inv(np.linalg.inv(T_M_C[candidate]) @ T_M_C[query]) @ T_from_to
            verified.append(base | {"pnp_inliers": int(count), "median_reprojection_px": float(np.median(reprojection)), "T_from_to": T_from_to, "delta": delta, "delta_translation_m": delta_translation_m, "covariance": covariance(object_array[inlier_ids], image_array[inlier_ids], rvec, tvec, camera), "odom_pnp_translation_difference_m": float(np.linalg.norm(disagreement[:3, 3])), "odom_pnp_rotation_difference_deg": angle_deg(disagreement[:3, :3]), "observations": [correspondences[int(index)] for index in inlier_ids if correspondences[int(index)]["query_keypoint_index"] >= 0]})
            kept += 1
            if kept >= args.top_k:
                break
    matcher_seconds = time.perf_counter() - matcher_start
    clusters: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for item in verified:
        clusters[(int(item["candidate_frame"]) // args.support_window_frames, int(item["query_frame"]) // args.support_window_frames)].append(item)
    accepted: list[dict[str, object]] = []
    for group in clusters.values():
        anchor = max(group, key=lambda item: int(item["pnp_inliers"]))
        consistent = [item for item in group if np.linalg.norm(item["delta"][:3, 3] - anchor["delta"][:3, 3]) <= 0.5 and angle_deg(item["delta"][:3, :3] @ anchor["delta"][:3, :3].T) <= 10.0]
        if len(consistent) < args.min_support:
            for item in group:
                rejected.append({key: value for key, value in item.items() if key not in ("T_from_to", "delta", "covariance")} | {"reason": "geometric_inconsistency", "support": len(consistent)})
            continue
        winner = max(consistent, key=lambda item: int(item["pnp_inliers"]))
        candidate, query = int(winner["candidate_frame"]), int(winner["query_frame"])
        accepted.append({
            "candidate_frame": candidate,
            "query_frame": query,
            "pnp_inliers": int(winner["pnp_inliers"]),
            "median_reprojection_px": float(winner["median_reprojection_px"]),
            "odom_pnp_translation_difference_m": float(winner["odom_pnp_translation_difference_m"]),
            "odom_pnp_rotation_difference_deg": float(winner["odom_pnp_rotation_difference_deg"]),
            "camera_from": {"timestamp_ns": int(vertices[candidate]["timestamp_ns"]), "pose": yaml_transform(T_M_C[candidate])},
            "camera_to": {"timestamp_ns": int(vertices[query]["timestamp_ns"]), "pose": yaml_transform(T_M_C[query])},
            "T_from_to": yaml_transform(winner["T_from_to"]),
            "switch_variable": 1.0,
            "switch_variable_variance": 1e-4,
            "covariance": winner["covariance"].reshape(-1).tolist(),
            "observations": winner["observations"],
        })
    # A returning camera yields many adjacent verified pairs.  Keep one strong
    # representative per temporal event, independently of its physical pose.
    accepted.sort(key=lambda edge: (-edge["pnp_inliers"], edge["median_reprojection_px"]))
    nms_accepted: list[dict[str, object]] = []
    for edge in accepted:
        first, second = sorted((int(edge["candidate_frame"]), int(edge["query_frame"])))
        if any(
            abs(first - min(int(kept["candidate_frame"]), int(kept["query_frame"]))) <= args.loop_nms_frames
            and abs(second - max(int(kept["candidate_frame"]), int(kept["query_frame"]))) <= args.loop_nms_frames
            for kept in nms_accepted
        ):
            continue
        nms_accepted.append(edge)
    accepted = nms_accepted
    (args.output / "verified_loops.yaml").write_text(yaml.safe_dump(accepted, sort_keys=False), encoding="utf-8")
    fields = sorted({key for row in rejected for key in row if not isinstance(row[key], np.ndarray)})
    with (args.output / "rejected_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key in fields} for row in rejected])
    with (args.output / "timing_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("frame_index", "salad_descriptor_ms", "covisible_salad_reference",
                  "adaptive_salad_threshold", "retrieved_candidates",
                  "similarity_gated_candidates", "lightglue_pnp_attempts", "lightglue_pnp_ms")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame, stats in enumerate(per_frame_stats):
            writer.writerow({
                "frame_index": frame, "salad_descriptor_ms": round(salad_per_frame_ms[frame], 4),
                "covisible_salad_reference": "" if np.isnan(covisible_salad_reference[frame]) else round(float(covisible_salad_reference[frame]), 6),
                "adaptive_salad_threshold": round(float(adaptive_salad_thresholds[frame]), 6),
                "retrieved_candidates": int(stats["retrieved_candidates"]),
                "similarity_gated_candidates": int(stats["similarity_gated_candidates"]),
                "lightglue_pnp_attempts": int(stats["lightglue_pnp_attempts"]),
                "lightglue_pnp_ms": round(float(stats["lightglue_pnp_ms"]), 4),
            })
    feature_timing_name = "feature_cache_load" if args.feature_cache else "superpoint_extraction"
    report = {"backend": "DINOv2-SALAD + SuperPoint-LightGlue ONNX + PnP", "retrieval": "faiss_exact_inner_product_adaptive_similarity_gate", "frames": len(rows), "top_k": args.top_k, "retrieval_pool_size": args.retrieval_pool_size, "covisibility_similarity_ratio": args.covisibility_similarity_ratio, "min_salad_similarity": args.min_salad_similarity, "max_loop_delta_translation_m": args.max_loop_delta_translation_m, "max_spatial_candidate_distance_m": args.max_spatial_candidate_distance_m, "temporal_exclusion_frames": args.temporal_exclusion_frames, "loop_nms_frames": args.loop_nms_frames, "covisibility_min_landmarks": args.covisibility_min_landmarks, "min_trajectory_separation_m": args.min_trajectory_separation_m, "min_pnp_inliers": args.min_pnp_inliers, "raw_pnp_verified": len(verified), "accepted_loops": len(accepted), "rejected": len(rejected), "lightglue_provider": sessions[0].get_providers()[0], "feature_cache": str(args.feature_cache) if args.feature_cache else None, "definitions": {"salad_similarity": "L2-normalized descriptor inner product (cosine similarity); larger means visually closer.", "adaptive_salad_threshold": "max(min_salad_similarity, covisibility_similarity_ratio * median SALAD similarity to the query frame's covisible neighbors).", "spatial_distance_m": "Euclidean distance between current optimized camera centers in M.", "trajectory_distance_m": "Absolute difference of accumulated optimized-camera path length along the trajectory.", "delta_translation_m": "Norm of translation of T_M_Cquery_from_PnP * inverse(T_M_Cquery_current); correction applied to the query camera pose, in meters."}, "timing_seconds": {"salad": salad_seconds, feature_timing_name: feature_seconds, "lightglue_pnp": matcher_seconds, "total": salad_seconds + feature_seconds + matcher_seconds}, "timing_per_frame_csv": "timing_per_frame.csv"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
