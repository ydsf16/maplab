#!/usr/bin/env python3
"""Learned cross-session loops with per-session SE(3) RANSAC consensus."""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
import yaml

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "sensor-recorder-loop-closure"))
from salad_lightglue_loop_closure import (  # noqa: E402
    angle_deg, covariance, keypoint_mapping, normalize_keypoints,
    quat_xyzw, read_csv, rotation_from_wxyz, salad_descriptors, transform,
    yaml_transform,
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--superpoint-model", type=Path, required=True)
    parser.add_argument("--lightglue-matcher-model", type=Path, required=True)
    parser.add_argument("--salad-repo", type=Path, default=Path(os.environ["PHONE_AI_SALAD_REPO"]))
    parser.add_argument("--salad-checkpoint", type=Path, default=Path(os.environ["PHONE_AI_SALAD_CHECKPOINT"]))
    parser.add_argument("--dinov2-repo", type=Path, default=Path(os.environ["PHONE_AI_DINOV2_REPO"]))
    parser.add_argument("--dinov2-checkpoint", type=Path, default=Path(os.environ["PHONE_AI_DINOV2_CHECKPOINT"]))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-salad-similarity", type=float, default=0.0)
    parser.add_argument("--min-lightglue-score", type=float, default=0.15)
    parser.add_argument("--pnp-ransac-px", type=float, default=3.0)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)
    parser.add_argument("--ransac-translation-m", type=float, default=0.75)
    parser.add_argument("--ransac-rotation-deg", type=float, default=8.0)
    parser.add_argument("--ransac-min-loops", type=int, default=3)
    parser.add_argument("--pgo-nms-frames", type=int, default=30,
                        help="One strongest RANSAC inlier per session-pair temporal event.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_session(spec: dict) -> dict:
    result = Path(spec["result"]).resolve()
    export = Path(spec["export"]).resolve()
    rows = read_csv(result / "normalized/keyframes.csv")
    vertices = read_csv(export / "vertices.csv")
    points = read_csv(export / "keypoints.csv")
    landmarks = read_csv(export / "landmarks.csv")
    if len(rows) != len(vertices):
        raise RuntimeError(f"{spec['id']}: keyframe / vertex count mismatch")
    report = json.loads((result / "maps/04_initial_vi_ba_intrinsics/report.json").read_text())
    cal = report["calibration"]
    T_C_I = transform(rotation_from_wxyz(cal["final_q_C_I_wxyz"]), np.array(cal["final_t_C_I_m"], dtype=np.float64))
    T_I_C = np.linalg.inv(T_C_I)
    poses = []
    for vertex in vertices:
        T_M_I = transform(rotation_from_wxyz([float(vertex[k]) for k in ("q_w", "q_x", "q_y", "q_z")]), np.array([float(vertex[k]) for k in ("p_x_m", "p_y_m", "p_z_m")]))
        poses.append(T_M_I @ T_I_C)
    frame_export = {}
    for frame in range(len(rows)):
        found = [row for row in points if int(row["vertex_index"]) == frame]
        frame_export[frame] = (np.array([[float(row["u_px"]), float(row["v_px"])] for row in found], dtype=np.float32), [row["landmark_id"] for row in found])
    cache_dir = result / "features/superpoint_lightglue/feature_cache"
    cache = {}
    for frame in range(len(rows)):
        with np.load(cache_dir / f"frame_{frame:06d}.npz") as stored:
            kp, desc = stored["keypoints"].astype(np.float32), stored["descriptors"].astype(np.float32)
        cache[frame] = (kp, keypoint_mapping(kp, frame_export[frame][0]), desc)
    return {"id": spec["id"], "result": result, "rows": rows, "vertices": vertices,
            "T_M_C": poses, "T_C_I": T_C_I, "T_I_C": T_I_C,
            "landmarks": {r["landmark_id"]: np.array([float(r[k]) for k in ("x_m", "y_m", "z_m")]) for r in landmarks},
            "frame_export": frame_export, "cache": cache}


def hypothesis(session_a: dict, frame_a: int, session_b: dict, frame_b: int,
               T_Cb_Ma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return session transform M_A<-M_B and direct IMU loop I_A<-I_B."""
    T_Ma_Mb = np.linalg.inv(T_Cb_Ma) @ np.linalg.inv(session_b["T_M_C"][frame_b])
    T_Ca_Cb = np.linalg.inv(session_a["T_M_C"][frame_a]) @ T_Ma_Mb @ session_b["T_M_C"][frame_b]
    return T_Ma_Mb, session_a["T_I_C"] @ T_Ca_Cb @ session_b["T_C_I"]


def consensus(raw: list[dict], trans: float, rot: float, minimum: int) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    grouped = defaultdict(list)
    for edge in raw:
        grouped[(edge["session_from"], edge["session_to"])].append(edge)
    for pair, edges in grouped.items():
        winner = []
        for candidate in edges:
            T = np.asarray(candidate["_session_transform"])
            inliers = []
            for edge in edges:
                error = np.linalg.inv(T) @ np.asarray(edge["_session_transform"])
                if np.linalg.norm(error[:3, 3]) <= trans and angle_deg(error[:3, :3]) <= rot:
                    inliers.append(edge)
            if len(inliers) > len(winner) or (len(inliers) == len(winner) and sum(x["pnp_inliers"] for x in inliers) > sum(x["pnp_inliers"] for x in winner)):
                winner = inliers
        temporal_blocks = {(x["camera_from"]["frame_index"] // 3, x["camera_to"]["frame_index"] // 3) for x in winner}
        if len(winner) < minimum or len(temporal_blocks) < 2:
            for edge in edges:
                edge["rejection_reason"] = "insufficient_session_pair_consensus"
                rejected.append(edge)
            continue
        winning_ids = {id(x) for x in winner}
        for edge in edges:
            if id(edge) in winning_ids:
                edge["session_pair_ransac_inlier_count"] = len(winner)
                accepted.append(edge)
            else:
                edge["rejection_reason"] = "cross_session_se3_ransac_outlier"
                rejected.append(edge)
    return accepted, rejected


def clean(edge: dict) -> dict:
    return {k: v for k, v in edge.items() if not k.startswith("_")}


def main() -> int:
    a = args(); a.output.mkdir(parents=True, exist_ok=True)
    specs = yaml.safe_load(a.sessions_yaml.read_text())
    sessions = [load_session(spec) for spec in specs["sessions"]]
    images, lookup = [], []
    for si, session in enumerate(sessions):
        for frame, row in enumerate(session["rows"]):
            images.append(session["result"] / "normalized" / row["image_path"]); lookup.append((si, frame))
    started = time.perf_counter()
    descriptors, _ = salad_descriptors(images, a.device, a.salad_repo, a.salad_checkpoint, a.dinov2_repo, a.dinov2_checkpoint, 16)
    np.save(a.output / "salad_descriptors.npy", descriptors)
    index = faiss.IndexFlatIP(descriptors.shape[1]); index.add(np.ascontiguousarray(descriptors.astype(np.float32)))
    _, neighbors = index.search(np.ascontiguousarray(descriptors.astype(np.float32)), len(images))
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    matcher = ort.InferenceSession(str(a.lightglue_matcher_model), providers=providers)
    raw, rejected = [], []
    for qi, (query_si, query_frame) in enumerate(lookup):
        query = sessions[query_si]; chosen = 0
        for ci in neighbors[qi]:
            candidate_si, candidate_frame = lookup[int(ci)]
            if candidate_si == query_si: continue
            score = float(descriptors[qi] @ descriptors[int(ci)])
            if score < a.min_salad_similarity: continue
            candidate = sessions[candidate_si]
            k0, ids0, d0 = candidate["cache"][candidate_frame]
            k1, ids1, d1 = query["cache"][query_frame]
            width, height = int(query["rows"][query_frame]["width_px"]), int(query["rows"][query_frame]["height_px"])
            matches, _, scores, _ = matcher.run(None, {"kpts0": normalize_keypoints(k0, width, height)[None], "kpts1": normalize_keypoints(k1, width, height)[None], "desc0": d0[None], "desc1": d1[None]})
            objects, pixels, observations = [], [], []
            for i0 in np.flatnonzero((matches[0] >= 0) & (scores[0] >= a.min_lightglue_score)):
                e0 = int(ids0[i0]); e1 = int(ids1[int(matches[0, i0])])
                if e0 < 0: continue
                landmark_id = candidate["frame_export"][candidate_frame][1][e0]
                if landmark_id in candidate["landmarks"]:
                    objects.append(candidate["landmarks"][landmark_id]); pixels.append(k1[int(matches[0, i0])]); observations.append({"candidate_landmark_id": landmark_id, "query_keypoint_index": e1})
            base = {"session_from": candidate["id"], "session_to": query["id"], "candidate_frame": candidate_frame, "query_frame": query_frame, "salad_similarity": score}
            if len(objects) < a.min_pnp_inliers:
                rejected.append(base | {"rejection_reason": "pnp_correspondences"}); continue
            K = np.array([[float(query["rows"][query_frame]["fx_px"]), 0, float(query["rows"][query_frame]["cx_px"])], [0, float(query["rows"][query_frame]["fy_px"]), float(query["rows"][query_frame]["cy_px"])], [0, 0, 1]], dtype=np.float64)
            objects, pixels = np.asarray(objects, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(objects, pixels, K, None, None, None, False, 2000, a.pnp_ransac_px, 0.999, cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < a.min_pnp_inliers:
                rejected.append(base | {"rejection_reason": "pnp_inliers", "pnp_inliers": 0 if inliers is None else int(len(inliers))}); continue
            ii = inliers.reshape(-1); cv2.solvePnPRefineLM(objects[ii], pixels[ii], K, None, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec); T_Ma_Mb, T_Ia_Ib = hypothesis(candidate, candidate_frame, query, query_frame, transform(R, tvec.reshape(3)))
            projected, _ = cv2.projectPoints(objects[ii], rvec, tvec, K, None)
            raw.append(base | {"pnp_inliers": int(len(ii)), "median_reprojection_px": float(np.median(np.linalg.norm(pixels[ii] - projected.reshape(-1, 2), axis=1))), "camera_from": {"frame_index": candidate_frame, "vertex_id": candidate["vertices"][candidate_frame]["vertex_id"], "timestamp_ns": int(candidate["vertices"][candidate_frame]["timestamp_ns"]), "pose": yaml_transform(candidate["T_M_C"][candidate_frame])}, "camera_to": {"frame_index": query_frame, "vertex_id": query["vertices"][query_frame]["vertex_id"], "timestamp_ns": int(query["vertices"][query_frame]["timestamp_ns"]), "pose": yaml_transform(query["T_M_C"][query_frame])}, "T_from_to": yaml_transform(T_Ia_Ib), "direct_constraint": True, "switch_variable": 1.0, "switch_variable_variance": 1e-4, "covariance": covariance(objects[ii], pixels[ii], rvec, tvec, K).reshape(-1).tolist(), "observations": [observations[int(i)] for i in ii if observations[int(i)]["query_keypoint_index"] >= 0], "_session_transform": T_Ma_Mb.tolist()})
            chosen += 1
            if chosen >= a.top_k: break
    accepted, ransac_rejected = consensus(raw, a.ransac_translation_m, a.ransac_rotation_deg, a.ransac_min_loops)
    pgo_edges = []
    by_event = defaultdict(list)
    for edge in accepted:
        key = (edge["session_from"], edge["session_to"],
               edge["camera_from"]["frame_index"] // a.pgo_nms_frames,
               edge["camera_to"]["frame_index"] // a.pgo_nms_frames)
        by_event[key].append(edge)
    for group in by_event.values():
        pgo_edges.append(max(group, key=lambda x: (x["pnp_inliers"], -x["median_reprojection_px"])))
    rejected.extend(ransac_rejected)
    (a.output / "raw_pnp_loops.yaml").write_text(yaml.safe_dump([clean(x) for x in raw], sort_keys=False))
    (a.output / "consensus_cross_session_loops.yaml").write_text(yaml.safe_dump([clean(x) for x in accepted], sort_keys=False))
    (a.output / "pgo_cross_session_loops.yaml").write_text(yaml.safe_dump([clean(x) for x in pgo_edges], sort_keys=False))
    with (a.output / "rejected_cross_session_loops.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_from", "session_to", "candidate_frame", "query_frame", "salad_similarity", "rejection_reason", "pnp_inliers"]); writer.writeheader(); writer.writerows([{k: x.get(k, "") for k in writer.fieldnames} for x in rejected])
    (a.output / "report.json").write_text(json.dumps({"sessions": [s["id"] for s in sessions], "raw_pnp_loops": len(raw), "session_pair_ransac_accepted": len(accepted), "pgo_event_nms_edges": len(pgo_edges), "pgo_nms_frames": a.pgo_nms_frames, "rejected": len(rejected), "seconds": time.perf_counter() - started, "top_k": a.top_k, "backend": "SALAD + SuperPoint-LightGlue + PnP + per-session-pair SE3 RANSAC"}, indent=2) + "\n")
    return 0

if __name__ == "__main__": raise SystemExit(main())
