#!/usr/bin/env python3
"""Build multi-frame SuperPoint + LightGlue tracks with ONNX Runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.frames: list[dict[int, int]] = []

    def add(self, frame: int, keypoint: int) -> int:
        node = len(self.parent)
        self.parent.append(node)
        self.frames.append({frame: keypoint})
        return node

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def merge(self, left: int, right: int) -> bool:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return True
        overlap = set(self.frames[left]).intersection(self.frames[right])
        if any(self.frames[left][frame] != self.frames[right][frame] for frame in overlap):
            return False
        if len(self.frames[left]) < len(self.frames[right]):
            left, right = right, left
        self.parent[right] = left
        self.frames[left].update(self.frames[right])
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pair-gap", type=int, default=3)
    parser.add_argument("--pose-gated-min-distance-m", type=float, default=0.15)
    parser.add_argument("--pose-gated-max-distance-m", type=float, default=2.0)
    parser.add_argument("--pose-gated-max-rotation-deg", type=float, default=20.0)
    parser.add_argument("--pose-gated-max-pairs-per-frame", type=int, default=4)
    parser.add_argument(
        "--disable-pose-gated-pairs", action="store_true",
        help="Use sequential sliding-window pairs only.",
    )
    parser.add_argument("--min-match-score", type=float, default=0.15)
    parser.add_argument("--ransac-threshold-px", type=float, default=1.0)
    parser.add_argument("--min-pair-inliers", type=int, default=15)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def load_gray(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unable to read image: {path}")
    if image.shape != (height, width):
        raise RuntimeError(
            f"unexpected image size {image.shape[::-1]} for {path}; expected {(width, height)}"
        )
    return (image.astype(np.float32) / 255.0)[None, None]


def geometric_inliers(
    points0: np.ndarray,
    points1: np.ndarray,
    row0: dict[str, str],
    row1: dict[str, str],
    threshold_px: float,
) -> np.ndarray:
    if len(points0) < 8:
        return np.zeros(len(points0), dtype=bool)
    k0 = np.array(
        [[float(row0["fx_px"]), 0, float(row0["cx_px"])],
         [0, float(row0["fy_px"]), float(row0["cy_px"])], [0, 0, 1]],
        dtype=np.float64,
    )
    k1 = np.array(
        [[float(row1["fx_px"]), 0, float(row1["cx_px"])],
         [0, float(row1["fy_px"]), float(row1["cy_px"])], [0, 0, 1]],
        dtype=np.float64,
    )
    normalized0 = cv2.undistortPoints(points0[:, None].astype(np.float64), k0, None)[:, 0]
    normalized1 = cv2.undistortPoints(points1[:, None].astype(np.float64), k1, None)[:, 0]
    focal = 0.25 * (k0[0, 0] + k0[1, 1] + k1[0, 0] + k1[1, 1])
    _, mask = cv2.findEssentialMat(
        normalized0, normalized1, focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=threshold_px / focal,
    )
    return np.zeros(len(points0), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)


def pose_gated_pairs(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> list[tuple[int, int, str, float, float]]:
    """Return unique sequential and ARKit-pose-gated frame pairs.

    ARKit/VIO poses only gate the expensive local matcher.  They never become
    optimization constraints: LightGlue and essential-matrix RANSAC still
    decide whether the pair contributes to a visual track.
    """
    positions = np.asarray(
        [[float(row[f"p_M_I_{axis}_m"]) for axis in "xyz"] for row in rows]
    )
    quaternions = np.asarray(
        [[float(row[f"q_M_I_{axis}"]) for axis in "wxyz"] for row in rows]
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    pairs: list[tuple[int, int, str, float, float]] = []
    matched_pairs: set[tuple[int, int]] = set()

    def append(frame0: int, frame1: int, pair_type: str, distance: float, rotation: float) -> None:
        pair = (frame0, frame1)
        if pair in matched_pairs:
            return
        matched_pairs.add(pair)
        pairs.append((frame0, frame1, pair_type, distance, rotation))

    for gap in range(1, args.max_pair_gap + 1):
        for frame0 in range(len(rows) - gap):
            frame1 = frame0 + gap
            dot = abs(float(np.dot(quaternions[frame0], quaternions[frame1])))
            rotation = math.degrees(2.0 * math.acos(min(1.0, dot)))
            append(
                frame0, frame1, "sequential",
                float(np.linalg.norm(positions[frame1] - positions[frame0])), rotation,
            )

    if args.disable_pose_gated_pairs:
        return pairs
    for frame1 in range(len(rows)):
        candidates: list[tuple[float, float, int]] = []
        for frame0 in range(frame1 - args.max_pair_gap - 1):
            pair = (frame0, frame1)
            if pair in matched_pairs:
                continue
            distance = float(np.linalg.norm(positions[frame1] - positions[frame0]))
            dot = abs(float(np.dot(quaternions[frame0], quaternions[frame1])))
            rotation = math.degrees(2.0 * math.acos(min(1.0, dot)))
            if not (args.pose_gated_min_distance_m <= distance <= args.pose_gated_max_distance_m):
                continue
            if rotation > args.pose_gated_max_rotation_deg:
                continue
            # Prefer a robust triangulation baseline while retaining a similar view.
            candidates.append((abs(distance - 0.6), rotation, frame0))
        for _, _, frame0 in sorted(candidates)[:args.pose_gated_max_pairs_per_frame]:
            distance = float(np.linalg.norm(positions[frame1] - positions[frame0]))
            dot = abs(float(np.dot(quaternions[frame0], quaternions[frame1])))
            rotation = math.degrees(2.0 * math.acos(min(1.0, dot)))
            append(frame0, frame1, "pose_gated", distance, rotation)
    return pairs


def main() -> int:
    args = parse_args()
    rows = read_rows(args.normalized_data / "keyframes.csv")
    width, height = int(rows[0]["width_px"]), int(rows[0]["height_px"])
    if (width, height) != (640, 480):
        raise RuntimeError(f"this frontend expects 640x480 input, got {width}x{height}")
    for row in rows:
        if (int(row["width_px"]), int(row["height_px"])) != (width, height):
            raise RuntimeError("keyframe image dimensions are not constant")
    if args.pose_gated_min_distance_m < 0.0 or args.pose_gated_max_distance_m <= args.pose_gated_min_distance_m:
        raise RuntimeError("invalid pose-gated distance range")
    if args.pose_gated_max_rotation_deg <= 0.0 or args.pose_gated_max_pairs_per_frame < 0:
        raise RuntimeError("invalid pose-gated matching settings")

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    available = ort.get_available_providers()
    if args.provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
    requested = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=requested)
    if args.provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider failed to activate: {session.get_providers()}")

    images = [
        load_gray(args.normalized_data / row["image_path"], width, height)
        for row in rows
    ]
    keypoints: list[np.ndarray | None] = [None] * len(rows)
    keypoint_scores: list[np.ndarray | None] = [None] * len(rows)
    nodes: dict[tuple[int, int], int] = {}
    union_find = UnionFind()
    pair_rows: list[dict[str, int | float]] = []
    rejected_conflicts = 0

    for frame0, frame1, pair_type, pose_distance, pose_rotation in pose_gated_pairs(rows, args):
        kpts0, kpts1, matches0, _, scores0, _ = session.run(
            None, {"image0": images[frame0], "image1": images[frame1]}
        )
        kpts0, kpts1 = kpts0[0].astype(np.float64), kpts1[0].astype(np.float64)
        for frame, current in ((frame0, kpts0), (frame1, kpts1)):
            if keypoints[frame] is None:
                keypoints[frame] = current
                keypoint_scores[frame] = np.zeros(len(current), dtype=np.float32)
            elif not np.array_equal(keypoints[frame], current):
                raise RuntimeError(f"SuperPoint output changed across pairs for frame {frame}")

        match_index = matches0[0].astype(np.int64)
        match_score = scores0[0].astype(np.float32)
        candidate0 = np.flatnonzero((match_index >= 0) & (match_score >= args.min_match_score))
        candidate1 = match_index[candidate0]
        mask = geometric_inliers(
            kpts0[candidate0], kpts1[candidate1], rows[frame0], rows[frame1],
            args.ransac_threshold_px,
        )
        inlier0, inlier1 = candidate0[mask], candidate1[mask]
        accepted = len(inlier0) >= args.min_pair_inliers
        if accepted:
            for index0, index1 in zip(inlier0, inlier1):
                score = float(match_score[index0])
                keypoint_scores[frame0][index0] = max(keypoint_scores[frame0][index0], score)
                keypoint_scores[frame1][index1] = max(keypoint_scores[frame1][index1], score)
                node0 = nodes.setdefault(
                    (frame0, int(index0)), union_find.add(frame0, int(index0))
                ) if (frame0, int(index0)) not in nodes else nodes[(frame0, int(index0))]
                node1 = nodes.setdefault(
                    (frame1, int(index1)), union_find.add(frame1, int(index1))
                ) if (frame1, int(index1)) not in nodes else nodes[(frame1, int(index1))]
                if not union_find.merge(node0, node1):
                    rejected_conflicts += 1
        pair_rows.append({
            "frame0": frame0, "frame1": frame1, "gap": frame1 - frame0,
            "pair_type": pair_type, "pose_distance_m": pose_distance,
            "pose_rotation_deg": pose_rotation,
            "lightglue_matches": int(len(candidate0)),
            "geometric_inliers": int(len(inlier0)), "accepted": int(accepted),
        })

    roots = Counter(union_find.find(node) for node in nodes.values())
    valid_roots = sorted(root for root, count in roots.items() if count >= 2)
    track_by_root = {root: track for track, root in enumerate(valid_roots)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "keypoints.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("vertex_index", "keypoint_index", "u_px", "v_px", "score", "track_id")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame, points in enumerate(keypoints):
            assert points is not None and keypoint_scores[frame] is not None
            for index, point in enumerate(points):
                node = nodes.get((frame, index))
                track = -1 if node is None else track_by_root.get(union_find.find(node), -1)
                writer.writerow({
                    "vertex_index": frame, "keypoint_index": index,
                    "u_px": float(point[0]), "v_px": float(point[1]),
                    "score": float(keypoint_scores[frame][index]), "track_id": track,
                })
    with (args.output / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)
    track_lengths = Counter(roots[root] for root in valid_roots)
    report = {
        "backend": "onnxruntime", "provider": session.get_providers()[0],
        "model": str(args.model), "model_sha256": "3d7479132d7b27dfb4d3cd69274c4542c0499cba9ce4bf7be8df4430600baf83",
        "image_size": [width, height], "frames": len(rows),
        "detected_keypoints": [len(points) for points in keypoints],
        "tracks": len(valid_roots), "track_length_histogram": dict(sorted(track_lengths.items())),
        "accepted_pairs": sum(int(row["accepted"]) for row in pair_rows),
        "total_pairs": len(pair_rows), "rejected_track_conflicts": rejected_conflicts,
        "pose_gated": {
            "enabled": not args.disable_pose_gated_pairs,
            "min_distance_m": args.pose_gated_min_distance_m,
            "max_distance_m": args.pose_gated_max_distance_m,
            "max_rotation_deg": args.pose_gated_max_rotation_deg,
            "max_pairs_per_frame": args.pose_gated_max_pairs_per_frame,
            "total_pairs": sum(row["pair_type"] == "pose_gated" for row in pair_rows),
            "accepted_pairs": sum(
                row["pair_type"] == "pose_gated" and int(row["accepted"])
                for row in pair_rows
            ),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
