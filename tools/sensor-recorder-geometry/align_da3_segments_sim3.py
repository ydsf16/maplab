#!/usr/bin/env python3
"""Put independent DA3 windows into the VI-BA world using robust Sim(3)."""
import argparse
import json
from pathlib import Path

import numpy as np


def c2w(extrinsics):
    if extrinsics.shape[-2:] == (4, 4):
        return np.linalg.inv(extrinsics)
    full = np.tile(np.eye(4), (len(extrinsics), 1, 1))
    full[:, :3, :4] = extrinsics
    return np.linalg.inv(full)


def umeyama(source, target):
    source_mean, target_mean = source.mean(0), target.mean(0)
    xs, xt = source - source_mean, target - target_mean
    covariance = xt.T @ xs / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular) @ sign) / np.mean(np.sum(xs * xs, axis=1)))
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def robust_sim3(source, target, seed):
    rng = np.random.default_rng(seed)
    best = None
    threshold = 0.20
    for _ in range(500):
        sample = rng.choice(len(source), size=4, replace=False)
        scale, rotation, translation = umeyama(source[sample], target[sample])
        error = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
        inliers = error < threshold
        score = (int(inliers.sum()), -float(np.median(error[inliers])) if inliers.any() else -np.inf)
        if best is None or score > best[0]:
            best = (score, inliers)
    inliers = best[1]
    if inliers.sum() < max(8, len(source) // 2):
        inliers = np.ones(len(source), dtype=bool)
    scale, rotation, translation = umeyama(source[inliers], target[inliers])
    error = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    return scale, rotation, translation, inliers, error


def angle_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=20)
    args = parser.parse_args()
    windows = sorted(args.windows_root.glob("window_*"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, window in enumerate(windows):
        prediction = np.load(window / "da3_output" / "exports" / "mini_npz" / "results.npz")
        target = c2w(np.load(window / "input" / "camera_params.npz")["extrinsics"])
        source = c2w(prediction["extrinsics"])
        scale, rotation, translation, inliers, errors = robust_sim3(source[:, :3, 3], target[:, :3, 3], index)
        transformed = np.tile(np.eye(4), (len(source), 1, 1))
        transformed[:, :3, :3] = rotation @ source[:, :3, :3]
        transformed[:, :3, 3] = (scale * (rotation @ source[:, :3, 3].T)).T + translation
        corrected = {name: prediction[name] for name in prediction.files}
        corrected["depth"] = prediction["depth"] * scale
        corrected["extrinsics"] = np.linalg.inv(transformed)[:, :3, :4].astype(np.float32)
        # Overlap is for alignment continuity; suppress duplicate TSDF observations.
        if index > 0:
            corrected["conf"] = prediction["conf"].copy()
            corrected["conf"][:args.overlap] = -np.inf
        output = args.output_root / f"window_{index:03d}_global_results.npz"
        np.savez_compressed(output, **corrected)
        orientation_error = [angle_degrees(target[i, :3, :3] @ transformed[i, :3, :3].T) for i in range(len(target))]
        reports.append({
            "window": index, "frames": len(source), "scale": scale,
            "inliers": int(inliers.sum()), "position_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "position_median_m": float(np.median(errors)),
            "orientation_median_deg": float(np.median(orientation_error)),
            "orientation_p95_deg": float(np.percentile(orientation_error, 95)),
            "output": str(output),
        })
    (args.output_root / "sim3_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
