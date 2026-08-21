#!/usr/bin/env python3
"""Estimate and apply robust relative depth scales across overlapping DA3 windows."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def robust_ratio(depth_a, conf_a, depth_b, conf_b):
    valid = (
        np.isfinite(depth_a) & np.isfinite(depth_b) &
        np.isfinite(conf_a) & np.isfinite(conf_b) &
        (depth_a > 0.1) & (depth_b > 0.1) &
        (depth_a < 10.0) & (depth_b < 10.0)
    )
    # Subsample keeps this estimator inexpensive while preserving the image-wide median.
    ratio = (depth_a[valid][::16] / depth_b[valid][::16])
    ratio = ratio[(ratio > 0.5) & (ratio < 2.0)]
    if ratio.size < 500:
        return None
    median = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - median)))
    if mad > 0:
        ratio = ratio[np.abs(ratio - median) <= 3.0 * 1.4826 * mad]
    if ratio.size < 250:
        return None
    return float(np.median(ratio)), int(ratio.size), float(np.median(np.abs(ratio - np.median(ratio))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    windows = sorted(args.windows_root.glob("window_*"))
    if len(windows) < 2:
        raise RuntimeError("need at least two DA3 windows")
    ids, data = [], []
    for window in windows:
        rows = list(csv.DictReader((window / "input" / "frames.csv").open()))
        slots = [int(row["record_slot"]) for row in rows]
        npz = np.load(window / "da3_output" / "exports" / "mini_npz" / "results.npz")
        data.append({k: npz[k] for k in npz.files})
        ids.append({slot: index for index, slot in enumerate(slots)})
    edges = []
    for i in range(len(windows) - 1):
        shared = sorted(set(ids[i]).intersection(ids[i + 1]))
        ratios = []
        for slot in shared:
            a, b = ids[i][slot], ids[i + 1][slot]
            estimate = robust_ratio(data[i]["depth"][a], data[i]["conf"][a], data[i + 1]["depth"][b], data[i + 1]["conf"][b])
            if estimate is not None:
                ratios.append(estimate)
        if not ratios:
            continue
        values = np.asarray([x[0] for x in ratios])
        edges.append({"from_window": i, "to_window": i + 1, "overlap_frames": len(ratios), "ratio_depth_from_over_to": float(np.median(values)), "ratio_mad": float(np.median(np.abs(values - np.median(values))))})
    if len(edges) != len(windows) - 1:
        raise RuntimeError(f"incomplete overlap graph: {len(edges)} edges for {len(windows)} windows")
    # Solve log(c_to) - log(c_from) = log(depth_from / depth_to), c_0 = 1.
    a, b = [], []
    for edge in edges:
        row = np.zeros(len(windows) - 1)
        if edge["from_window"] > 0: row[edge["from_window"] - 1] -= 1.0
        row[edge["to_window"] - 1] += 1.0
        a.append(row); b.append(np.log(edge["ratio_depth_from_over_to"]))
    log_scales = np.r_[0.0, np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)[0]]
    scales = np.exp(log_scales)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_npzs = []
    for index, item in enumerate(data):
        corrected = dict(item); corrected["depth"] = item["depth"] * scales[index]
        path = args.output_root / f"window_{index:03d}_results.npz"
        np.savez_compressed(path, **corrected); output_npzs.append(str(path))
    report = {"method": "overlap_frame_median_depth_ratio_global_log_scale", "windows": len(windows), "scales": scales.tolist(), "edges": edges, "output_npzs": output_npzs}
    (args.output_root / "scale_consistency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
