#!/usr/bin/env python3
"""Sparse, fully unconditioned DA3 reconstruction followed by TSDF fusion."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--da3-repo", type=Path, default=Path(os.environ["PHONE_AI_DA3_REPO"]))
    parser.add_argument("--model", type=Path, default=Path(os.environ["PHONE_AI_DA3_MODEL"]))
    parser.add_argument("--fuser", type=Path, required=True)
    return parser.parse_args()


def extract_uniform_images(data: Path, frames_csv: Path, output: Path, count: int):
    rows = list(csv.DictReader(frames_csv.open()))
    probe = cv2.VideoCapture(str(data / "wide.mp4"))
    video_frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    rows = [row for row in rows if 0 <= int(row["record_slot"]) < video_frame_count]
    # Some iPhone recordings expose 1-2 trailing metadata rows beyond frames
    # reliably decodable by OpenCV. Keep the sampling set inside the video body.
    rows = rows[:-4]
    if count < 3 or count > len(rows):
        raise ValueError(f"frames must be in [3, {len(rows)}]")
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int32)
    selected = [rows[index] for index in indices]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(data / "wide.mp4"))
    records = []
    for index, row in enumerate(selected):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["record_slot"]))
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode record slot {row['record_slot']}")
        name = f"frame_{index:06d}.jpg"
        cv2.imwrite(str(images / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({
            "da3_index": index,
            "record_slot": int(row["record_slot"]),
            "timestamp_ns": int(row["timestamp_ns"]),
            "image": f"images/{name}",
        })
    capture.release()
    with (output / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return images, records


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output must not exist: {args.output}")
    args.output.mkdir(parents=True)
    images, records = extract_uniform_images(
        args.data, args.normalized / "frames.csv", args.output / "input", args.frames
    )
    sys.path.insert(0, str(args.da3_repo / "src"))
    from depth_anything_3.api import DepthAnything3

    start = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model)).to("cuda")
    load_seconds = time.perf_counter() - start
    inference_start = time.perf_counter()
    model.inference(
        [str(path) for path in sorted(images.glob("*.jpg"))],
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=False,
        process_res=args.process_res,
        process_res_method="upper_bound_resize",
        export_dir=str(args.output / "da3_output"),
        export_format="mini_npz",
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    npz = args.output / "da3_output" / "exports" / "mini_npz" / "results.npz"
    data = np.load(npz)
    expected = {"depth", "conf", "intrinsics", "extrinsics"}
    if not expected.issubset(data.files):
        raise RuntimeError(f"DA3 output lacks fields: {expected - set(data.files)}")
    if len(data["depth"]) != len(records):
        raise RuntimeError("DA3 output frame count mismatch")
    subprocess.run(
        [sys.executable, str(args.fuser), "--npz", str(npz), "--images", str(images),
         "--output", str(args.output / "tsdf")],
        check=True,
    )
    (args.output / "run_stats.json").write_text(json.dumps({
        "frames": args.frames,
        "process_res": args.process_res,
        "conditioned_on_pose": False,
        "conditioned_on_intrinsics": False,
        "scale_source": "DA3 self-estimated; arbitrary scale",
        "pose_source": "DA3 self-estimated",
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
