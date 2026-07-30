#!/usr/bin/env python3
"""Run multiple DA3 windows while keeping one model instance resident on the GPU."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/da3/repo"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/da3/models/DA3-GIANT-1.1"))
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise ValueError("--input and --output must occur equally often")

    sys.path.insert(0, str(args.repo / "src"))
    from depth_anything_3.api import DepthAnything3

    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model)).to("cuda")
    load_seconds = time.perf_counter() - started
    window_stats = []
    for index, (input_dir, output_dir) in enumerate(zip(args.input, args.output)):
        camera = np.load(input_dir / "camera_params.npz")
        images = [str(path) for path in sorted((input_dir / "images").glob("*.jpg"))]
        output_dir.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        model.inference(
            images,
            extrinsics=camera["extrinsics"],
            intrinsics=camera["intrinsics"],
            align_to_input_ext_scale=True,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=str(output_dir),
            export_format="mini_npz",
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        stats = {
            "window_index": index,
            "frames": len(images),
            "process_res": args.process_res,
            "model_load_seconds_shared": load_seconds,
            "inference_seconds": elapsed,
            "model": str(args.model),
            "known_pose": True,
        }
        (output_dir / "run_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        window_stats.append(stats)
    summary = {
        "windows": len(window_stats),
        "frames": sum(stat["frames"] for stat in window_stats),
        "model_load_seconds": load_seconds,
        "inference_seconds": sum(stat["inference_seconds"] for stat in window_stats),
        "total_seconds": time.perf_counter() - started,
        "process_res": args.process_res,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
