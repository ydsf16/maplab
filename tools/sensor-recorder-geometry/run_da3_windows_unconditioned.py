#!/usr/bin/env python3
"""Run DA3 windows without supplying external intrinsics or poses."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--repo", type=Path, default=Path(os.environ["PHONE_AI_DA3_REPO"]))
    parser.add_argument("--model", type=Path, default=Path(os.environ["PHONE_AI_DA3_MODEL"]))
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()
    if len(args.input) != len(args.output):
        raise ValueError("input/output count mismatch")
    sys.path.insert(0, str(args.repo / "src"))
    from depth_anything_3.api import DepthAnything3
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(str(args.model)).to("cuda")
    load_seconds = time.perf_counter() - started
    stats = []
    for index, (input_dir, output_dir) in enumerate(zip(args.input, args.output)):
        images = [str(path) for path in sorted((input_dir / "images").glob("*.jpg"))]
        output_dir.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        model.inference(
            images, extrinsics=None, intrinsics=None, align_to_input_ext_scale=False,
            process_res=args.process_res, process_res_method="upper_bound_resize",
            export_dir=str(output_dir), export_format="mini_npz",
        )
        torch.cuda.synchronize()
        result = output_dir / "exports" / "mini_npz" / "results.npz"
        for _ in range(40):
            if result.is_file():
                break
            time.sleep(0.25)
        if not result.is_file():
            raise RuntimeError(f"DA3 export did not appear: {result}")
        item = {"window": index, "frames": len(images), "inference_seconds": time.perf_counter() - began}
        stats.append(item)
    print(json.dumps({"windows": stats, "model_load_seconds": load_seconds}, indent=2))


if __name__ == "__main__":
    main()
