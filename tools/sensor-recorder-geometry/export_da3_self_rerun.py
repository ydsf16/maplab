#!/usr/bin/env python3
"""Export a DA3 self-estimated TSDF result without claiming a SLAM world frame."""
import argparse
from pathlib import Path

import numpy as np
import rerun as rr


def read_points(path: Path):
    types = {"float": "<f4", "double": "<f8", "uchar": "u1"}
    props = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline().decode().strip()
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property ") and line.split()[1] != "list":
                props.append((line.split()[2], types[line.split()[1]]))
            elif line == "end_header":
                break
        array = np.fromfile(stream, dtype=np.dtype(props), count=count)
    points = np.column_stack([array["x"], array["y"], array["z"]]).astype(np.float32)
    colors = np.column_stack([array["red"], array["green"], array["blue"]]).astype(np.uint8)
    return points, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--npz", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points, colors = read_points(args.ply)
    ext = np.concatenate([np.load(path)["extrinsics"] for path in args.npz])
    homogeneous = np.tile(np.eye(4), (len(ext), 1, 1))
    homogeneous[:, :3, :4] = ext
    centers = np.linalg.inv(homogeneous)[:, :3, 3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("PhoneAI_DA3_self_estimated", spawn=False)
    rr.save(str(args.output))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/tsdf/pointcloud", rr.Points3D(points, colors=colors, radii=0.004), static=True)
    rr.log("world/da3_self_estimated_trajectory", rr.LineStrips3D([centers], colors=[[255, 180, 45]], radii=0.02), static=True)
    rr.log("world/da3_self_estimated_frames", rr.Points3D(centers, colors=[[70, 220, 120]], radii=0.04), static=True)
    rr.log("metadata", rr.TextDocument(
        f"points={len(points)}\nframes={len(centers)}\n"
        "pose=DA3 self-estimated\nintrinsics=DA3 self-estimated\n"
        "scale=arbitrary; not aligned to VI-BA / Maplab world"
    ), static=True)


if __name__ == "__main__":
    main()
