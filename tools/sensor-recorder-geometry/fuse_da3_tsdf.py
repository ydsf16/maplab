#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, action="append", required=True,
                        help="DA3 results.npz; repeat once per window")
    parser.add_argument("--images", type=Path, action="append", required=True,
                        help="Image directory matching each --npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--trunc", type=float, default=0.04)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--min-component-triangles", type=int, default=1000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if len(args.npz) != len(args.images):
        raise ValueError("--npz and --images must be provided equally often")
    windows=[]; confidences=[]
    for npz_path, image_dir in zip(args.npz,args.images):
        data=np.load(npz_path)
        depths=data["depth"].astype(np.float32); conf=data["conf"].astype(np.float32)
        intrinsics=data["intrinsics"].astype(np.float64); extrinsics=data["extrinsics"].astype(np.float64)
        image_paths=sorted(image_dir.glob("frame_*.jpg"))
        if len(image_paths) != len(depths):
            raise ValueError(f"Image/depth count mismatch in {npz_path}: {len(image_paths)} vs {len(depths)}")
        windows.append((depths,conf,intrinsics,extrinsics,image_paths)); confidences.append(conf[np.isfinite(conf)])
    threshold=float(np.percentile(np.concatenate(confidences),args.conf_percentile))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    t0 = time.perf_counter()
    valid_pixel_fractions = []
    height=width=None; total_frames=0
    for depths,conf,intrinsics,extrinsics,image_paths in windows:
        height,width=depths.shape[1:]
        for i,image_path in enumerate(image_paths):
            depth=depths[i].copy()
            valid=np.isfinite(depth) & (depth > 0) & (depth <= args.max_depth) & (conf[i] >= threshold)
            depth[~valid]=0.0; valid_pixel_fractions.append(float(valid.mean())); total_frames+=1
            rgb=np.asarray(Image.open(image_path).convert("RGB").resize((width,height),Image.Resampling.LANCZOS))
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(rgb)),o3d.geometry.Image(np.ascontiguousarray(depth)),depth_scale=1.0,depth_trunc=args.max_depth,convert_rgb_to_intensity=False)
            k=intrinsics[i]; intrinsic=o3d.camera.PinholeCameraIntrinsic(width,height,k[0,0],k[1,1],k[0,2],k[1,2]); ext=np.eye(4,dtype=np.float64); ext[:3,:4]=extrinsics[i]; volume.integrate(rgbd,intrinsic,ext)

    raw_mesh = volume.extract_triangle_mesh()
    raw_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output / "tsdf_mesh_raw.ply"), raw_mesh, write_ascii=False)

    mesh = o3d.geometry.TriangleMesh(raw_mesh)
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        remove = counts[labels] < args.min_component_triangles
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(args.output / "tsdf_mesh_clean.ply"), mesh, write_ascii=False)

    point_count = min(1_000_000, max(100_000, len(mesh.triangles)))
    pointcloud = mesh.sample_points_uniformly(number_of_points=point_count) if len(mesh.triangles) else o3d.geometry.PointCloud()
    o3d.io.write_point_cloud(str(args.output / "tsdf_pointcloud.ply"), pointcloud, write_ascii=False)

    import trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    colors_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8) if len(colors) else None
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors_u8, process=False)
    tri.export(args.output / "tsdf_mesh_clean.glb")

    bounds = mesh.get_axis_aligned_bounding_box()
    stats = {
        "frames": total_frames,
        "windows": len(windows),
        "depth_shape": [height, width],
        "voxel_length_m": args.voxel,
        "sdf_trunc_m": args.trunc,
        "max_depth_m": args.max_depth,
        "confidence_percentile": args.conf_percentile,
        "confidence_threshold": threshold,
        "valid_pixel_fraction_mean": float(np.mean(valid_pixel_fractions)),
        "raw_vertices": len(raw_mesh.vertices),
        "raw_triangles": len(raw_mesh.triangles),
        "clean_vertices": len(mesh.vertices),
        "clean_triangles": len(mesh.triangles),
        "sampled_points": len(pointcloud.points),
        "bounds_min_m": bounds.get_min_bound().tolist() if len(mesh.vertices) else [],
        "bounds_max_m": bounds.get_max_bound().tolist() if len(mesh.vertices) else [],
        "total_seconds": time.perf_counter() - t0,
    }
    (args.output / "tsdf_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
