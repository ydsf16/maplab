#!/usr/bin/env python3
"""Run Mosaic3D semantic inference on an arbitrary colored PLY point cloud.

This adapter intentionally bypasses the benchmark datamodule.  It reproduces
the official validation preprocessing and loads the official Mosaic3D encoder
and RECAP-CLIP text tower directly from the released Lightning checkpoint.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import types
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch


DEFAULT_CLASSES = [
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "chair",
    "desk",
    "table",
    "cabinet",
    "shelf",
    "sofa",
    "bed",
    "monitor",
    "television",
    "keyboard",
    "lamp",
    "plant",
    "curtain",
    "picture",
    "box",
    "other",
]


PALETTE = np.asarray(
    [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [255, 127, 14],
        [214, 39, 40],
        [140, 86, 75],
        [148, 103, 189],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [23, 190, 207],
        [44, 160, 44],
        [255, 215, 0],
        [127, 127, 127],
        [50, 205, 50],
        [206, 109, 189],
        [82, 84, 163],
        [158, 218, 229],
        [96, 96, 96],
    ],
    dtype=np.uint8,
)


def semantic_color(class_id: int) -> np.ndarray:
    """Return a stable, visually distinct RGB color for a semantic ID."""

    if class_id < len(PALETTE):
        return PALETTE[class_id]
    hue = (class_id * 0.618033988749895) % 1.0
    saturation = 0.58 + 0.18 * ((class_id % 3) / 2)
    value = 0.82 + 0.14 * ((class_id % 2))
    return np.asarray(colorsys.hsv_to_rgb(hue, saturation, value)) * 255


def load_class_catalog(path: Path | None, profile: str) -> list[dict]:
    if path is None:
        return [{"id": index, "name": name, "group": "legacy"} for index, name in enumerate(DEFAULT_CLASSES)]
    catalog = json.loads(path.read_text())
    records = catalog["classes"] if isinstance(catalog, dict) else catalog
    selected = [
        item
        for item in records
        if profile == "all" or profile in item.get("profiles", ["all"])
    ]
    if not selected:
        raise ValueError(f"class catalog has no entries for profile {profile!r}")
    ids = [int(item["id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic IDs must be unique")
    return selected


def install_torch_scatter_fallback() -> None:
    """Provide the small torch_scatter subset used by the Mosaic3D encoder."""

    if "torch_scatter" in sys.modules:
        return
    try:
        __import__("torch_scatter")
        return
    except ImportError:
        pass

    module = types.ModuleType("torch_scatter")

    def scatter(src, index, dim=0, out=None, dim_size=None, reduce="sum"):
        if dim != 0 or out is not None:
            raise NotImplementedError("fallback scatter supports dim=0 and out=None")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        shape = (dim_size,) + tuple(src.shape[1:])
        result = torch.zeros(shape, dtype=src.dtype, device=src.device)
        result.index_add_(0, index.long(), src)
        if reduce in ("sum", "add"):
            return result
        if reduce == "mean":
            count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            count.index_add_(0, index.long(), torch.ones_like(index, dtype=src.dtype))
            count = count.clamp_min_(1)
            return result / count.reshape((-1,) + (1,) * (src.ndim - 1))
        raise NotImplementedError(f"fallback scatter reduce={reduce!r}")

    def segment_csr(src, indptr, reduce="sum"):
        chunks = []
        for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            part = src[start:end]
            chunks.append(part.mean(0) if reduce == "mean" else part.sum(0))
        return torch.stack(chunks) if chunks else src.new_empty((0,) + src.shape[1:])

    module.scatter = scatter
    module.segment_csr = segment_csr
    sys.modules["torch_scatter"] = module


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    return points, colors


def write_semantic_ply(
    path: Path,
    points: np.ndarray,
    source_colors: np.ndarray,
    labels: np.ndarray,
    semantic_colors: np.ndarray,
    confidence: np.ndarray,
    cosine: np.ndarray,
    margin: np.ndarray,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("source_red", "u1"),
        ("source_green", "u1"),
        ("source_blue", "u1"),
        ("semantic_id", "u2"),
        ("confidence", "f4"),
        ("cosine", "f4"),
        ("margin", "f4"),
    ]
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points.T
    data["red"], data["green"], data["blue"] = semantic_colors.T
    data["source_red"], data["source_green"], data["source_blue"] = source_colors.T
    data["semantic_id"] = labels
    data["confidence"] = confidence
    data["cosine"] = cosine
    data["margin"] = margin
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def build_network(
    repo: Path,
    checkpoint_state: dict[str, torch.Tensor],
    backbone_name: str,
) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    install_torch_scatter_fallback()

    from src.models.networks.ppt.model import PPT
    if backbone_name == "spunet34c":
        from src.models.networks.spunet.spconv_unet_v1m3_pdnorm import SpUNetBase

        backbone_type = SpUNetBase
        channels = [32, 64, 128, 256, 256, 128, 96, 96]
        layers = [2, 3, 4, 6, 2, 2, 2, 2]
    elif backbone_name == "spunet101":
        from src.models.networks.spunet.spconv_unet_v1m1_base import SpUNetBottleneck

        backbone_type = SpUNetBottleneck
        channels = [32, 64, 128, 256, 256, 128, 96, 96]
        layers = [2, 3, 4, 23, 2, 2, 2, 2]
    else:
        raise ValueError(f"unsupported backbone: {backbone_name}")

    conditions = ["ScanNet", "ARKitScenes", "ScanNetPP"]
    backbone = partial(
        backbone_type,
        in_channels=3,
        out_channels=768,
        base_channels=32,
        channels=channels,
        layers=layers,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=True,
        conditions=conditions,
        zero_init=True,
    )
    network = PPT(backbone=backbone, conditions=conditions, context_channels=256)
    state = {key.removeprefix("net."): value for key, value in checkpoint_state.items() if key.startswith("net.")}
    incompatible = network.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"network state mismatch: {incompatible}")
    return network


def encode_classes(
    clip_config_dir: Path,
    checkpoint_state: dict[str, torch.Tensor],
    class_names: list[str],
) -> torch.Tensor:
    from open_clip.model import CLIP
    from open_clip.tokenizer import HFTokenizer

    config = json.loads((clip_config_dir / "open_clip_config.json").read_text())
    model = CLIP(**config["model_cfg"])
    state = {
        key.removeprefix("clip_encoder."): value
        for key, value in checkpoint_state.items()
        if key.startswith("clip_encoder.")
    }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"CLIP state mismatch: {incompatible}")
    tokenizer = HFTokenizer(
        str(clip_config_dir),
        context_length=config["model_cfg"]["text_cfg"]["context_length"],
        strip_sep_token=True,
    )
    prompts = ["other" if name == "other" else f"a {name} in a scene" for name in class_names]
    model.eval()
    with torch.inference_mode():
        tokens = tokenizer(prompts)
        embeddings = model.encode_text(tokens).float()
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    del model
    return embeddings.cpu()


def prepare_coordinates(points: np.ndarray, world_coordinate: str) -> tuple[np.ndarray, dict[str, list[float]]]:
    if world_coordinate == "arkit_y_up":
        points_z_up = np.column_stack(
            [points[:, 0], -points[:, 2], points[:, 1]]
        ).astype(np.float32)
        transform = ["x", "-z", "y"]
    elif world_coordinate == "maplab_z_up":
        points_z_up = points.astype(np.float32)
        transform = ["x", "y", "z"]
    else:
        raise ValueError(f"Unsupported world coordinate: {world_coordinate}")
    minimum = points_z_up.min(0)
    maximum = points_z_up.max(0)
    shift = np.asarray(
        [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]],
        dtype=np.float32,
    )
    return points_z_up - shift, {
        "world_to_model": transform,
        "input_world_coordinate": world_coordinate,
        "center_shift_z_up": shift.tolist(),
    }


def make_preview(path: Path, points: np.ndarray, colors: np.ndarray, seed: int = 7) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    count = min(len(points), 120_000)
    idx = rng.choice(len(points), count, replace=False)
    sample = points[idx]
    sample_colors = colors[idx].astype(np.float32) / 255.0
    fig = plt.figure(figsize=(14, 7), dpi=150)
    for panel, (elev, azim, title) in enumerate(
        [(22, -65, "Semantic perspective"), (90, -90, "Semantic top view")], start=1
    ):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.scatter(sample[:, 0], sample[:, 2], sample[:, 1], c=sample_colors, s=0.15, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y-up")
        ax.set_box_aspect(np.maximum(np.ptp(sample[:, [0, 2, 1]], axis=0), 0.1))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-config-dir", type=Path, required=True)
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--class-catalog", type=Path)
    parser.add_argument("--profile", default="all")
    parser.add_argument("--backbone", choices=["spunet34c", "spunet101"], default="spunet34c")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--world-coordinate", choices=["arkit_y_up", "maplab_z_up"], default="arkit_y_up")
    args = parser.parse_args()

    if args.classes and args.class_catalog:
        parser.error("use either --classes or --class-catalog")
    if args.classes:
        class_catalog = [
            {"id": index, "name": name, "group": "custom"}
            for index, name in enumerate(args.classes)
        ]
    else:
        class_catalog = load_class_catalog(args.class_catalog, args.profile)
    class_names = [item["name"] for item in class_catalog]
    semantic_ids = torch.tensor([int(item["id"]) for item in class_catalog], dtype=torch.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.input_ply}", flush=True)
    points, source_colors = read_ply(args.input_ply)
    model_points, transform_info = prepare_coordinates(points, args.world_coordinate)
    print(f"Loaded {len(points):,} points", flush=True)

    print("Loading official checkpoint", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    checkpoint_state = checkpoint["state_dict"]
    print("Encoding open-vocabulary class prompts", flush=True)
    text_embeddings = encode_classes(args.clip_config_dir, checkpoint_state, class_names)
    print(f"Building Mosaic3D {args.backbone}", flush=True)
    network = build_network(args.repo, checkpoint_state, args.backbone)
    del checkpoint, checkpoint_state

    device = torch.device("cuda")
    network.eval().to(device)
    coord = torch.from_numpy(model_points).to(device)
    color = torch.from_numpy(source_colors.astype(np.float32) / 127.5 - 1.0).to(device)
    batch = torch.zeros(len(points), dtype=torch.int32, device=device)
    input_dict = {
        "coord": coord,
        "feat": color,
        "batch": batch,
        "grid_size": args.grid_size,
        "condition": ["ARKitScenes"],
    }

    torch.cuda.reset_peak_memory_stats()
    print("Running Mosaic3D encoder", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output = network(input_dict)
        voxel_features = output.sparse_conv_feat.features.float()
        voxel_features = torch.nn.functional.normalize(voxel_features, dim=-1)
        logits = voxel_features @ text_embeddings.to(device).T
        top2 = logits.topk(2, dim=1)
        voxel_candidate_indices = top2.indices[:, 0]
        voxel_labels = semantic_ids.to(device)[voxel_candidate_indices]
        voxel_cosine = top2.values[:, 0]
        voxel_margin = top2.values[:, 0] - top2.values[:, 1]
        voxel_confidence = torch.softmax(logits / args.temperature, dim=1).max(dim=1).values
        point_map = output.v2p_map
        labels = voxel_labels[point_map].cpu().numpy().astype(np.uint16)
        cosine = voxel_cosine[point_map].cpu().numpy().astype(np.float32)
        margin = voxel_margin[point_map].cpu().numpy().astype(np.float32)
        confidence = voxel_confidence[point_map].cpu().numpy().astype(np.float32)

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    voxel_count = int(len(voxel_labels))
    color_lut = np.zeros((int(semantic_ids.max()) + 1, 3), dtype=np.uint8)
    for class_id in semantic_ids.tolist():
        color_lut[class_id] = semantic_color(class_id).astype(np.uint8)
    semantic_colors = color_lut[labels]
    semantic_ply = args.output_dir / "semantic_colored.ply"
    write_semantic_ply(
        semantic_ply,
        points,
        source_colors,
        labels,
        semantic_colors,
        confidence,
        cosine,
        margin,
    )
    np.save(args.output_dir / "semantic_labels.npy", labels)
    np.savez_compressed(
        args.output_dir / "semantic_scene.npz",
        points=points,
        source_colors=source_colors,
        semantic_colors=semantic_colors,
        labels=labels,
        confidence=confidence.astype(np.float16),
        cosine=cosine.astype(np.float16),
        margin=margin.astype(np.float16),
    )

    counts = Counter(labels.tolist())
    class_stats = []
    for item in class_catalog:
        class_id = int(item["id"])
        name = item["name"]
        mask = labels == class_id
        count = int(counts.get(class_id, 0))
        class_stats.append(
            {
                "id": class_id,
                "class": name,
                "group": item.get("group", "other"),
                "points": count,
                "fraction": count / len(labels),
                "mean_confidence": float(confidence[mask].mean()) if count else None,
                "mean_cosine": float(cosine[mask].mean()) if count else None,
            }
        )
    metadata = {
        "model": f"Mosaic3D {args.backbone} sc+ar+sc++",
        "checkpoint": str(args.checkpoint),
        "input_ply": str(args.input_ply),
        "point_count": len(points),
        "voxel_count": voxel_count,
        "grid_size_m": args.grid_size,
        "condition": "ARKitScenes",
        "class_profile": args.profile,
        "temperature": args.temperature,
        "peak_gpu_gib": peak_gib,
        "coordinate_transform": transform_info,
        "classes": class_stats,
    }
    (args.output_dir / "semantic_stats.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "class_catalog.json").write_text(
        json.dumps(class_catalog, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "class_names.json").write_text(
        json.dumps({str(item["id"]): item["name"] for item in class_catalog}, ensure_ascii=False, indent=2) + "\n"
    )
    make_preview(args.output_dir / "semantic_preview.png", points, semantic_colors)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote {semantic_ply}", flush=True)


if __name__ == "__main__":
    main()
