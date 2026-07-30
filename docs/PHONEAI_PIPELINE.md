# PhoneAI offline spatial pipeline

This repository turns a Sensor Recorder Pro ARKit recording into a globally
consistent visual-inertial trajectory, a dense TSDF scene, and a pure-3D
semantic point cloud.

## Coordinate contract

- The VI-Map and all exported poses use Maplab's right-handed map frame:
  Z is up and gravity is along negative Z.
- `image_poses_tum.txt` stores `T_M_C` in metres.
- DA3 receives `T_C_M = inverse(T_M_C)` and final VI-BA intrinsics, scaled
  from the 640x480 SLAM image to the recorded video resolution.
- Mosaic3D receives the TSDF directly in `maplab_z_up`; no Y-up conversion is
  applied. Rerun declares `RIGHT_HAND_Z_UP`.

## Single-session commands

```bash
# 1. Import, learned features, local matching, initial VI-BA, loop closure,
#    PGO-gated observation fusion, final VI-BA, and dense IMU/camera poses.
bash process.sh full --data /root/data/recorder/SR_xxx --output /root/data/maplab_results/SR_xxx_full --force

# 2. Windowed DA3 and globally fused TSDF.
bash process.sh geometry --slam-output /root/data/maplab_results/SR_xxx_full --data /root/data/recorder/SR_xxx --output /root/data/maplab_results/SR_xxx_full/geometry --window-size 20 --window-overlap 4

# 3. Pure-3D open-vocabulary semantic point cloud from TSDF.
bash process.sh semantics --geometry-output /root/data/maplab_results/SR_xxx_full/geometry --output /root/data/maplab_results/SR_xxx_full/semantics_mosaic3d --profile indoor
```

`geometry` loads DA3 once and processes all windows sequentially. Depth from
every window is integrated into one TSDF; a window never defines a map boundary.

## Multi-session commands

Run `full` independently for every session, then combine their outputs with:

```bash
bash tools/sensor-recorder-multisession/run_multisession.sh \
  --session /root/data/maplab_results/SR_a_full \
  --session /root/data/maplab_results/SR_b_full \
  --session /root/data/maplab_results/SR_c_full \
  --output /root/data/maplab_results/combined
```

The multi-session stage uses SALAD retrieval, cached SuperPoint+LightGlue,
PnP verification, session-pair SE(3) RANSAC, Maplab multi-mission PGO, and a
final joint VI-BA. It exports an integrated VI-Map, per-session dense IMU and
camera TUM trajectories, and a colour-coded Rerun recording.

Then run global geometry and pure-3D semantics:

```bash
# Each session keeps its own final intrinsics and joint-map camera poses.
bash process.sh multisession-geometry \
  --joint-output /root/data/maplab_results/combined \
  --output /root/data/maplab_results/combined/25_multisession_geometry

# Mosaic3D operates once on the globally fused TSDF.
bash process.sh semantics \
  --geometry-output /root/data/maplab_results/combined/25_multisession_geometry \
  --output /root/data/maplab_results/combined/26_multisession_semantics_mosaic3d \
  --profile indoor
```

`multisession-geometry` selects and windows frames within each Mission only,
keeps the DA3 model resident across all windows, and integrates every depth
prediction into one global TSDF in the joint Maplab frame.

## External model assets

Model weights remain outside Git:

- DA3: `/root/autodl-tmp/da3/models/DA3-GIANT-1.1`
- Mosaic3D: `/root/autodl-tmp/mosaic3d/models/sc+ar+sc++.ckpt`
- Mosaic3D RECAP-CLIP config: `/root/autodl-tmp/mosaic3d/models/recap_clip`

The semantic stage requires `spconv`, `open-clip-torch`, `timm`,
`transformers`, and `jaxtyping` in the DA3 Python environment. The code and
commands are versioned here; recordings, maps, weights, RRDs, and meshes are
kept outside Git.
