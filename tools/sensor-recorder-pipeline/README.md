# Sensor Recorder pipeline

Stable command-line entry point for importing Sensor Recorder Pro sessions into
maplab. Raw recordings are treated as immutable inputs.

```bash
bash process.sh single \
  --data ~/data/recorder/SR_2026-07-21_12-41-05 \
  --output ~/data/maplab_results/SR_2026-07-21_12-41-05
```

The pipeline provides four stages:

1. `validate`: validates the recording contract and writes a report.
2. `normalize`: converts timestamps, poses and raw IMU into a versioned,
   maplab-oriented interchange format.
3. `create_vimap`: invokes the native `sensor_recorder_to_vimap` executable.
4. `export_rerun`: writes and verifies a Rerun recording containing the
   trajectory, camera model, velocity, VIWLS graph and keyframe images.

`create_vimap` requires `ffmpeg`. It decodes the video frames selected by
`frame_index` and stores one grayscale `kRawImage` resource on every VI-Map
vertex. Each VIWLS edge contains the original IMU samples inside its time
interval and linearly interpolated measurements at both vertex timestamps.
The iPhone accelerometer samples are negated during normalization because
maplab expects specific force, while Sensor Recorder reports the gravity
direction at rest.

The initial vertex velocity is derived from the ARKit position trajectory.
Gyroscope and accelerometer biases are initialized to zero and are intended to
be estimated during visual-inertial optimization.

The default command produces both `maps/00_imported/vi_map` and
`rerun/<recording-name>_vimap.rrd`. Install `rerun-sdk==0.33.0` in the Python
environment used by `process.sh`, or select another interpreter with
`RERUN_PYTHON`.

Use `--to normalize` when the native maplab executable has not been built yet.
Each completed stage has a fingerprint and is skipped on an unchanged rerun.

Build the native target in a configured maplab catkin workspace with:

```bash
catkin build sensor_recorder_importer
```

Run the tunable ORB/BRISK frontend and triangulation without BA with:

```bash
sensor_recorder_brisk_ba \
  --map=/path/to/vi_map \
  --report=/path/to/report.json \
  --run_frontend=true \
  --run_ba=false \
  --frontend_fast_threshold=5 \
  --frontend_pyramid_levels=4 \
  --frontend_nms_radius=4 \
  --frontend_max_features=1000
```

For 1920x1440 iPhone frames, the validated 10 Hz frontend profile is:

```bash
sensor_recorder_brisk_ba \
  --map=/path/to/vi_map \
  --report=/path/to/report.json \
  --run_frontend=true \
  --run_ba=false \
  --frontend_fast_threshold=5 \
  --frontend_pyramid_levels=4 \
  --frontend_nms_radius=2 \
  --frontend_max_features=2000 \
  --feature_tracking_detector_orb_num_features=4000 \
  --gyro_matcher_small_search_distance_px=30 \
  --gyro_matcher_large_search_distance_px=60 \
  --gyro_lk_candidate_ratio=0.8 \
  --gyro_lk_max_pyramid_levels=3 \
  --gyro_lk_window_size=31
```

The BRISK matching-bit thresholds and Lowe ratio are also runtime flags:
`--gyro_matcher_matching_bits_ratio_relaxed`,
`--gyro_matcher_matching_bits_ratio_strict`, and
`--gyro_matcher_lowe_ratio`.

The frontend report contains every frame's detected keypoint count and every
adjacent frame pair's RANSAC inlier/outlier counts. When this report is passed
to `export_vimap_rerun.py`, the same values are available as Rerun scalar
timelines and all detected keypoints are overlaid on the keyframe images.
Rejected landmarks are omitted from the default 3D view; pass
`--include-bad-landmarks` for a dedicated debugging export.

`configs/iphone_arkit_dense.json` imports every IMU-covered camera frame as a
VI-Map vertex. Use it when validating the built-in adjacent-frame tracker; the
default configuration keeps every third frame for the normal keyframe map.

`sensor_recorder_vimap_export` writes optimized poses, landmarks and 2D
keypoints for `export_vimap_rerun.py`. The Rerun recording then contains both
the initial trajectory and the optimized VI-Map trajectory. When calibration
is enabled, the report and Rerun export use the optimized intrinsics and
camera-IMU transform rather than the initial configuration values.

The default configuration is `configs/iphone_arkit.json`. It records all frame
conventions explicitly. Initial camera-to-IMU translation is zero; the initial
rotation matches Landscape Right capture and is intended to be refined later.

## SuperPoint + LightGlue ONNX frontend

Use `configs/iphone_arkit_640.json` to resize the keyframe images to 640x480.
The importer scales `fx`, `fy`, `cx`, and `cy` by the same factors and keeps the
raw recording unchanged.

The validated frontend uses the fixed-size
`superpoint_2048_lightglue_end2end.onnx` model from LightGlue-ONNX. The model is
a runtime dependency and is not committed to this repository. Install
`onnxruntime-gpu[cuda,cudnn]==1.23.2`, `numpy`, and
`opencv-python-headless`, then set `LIGHTGLUE_ONNX_MODEL` if the model is stored
outside the default AutoDL path.

```bash
bash process.sh single \
  --data /root/data/recorder/SR_2026-07-22_22-24-31 \
  --output /root/data/maplab_results/SR_2026-07-22_22-24-31 \
  --config configs/iphone_arkit_640.json \
  --to create_vimap

bash process.sh superpoint-lightglue \
  --output /root/data/maplab_results/SR_2026-07-22_22-24-31

# Optional Visual-only BA with 3 px iterative outlier rejection.
bash process.sh superpoint-lightglue \
  --output /root/data/maplab_results/SR_2026-07-22_22-24-31 \
  --visual-ba
```

The ONNX frontend matches keyframes at offsets 1, 2, and 3, filters matches
with Essential Matrix RANSAC, creates conflict-free multi-frame tracks, imports
them as `kSuperPoint` observations, triangulates without BA, and automatically
exports a verified Rerun recording. The current VI-Map stores a one-byte
placeholder descriptor per keypoint because subsequent matching remains in the
external ONNX frontend; keypoints, scores, and track IDs are stored normally.
The default match-score threshold is 0.15 and the Essential Matrix RANSAC
threshold is 1.0 pixel at 640x480.

On AutoDL, the default wrapper runs the native importer inside the isolated
Ubuntu 20.04/ROS Noetic runtime. Its paths can be overridden with
`MAPLAB_RUNTIME_ROOT`, `MAPLAB_RUNTIME_WORKSPACE`, and
`MAPLAB_IMPORTER_BINARY`. The user-facing process does not start ROS.

## Learning-based loop closure

Run it after `04_visual_ba_intrinsics` exists:

```bash
bash process.sh loop-closure --output /root/data/maplab_results/SR_2026-07-22_22-24-31_sp_lg
```

DINOv2-SALAD retrieves the top 20 candidates. SuperPoint-LightGlue ONNX and
2D-3D PnP verify them. Temporal neighbours and pairs separated by less than 0.5 m
along the current visual-odometry trajectory are rejected before local matching.
A loop needs at least 20 PnP inliers,
3 px RANSAC, grid coverage, and two consistent pairs in a temporal cluster.

The frontend writes `loops/salad_lightglue_pnp/verified_loops.yaml` and full
reject diagnostics without modifying stages 00--05. Verified loops produce
`06_posegraph_salad_lightglue`, `07_global_visual_ba_loops`, and
`08_visual_inertial_ba_loops_preview`. The last result remains a preview while
the accelerometer sign is unresolved.

SALAD uses local source and two external weight files so AutoDL needs no GitHub
access at runtime. Defaults are `/root/autodl-tmp/third_party/{salad,dinov2}`
with `dino_salad.ckpt` and `dinov2_vitb14_pretrain.pth`; weights stay out of Git.
