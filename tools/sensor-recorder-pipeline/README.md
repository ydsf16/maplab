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

The default configuration is `configs/iphone_arkit.json`. It records all frame
conventions explicitly. Initial camera-to-IMU translation is zero; the initial
rotation matches Landscape Right capture and is intended to be refined later.

On AutoDL, the default wrapper runs the native importer inside the isolated
Ubuntu 20.04/ROS Noetic runtime. Its paths can be overridden with
`MAPLAB_RUNTIME_ROOT`, `MAPLAB_RUNTIME_WORKSPACE`, and
`MAPLAB_IMPORTER_BINARY`. The user-facing process does not start ROS.
