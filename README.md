# PhoneAI: Sensor Recorder Pro offline spatial pipeline

This fork turns an iPhone Sensor Recorder Pro ARKit recording into three
incremental spatial representations:

```text
RGB + ARKit pose + IMU
        -> globally consistent visual-inertial trajectories
        -> DA3 depth + global TSDF geometry
        -> Mosaic3D pure-3D semantic point cloud
```

It retains Maplab as the visual-inertial and multi-Mission backend, and adds
SuperPoint + LightGlue matching, SALAD loop retrieval, DA3 geometry, Mosaic3D
semantic inference, TUM exports, and Rerun visualization.

## Quick start

`process.sh` is the public processing entry point. Build the Maplab runtime and
install the external model assets before running it. Raw recordings and results
are deliberately outside Git.

### Single trajectory

```bash
# Trajectory: import -> learned matching -> loop closure -> PGO -> final VI-BA.
bash process.sh full \
  --data /root/data/recorder/SR_xxx \
  --output /root/data/maplab_results/SR_xxx_full \
  --force

# Dense geometry: final VI-BA pose -> DA3 -> one global TSDF.
bash process.sh geometry \
  --slam-output /root/data/maplab_results/SR_xxx_full \
  --data /root/data/recorder/SR_xxx \
  --output /root/data/maplab_results/SR_xxx_full/geometry

# Pure 3D semantics: TSDF point cloud -> Mosaic3D.
bash process.sh semantics \
  --geometry-output /root/data/maplab_results/SR_xxx_full/geometry \
  --output /root/data/maplab_results/SR_xxx_full/semantics_mosaic3d \
  --profile indoor
```

DA3 uses 20-frame windows with 4-frame overlap by default. The model remains
resident on the GPU, and all window depths fuse into one TSDF.

### Multiple trajectories

First run `full` once for every recording. Then register the sessions and
perform joint visual-inertial optimization:

```bash
bash tools/sensor-recorder-multisession/run_multisession.sh \
  --session /root/data/maplab_results/SR_a_full \
  --session /root/data/maplab_results/SR_b_full \
  --session /root/data/maplab_results/SR_c_full \
  --output /root/data/maplab_results/combined

# Per-Mission DA3 windows, one joint TSDF, then one global 3D semantic map.
bash process.sh multisession-geometry \
  --joint-output /root/data/maplab_results/combined \
  --output /root/data/maplab_results/combined/25_multisession_geometry

bash process.sh semantics \
  --geometry-output /root/data/maplab_results/combined/25_multisession_geometry \
  --output /root/data/maplab_results/combined/26_multisession_semantics_mosaic3d \
  --profile indoor
```

Cross-session registration uses SALAD retrieval, SuperPoint + LightGlue,
2D-3D PnP verification, session-pair SE(3) RANSAC, Maplab PGO, PGO-gated
observation fusion, and joint VI-BA. DA3 windows never cross Mission
boundaries; every Mission uses its own final intrinsics and global camera pose.

## Outputs

| Stage | Main output | Meaning |
| --- | --- | --- |
| `full` | `maps/08_visual_inertial_ba_loops_preview/vi_map` | Final single-trajectory VI-Map |
| `full` | `poses/imu_poses_tum.txt` | Dense IMU pose, `T_M_I`, TUM format |
| `full` | `poses/image_poses_tum.txt` | Dense camera pose, `T_M_C`, TUM format |
| `full` | `rerun_*.rrd` | SLAM diagnostics, loops, tracks and trajectories |
| `geometry` | `tsdf/tsdf_mesh_clean.glb` | Colored global TSDF mesh |
| `geometry` | `rerun_geometry.rrd` | TSDF and final VI-BA trajectory |
| `semantics` | `semantic_colored.ply` | RGB, semantic ID, confidence and class scores per point |
| `semantics` | `rerun_semantic.rrd` | Global semantic point cloud |
| multi-session | `joint_vimap/` and `poses/<session>/` | Joint map and per-session TUM trajectories in one map frame |

All poses use Maplab's right-handed Z-up map frame, in metres. Gravity is
negative Z. Camera pose is `T_M_C`; DA3 receives its inverse, `T_C_M`.

## External assets

The repository contains the pipeline code, not model weights or recordings.
Expected runtime assets are:

- DA3 GIANT: `/root/autodl-tmp/da3/models/DA3-GIANT-1.1`
- Mosaic3D checkpoint: `/root/autodl-tmp/mosaic3d/models/sc+ar+sc++.ckpt`
- Mosaic3D RECAP-CLIP configuration: `/root/autodl-tmp/mosaic3d/models/recap_clip`

Mosaic3D requires `spconv`, `open-clip-torch`, `timm`, `transformers`, and
`jaxtyping` in the DA3 Python environment. For the full stage-by-stage contract,
see [docs/PHONEAI_PIPELINE.md](docs/PHONEAI_PIPELINE.md).

---

## Upstream Maplab

<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/logos/maplab_new.png" width="500">


*Ubuntu 18.04+ROS melodic*: [![Build Status](https://jenkins.asl.ethz.ch/buildStatus/icon?job=maplab_nightly)](https://jenkins.asl.ethz.ch/job/maplab_nightly)
[![Documentation Status](https://jenkins.asl.ethz.ch/buildStatus/icon?job=maplab_docs&subject=docs)](https://jenkins.asl.ethz.ch/job/maplab_docs)

## News
 * **November 2022:** maplab 2.0 initial release with new features and sensors. [Paper](https://arxiv.org/pdf/2212.00654.pdf).
 * **July 2018:** Check out our release candidate with improved localization and lots of new features! [Release 1.3](https://github.com/ethz-asl/maplab/releases/tag/1.3).
 * **May 2018:** maplab was presented at [ICRA](https://icra2018.org/) in Brisbane. [Paper](https://arxiv.org/abs/1711.10250) / [Initial Release](https://github.com/ethz-asl/maplab/releases/tag/initial_release).



## Description

This repository contains **maplab 2.0**, an open research-oriented mapping framework, written in C++,  for multi-session and multi-robot mapping. For the original maplab release from 2018 the source code and documentation is available [here](https://github.com/ethz-asl/maplab/releases/tag/1.3).

**For documentation, tutorials and datasets, please visit the [wiki](https://maplab.asl.ethz.ch/index.html).**

## Features

### Robust visual-inertial odometry with localization
<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/rovio_stairs.gif" width="400"> <img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/rviz_cla_vs.gif" width="400">

### Large-scale multisession mapping and optimization
<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/largescale.gif" width="400"> <img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/cla.png" width="400">

### Multi-robot mapping and online operation
<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/arche.jpg" width="400"> <img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/multirobot.gif" width="400">

### Dense reconstruction
<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/stereo.png" width="400"> <img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/pmvs.png" width="400">

### A research platform extensively tested on real robots
<img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/topomap.png" width="400"> <img src="https://raw.githubusercontent.com/ethz-asl/maplab/master/docs/pages/readme_images/robots.jpg" width="400">

## Installation and getting started

The following articles help you with getting started with maplab and ROVIOLI:

- [Installation on Ubuntu 18.04 or 20.04](https://maplab.asl.ethz.ch/docs/master/pages/installation/A_Installation-Ubuntu.html)
- [Introduction to the maplab framework](https://maplab.asl.ethz.ch/docs/master/pages/overview_and_introduction/A_The-Maplab-Framework.html)
- [Running ROVIOLI in VIO mode](https://maplab.asl.ethz.ch/docs/master/pages/tutorials-rovioli/B_Running-ROVIOLI-in-VIO-mode.html)
- [Basic console usage](https://maplab.asl.ethz.ch/docs/master/pages/tutorials-maplab/basics/A_Basic-Console-Usage.html)
- [Console map management](https://maplab.asl.ethz.ch/docs/master/pages/tutorials-maplab/basics/C_Console-map-management.html)

**More detailed information can be found in the [wiki pages](https://maplab.asl.ethz.ch/index.html).**

## Research Results

The maplab framework has been used as an experimental platform for numerous scientific publications. For a complete list of publications please refer to [Research based on maplab](https://maplab.asl.ethz.ch/docs/master/pages/overview_and_introduction/C_Related-Research.html#additional-citations).

## Citing

Please cite the following papers [maplab](https://arxiv.org/abs/1711.10250) and [maplab 2.0](https://arxiv.org/abs/2212.00654) when using our framework for your research:

```bibtex
@article{schneider2018maplab,
  title={{maplab: An Open Framework for Research in Visual-inertial Mapping and Localization}},
  author={T. Schneider and M. T. Dymczyk and M. Fehr and K. Egger and S. Lynen and I. Gilitschenski and R. Siegwart},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={3},
  pages={1418--1425},
  year={2018},
  doi={10.1109/LRA.2018.2800113}
}
```

```bibtex
@article{cramariuc2022maplab,
  title={{maplab 2.0 – A Modular and Multi-Modal Mapping Framework}},
  author={A. Cramariuc and L. Bernreiter and F. Tschopp and M. Fehr and V. Reijgwart and J. Nieto and R. Siegwart and C. Cadena},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={2},
  pages={520-527},
  year={2023},
  doi={10.1109/LRA.2022.3227865}
}
```

### Additional Citations

Certain components of maplab are directly based on [other publications](https://maplab.asl.ethz.ch/docs/master/pages/overview_and_introduction/B_Citing-Maplab).


## Credits

 * Thomas Schneider
 * Marcin Dymczyk
 * Marius Fehr
 * Kevin Egger
 * Simon Lynen
 * Mathias Bürki
 * Titus Cieslewski
 * Timo Hinzmann
 * Mathias Gehrig
 * Florian Tschopp
 * Andrei Cramariuc
 * Lukas Bernreiter

For a complete list of contributors, have a look at [CONTRIBUTORS.md](https://github.com/ethz-asl/maplab/blob/master/CONTRIBUTORS.md)
