# Sensor Recorder Pro → Maplab 离线后处理

该流水线把 Sensor Recorder Pro 的 ARKit 图像、Pose 与原始 IMU 转为
VI-Map，并完成学习型局部特征、单轨迹 VI-BA、安全回环、最终 VI-BA、
TUM Pose 与 Rerun 可视化。原始录制目录保持只读。

## 一键处理

```bash
bash process.sh full \
  --data /root/data/recorder/SR_2026-07-29_00-10-03 \
  --output /root/data/maplab_results/SR_2026-07-29_00-10-03_full \
  --force
```

默认配置为 `configs/iphone_arkit_640.json`。图像缩放到 640×480，内参按
相同比例缩放；关键帧按 0.25 m、10° 或最长 1 s 自适应选择。因此任意相邻
关键帧间都可通过 IMU 积分恢复密集位姿。

`full` 结束后写入 `pipeline_timing.tsv`，包含导入、前端与初始 VI-BA、
回环/PGO/最终 VI-BA、Pose/Rerun 导出的壁钟时间。

## 流程与产物

| 阶段 | 工作 | 主要产物 |
|---|---|---|
| `00_imported` | 归一化 ARKit Pose、原始 ACC/Gyro，创建 VI-Map 与 VIWLS 边 | `normalized/`、`maps/00_imported/vi_map` |
| 前端 | SuperPoint 提点、LightGlue 匹配、Essential Matrix RANSAC、多帧 track | `features/superpoint_lightglue/` |
| `04_initial_vi_ba_intrinsics` | 初始 VI-BA，优化位姿、速度、bias、内参与外参 | 初始优化 VI-Map、Rerun |
| Loop/PGO | SALAD 检索、LightGlue + PnP、switchable PoseGraph | `loops/salad_lightglue_pnp/`、`06_posegraph...` |
| `08_visual_inertial_ba_loops_preview` | 仅融合 PGO 接受回环的 2D–3D 观测后进行最终 VI-BA | 最终优化 VI-Map、Rerun |
| Pose export | Maplab RK4 IMU 积分与相机外参变换 | `poses/`、密集轨迹 Rerun |

## 坐标、时间与 IMU

- 优化状态使用 `T_M_I`：IMU 在 Map 坐标系中的位姿。
- 相机轨迹使用 `T_M_C = T_M_I · inverse(T_C_I)`。
- TUM 四元数顺序为 `qx qy qz qw`；时间是录制起点相对秒。
- VIWLS 保存每个关键帧区间内的原始 IMU 样本，并在关键帧边界插入线性插值
  的 IMU 测量。
- 归一化阶段将 iPhone ACC 转为 Maplab 需要的 specific force；Gyro 保持
  角速度单位 `rad/s`。加速度符号问题的实验应通过阶段报告与 Rerun 独立比较。

## 学习型视觉前端

使用分离 ONNX 模型：`superpoint_2048.onnx` 与 `superpoint_lightglue.onnx`。

- 每帧最多 2,048 个 SuperPoint 特征，并持久化到 `feature_cache/`。
- 匹配集合包括滑动窗口（间隔 1–3）与 ARKit Pose 门控的非相邻帧对；配对去重。
- 默认 4 个独立 LightGlue 会话并行执行，随后使用 OpenCV
  `findEssentialMat(..., RANSAC)` 做 2D–2D 几何验证。
- Essential RANSAC：1 px、置信度 0.995、最多 500 次迭代；至少 15 个内点
  才写入 track。
- `pairs.csv` 记录每个帧对的 LightGlue、Essential RANSAC 与总耗时；
  `timing_per_frame.csv` 汇总单帧耗时。

## 安全回环与观测融合

回环前端独立于 Maplab 的二进制词袋：

1. 在 `04_initial_vi_ba_intrinsics` 构建共视图。共享至少 20 个有效 Landmark
   的候选直接跳过，无需 LightGlue/PnP。
2. SALAD 对其余帧检索 Top-10。候选必须超过按共视邻居相似度计算的自适应阈值。
3. 排除时间邻居、累计轨迹距离小于 0.5 m、当前空间距离超过 10 m 的候选。
4. LightGlue 匹配后使用 OpenCV `solvePnPRansac`（3 px、2,000 次、0.999，至少
   20 内点），并要求网格覆盖与多帧支持。
5. PGO 中每条边使用 switchable SE(3) 约束；仅 `switch ≥ 0.8` 且
   `Mahalanobis² ≤ 12.59` 的边进入观测融合。
6. 最终 BA 不保留 Loop 的 SE(3) 边。它只消费 PGO 接受回环中的 PnP 2D–3D
   内点：合并对应 Landmark 或增加观测，重新三角化受影响 track 后进行 VI-BA。

关键诊断文件：

- `verified_loops.yaml`：PnP 通过的候选。
- `accepted_loops_after_pgo.yaml`：唯一允许进入 Landmark 融合的回环。
- `rejected_candidates.csv`：共视、轨迹、检索、PnP、PGO 等拒绝原因。
- `loop_report.json`：检索门限、距离定义、回环耗时与计数。

## 导出 TUM Pose 与 Rerun

```bash
bash process.sh export-poses \
  --output /root/data/maplab_results/SR_2026-07-29_00-10-03_full
```

`--stage auto` 默认使用最终 `08` VI-BA；无有效回环时回退到 `04`。也可显式选择
`--stage initial` 或 `--stage final`。

输出：

- `poses/imu_poses_tum.txt`：每个 VIWLS IMU 时间戳的 `T_M_I`。
- `poses/image_poses_tum.txt`：每个处于 IMU 覆盖范围的图像时间戳的 `T_M_C`。
- `poses/pose_export_report.txt`：数量、覆盖范围与越界图像帧计数。
- `rerun/<recording>_dense_poses.rrd`：橙色高频 IMU 轨迹与绿色图像相机轨迹。

两份 TUM 文件都是：

```text
timestamp tx ty tz qx qy qz qw
```

## 运行环境

用户接口不需要启动 ROS。包装脚本在 AutoDL 的 Ubuntu 20.04 / ROS Noetic
proot 中调用原生 Maplab 工具。可通过以下环境变量覆盖路径：

```text
MAPLAB_RUNTIME_ROOT
MAPLAB_RUNTIME_WORKSPACE
LIGHTGLUE_PYTHON
SALAD_PYTHON
RERUN_PYTHON
```

模型和权重不提交进 Git。默认路径位于
`/root/autodl-tmp/third_party/LightGlue-ONNX-v1/weights/` 与
`/root/autodl-tmp/third_party/{salad,dinov2}`。

在 Mac 上可用 evo 快速检查 TUM 轨迹：

```bash
evo_traj tum poses/image_poses_tum.txt -p
evo_traj tum poses/imu_poses_tum.txt -p
```
