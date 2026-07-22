#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <feature-tracking/feature-tracking-types.h>
#include <feature-tracking/vo-feature-tracking-pipeline.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <landmark-triangulation/landmark-triangulation.h>
#include <map-optimization/vi-map-optimizer.h>
#include <map-optimization/vi-optimization-builder.h>
#include <map-resources/resource-common.h>
#include <sensors/imu.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map/vertex.h>
#include <vi-map-helpers/vi-map-manipulation.h>

DEFINE_string(map, "", "VI-Map folder to process in place.");
DEFINE_string(report, "", "Path to write the frontend and BA JSON report.");
DEFINE_int32(ba_iterations, 30, "Maximum number of visual BA iterations.");
DEFINE_bool(run_frontend, true, "Extract and match features before BA.");
DEFINE_bool(use_imu, false, "Add VIWLS IMU factors to the BA problem.");
DEFINE_bool(
    optimize_extrinsics, false,
    "Optimize camera-IMU rotation and translation during BA.");
DEFINE_bool(optimize_biases, true, "Optimize accelerometer and gyro biases.");
DEFINE_bool(optimize_velocity, true, "Optimize keyframe velocities.");

namespace {

void loadRawImage(vi_map::VIMap* map, vi_map::Vertex* vertex) {
  CHECK_NOTNULL(map);
  CHECK_NOTNULL(vertex);
  cv::Mat image;
  CHECK(map->getFrameResource(
      *vertex, 0u, backend::ResourceType::kRawImage, &image));
  CHECK(!image.empty());
  vertex->getVisualFrameShared(0u)->setRawImage(image);
}

size_t countKeypoints(const pose_graph::VertexIdList& vertex_ids,
                      const vi_map::VIMap& map) {
  size_t count = 0u;
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    count += map.getVertex(vertex_id)
                 .getVisualFrame(0u)
                 .getNumKeypointMeasurements();
  }
  return count;
}

void writeReport(
    const std::string& path, const size_t vertices, const size_t keypoints,
    const size_t inlier_matches, const size_t outlier_matches,
    const size_t landmarks, const double initial_cost, const double final_cost,
    const size_t iterations, const double pose_rms_delta_m,
    const double pose_max_delta_m, const double velocity_rms_delta_m_s,
    const double accel_bias_rms_delta, const double gyro_bias_rms_delta) {
  std::ofstream stream(path);
  CHECK(stream.good()) << "Unable to write report: " << path;
  stream << "{\n"
         << "  \"status\": \"created_and_verified\",\n"
         << "  \"frontend\": \""
         << (FLAGS_run_frontend ? "maplab_orb_brisk_gyro_tracker"
                                : "existing_vimap_tracks")
         << "\",\n"
         << "  \"vertices\": " << vertices << ",\n"
         << "  \"keypoints\": " << keypoints << ",\n"
         << "  \"inlier_matches\": " << inlier_matches << ",\n"
         << "  \"outlier_matches\": " << outlier_matches << ",\n"
         << "  \"landmarks\": " << landmarks << ",\n"
         << "  \"ba\": {\n"
         << "    \"type\": \""
         << (FLAGS_use_imu ? "visual_inertial" : "visual") << "\",\n"
         << "    \"imu_factors\": " << (FLAGS_use_imu ? "true" : "false")
         << ",\n"
         << "    \"optimize_extrinsics\": "
         << (FLAGS_optimize_extrinsics ? "true" : "false") << ",\n"
         << "    \"iterations\": " << iterations << ",\n"
         << "    \"initial_cost\": " << initial_cost << ",\n"
         << "    \"final_cost\": " << final_cost << ",\n"
         << "    \"pose_rms_delta_m\": " << pose_rms_delta_m << ",\n"
         << "    \"pose_max_delta_m\": " << pose_max_delta_m << ",\n"
         << "    \"velocity_rms_delta_m_s\": " << velocity_rms_delta_m_s
         << ",\n"
         << "    \"accel_bias_rms_delta\": " << accel_bias_rms_delta
         << ",\n"
         << "    \"gyro_bias_rms_delta\": " << gyro_bias_rms_delta << "\n"
         << "  }\n"
         << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_map.empty()) << "--map is required";
  CHECK(!FLAGS_report.empty()) << "--report is required";
  CHECK_GT(FLAGS_ba_iterations, 0);

  vi_map::VIMap map;
  CHECK(vi_map::serialization::loadMapFromFolder(FLAGS_map, &map));
  CHECK(vi_map::checkMapConsistency(map));

  vi_map::MissionIdList mission_ids;
  map.getAllMissionIds(&mission_ids);
  CHECK_EQ(mission_ids.size(), 1u);
  const vi_map::MissionId mission_id = mission_ids.front();
  pose_graph::VertexIdList vertex_ids;
  map.getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);
  CHECK_GE(vertex_ids.size(), 2u);

  std::vector<Eigen::Vector3d> positions_before;
  std::vector<Eigen::Vector3d> velocities_before;
  std::vector<Eigen::Vector3d> accel_biases_before;
  std::vector<Eigen::Vector3d> gyro_biases_before;
  positions_before.reserve(vertex_ids.size());
  velocities_before.reserve(vertex_ids.size());
  accel_biases_before.reserve(vertex_ids.size());
  gyro_biases_before.reserve(vertex_ids.size());
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    const vi_map::Vertex& vertex = map.getVertex(vertex_id);
    positions_before.emplace_back(vertex.get_T_M_I().getPosition());
    velocities_before.emplace_back(vertex.get_v_M());
    accel_biases_before.emplace_back(vertex.getAccelBias());
    gyro_biases_before.emplace_back(vertex.getGyroBias());
  }

  size_t inlier_match_count = 0u;
  size_t outlier_match_count = 0u;
  if (FLAGS_run_frontend) {
    aslam::NCamera::ConstPtr ncamera =
        map.getVertex(vertex_ids.front()).getNCameras();
    feature_tracking::FeatureTrackingExtractorSettings extractor_settings;
    feature_tracking::FeatureTrackingDetectorSettings detector_settings;
    feature_tracking::FeatureTrackingOutlierSettings outlier_settings;
    feature_tracking::VOFeatureTrackingPipeline tracker(
        ncamera, extractor_settings, detector_settings, outlier_settings);

    vi_map::Vertex* previous_vertex = &map.getVertex(vertex_ids.front());
    loadRawImage(&map, previous_vertex);
    tracker.initializeFirstNFrame(
        previous_vertex->getVisualNFrameShared().get());
    previous_vertex->expandVisualObservationContainersIfNecessary();
    for (size_t index = 1u; index < vertex_ids.size(); ++index) {
      vi_map::Vertex* current_vertex = &map.getVertex(vertex_ids[index]);
      loadRawImage(&map, current_vertex);
      const aslam::Quaternion q_Bkp1_Bk =
          current_vertex->get_T_M_I().getRotation().inverse() *
          previous_vertex->get_T_M_I().getRotation();
      aslam::FrameToFrameMatchesList inlier_matches;
      aslam::FrameToFrameMatchesList outlier_matches;
      tracker.trackFeaturesNFrame(
          q_Bkp1_Bk, current_vertex->getVisualNFrameShared().get(),
          previous_vertex->getVisualNFrameShared().get(), &inlier_matches,
          &outlier_matches);
      CHECK_EQ(inlier_matches.size(), 1u);
      CHECK_EQ(outlier_matches.size(), 1u);
      inlier_match_count += inlier_matches.front().size();
      outlier_match_count += outlier_matches.front().size();
      current_vertex->expandVisualObservationContainersIfNecessary();
      previous_vertex->getVisualNFrameShared()->releaseRawImagesOfAllFrames();
      previous_vertex = current_vertex;
    }
    previous_vertex->getVisualNFrameShared()->releaseRawImagesOfAllFrames();
  }

  const size_t keypoint_count = countKeypoints(vertex_ids, map);
  if (FLAGS_run_frontend) {
    vi_map_helpers::VIMapManipulation manipulation(&map);
    const size_t landmark_count =
        manipulation.initializeLandmarksFromUnusedFeatureTracksOfMission(
            mission_id);
    CHECK_GT(landmark_count, 0u);
    landmark_triangulation::retriangulateLandmarksOfMission(mission_id, &map);
  }
  CHECK_GT(map.numLandmarks(), 0u);
  CHECK(vi_map::checkMapConsistency(map));

  map_optimization::ViProblemOptions options =
      map_optimization::ViProblemOptions::initFromGFlags();
  options.add_visual_constraints = true;
  options.add_inertial_constraints = FLAGS_use_imu;
  options.fix_gyro_bias = !FLAGS_optimize_biases;
  options.fix_accel_bias = !FLAGS_optimize_biases;
  options.fix_velocity = !FLAGS_optimize_velocity;
  if (FLAGS_use_imu) {
    options.gravity_magnitude =
        map.getMissionImu(mission_id).getGravityMagnitudeMps2();
  }
  options.fix_intrinsics = true;
  options.fix_extrinsics_rotation = !FLAGS_optimize_extrinsics;
  options.fix_extrinsics_translation = !FLAGS_optimize_extrinsics;
  options.solver_options.max_num_iterations = FLAGS_ba_iterations;
  options.enable_visual_outlier_rejection = true;

  constexpr bool kEnableSignalHandler = false;
  map_optimization::VIMapOptimizer optimizer(nullptr, kEnableSignalHandler);
  map_optimization::OptimizationProblemResult optimization_result;
  CHECK(optimizer.optimize(
      options, vi_map::MissionIdSet{mission_id}, &map, &optimization_result));
  CHECK(!optimization_result.solver_summaries.empty());
  const ceres::Solver::Summary& initial_summary =
      optimization_result.solver_summaries.front();
  const ceres::Solver::Summary& final_summary =
      optimization_result.solver_summaries.back();
  size_t total_iterations = 0u;
  for (const ceres::Solver::Summary& summary :
       optimization_result.solver_summaries) {
    total_iterations += summary.iterations.size();
  }

  double squared_pose_delta_sum = 0.0;
  double squared_velocity_delta_sum = 0.0;
  double squared_accel_bias_delta_sum = 0.0;
  double squared_gyro_bias_delta_sum = 0.0;
  double max_pose_delta = 0.0;
  for (size_t index = 0u; index < vertex_ids.size(); ++index) {
    const double delta =
        (map.getVertex(vertex_ids[index]).get_T_M_I().getPosition() -
         positions_before[index])
            .norm();
    squared_pose_delta_sum += delta * delta;
    max_pose_delta = std::max(max_pose_delta, delta);
    const vi_map::Vertex& vertex = map.getVertex(vertex_ids[index]);
    squared_velocity_delta_sum +=
        (vertex.get_v_M() - velocities_before[index]).squaredNorm();
    squared_accel_bias_delta_sum +=
        (vertex.getAccelBias() - accel_biases_before[index]).squaredNorm();
    squared_gyro_bias_delta_sum +=
        (vertex.getGyroBias() - gyro_biases_before[index]).squaredNorm();
  }
  const double rms_pose_delta =
      std::sqrt(squared_pose_delta_sum / vertex_ids.size());
  const double rms_velocity_delta =
      std::sqrt(squared_velocity_delta_sum / vertex_ids.size());
  const double rms_accel_bias_delta =
      std::sqrt(squared_accel_bias_delta_sum / vertex_ids.size());
  const double rms_gyro_bias_delta =
      std::sqrt(squared_gyro_bias_delta_sum / vertex_ids.size());

  CHECK(vi_map::checkMapConsistency(map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(
      FLAGS_map, save_config, &map));

  writeReport(
      FLAGS_report, vertex_ids.size(), keypoint_count, inlier_match_count,
      outlier_match_count, map.numLandmarks(), initial_summary.initial_cost,
      final_summary.final_cost, total_iterations, rms_pose_delta,
      max_pose_delta, rms_velocity_delta, rms_accel_bias_delta,
      rms_gyro_bias_delta);
  LOG(INFO) << "BRISK frontend and "
            << (FLAGS_use_imu ? "visual-inertial" : "visual")
            << " BA complete: " << keypoint_count
            << " keypoints, " << inlier_match_count << " inlier matches, "
            << map.numLandmarks() << " landmarks, cost "
            << initial_summary.initial_cost << " -> "
            << final_summary.final_cost;
  return 0;
}
