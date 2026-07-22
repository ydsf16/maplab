#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
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
#include <sensors/external-features.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map/vertex.h>
#include <vi-map-helpers/vi-map-manipulation.h>

DECLARE_int32(feature_tracking_detector_orb_num_features);
DECLARE_int32(gyro_matcher_small_search_distance_px);
DECLARE_int32(gyro_matcher_large_search_distance_px);
DECLARE_double(gyro_matcher_matching_bits_ratio_relaxed);
DECLARE_double(gyro_matcher_matching_bits_ratio_strict);
DECLARE_double(gyro_matcher_lowe_ratio);
DECLARE_double(gyro_lk_candidate_ratio);
DECLARE_int32(gyro_lk_window_size);
DECLARE_int32(gyro_lk_max_pyramid_levels);

DEFINE_string(map, "", "VI-Map folder to process in place.");
DEFINE_string(report, "", "Path to write the frontend and BA JSON report.");
DEFINE_string(
    tracks_csv, "",
    "SuperPoint keypoints and LightGlue track IDs to import instead of BRISK.");
DEFINE_int32(ba_iterations, 30, "Maximum number of visual BA iterations.");
DEFINE_bool(run_frontend, true, "Extract and match features before BA.");
DEFINE_bool(run_ba, true, "Run bundle adjustment after triangulation.");
DEFINE_bool(use_imu, false, "Add VIWLS IMU factors to the BA problem.");
DEFINE_bool(
    optimize_extrinsics, false,
    "Optimize camera-IMU rotation and translation during BA.");
DEFINE_bool(optimize_biases, true, "Optimize accelerometer and gyro biases.");
DEFINE_bool(optimize_velocity, true, "Optimize keyframe velocities.");
DEFINE_int32(
    frontend_fast_threshold, 5,
    "FAST threshold used by the ORB detector in the offline frontend.");
DEFINE_int32(
    frontend_pyramid_levels, 4,
    "Number of ORB image pyramid levels in the offline frontend.");
DEFINE_double(
    frontend_nms_radius, 4.0,
    "Non-maximum suppression radius in pixels in the offline frontend.");
DEFINE_int32(
    frontend_max_features, 1000,
    "Maximum number of detected features retained per camera frame.");

namespace {

struct TriangulationStats {
  size_t good_landmarks = 0u;
  size_t bad_landmarks = 0u;
  size_t observations = 0u;
  size_t positive_depth_observations = 0u;
  double median_reprojection_error_px = 0.0;
  double p95_reprojection_error_px = 0.0;
};

struct PairFrontendStats {
  size_t previous_index = 0u;
  size_t current_index = 0u;
  size_t inlier_matches = 0u;
  size_t outlier_matches = 0u;
};

struct ImportedKeypoint {
  size_t keypoint_index = 0u;
  double u_px = 0.0;
  double v_px = 0.0;
  double score = 0.0;
  int track_id = -1;
};

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.emplace_back(field);
  }
  return fields;
}

void importSuperPointTracks(
    const std::string& path, const pose_graph::VertexIdList& vertex_ids,
    vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  std::ifstream stream(path);
  CHECK(stream.good()) << "Unable to read tracks CSV: " << path;
  std::string line;
  CHECK(std::getline(stream, line));
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  CHECK_EQ(
      line, "vertex_index,keypoint_index,u_px,v_px,score,track_id")
      << "Unexpected tracks CSV header";
  std::vector<std::vector<ImportedKeypoint>> per_vertex(vertex_ids.size());
  size_t line_number = 1u;
  while (std::getline(stream, line)) {
    ++line_number;
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = splitCsvLine(line);
    CHECK_EQ(fields.size(), 6u) << "Malformed tracks CSV line " << line_number;
    const size_t vertex_index = std::stoul(fields[0]);
    CHECK_LT(vertex_index, per_vertex.size());
    ImportedKeypoint keypoint;
    keypoint.keypoint_index = std::stoul(fields[1]);
    keypoint.u_px = std::stod(fields[2]);
    keypoint.v_px = std::stod(fields[3]);
    keypoint.score = std::stod(fields[4]);
    keypoint.track_id = std::stoi(fields[5]);
    CHECK_EQ(keypoint.keypoint_index, per_vertex[vertex_index].size())
        << "Keypoints must be contiguous and ordered at line " << line_number;
    per_vertex[vertex_index].emplace_back(keypoint);
  }

  for (size_t vertex_index = 0u; vertex_index < vertex_ids.size();
       ++vertex_index) {
    const std::vector<ImportedKeypoint>& input = per_vertex[vertex_index];
    CHECK(!input.empty()) << "No SuperPoint features for vertex " << vertex_index;
    const Eigen::Index count = static_cast<Eigen::Index>(input.size());
    Eigen::Matrix2Xd measurements(2, count);
    Eigen::VectorXd uncertainties = Eigen::VectorXd::Constant(count, 1.0);
    Eigen::VectorXd scores(count);
    Eigen::VectorXi track_ids(count);
    for (Eigen::Index index = 0; index < count; ++index) {
      measurements(0, index) = input[index].u_px;
      measurements(1, index) = input[index].v_px;
      scores(index) = input[index].score;
      track_ids(index) = input[index].track_id;
    }
    // Tracks are produced by the external ONNX frontend. The descriptor byte
    // is a placeholder needed by VisualFrame to carry the kSuperPoint type;
    // matching continues to use the ONNX model, not these placeholder bytes.
    aslam::VisualFrame::DescriptorsT descriptors(1, count);
    descriptors.setZero();
    vi_map::Vertex* vertex = &map->getVertex(vertex_ids[vertex_index]);
    aslam::VisualFrame::Ptr frame = vertex->getVisualFrameShared(0u);
    frame->setKeypointMeasurements(measurements);
    frame->setKeypointMeasurementUncertainties(uncertainties);
    frame->setKeypointScores(scores);
    frame->setTrackIds(track_ids);
    frame->setDescriptors(
        descriptors, 0u, static_cast<int>(vi_map::FeatureType::kSuperPoint));
    vertex->resetObservedLandmarkIdsToInvalid();
    vertex->expandVisualObservationContainersIfNecessary();
  }
}

TriangulationStats computeTriangulationStats(const vi_map::VIMap& map) {
  TriangulationStats stats;
  std::vector<double> reprojection_errors;
  vi_map::LandmarkIdList landmark_ids;
  map.getAllLandmarkIds(&landmark_ids);
  for (const vi_map::LandmarkId& landmark_id : landmark_ids) {
    const vi_map::Landmark& landmark = map.getLandmark(landmark_id);
    if (landmark.getQuality() == vi_map::Landmark::Quality::kGood) {
      ++stats.good_landmarks;
    } else {
      ++stats.bad_landmarks;
    }
    for (const vi_map::KeypointIdentifier& observation :
         landmark.getObservations()) {
      const vi_map::Vertex& vertex =
          map.getVertex(observation.frame_id.vertex_id);
      const size_t frame_index = observation.frame_id.frame_index;
      const Eigen::Vector3d p_C =
          map.getLandmark_p_C_fi(landmark_id, vertex, frame_index);
      ++stats.observations;
      if (p_C.z() <= 0.0) {
        continue;
      }
      ++stats.positive_depth_observations;
      Eigen::Vector2d projected;
      vertex.getCamera(frame_index)->project3(p_C, &projected);
      const Eigen::Vector2d measured =
          vertex.getVisualFrame(frame_index)
              .getKeypointMeasurements()
              .col(observation.keypoint_index);
      reprojection_errors.emplace_back((projected - measured).norm());
    }
  }
  if (!reprojection_errors.empty()) {
    std::sort(reprojection_errors.begin(), reprojection_errors.end());
    stats.median_reprojection_error_px =
        reprojection_errors[reprojection_errors.size() / 2u];
    const size_t p95_index = std::min(
        reprojection_errors.size() - 1u,
        static_cast<size_t>(0.95 * reprojection_errors.size()));
    stats.p95_reprojection_error_px = reprojection_errors[p95_index];
  }
  return stats;
}

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
    const double accel_bias_rms_delta, const double gyro_bias_rms_delta,
    const TriangulationStats& triangulation_stats,
    const std::vector<size_t>& frame_keypoint_counts,
    const std::vector<PairFrontendStats>& pair_stats) {
  std::ofstream stream(path);
  CHECK(stream.good()) << "Unable to write report: " << path;
  stream << "{\n"
         << "  \"status\": \"created_and_verified\",\n"
         << "  \"frontend\": \""
         << (FLAGS_run_frontend
                 ? "maplab_orb_brisk_gyro_tracker"
                 : (FLAGS_tracks_csv.empty() ? "existing_vimap_tracks"
                                             : "superpoint_lightglue_onnx"))
         << "\",\n"
         << "  \"vertices\": " << vertices << ",\n"
         << "  \"keypoints\": " << keypoints << ",\n"
         << "  \"inlier_matches\": " << inlier_matches << ",\n"
         << "  \"outlier_matches\": " << outlier_matches << ",\n"
         << "  \"landmarks\": " << landmarks << ",\n"
         << "  \"frontend_settings\": {\n"
         << "    \"fast_threshold\": " << FLAGS_frontend_fast_threshold
         << ",\n"
         << "    \"pyramid_levels\": " << FLAGS_frontend_pyramid_levels
         << ",\n"
         << "    \"nms_radius_px\": " << FLAGS_frontend_nms_radius
         << ",\n"
         << "    \"max_features\": " << FLAGS_frontend_max_features
         << ",\n"
         << "    \"orb_candidate_features\": "
         << FLAGS_feature_tracking_detector_orb_num_features << ",\n"
         << "    \"matcher_small_search_px\": "
         << FLAGS_gyro_matcher_small_search_distance_px << ",\n"
         << "    \"matcher_large_search_px\": "
         << FLAGS_gyro_matcher_large_search_distance_px << ",\n"
         << "    \"matching_bits_ratio_relaxed\": "
         << FLAGS_gyro_matcher_matching_bits_ratio_relaxed << ",\n"
         << "    \"matching_bits_ratio_strict\": "
         << FLAGS_gyro_matcher_matching_bits_ratio_strict << ",\n"
         << "    \"lowe_ratio\": " << FLAGS_gyro_matcher_lowe_ratio
         << ",\n"
         << "    \"lk_candidate_ratio\": " << FLAGS_gyro_lk_candidate_ratio
         << ",\n"
         << "    \"lk_window_size\": " << FLAGS_gyro_lk_window_size
         << ",\n"
         << "    \"lk_max_pyramid_levels\": "
         << FLAGS_gyro_lk_max_pyramid_levels << "\n"
         << "  },\n"
         << "  \"frame_keypoint_counts\": [";
  for (size_t index = 0u; index < frame_keypoint_counts.size(); ++index) {
    stream << (index == 0u ? "" : ", ") << frame_keypoint_counts[index];
  }
  stream << "],\n"
         << "  \"pair_match_counts\": [\n";
  for (size_t index = 0u; index < pair_stats.size(); ++index) {
    const PairFrontendStats& pair = pair_stats[index];
    stream << "    {\"previous_index\": " << pair.previous_index
           << ", \"current_index\": " << pair.current_index
           << ", \"inliers\": " << pair.inlier_matches
           << ", \"outliers\": " << pair.outlier_matches << "}"
           << (index + 1u == pair_stats.size() ? "\n" : ",\n");
  }
  stream << "  ],\n"
         << "  \"triangulation\": {\n"
         << "    \"good_landmarks\": " << triangulation_stats.good_landmarks
         << ",\n"
         << "    \"bad_landmarks\": " << triangulation_stats.bad_landmarks
         << ",\n"
         << "    \"observations\": " << triangulation_stats.observations
         << ",\n"
         << "    \"positive_depth_observations\": "
         << triangulation_stats.positive_depth_observations << ",\n"
         << "    \"positive_depth_ratio\": "
         << (triangulation_stats.observations > 0u
                 ? static_cast<double>(
                       triangulation_stats.positive_depth_observations) /
                       triangulation_stats.observations
                 : 0.0)
         << ",\n"
         << "    \"median_reprojection_error_px\": "
         << triangulation_stats.median_reprojection_error_px << ",\n"
         << "    \"p95_reprojection_error_px\": "
         << triangulation_stats.p95_reprojection_error_px << "\n"
         << "  },\n"
         << "  \"ba\": {\n"
         << "    \"type\": \""
         << (!FLAGS_run_ba ? "disabled"
                           : (FLAGS_use_imu ? "visual_inertial" : "visual"))
         << "\",\n"
         << "    \"imu_factors\": "
         << (FLAGS_run_ba && FLAGS_use_imu ? "true" : "false")
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
  CHECK_GE(FLAGS_frontend_fast_threshold, 0);
  CHECK_GT(FLAGS_frontend_pyramid_levels, 0);
  CHECK_GT(FLAGS_frontend_nms_radius, 0.0);
  CHECK_GT(FLAGS_frontend_max_features, 0);

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
  std::vector<size_t> frame_keypoint_counts;
  std::vector<PairFrontendStats> pair_stats;
  CHECK(FLAGS_tracks_csv.empty() || !FLAGS_run_frontend)
      << "--tracks_csv and --run_frontend=true are mutually exclusive";
  if (!FLAGS_tracks_csv.empty()) {
    importSuperPointTracks(FLAGS_tracks_csv, vertex_ids, &map);
  }
  if (FLAGS_run_frontend) {
    aslam::NCamera::ConstPtr ncamera =
        map.getVertex(vertex_ids.front()).getNCameras();
    feature_tracking::FeatureTrackingExtractorSettings extractor_settings;
    feature_tracking::FeatureTrackingDetectorSettings detector_settings;
    feature_tracking::FeatureTrackingOutlierSettings outlier_settings;
    detector_settings.orb_detector_fast_threshold =
        FLAGS_frontend_fast_threshold;
    detector_settings.orb_detector_pyramid_levels =
        FLAGS_frontend_pyramid_levels;
    detector_settings.detector_nonmaxsuppression_radius =
        FLAGS_frontend_nms_radius;
    detector_settings.max_feature_count =
        static_cast<size_t>(FLAGS_frontend_max_features);
    feature_tracking::VOFeatureTrackingPipeline tracker(
        ncamera, extractor_settings, detector_settings, outlier_settings);

    vi_map::Vertex* previous_vertex = &map.getVertex(vertex_ids.front());
    loadRawImage(&map, previous_vertex);
    tracker.initializeFirstNFrame(
        previous_vertex->getVisualNFrameShared().get());
    previous_vertex->expandVisualObservationContainersIfNecessary();
    frame_keypoint_counts.reserve(vertex_ids.size());
    pair_stats.reserve(vertex_ids.size() - 1u);
    frame_keypoint_counts.emplace_back(
        previous_vertex->getVisualFrame(0u).getNumKeypointMeasurements());
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
      frame_keypoint_counts.emplace_back(
          current_vertex->getVisualFrame(0u).getNumKeypointMeasurements());
      pair_stats.push_back(PairFrontendStats{
          index - 1u, index, inlier_matches.front().size(),
          outlier_matches.front().size()});
      previous_vertex->getVisualNFrameShared()->releaseRawImagesOfAllFrames();
      previous_vertex = current_vertex;
    }
    previous_vertex->getVisualNFrameShared()->releaseRawImagesOfAllFrames();
  }

  const size_t keypoint_count = countKeypoints(vertex_ids, map);
  if (!FLAGS_run_frontend) {
    frame_keypoint_counts.reserve(vertex_ids.size());
    for (const pose_graph::VertexId& vertex_id : vertex_ids) {
      frame_keypoint_counts.emplace_back(
          map.getVertex(vertex_id)
              .getVisualFrame(0u)
              .getNumKeypointMeasurements());
    }
  }
  if (FLAGS_run_frontend || !FLAGS_tracks_csv.empty()) {
    vi_map_helpers::VIMapManipulation manipulation(&map);
    const size_t landmark_count =
        manipulation.initializeLandmarksFromUnusedFeatureTracksOfMission(
            mission_id);
    CHECK_GT(landmark_count, 0u);
    landmark_triangulation::retriangulateLandmarksOfMission(mission_id, &map);
  }
  CHECK_GT(map.numLandmarks(), 0u);
  CHECK(vi_map::checkMapConsistency(map));
  const TriangulationStats triangulation_stats =
      computeTriangulationStats(map);

  if (!FLAGS_run_ba) {
    backend::SaveConfig save_config;
    save_config.overwrite_existing_files = true;
    CHECK(vi_map::serialization::saveMapToFolder(
        FLAGS_map, save_config, &map));
    writeReport(
        FLAGS_report, vertex_ids.size(), keypoint_count, inlier_match_count,
        outlier_match_count, map.numLandmarks(), 0.0, 0.0, 0u, 0.0, 0.0,
        0.0, 0.0, 0.0, triangulation_stats, frame_keypoint_counts, pair_stats);
    LOG(INFO) << "BRISK frontend and triangulation complete without BA: "
              << map.numLandmarks() << " landmarks, median reprojection error "
              << triangulation_stats.median_reprojection_error_px << " px";
    return 0;
  }

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
  const TriangulationStats post_ba_triangulation_stats =
      computeTriangulationStats(map);
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(
      FLAGS_map, save_config, &map));

  writeReport(
      FLAGS_report, vertex_ids.size(), keypoint_count, inlier_match_count,
      outlier_match_count, map.numLandmarks(), initial_summary.initial_cost,
      final_summary.final_cost, total_iterations, rms_pose_delta,
      max_pose_delta, rms_velocity_delta, rms_accel_bias_delta,
      rms_gyro_bias_delta, post_ba_triangulation_stats, frame_keypoint_counts,
      pair_stats);
  LOG(INFO) << "BRISK frontend and "
            << (FLAGS_use_imu ? "visual-inertial" : "visual")
            << " BA complete: " << keypoint_count
            << " keypoints, " << inlier_match_count << " inlier matches, "
            << map.numLandmarks() << " landmarks, cost "
            << initial_summary.initial_cost << " -> "
            << final_summary.final_cost;
  return 0;
}
