#include <fstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map-optimization/optimization-problem.h>
#include <map-optimization/solver-options.h>
#include <map-optimization/solver.h>
#include <map-optimization/vi-optimization-builder.h>
#include <maplab-common/pose_types.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/loopclosure-edge.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <yaml-cpp/yaml.h>

DEFINE_string(map, "", "Input VI-Map folder; updated in place.");
DEFINE_string(loops_yaml, "", "Verified external loop closure edges YAML.");
DEFINE_string(accepted_loops_yaml, "", "PGO-gated loop observations YAML.");
DEFINE_double(min_switch_variable, 0.8, "Minimum optimized switch value.");
DEFINE_double(max_mahalanobis_squared, 12.59, "Maximum 6D PGO residual.");

namespace {

pose::Transformation adaptTransformation(
    const pose::Transformation& T_M_S, const pose::Transformation& T_M_T,
    const pose::Transformation& T_M_S2, const pose::Transformation& T_M_T2,
    const pose::Transformation& T_S_T) {
  return (T_M_S2.inverse() * T_M_S) * T_S_T * (T_M_T.inverse() * T_M_T2);
}

pose::Transformation transformFromYaml(const YAML::Node& yaml) {
  const std::vector<float> translation = yaml["translation_m"].as<std::vector<float>>();
  const std::vector<float> quaternion = yaml["rotation_xyzw"].as<std::vector<float>>();
  CHECK_EQ(translation.size(), 3u);
  CHECK_EQ(quaternion.size(), 4u);
  kindr::minimal::RotationQuaternion rotation(
      quaternion[3], quaternion[0], quaternion[1], quaternion[2]);
  kindr::minimal::Position position;
  position << translation[0], translation[1], translation[2];
  return pose::Transformation(rotation, position);
}

struct ExternalLoop {
  uint64_t from_timestamp_ns;
  uint64_t to_timestamp_ns;
  pose::Transformation T_M_from;
  pose::Transformation T_M_to;
  pose::Transformation T_from_to;
  double switch_variable;
  double switch_variable_variance;
  Eigen::Matrix<double, 6, 6> covariance;
  YAML::Node source;
};

std::vector<ExternalLoop> loadLoops(const std::string& filename) {
  std::vector<ExternalLoop> loops;
  for (const YAML::Node& node : YAML::LoadFile(filename)) {
    ExternalLoop loop;
    loop.from_timestamp_ns = node["camera_from"]["timestamp_ns"].as<uint64_t>();
    loop.source = node;
    loop.to_timestamp_ns = node["camera_to"]["timestamp_ns"].as<uint64_t>();
    loop.T_M_from = transformFromYaml(node["camera_from"]["pose"]);
    loop.T_M_to = transformFromYaml(node["camera_to"]["pose"]);
    loop.T_from_to = transformFromYaml(node["T_from_to"]);
    loop.switch_variable = node["switch_variable"].as<double>();
    loop.switch_variable_variance = node["switch_variable_variance"].as<double>();
    const std::vector<double> covariance = node["covariance"].as<std::vector<double>>();
    CHECK_EQ(covariance.size(), 36u);
    for (size_t index = 0u; index < covariance.size(); ++index) {
      loop.covariance(index / 6u, index % 6u) = covariance[index];
    }
    loops.emplace_back(std::move(loop));
  }
  return loops;
}

struct AttachedLoop { pose_graph::EdgeId edge_id; ExternalLoop loop; };
std::vector<AttachedLoop> addExternalLoops(
    const std::vector<ExternalLoop>& edges, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  vi_map_helpers::VIMapQueries queries(*map);
  constexpr uint64_t kTimestampToleranceNs = static_cast<uint64_t>(20e6);
  std::vector<AttachedLoop> added;
  for (const ExternalLoop& edge : edges) {
    pose_graph::VertexId from_id, to_id;
    uint64_t from_delta = 0u, to_delta = 0u;
    if (!queries.getClosestVertexIdByTimestamp(
            edge.from_timestamp_ns, kTimestampToleranceNs, &from_id,
            &from_delta) ||
        !queries.getClosestVertexIdByTimestamp(
            edge.to_timestamp_ns, kTimestampToleranceNs, &to_id, &to_delta)) {
      LOG(WARNING) << "Skipping loop without matching vertices at timestamps "
                   << edge.from_timestamp_ns << " and " << edge.to_timestamp_ns;
      continue;
    }
    const vi_map::Vertex& from_vertex = map->getVertex(from_id);
    const vi_map::Vertex& to_vertex = map->getVertex(to_id);
    const pose::Transformation T_loop = adaptTransformation(
        edge.T_M_from, edge.T_M_to, from_vertex.get_T_M_I(),
        to_vertex.get_T_M_I(), edge.T_from_to);
    pose_graph::EdgeId edge_id;
    aslam::generateId(&edge_id);
    map->addEdge(aligned_unique<vi_map::LoopClosureEdge>(
        edge_id, from_id, to_id, edge.switch_variable,
        edge.switch_variable_variance, T_loop, edge.covariance));
    added.push_back({edge_id, edge});
  }
  return added;
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_map.empty()) << "--map is required";
  CHECK(!FLAGS_loops_yaml.empty()) << "--loops_yaml is required";

  vi_map::VIMap map;
  CHECK(vi_map::serialization::loadMapFromFolder(FLAGS_map, &map));
  const std::vector<ExternalLoop> edges = loadLoops(FLAGS_loops_yaml);
  const std::vector<AttachedLoop> added = addExternalLoops(edges, &map);
  CHECK_GT(added.size(), 0u) << "No external loop was attached to the VI-Map";

  vi_map::MissionIdList mission_list;
  map.getAllMissionIds(&mission_list);
  vi_map::MissionIdSet missions(mission_list.begin(), mission_list.end());
  ceres::Solver::Options options = map_optimization::initSolverOptionsFromFlags();
  map_optimization::ViProblemOptions problem_options =
      map_optimization::ViProblemOptions::initFromGFlags();
  problem_options.fix_accel_bias = true;
  problem_options.fix_gyro_bias = true;
  problem_options.fix_velocity = true;
  problem_options.fix_intrinsics = true;
  problem_options.fix_extrinsics_rotation = true;
  problem_options.fix_extrinsics_translation = true;
  problem_options.fix_landmark_positions = true;
  problem_options.add_loop_closure_edges = true;
  map_optimization::OptimizationProblem::UniquePtr problem(
      map_optimization::constructOptimizationProblem(missions, problem_options, &map));
  CHECK(problem);
  // Keep loop edges until their switch variables and residuals have been read.
  // They are explicitly removed below and never enter visual or inertial BA.
  map_optimization::solve(options, problem.get());
  LOG(INFO) << "External PGO retained " << added.size()
            << " loop edges for switch gating.";
  YAML::Node accepted(YAML::NodeType::Sequence);
  YAML::Node rejected(YAML::NodeType::Sequence);
  for (const AttachedLoop& attached : added) {
    const vi_map::LoopClosureEdge& edge =
        map.getEdgeAs<vi_map::LoopClosureEdge>(attached.edge_id);
    const pose::Transformation T_current =
        map.getVertex(edge.from()).get_T_M_I().inverse() *
        map.getVertex(edge.to()).get_T_M_I();
    const pose::Transformation T_error = edge.get_T_A_B().inverse() * T_current;
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() = T_error.getPosition();
    residual.tail<3>() = T_error.getRotation().log();
    const double mahalanobis_squared = residual.transpose() *
        attached.loop.covariance.inverse() * residual;
    YAML::Node decision = attached.loop.source;
    decision["pgo_switch_variable"] = edge.getSwitchVariable();
    decision["pgo_mahalanobis_squared"] = mahalanobis_squared;
    if (edge.getSwitchVariable() >= FLAGS_min_switch_variable &&
        mahalanobis_squared <= FLAGS_max_mahalanobis_squared) {
      accepted.push_back(decision);
    } else {
      decision["pgo_rejection_reason"] =
          edge.getSwitchVariable() < FLAGS_min_switch_variable ?
          "switch_rejected" : "pgo_residual_rejected";
      rejected.push_back(decision);
    }
  }
  if (!FLAGS_accepted_loops_yaml.empty()) {
    YAML::Node document;
    document["accepted"] = accepted;
    document["rejected"] = rejected;
    YAML::Emitter emitter;
    emitter << document;
    std::ofstream output(FLAGS_accepted_loops_yaml);
    CHECK(output.good()); output << emitter.c_str();
  }
  map.removeLoopClosureEdges();
  CHECK(vi_map::checkMapConsistency(map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(FLAGS_map, save_config, &map));
  LOG(INFO) << "Pose graph relaxation complete with " << added.size() << " loop edges.";
  return 0;
}
