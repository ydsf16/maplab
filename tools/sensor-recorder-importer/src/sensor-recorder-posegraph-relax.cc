#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map-optimization/solver-options.h>
#include <map-optimization/vi-map-relaxation.h>
#include <maplab-common/pose_types.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/loopclosure-edge.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <yaml-cpp/yaml.h>

DEFINE_string(map, "", "Input VI-Map folder; updated in place.");
DEFINE_string(loops_yaml, "", "Verified external loop closure edges YAML.");

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
};

std::vector<ExternalLoop> loadLoops(const std::string& filename) {
  std::vector<ExternalLoop> loops;
  for (const YAML::Node& node : YAML::LoadFile(filename)) {
    ExternalLoop loop;
    loop.from_timestamp_ns = node["camera_from"]["timestamp_ns"].as<uint64_t>();
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

size_t addExternalLoops(
    const std::vector<ExternalLoop>& edges, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  vi_map_helpers::VIMapQueries queries(*map);
  constexpr uint64_t kTimestampToleranceNs = static_cast<uint64_t>(20e6);
  size_t added = 0u;
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
    ++added;
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
  const size_t added = addExternalLoops(edges, &map);
  CHECK_GT(added, 0u) << "No external loop was attached to the VI-Map";

  vi_map::MissionIdList mission_list;
  map.getAllMissionIds(&mission_list);
  vi_map::MissionIdSet missions(mission_list.begin(), mission_list.end());
  map_optimization::VIMapRelaxation relaxation(nullptr, false);
  ceres::Solver::Options options = map_optimization::initSolverOptionsFromFlags();
  CHECK(relaxation.solveRelaxation(options, missions, &map));
  map.removeLoopClosureEdges();
  CHECK(vi_map::checkMapConsistency(map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(FLAGS_map, save_config, &map));
  LOG(INFO) << "Pose graph relaxation complete with " << added << " loop edges.";
  return 0;
}
