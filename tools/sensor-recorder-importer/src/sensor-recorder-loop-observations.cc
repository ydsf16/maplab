#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <landmark-triangulation/landmark-triangulation.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <yaml-cpp/yaml.h>

DEFINE_string(map, "", "Input VI-Map folder; updated in place.");
DEFINE_string(loops_yaml, "", "Verified loop YAML with PnP inlier observations.");
DEFINE_int32(min_merge_support, 1,
             "Minimum verified PnP inlier correspondences supporting a merge.");

namespace {

struct Observation {
  pose_graph::VertexId query_vertex_id;
  bool has_explicit_vertex_id = false;
  uint64_t query_timestamp_ns;
  vi_map::LandmarkId candidate_landmark_id;
  unsigned int query_keypoint_index;
};

std::vector<Observation> loadObservations(const std::string& filename) {
  std::vector<Observation> observations;
  const YAML::Node root = YAML::LoadFile(filename);
  const YAML::Node loops = root["accepted"] ? root["accepted"] : root;
  for (const YAML::Node& loop : loops) {
    const uint64_t query_timestamp_ns =
        loop["camera_to"]["timestamp_ns"].as<uint64_t>();
    const YAML::Node pnp_observations = loop["observations"];
    if (!pnp_observations) {
      continue;
    }
    for (const YAML::Node& node : pnp_observations) {
      Observation observation;
      if (loop["camera_to"]["vertex_id"]) {
        CHECK(observation.query_vertex_id.fromHexString(
            loop["camera_to"]["vertex_id"].as<std::string>()));
        observation.has_explicit_vertex_id = true;
      }
      observation.query_timestamp_ns = query_timestamp_ns;
      CHECK(observation.candidate_landmark_id.fromHexString(
          node["candidate_landmark_id"].as<std::string>()));
      observation.query_keypoint_index =
          node["query_keypoint_index"].as<unsigned int>();
      observations.emplace_back(std::move(observation));
    }
  }
  return observations;
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_map.empty()) << "--map is required";
  CHECK(!FLAGS_loops_yaml.empty()) << "--loops_yaml is required";
  CHECK_GT(FLAGS_min_merge_support, 0);

  vi_map::VIMap map;
  CHECK(vi_map::serialization::loadMapFromFolder(FLAGS_map, &map));
  vi_map_helpers::VIMapQueries queries(map);
  constexpr uint64_t kTimestampToleranceNs = static_cast<uint64_t>(20e6);

  struct ResolvedObservation {
    pose_graph::VertexId query_vertex_id;
    vi_map::LandmarkId candidate_landmark_id;
    unsigned int query_keypoint_index;
  };
  std::vector<ResolvedObservation> resolved;
  size_t skipped = 0u;
  for (const Observation& observation : loadObservations(FLAGS_loops_yaml)) {
    pose_graph::VertexId query_vertex_id;
    uint64_t timestamp_delta = 0u;
    if (observation.has_explicit_vertex_id) {
      query_vertex_id = observation.query_vertex_id;
      if (!map.hasVertex(query_vertex_id) ||
          !map.hasLandmark(observation.candidate_landmark_id)) {
        ++skipped;
        continue;
      }
    } else if (!queries.getClosestVertexIdByTimestamp(
            observation.query_timestamp_ns, kTimestampToleranceNs,
            &query_vertex_id, &timestamp_delta) ||
        !map.hasLandmark(observation.candidate_landmark_id)) {
      ++skipped;
      continue;
    }
    const vi_map::Vertex& vertex = map.getVertex(query_vertex_id);
    if (observation.query_keypoint_index >=
        vertex.observedLandmarkIdsSize(0u)) {
      ++skipped;
      continue;
    }
    resolved.push_back(
        {query_vertex_id, observation.candidate_landmark_id,
         observation.query_keypoint_index});
  }

  using LandmarkPair = std::pair<vi_map::LandmarkId, vi_map::LandmarkId>;
  std::map<LandmarkPair, int> merge_support;
  for (const ResolvedObservation& observation : resolved) {
    const vi_map::LandmarkId existing = map.getVertex(observation.query_vertex_id)
        .getObservedLandmarkId(0u, observation.query_keypoint_index);
    if (existing.isValid() && existing != observation.candidate_landmark_id) {
      ++merge_support[{existing, observation.candidate_landmark_id}];
    }
  }

  size_t merged = 0u;
  for (const auto& item : merge_support) {
    const vi_map::LandmarkId& to_merge = item.first.first;
    const vi_map::LandmarkId& into = item.first.second;
    if (item.second < FLAGS_min_merge_support || !map.hasLandmark(to_merge) ||
        !map.hasLandmark(into)) {
      continue;
    }
    // This pair is a PnP RANSAC inlier: candidate 3D landmark projected onto
    // the query 2D feature, which already owns `to_merge`.  Identity comes
    // from that visual correspondence, never from post-PGO point proximity.
    map.mergeLandmarks(to_merge, into);
    ++merged;
  }

  size_t attached = 0u;
  for (const ResolvedObservation& observation : resolved) {
    if (!map.hasLandmark(observation.candidate_landmark_id)) {
      continue;
    }
    const vi_map::LandmarkId existing = map.getVertex(observation.query_vertex_id)
        .getObservedLandmarkId(0u, observation.query_keypoint_index);
    if (!existing.isValid()) {
      map.associateKeypointWithExistingLandmark(
          observation.query_vertex_id, 0u, observation.query_keypoint_index,
          observation.candidate_landmark_id);
      ++attached;
    }
  }

  landmark_triangulation::retriangulateLandmarks(&map);
  CHECK(vi_map::checkMapConsistency(map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(FLAGS_map, save_config, &map));
  LOG(INFO) << "Applied " << attached << " cross-loop observations, merged "
            << merged << " landmark pairs, skipped " << skipped << ".";
  return 0;
}
