#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <aslam/cameras/ncamera.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <landmark-triangulation/pose-interpolator.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map/viwls-edge.h>

DEFINE_string(map, "", "VI-Map folder to export.");
DEFINE_string(image_timestamps, "", "Normalized frames.csv with timestamp_ns.");
DEFINE_string(output, "", "Directory for TUM pose files.");

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.emplace_back(field);
  }
  return fields;
}

std::vector<int64_t> readImageTimestamps(const std::string& filename) {
  std::ifstream input(filename);
  CHECK(input.good()) << "Unable to read " << filename;
  std::string line;
  CHECK(std::getline(input, line));
  const std::vector<std::string> header = split(line);
  const auto timestamp_it =
      std::find(header.begin(), header.end(), "timestamp_ns");
  CHECK(timestamp_it != header.end()) << "timestamp_ns is missing from " << filename;
  const size_t timestamp_column = std::distance(header.begin(), timestamp_it);

  std::vector<int64_t> timestamps;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> fields = split(line);
    CHECK_GT(fields.size(), timestamp_column) << "Malformed row in " << filename;
    timestamps.emplace_back(std::stoll(fields[timestamp_column]));
  }
  return timestamps;
}

std::vector<int64_t> collectImuTimestamps(
    const vi_map::VIMap& map, const vi_map::MissionId& mission_id) {
  pose_graph::VertexIdList vertex_ids;
  map.getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);
  std::set<int64_t> timestamp_set;
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    pose_graph::EdgeIdSet outgoing_edges;
    map.getVertex(vertex_id).getOutgoingEdges(&outgoing_edges);
    for (const pose_graph::EdgeId& edge_id : outgoing_edges) {
      if (map.getEdgeType(edge_id) != pose_graph::Edge::EdgeType::kViwls) {
        continue;
      }
      const vi_map::ViwlsEdge& edge = map.getEdgeAs<vi_map::ViwlsEdge>(edge_id);
      const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& timestamps =
          edge.getImuTimestamps();
      for (Eigen::Index index = 0; index < timestamps.cols(); ++index) {
        timestamp_set.insert(timestamps(index));
      }
    }
  }
  return std::vector<int64_t>(timestamp_set.begin(), timestamp_set.end());
}

std::vector<int64_t> keepInRange(
    const std::vector<int64_t>& timestamps, const int64_t min_timestamp_ns,
    const int64_t max_timestamp_ns, size_t* dropped) {
  CHECK_NOTNULL(dropped);
  *dropped = 0u;
  std::vector<int64_t> valid;
  valid.reserve(timestamps.size());
  for (const int64_t timestamp : timestamps) {
    if (timestamp < min_timestamp_ns || timestamp > max_timestamp_ns) {
      ++*dropped;
    } else {
      valid.emplace_back(timestamp);
    }
  }
  return valid;
}

void writeTum(
    const std::string& filename, const std::vector<int64_t>& timestamps,
    const aslam::TransformationVector& T_M_I,
    const aslam::Transformation* T_C_I) {
  CHECK_EQ(timestamps.size(), T_M_I.size());
  std::ofstream output(filename);
  CHECK(output.good()) << "Unable to write " << filename;
  output << std::setprecision(17);
  for (size_t index = 0u; index < timestamps.size(); ++index) {
    const aslam::Transformation T_M_body =
        T_C_I == nullptr ? T_M_I[index] : T_M_I[index] * T_C_I->inverse();
    const Eigen::Vector3d& p_M_body = T_M_body.getPosition();
    const Eigen::Quaterniond q_M_body =
        T_M_body.getRotation().toImplementation();
    output << static_cast<double>(timestamps[index]) * 1e-9 << ' '
           << p_M_body.x() << ' ' << p_M_body.y() << ' ' << p_M_body.z() << ' '
           << q_M_body.x() << ' ' << q_M_body.y() << ' ' << q_M_body.z() << ' '
           << q_M_body.w() << '\n';
  }
}

aslam::TransformationVector interpolatePoses(
    const vi_map::VIMap& map, const vi_map::MissionId& mission_id,
    const std::vector<int64_t>& timestamps) {
  Eigen::Matrix<int64_t, 1, Eigen::Dynamic> requested(1, timestamps.size());
  for (size_t index = 0u; index < timestamps.size(); ++index) {
    requested(index) = timestamps[index];
  }
  landmark_triangulation::PoseInterpolator interpolator;
  aslam::TransformationVector poses_M_I;
  interpolator.getPosesAtTime(map, mission_id, requested, &poses_M_I);
  return poses_M_I;
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_map.empty()) << "--map is required";
  CHECK(!FLAGS_image_timestamps.empty()) << "--image_timestamps is required";
  CHECK(!FLAGS_output.empty()) << "--output is required";
  CHECK(common::createPath(FLAGS_output));

  vi_map::VIMap map;
  CHECK(vi_map::serialization::loadMapFromFolder(FLAGS_map, &map));
  vi_map::MissionIdList mission_ids;
  map.getAllMissionIds(&mission_ids);
  CHECK_EQ(mission_ids.size(), 1u);
  const vi_map::MissionId mission_id = mission_ids.front();
  const aslam::NCamera& ncamera = map.getMissionNCamera(mission_id);
  CHECK_EQ(ncamera.getNumCameras(), 1u);
  const aslam::Transformation T_C_I = ncamera.get_T_C_B(0u);

  const std::vector<int64_t> imu_timestamps =
      collectImuTimestamps(map, mission_id);
  CHECK_GT(imu_timestamps.size(), 1u) << "No usable VIWLS IMU samples.";
  const int64_t min_timestamp_ns = imu_timestamps.front();
  const int64_t max_timestamp_ns = imu_timestamps.back();
  const aslam::TransformationVector imu_poses_M_I =
      interpolatePoses(map, mission_id, imu_timestamps);
  writeTum(
      FLAGS_output + "/imu_poses_tum.txt", imu_timestamps, imu_poses_M_I,
      nullptr);

  const std::vector<int64_t> all_image_timestamps =
      readImageTimestamps(FLAGS_image_timestamps);
  size_t dropped_image_timestamps = 0u;
  const std::vector<int64_t> image_timestamps = keepInRange(
      all_image_timestamps, min_timestamp_ns, max_timestamp_ns,
      &dropped_image_timestamps);
  CHECK(!image_timestamps.empty()) << "No image timestamps overlap VIWLS IMU coverage.";
  const aslam::TransformationVector image_poses_M_I =
      interpolatePoses(map, mission_id, image_timestamps);
  writeTum(
      FLAGS_output + "/image_poses_tum.txt", image_timestamps,
      image_poses_M_I, &T_C_I);

  std::ofstream report(FLAGS_output + "/pose_export_report.txt");
  CHECK(report.good());
  report << "imu_pose_definition=T_M_I (IMU pose in map frame)\n"
         << "image_pose_definition=T_M_C (camera pose in map frame)\n"
         << "tum_columns=timestamp_s tx ty tz qx qy qz qw\n"
         << "imu_pose_count=" << imu_timestamps.size() << '\n'
         << "image_pose_count=" << image_timestamps.size() << '\n'
         << "image_timestamps_dropped_outside_imu_coverage=" << dropped_image_timestamps << '\n'
         << "imu_coverage_start_ns=" << min_timestamp_ns << '\n'
         << "imu_coverage_end_ns=" << max_timestamp_ns << '\n';
  LOG(INFO) << "Exported " << imu_timestamps.size() << " IMU poses and "
            << image_timestamps.size() << " image poses to " << FLAGS_output;
  return 0;
}
