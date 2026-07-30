#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>

#include <aslam/frames/visual-frame.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <maplab-common/file-system-tools.h>
#include <vi-map/landmark.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map/vertex.h>

DEFINE_string(map, "", "VI-Map folder to export.");
DEFINE_string(output, "", "Directory for Rerun interchange CSV files.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_map.empty()) << "--map is required";
  CHECK(!FLAGS_output.empty()) << "--output is required";
  CHECK(common::createPath(FLAGS_output));

  vi_map::VIMap map;
  CHECK(vi_map::serialization::loadMapFromFolder(FLAGS_map, &map));
  vi_map::MissionIdList mission_ids;
  map.getAllMissionIds(&mission_ids);
  pose_graph::VertexIdList vertex_ids;
  std::vector<std::string> vertex_mission_ids;
  for (const vi_map::MissionId& mission_id : mission_ids) {
    pose_graph::VertexIdList mission_vertices;
    map.getAllVertexIdsInMissionAlongGraph(mission_id, &mission_vertices);
    vertex_ids.insert(vertex_ids.end(), mission_vertices.begin(), mission_vertices.end());
    vertex_mission_ids.insert(vertex_mission_ids.end(), mission_vertices.size(),
                              mission_id.hexString());
  }

  std::ofstream vertices(FLAGS_output + "/vertices.csv");
  std::ofstream keypoints(FLAGS_output + "/keypoints.csv");
  std::ofstream landmarks(FLAGS_output + "/landmarks.csv");
  CHECK(vertices.good());
  CHECK(keypoints.good());
  CHECK(landmarks.good());
  vertices << std::setprecision(17);
  keypoints << std::setprecision(17);
  landmarks << std::setprecision(17);
  vertices << "vertex_index,mission_id,vertex_id,timestamp_ns,p_x_m,p_y_m,p_z_m,q_w,q_x,q_y,q_z,"
              "v_x_m_s,v_y_m_s,v_z_m_s,accel_bias_x,accel_bias_y,"
              "accel_bias_z,gyro_bias_x,gyro_bias_y,gyro_bias_z\n";
  keypoints << "vertex_index,u_px,v_px,has_landmark,landmark_id\n";
  landmarks << "landmark_id,x_m,y_m,z_m,observation_count,quality\n";

  size_t keypoint_count = 0u;
  size_t observation_count = 0u;
  for (size_t vertex_index = 0u; vertex_index < vertex_ids.size();
       ++vertex_index) {
    const vi_map::Vertex& vertex = map.getVertex(vertex_ids[vertex_index]);
    const Eigen::Vector3d& p = vertex.get_p_M_I();
    const Eigen::Quaterniond& q = vertex.get_q_M_I();
    const Eigen::Vector3d& v = vertex.get_v_M();
    const Eigen::Vector3d& accel_bias = vertex.getAccelBias();
    const Eigen::Vector3d& gyro_bias = vertex.getGyroBias();
    const int64_t timestamp_ns = vertex.getVisualFrame(0u).getTimestampNanoseconds();
    vertices << vertex_index << ',' << vertex_mission_ids[vertex_index] << ','
             << vertex_ids[vertex_index].hexString()
             << ',' << timestamp_ns << ',' << p.x() << ','
             << p.y() << ',' << p.z() << ',' << q.w() << ',' << q.x() << ','
             << q.y() << ',' << q.z() << ',' << v.x() << ',' << v.y() << ','
             << v.z() << ',' << accel_bias.x() << ',' << accel_bias.y() << ','
             << accel_bias.z() << ',' << gyro_bias.x() << ',' << gyro_bias.y()
             << ',' << gyro_bias.z() << '\n';

    const aslam::VisualFrame& frame = vertex.getVisualFrame(0u);
    if (!frame.hasKeypointMeasurements()) {
      continue;
    }
    const Eigen::Matrix2Xd& measurements = frame.getKeypointMeasurements();
    for (Eigen::Index keypoint_index = 0; keypoint_index < measurements.cols();
         ++keypoint_index) {
      const vi_map::LandmarkId landmark_id =
          vertex.getObservedLandmarkId(0u, keypoint_index);
      const bool has_landmark = landmark_id.isValid();
      keypoints << vertex_index << ',' << measurements(0, keypoint_index) << ','
                << measurements(1, keypoint_index) << ',' << has_landmark
                << ',' << (has_landmark ? landmark_id.hexString() : "")
                << '\n';
      ++keypoint_count;
      observation_count += has_landmark ? 1u : 0u;
    }
  }

  vi_map::LandmarkIdList landmark_ids;
  map.getAllLandmarkIds(&landmark_ids);
  for (const vi_map::LandmarkId& landmark_id : landmark_ids) {
    const Eigen::Vector3d p = map.getLandmark_G_p_fi(landmark_id);
    landmarks << landmark_id.hexString() << ',' << p.x() << ',' << p.y() << ','
              << p.z() << ','
              << map.getLandmark(landmark_id).numberOfObservations() << ','
              << static_cast<int>(map.getLandmark(landmark_id).getQuality())
              << '\n';
  }
  LOG(INFO) << "Exported " << vertex_ids.size() << " vertices, "
            << landmark_ids.size() << " landmarks, " << keypoint_count
            << " keypoints and " << observation_count
            << " landmark observations to " << FLAGS_output;
  return 0;
}
