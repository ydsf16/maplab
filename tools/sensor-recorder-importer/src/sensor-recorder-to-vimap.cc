#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/memory.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <maplab-common/file-system-tools.h>
#include <map-resources/resource-common.h>
#include <opencv2/imgcodecs.hpp>
#include <sensors/imu.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/sensor-manager.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <vi-map/vertex.h>
#include <vi-map/viwls-edge.h>

DEFINE_string(
    normalized_data, "",
    "Directory containing recording.json, calibration.csv, keyframes.csv and "
    "imu.csv.");
DEFINE_string(output_map, "", "Output folder for the native VI-Map.");

namespace {

struct CsvTable {
  std::unordered_map<std::string, size_t> columns;
  std::vector<std::vector<std::string>> rows;
};

struct Keyframe {
  int64_t timestamp_ns;
  Eigen::Vector3d position;
  Eigen::Quaterniond quaternion;
  Eigen::Vector3d velocity;
  double fx;
  double fy;
  double cx;
  double cy;
  uint32_t width;
  uint32_t height;
  std::string image_path;
};

struct ImuSample {
  int64_t timestamp_ns;
  Eigen::Matrix<double, 6, 1> measurement;
};

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> values;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) {
    if (!value.empty() && value.back() == '\r') {
      value.pop_back();
    }
    values.emplace_back(value);
  }
  if (!line.empty() && line.back() == ',') {
    values.emplace_back();
  }
  return values;
}

CsvTable readCsv(const std::string& path) {
  std::ifstream stream(path);
  CHECK(stream.good()) << "Unable to open CSV: " << path;

  std::string line;
  CHECK(static_cast<bool>(std::getline(stream, line)))
      << "CSV has no header: " << path;
  const std::vector<std::string> header = splitCsvLine(line);
  CHECK(!header.empty()) << "CSV has an empty header: " << path;

  CsvTable table;
  for (size_t index = 0u; index < header.size(); ++index) {
    CHECK(table.columns.emplace(header[index], index).second)
        << "Duplicate CSV column '" << header[index] << "' in " << path;
  }
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> row = splitCsvLine(line);
    CHECK_EQ(row.size(), header.size())
        << "CSV row has the wrong number of fields in " << path;
    table.rows.emplace_back(std::move(row));
  }
  CHECK(!table.rows.empty()) << "CSV has no data: " << path;
  return table;
}

const std::string& value(
    const CsvTable& table, const std::vector<std::string>& row,
    const std::string& column) {
  const auto iterator = table.columns.find(column);
  CHECK(iterator != table.columns.end()) << "Missing CSV column: " << column;
  return row.at(iterator->second);
}

double doubleValue(
    const CsvTable& table, const std::vector<std::string>& row,
    const std::string& column) {
  return std::stod(value(table, row, column));
}

int64_t int64Value(
    const CsvTable& table, const std::vector<std::string>& row,
    const std::string& column) {
  return std::stoll(value(table, row, column));
}

std::vector<Keyframe> readKeyframes(const std::string& path) {
  const CsvTable table = readCsv(path);
  std::vector<Keyframe> keyframes;
  keyframes.reserve(table.rows.size());
  for (const std::vector<std::string>& row : table.rows) {
    Keyframe keyframe;
    keyframe.timestamp_ns = int64Value(table, row, "timestamp_ns");
    keyframe.position = Eigen::Vector3d(
        doubleValue(table, row, "p_M_I_x_m"),
        doubleValue(table, row, "p_M_I_y_m"),
        doubleValue(table, row, "p_M_I_z_m"));
    keyframe.quaternion = Eigen::Quaterniond(
        doubleValue(table, row, "q_M_I_w"),
        doubleValue(table, row, "q_M_I_x"),
        doubleValue(table, row, "q_M_I_y"),
        doubleValue(table, row, "q_M_I_z"));
    keyframe.quaternion.normalize();
    keyframe.velocity = Eigen::Vector3d(
        doubleValue(table, row, "v_M_I_x_m_s"),
        doubleValue(table, row, "v_M_I_y_m_s"),
        doubleValue(table, row, "v_M_I_z_m_s"));
    keyframe.fx = doubleValue(table, row, "fx_px");
    keyframe.fy = doubleValue(table, row, "fy_px");
    keyframe.cx = doubleValue(table, row, "cx_px");
    keyframe.cy = doubleValue(table, row, "cy_px");
    keyframe.width =
        static_cast<uint32_t>(int64Value(table, row, "width_px"));
    keyframe.height =
        static_cast<uint32_t>(int64Value(table, row, "height_px"));
    const auto image_path_column = table.columns.find("image_path");
    if (image_path_column != table.columns.end()) {
      keyframe.image_path = row.at(image_path_column->second);
    }
    if (!keyframes.empty()) {
      CHECK_GT(keyframe.timestamp_ns, keyframes.back().timestamp_ns)
          << "Keyframe timestamps must be strictly increasing.";
    }
    keyframes.emplace_back(keyframe);
  }
  CHECK_GE(keyframes.size(), 2u);
  return keyframes;
}

std::vector<ImuSample> readImu(const std::string& path) {
  const CsvTable table = readCsv(path);
  std::vector<ImuSample> samples;
  samples.reserve(table.rows.size());
  for (const std::vector<std::string>& row : table.rows) {
    ImuSample sample;
    sample.timestamp_ns = int64Value(table, row, "timestamp_ns");
    sample.measurement << doubleValue(table, row, "ax_m_s2"),
        doubleValue(table, row, "ay_m_s2"),
        doubleValue(table, row, "az_m_s2"),
        doubleValue(table, row, "gx_rad_s"),
        doubleValue(table, row, "gy_rad_s"),
        doubleValue(table, row, "gz_rad_s");
    if (!samples.empty()) {
      CHECK_GT(sample.timestamp_ns, samples.back().timestamp_ns)
          << "IMU timestamps must be strictly increasing.";
    }
    samples.emplace_back(sample);
  }
  CHECK_GE(samples.size(), 2u);
  return samples;
}

ImuSample interpolateImuSample(
    const std::vector<ImuSample>& samples, const int64_t timestamp_ns) {
  const auto upper = std::lower_bound(
      samples.begin(), samples.end(), timestamp_ns,
      [](const ImuSample& sample, const int64_t timestamp) {
        return sample.timestamp_ns < timestamp;
      });
  CHECK(upper != samples.end()) << "IMU does not cover edge end timestamp.";
  if (upper->timestamp_ns == timestamp_ns) {
    return *upper;
  }
  CHECK(upper != samples.begin()) << "IMU does not cover edge start timestamp.";
  const ImuSample& right = *upper;
  const ImuSample& left = *(upper - 1);
  const double alpha = static_cast<double>(timestamp_ns - left.timestamp_ns) /
                       static_cast<double>(right.timestamp_ns - left.timestamp_ns);
  ImuSample result;
  result.timestamp_ns = timestamp_ns;
  result.measurement =
      (1.0 - alpha) * left.measurement + alpha * right.measurement;
  return result;
}

std::vector<ImuSample> imuSamplesForEdge(
    const std::vector<ImuSample>& samples, const int64_t start_timestamp_ns,
    const int64_t end_timestamp_ns) {
  CHECK_LT(start_timestamp_ns, end_timestamp_ns);
  std::vector<ImuSample> edge_samples;
  edge_samples.emplace_back(interpolateImuSample(samples, start_timestamp_ns));
  auto sample = std::upper_bound(
      samples.begin(), samples.end(), start_timestamp_ns,
      [](const int64_t timestamp, const ImuSample& candidate) {
        return timestamp < candidate.timestamp_ns;
      });
  for (; sample != samples.end() && sample->timestamp_ns < end_timestamp_ns;
       ++sample) {
    edge_samples.emplace_back(*sample);
  }
  edge_samples.emplace_back(interpolateImuSample(samples, end_timestamp_ns));
  CHECK_GE(edge_samples.size(), 2u);
  return edge_samples;
}

struct Calibration {
  aslam::Transformation T_C_I;
  vi_map::ImuSigmas imu_sigmas;
  double gravity_m_s2;
};

Calibration readCalibration(const std::string& path) {
  const CsvTable table = readCsv(path);
  CHECK_EQ(table.rows.size(), 1u);
  const std::vector<std::string>& row = table.rows.front();
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  for (int row_index = 0; row_index < 3; ++row_index) {
    for (int column_index = 0; column_index < 3; ++column_index) {
      matrix(row_index, column_index) = doubleValue(
          table, row,
          "R_C_I_" + std::to_string(row_index) +
              std::to_string(column_index));
    }
  }
  matrix(0, 3) = doubleValue(table, row, "t_C_I_x_m");
  matrix(1, 3) = doubleValue(table, row, "t_C_I_y_m");
  matrix(2, 3) = doubleValue(table, row, "t_C_I_z_m");

  Calibration calibration;
  calibration.T_C_I =
      aslam::Transformation::constructAndRenormalizeRotation(matrix);
  calibration.gravity_m_s2 = doubleValue(table, row, "gravity_m_s2");
  calibration.imu_sigmas.gyro_noise_density =
      doubleValue(table, row, "gyro_noise_density");
  calibration.imu_sigmas.gyro_bias_random_walk_noise_density = doubleValue(
      table, row, "gyro_bias_random_walk_noise_density");
  calibration.imu_sigmas.acc_noise_density =
      doubleValue(table, row, "acc_noise_density");
  calibration.imu_sigmas.acc_bias_random_walk_noise_density = doubleValue(
      table, row, "acc_bias_random_walk_noise_density");
  return calibration;
}

aslam::VisualNFrame::Ptr createNFrame(
    int64_t timestamp_ns, const aslam::NCamera::Ptr& camera_rig) {
  aslam::VisualNFrame::Ptr nframe(new aslam::VisualNFrame(camera_rig));
  nframe->setId(aslam::createRandomId<aslam::NFramesId>());
  std::shared_ptr<aslam::VisualFrame> frame(new aslam::VisualFrame());
  frame->setCameraGeometry(camera_rig->getCameraShared(0u));
  frame->setId(aslam::createRandomId<aslam::FrameId>());
  frame->setTimestampNanoseconds(timestamp_ns);
  frame->clearKeypointChannels();
  nframe->setFrame(0u, frame);
  CHECK(nframe->areAllFramesSet());
  return nframe;
}

aslam::Transformation transformationFromKeyframe(const Keyframe& keyframe) {
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.block<3, 3>(0, 0) = keyframe.quaternion.toRotationMatrix();
  matrix.block<3, 1>(0, 3) = keyframe.position;
  return aslam::Transformation::constructAndRenormalizeRotation(matrix);
}

}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_normalized_data.empty()) << "--normalized_data is required";
  CHECK(!FLAGS_output_map.empty()) << "--output_map is required";
  CHECK(!vi_map::serialization::hasMapOnFileSystem(FLAGS_output_map))
      << "Output already contains a VI-Map: " << FLAGS_output_map;

  const std::string keyframes_path = common::concatenateFolderAndFileName(
      FLAGS_normalized_data, "keyframes.csv");
  const std::string imu_path = common::concatenateFolderAndFileName(
      FLAGS_normalized_data, "imu.csv");
  const std::string calibration_path = common::concatenateFolderAndFileName(
      FLAGS_normalized_data, "calibration.csv");
  const std::vector<Keyframe> keyframes = readKeyframes(keyframes_path);
  const std::vector<ImuSample> imu_samples = readImu(imu_path);
  const Calibration calibration = readCalibration(calibration_path);

  CHECK_LE(imu_samples.front().timestamp_ns, keyframes.front().timestamp_ns);
  CHECK_GE(imu_samples.back().timestamp_ns, keyframes.back().timestamp_ns);

  const Keyframe& reference = keyframes.front();
  std::shared_ptr<aslam::Camera> camera(new aslam::PinholeCamera(
      reference.fx, reference.fy, reference.cx, reference.cy, reference.width,
      reference.height));
  camera->setId(aslam::createRandomId<aslam::SensorId>());

  std::vector<std::shared_ptr<aslam::Camera>> cameras{camera};
  aslam::TransformationVector T_C_I_vector{calibration.T_C_I};
  const aslam::NCameraId ncamera_id =
      aslam::createRandomId<aslam::NCameraId>();
  aslam::NCamera::UniquePtr ncamera = aligned_unique<aslam::NCamera>(
      ncamera_id, T_C_I_vector, cameras, "SensorRecorderPro wide camera");

  const aslam::SensorId imu_sensor_id =
      aslam::createRandomId<aslam::SensorId>();
  vi_map::Imu::UniquePtr imu_sensor =
      aligned_unique<vi_map::Imu>(imu_sensor_id, "iphone_imu");
  imu_sensor->setImuSigmas(calibration.imu_sigmas);
  imu_sensor->setGravityMagnitude(calibration.gravity_m_s2);

  vi_map::VIMap map(FLAGS_output_map);
  map.useMapResourceFolder();
  map.getSensorManager().addSensorAsBase<vi_map::Imu>(std::move(imu_sensor));
  map.getSensorManager().addSensor<aslam::NCamera>(
      std::move(ncamera), imu_sensor_id, aslam::Transformation());

  vi_map::MissionId mission_id;
  aslam::generateId(&mission_id);
  Eigen::Matrix<double, 6, 6> baseframe_covariance =
      Eigen::Matrix<double, 6, 6>::Zero();
  map.addNewMissionWithBaseframe(
      mission_id, aslam::Transformation(), baseframe_covariance,
      vi_map::Mission::BackBone::kViwls);
  aslam::SensorIdSet sensor_ids;
  map.getSensorManager().getAllSensorIds(&sensor_ids);
  map.associateMissionSensors(sensor_ids, mission_id);

  aslam::NCamera::Ptr camera_rig =
      map.getSensorManager().getSensorPtr<aslam::NCamera>(ncamera_id);
  CHECK(camera_rig);

  pose_graph::VertexId previous_vertex_id;
  std::vector<pose_graph::VertexId> vertex_ids;
  vertex_ids.reserve(keyframes.size());
  size_t raw_image_resource_count = 0u;
  for (size_t keyframe_index = 0u; keyframe_index < keyframes.size();
       ++keyframe_index) {
    const Keyframe& keyframe = keyframes[keyframe_index];
    pose_graph::VertexId vertex_id;
    aslam::generateId(&vertex_id);
    Eigen::Matrix<double, 6, 1> imu_biases =
        Eigen::Matrix<double, 6, 1>::Zero();
    std::vector<std::vector<vi_map::LandmarkId>> observed_landmarks(1u);
    vi_map::Vertex::UniquePtr vertex(new vi_map::Vertex(
        vertex_id, imu_biases,
        createNFrame(keyframe.timestamp_ns, camera_rig), observed_landmarks,
        mission_id));
    vertex->set_T_M_I(transformationFromKeyframe(keyframe));
    vertex->set_v_M(keyframe.velocity);
    vi_map::Vertex* vertex_ptr = vertex.get();
    map.addVertex(std::move(vertex));
    vertex_ids.emplace_back(vertex_id);

    if (!keyframe.image_path.empty()) {
      const std::string image_path = common::concatenateFolderAndFileName(
          FLAGS_normalized_data, keyframe.image_path);
      const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
      CHECK(!image.empty()) << "Unable to load keyframe image: " << image_path;
      CHECK_EQ(static_cast<uint32_t>(image.cols), keyframe.width);
      CHECK_EQ(static_cast<uint32_t>(image.rows), keyframe.height);
      map.storeFrameResource(
          image, 0u, backend::ResourceType::kRawImage, vertex_ptr);
      ++raw_image_resource_count;
    }

    if (keyframe_index == 0u) {
      map.getMission(mission_id).setRootVertexId(vertex_id);
    } else {
      const std::vector<ImuSample> edge_samples = imuSamplesForEdge(
          imu_samples, keyframes[keyframe_index - 1u].timestamp_ns,
          keyframe.timestamp_ns);
      const size_t measurement_count = edge_samples.size();
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic> imu_timestamps(
          1, measurement_count);
      Eigen::Matrix<double, 6, Eigen::Dynamic> imu_measurements(
          6, measurement_count);
      for (size_t offset = 0u; offset < measurement_count; ++offset) {
        const ImuSample& sample = edge_samples[offset];
        imu_timestamps(offset) = sample.timestamp_ns;
        imu_measurements.col(offset) = sample.measurement;
      }
      pose_graph::EdgeId edge_id;
      aslam::generateId(&edge_id);
      map.addEdge(vi_map::ViwlsEdge::UniquePtr(new vi_map::ViwlsEdge(
          edge_id, previous_vertex_id, vertex_id, imu_timestamps,
          imu_measurements)));
    }
    previous_vertex_id = vertex_id;
  }

  CHECK(vi_map::checkMapConsistency(map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = false;
  CHECK(vi_map::serialization::saveMapToFolder(
      FLAGS_output_map, save_config, &map));

  vi_map::VIMap reloaded_map;
  CHECK(vi_map::serialization::loadMapFromFolder(
      FLAGS_output_map, &reloaded_map));
  CHECK(vi_map::checkMapConsistency(reloaded_map));
  CHECK_EQ(reloaded_map.numVertices(), keyframes.size());
  CHECK_EQ(reloaded_map.numEdges(), keyframes.size() - 1u);
  size_t reloaded_image_count = 0u;
  for (const pose_graph::VertexId& vertex_id : vertex_ids) {
    const vi_map::Vertex& vertex = reloaded_map.getVertex(vertex_id);
    cv::Mat image;
    if (reloaded_map.getFrameResource(
            vertex, 0u, backend::ResourceType::kRawImage, &image)) {
      CHECK(!image.empty());
      ++reloaded_image_count;
    }
  }
  CHECK_EQ(reloaded_image_count, raw_image_resource_count);

  LOG(INFO) << "Created and reloaded VI-Map with "
            << reloaded_map.numVertices() << " vertices and "
            << reloaded_map.numEdges() << " VIWLS edges at "
            << FLAGS_output_map << " with " << reloaded_image_count
            << " raw image resources";
  return 0;
}
