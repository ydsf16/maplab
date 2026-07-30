#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>

DEFINE_string(maps, "", "Comma-separated input VI-Map folders.");
DEFINE_string(output, "", "Output folder for the joint multi-mission VI-Map.");

namespace {
std::vector<std::string> split(const std::string& value) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) result.emplace_back(item);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  CHECK(!FLAGS_maps.empty()) << "--maps is required";
  CHECK(!FLAGS_output.empty()) << "--output is required";
  const std::vector<std::string> map_folders = split(FLAGS_maps);
  CHECK_GE(map_folders.size(), 2u) << "at least two maps are required";
  vi_map::VIMap joint_map;
  for (const std::string& map_folder : map_folders) {
    vi_map::VIMap source_map;
    CHECK(vi_map::serialization::loadMapFromFolder(map_folder, &source_map))
        << "failed to load " << map_folder;
    CHECK(joint_map.mergeAllMissionsFromMap(source_map))
        << "failed to merge " << map_folder;
  }
  CHECK(vi_map::checkMapConsistency(joint_map));
  backend::SaveConfig save_config;
  save_config.overwrite_existing_files = true;
  CHECK(vi_map::serialization::saveMapToFolder(
      FLAGS_output, save_config, &joint_map));
  LOG(INFO) << "Merged " << map_folders.size() << " maps into "
            << FLAGS_output << " with " << joint_map.numMissions()
            << " missions.";
  return 0;
}
