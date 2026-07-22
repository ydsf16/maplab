cmake_minimum_required(VERSION 2.8)
project(sensor_recorder_importer)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cs_add_executable(sensor_recorder_to_vimap
                  src/sensor-recorder-to-vimap.cc)

cs_install()
cs_export()
