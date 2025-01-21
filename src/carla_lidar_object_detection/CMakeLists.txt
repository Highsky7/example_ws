cmake_minimum_required(VERSION 3.0.2)
project(carla_lidar_object_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
)

find_package(PCL REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(lidar_object_detection_node src/lidar_object_detection_node.cpp)
target_link_libraries(lidar_object_detection_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
