cmake_minimum_required(VERSION 3.0.2)
project(dolidar_ysm)

# C++ 표준 설정 (ROS Noetic 이상에서는 기본적으로 C++14 이상 지원)
add_compile_options(-std=c++14)

# ROS 패키지 의존성 찾기
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
)

# Open3D 및 Eigen 의존성 찾기
find_package(Open3D REQUIRED)
find_package(Eigen3 REQUIRED)

# catkin_package 설정
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs
  DEPENDS Open3D Eigen3
)

# include 디렉토리 설정
include_directories(
  include
  ${catkin_INCLUDE_DIRS}      # ROS 헤더 경로
  ${Open3D_INCLUDE_DIRS}      # Open3D 헤더 경로
  ${EIGEN3_INCLUDE_DIRS}      # Eigen 헤더 경로
)

# 실행 파일 추가
add_executable(lidar_process_node src/lidar_process_node.cpp)

# Open3D와 catkin 링킹
target_link_libraries(lidar_process_node
  ${catkin_LIBRARIES}
  ${Open3D_LIBRARIES}
)
