cmake_minimum_required(VERSION 3.0.2)
project(lidar_example_pkg)

## ROS 및 Catkin 패키지 의존성 설정
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
)

## Catkin 패키지 설정
catkin_package(
  CATKIN_DEPENDS rospy roscpp sensor_msgs visualization_msgs pcl_ros pcl_conversions
)

## 빌드 경로 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  /opt/ros/noetic/include  # ROS 헤더 경로 추가
  /usr/include/pcl-1.10    # PCL 헤더 경로 추가
)

## Python 스크립트 설치
catkin_install_python(
  PROGRAMS
    scripts/22.py
    scripts/cuda2.py
    scripts/lidar_processor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## C++ 파일 컴파일 설정
add_executable(lidar_processor_cpp src/22_with_roi.cpp)

## C++ 실행 파일에 라이브러리 링크
target_link_libraries(lidar_processor_cpp
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  pcl_common
  pcl_io
  pcl_filters
  pcl_segmentation
  pcl_search
)

## C++ 바이너리 설치 설정
install(
  TARGETS lidar_processor_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
