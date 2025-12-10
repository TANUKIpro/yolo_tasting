cmake_minimum_required(VERSION 3.0.2)
project(video_publisher)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)

catkin_package()

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install scripts
install(PROGRAMS scripts/video_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
