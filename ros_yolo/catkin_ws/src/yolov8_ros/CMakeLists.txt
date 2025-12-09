cmake_minimum_required(VERSION 3.0.2)
project(yolov8_ros)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
    FILES
    RecognitionObject.msg
    RecognitionObjectArray.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    std_msgs yolov8_ros
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
   CATKIN_DEPENDS rospy std_msgs message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)
