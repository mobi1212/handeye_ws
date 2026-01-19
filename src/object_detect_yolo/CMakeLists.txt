cmake_minimum_required(VERSION 3.0.2)
project(object_detect_yolo)

## 找 catkin 套件
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    image_transport
    rospy
    sensor_msgs
    std_msgs
    geometry_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## 安裝 Python 節點（ROS 版 detect_node.py 放在 scripts/）
catkin_install_python(
  PROGRAMS
    scripts/detect_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

## 如果之後有要加測試再打開
# catkin_add_nosetests(test)

