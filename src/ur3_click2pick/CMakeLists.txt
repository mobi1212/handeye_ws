cmake_minimum_required(VERSION 3.0.2)
project(ur3_click2pick)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()

catkin_install_python(
  PROGRAMS scripts/click_to_pick_cv.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
