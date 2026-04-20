cmake_minimum_required(VERSION 3.0.2)
project(ur3_handover)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/brain_node.py
    scripts/client_camera.py
    scripts/semantic_grasp_controller.py
    scripts/robotiq_gripper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
