#!/usr/bin/env python3
import os
import sys
import rospy
import moveit_commander
import yaml
from geometry_msgs.msg import Pose

SAVE_DIR = os.path.expanduser("~/handeye_ws/poses")  # 儲存路徑

def find_next_filename():
    """找下一個可用的 poseX 檔名，補齊中間缺號"""
    existing = []
    if os.path.exists(SAVE_DIR):
        for f in os.listdir(SAVE_DIR):
            if f.startswith("pose") and f.endswith(".yaml"):
                try:
                    num = int(f[4:-5])  # poseX.yaml → X
                    existing.append(num)
                except ValueError:
                    continue
    else:
        os.makedirs(SAVE_DIR)

    existing = sorted(existing)
    candidate = 1
    for n in existing:
        if n != candidate:
            return f"pose{candidate}.yaml"
        candidate += 1
    return f"pose{candidate}.yaml"

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("save_current_pose", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("manipulator")

    # 取得當前 pose
    current_pose = arm.get_current_pose().pose

    # 找到下一個檔名
    filename = os.path.join(SAVE_DIR, find_next_filename())

    # 轉成 dict
    pose_dict = {
        "position": {
            "x": current_pose.position.x,
            "y": current_pose.position.y,
            "z": current_pose.position.z,
        },
        "orientation": {
            "x": current_pose.orientation.x,
            "y": current_pose.orientation.y,
            "z": current_pose.orientation.z,
            "w": current_pose.orientation.w,
        }
    }

    # 存成 YAML
    with open(filename, "w") as f:
        yaml.dump(pose_dict, f)

    rospy.loginfo(f"Pose saved to {filename}")

if __name__ == "__main__":
    main()
	
