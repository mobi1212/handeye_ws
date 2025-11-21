#!/usr/bin/env python3
import sys, math, rospy
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from robotiq_gripper import RobotiqGripper # 確認此模組已放在 PYTHONPATH 或同資料夾

# =========================
# 全域參數（可被 ROS param 覆蓋）
# =========================
CONFIG = {
    # 幾何與規劃
    "approach_z": 0.10,          # 上方 10 cm
    "lift_z":     0.12,          # 抬升 12 cm
    "eef_step":   0.01,          # Cartesian 插值步長 (m)
    "vel_scale":  0.10,          # 手臂速度比例 0~1
    "acc_scale":  0.10,          # 手臂加速度比例 0~1
    "ee_rpy_deg": (0.0, 180.0, 0.0),  # 末端固定姿態 (roll, pitch, yaw) 度

    # 夾爪（TCP / URCap Socket）
    "gripper_ip": "192.168.86.7",
    "gripper_port": 63352,
    "gripper_speed": 100,        # 0~255
    "gripper_force": 80,         # 0~255

    # 夾爪等待時間縮放（無回授時的經驗公式）
    "wait_min": 0.2,             # 下限
    "wait_max": 1.2,             # 上限
    "wait_base": 0.15,           # 基礎常數
    "wait_k": 0.9,               # 與速度反比的係數
    "wait_move_extra": 0.2,      # 依行程比例加成的偏移
}

def cfg(name):  # 允許用 ROS param 覆蓋：/pick_object/<key>
    ns = f"/pick_object/{name}"
    val = rospy.get_param(ns, CONFIG[name])
    CONFIG[name] = val
    return val

# =============== 夾爪控制（TCP）===============
class TCPGripper:
    def __init__(self, host="192.168.86.7", port=63352):
        self.host = host
        self.port = port
        self.g = RobotiqGripper()
        self._speed = 128
        self._force = 64

    def connect(self, timeout=5.0):
        try:
            self.g.connect(self.host, self.port)
            self.g.activate()
            rospy.loginfo("[Gripper] activated, waiting...")
            rospy.sleep(3.0)  # 若第一次上電可改 4~5
        except Exception as e:
            rospy.logerr(f"[Gripper] 連線/啟動失敗：{e}")
            raise

    def set_speed(self, v):  # 0~255
        self._speed = max(0, min(255, int(v)))

    def set_force(self, f):  # 0~255
        self._force = max(0, min(255, int(f)))

    def open(self):
        self.g.move_and_wait_for_pos(position=0,   speed=self._speed, force=self._force)

    def close(self, pos=255):
        self.g.move_and_wait_for_pos(position=max(0, min(255, int(pos))),
                                     speed=self._speed, force=self._force)

# =============== 等待時間（依速度/行程縮放）===============
def grip_wait(speed, last_pos=None, target_pos=None):
    wait_min  = cfg("wait_min")
    wait_max  = cfg("wait_max")
    wait_base = cfg("wait_base")
    wait_k    = cfg("wait_k")
    extra     = cfg("wait_move_extra")

    base = max(wait_min, min(wait_max, wait_base + wait_k * (100.0 / max(1, speed))))
    if last_pos is not None and target_pos is not None:
        span = abs(int(target_pos) - int(last_pos)) / 255.0
        base *= (span + extra)
    rospy.sleep(base)

# =============== MoveIt 笛卡兒移動 ===============
def cartesian_move(arm, waypoints):
    eef_step = cfg("eef_step")
    arm.set_start_state_to_current_state()
    # 第三個參數是 jump_threshold，要給 True，否則會被當 1.0 rad
    plan, fraction = arm.compute_cartesian_path(waypoints, eef_step, True)
    if fraction < 0.99:
        rospy.logwarn(f"Cartesian path incomplete, fraction={fraction:.2f}")
        return False
    rospy.loginfo(f"Executing Cartesian path, fraction={fraction:.2f}")
    arm.execute(plan, wait=True)
    arm.stop(); arm.clear_pose_targets()
    return True

# =============== 主流程 ===============
def main():
    if len(sys.argv) < 4:
        print("用法: rosrun my_pickplace pick_object.py <x> <y> <z>")
        return
    x, y, z = map(float, sys.argv[1:4])

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("pick_object", anonymous=True)

    arm = moveit_commander.MoveGroupCommander("manipulator")
    arm.set_max_velocity_scaling_factor(cfg("vel_scale"))
    arm.set_max_acceleration_scaling_factor(cfg("acc_scale"))

    # 固定末端姿態
    r_deg, p_deg, y_deg = cfg("ee_rpy_deg")
    qx, qy, qz, qw = quaternion_from_euler(
        math.radians(r_deg), math.radians(p_deg), math.radians(y_deg)
    )

    def make_pose(px, py, pz):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = px, py, pz
        pose.orientation.x = qx; pose.orientation.y = qy
        pose.orientation.z = qz; pose.orientation.w = qw
        return pose

    rospy.loginfo(f"目標: ({x:.3f}, {y:.3f}, {z:.3f})")

    # 夾爪初始化
    gripper = TCPGripper(host=cfg("gripper_ip"), port=cfg("gripper_port"))
    gripper.connect()
    speed = int(cfg("gripper_speed"))
    force = int(cfg("gripper_force"))
    gripper.set_speed(speed)
    gripper.set_force(force)

    try:
        # Step1: Approach（用 set_pose_target + go，比直接 go(pose) 穩）
        approach_z = cfg("approach_z")
        rospy.loginfo("Step1: Approach...")
        arm.set_start_state_to_current_state()
        arm.set_pose_target(make_pose(x, y, z + approach_z))
        if not arm.go(wait=True):
            rospy.logwarn("Approach failed"); return
        arm.stop(); arm.clear_pose_targets()
        rospy.loginfo("Step1 success")

        # Step2: Descend（笛卡兒）
        rospy.loginfo("Step2: Descend...")
        if not cartesian_move(arm, [make_pose(x, y, z)]): return
        rospy.loginfo("Step2 success")

        # 夾取
        rospy.loginfo("Gripper: close")
        gripper.close(pos=150)
        grip_wait(speed=speed)

        # Step3: Lift（笛卡兒）
        lift_z = cfg("lift_z")
        rospy.loginfo("Step3: Lift...")
        if not cartesian_move(arm, [make_pose(x, y, z + lift_z)]): return
        rospy.loginfo("Step3 success")

        gripper.open()

        rospy.loginfo("流程完成")

    finally:
        moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
