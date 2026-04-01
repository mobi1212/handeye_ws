#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-to-Pick (含 TCP offset 修正的版本)

動作順序：
  1) 物體位置 + 姿態 -> 算出 ee_z
  2) 用 tcp_offset 修正出「法蘭應到達的 grasp_xyz」
  3) 沿 -ee_z 方向拉出 pre-grasp
  4) go(pre-grasp) 關節空間
  5) Cartesian 從 pre-grasp 前進到 grasp
  6) 夾緊
  7) 統一往 base_link +Z 方向抬升（retreat up）
  8) 移動到固定放置姿勢 final pose
  9) 開夾
 10) 再往上抬一小段（避免撞到物體）
"""

import os, sys, math, numpy as np
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler, quaternion_matrix
from moveit_commander import MoveGroupCommander, roscpp_initialize

from robotiq_gripper import RobotiqGripper


class PoseToPick:
    def __init__(self):
        # ---- 參數 ----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.move_group  = rospy.get_param("~move_group", "manipulator")

        # 使用者輸入：物體位置 + 夾爪姿態（RPY, deg）
        self.target_xyz      = rospy.get_param("~target_xyz",      [0.4, 0.0, 0.05])
        self.target_rpy_deg  = rospy.get_param("~target_rpy_deg",  [0.0, 180.0, 0.0])

        # TCP 偏移（夾爪指尖中心到 MoveIt end-effector 的距離, m）
        self.tcp_offset = float(rospy.get_param("~tcp_offset", 0.16))

        # pre-grasp 距離（沿 -ee_z 方向），retreat_dist 現在僅保留參數，不再使用
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))  # m
        self.retreat_dist  = float(rospy.get_param("~retreat_dist",  0.10))  # m（未使用）

        # 統一往上抬升的高度（grasp 之後）
        self.retreat_up_height    = float(rospy.get_param("~retreat_up_height", 0.10))  # m
        # 放開之後再往上抬升一小段
        self.post_open_up_height  = float(rospy.get_param("~post_open_up_height", 0.1))  # m

        self.eef_step    = float(rospy.get_param("~eef_step",   0.01))
        self.vel_scale   = float(rospy.get_param("~vel_scale",  0.10))
        self.acc_scale   = float(rospy.get_param("~acc_scale",  0.10))

        # 固定放置位置與姿態（final drop pose）
        # 位置與姿態若之後想改，可改成 rosparam
        self.final_xyz     = [0.0, 0.4, 0.17]
        self.final_rpy_deg = [180.0, 0.0, 30.0]

        # 夾爪 TCP
        self.grip_ip     = rospy.get_param("~gripper_ip",   "192.168.86.7")
        self.grip_port   = int(rospy.get_param("~gripper_port", 63352))
        self.grip_speed  = int(rospy.get_param("~gripper_speed", 100))  # 0~255
        self.grip_force  = int(rospy.get_param("~gripper_force", 80))   # 0~255
        self.wait_min    = float(rospy.get_param("~wait_min",  0.2))
        self.wait_max    = float(rospy.get_param("~wait_max",  1.2))
        self.wait_base   = float(rospy.get_param("~wait_base", 0.15))
        self.wait_k      = float(rospy.get_param("~wait_k",    0.9))
        self.wait_move_extra = float(rospy.get_param("~wait_move_extra", 0.2))

        # ---- MoveIt ----
        roscpp_initialize([])
        self.group = MoveGroupCommander(self.move_group)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)

        rospy.loginfo("[p2p] base=%s group=%s", self.base_frame, self.move_group)
        rospy.loginfo("[p2p] target_xyz(object) = (%.3f, %.3f, %.3f)", *self.target_xyz)
        rospy.loginfo("[p2p] target_rpy_deg     = (%.1f, %.1f, %.1f)", *self.target_rpy_deg)
        rospy.loginfo("[p2p] tcp_offset=%.3f approach=%.3f retreat_up=%.3f post_open_up=%.3f",
                      self.tcp_offset, self.approach_dist,
                      self.retreat_up_height, self.post_open_up_height)

        # ---- Gripper ----
        self.g = RobotiqGripper()
        self._last_grip_pos = None
        self.init_gripper()

    # ---------- gripper helpers ----------
    def init_gripper(self):
        try:
            rospy.loginfo("[grip] connecting %s:%d ...", self.grip_ip, self.grip_port)
            self.g.connect(self.grip_ip, self.grip_port)
            self.g.activate()
            rospy.loginfo("[grip] activated")
            rospy.sleep(2.0)
            self.g.move_and_wait_for_pos(position=0,
                                         speed=self.grip_speed,
                                         force=self.grip_force)
            self._last_grip_pos = 0
            rospy.loginfo("[grip] opened to 0")
        except Exception as e:
            rospy.logwarn("[grip] connect/activate failed: %s", e)
            self.g = None
            self._last_grip_pos = None

    def grip_wait(self, speed, last_pos=None, target_pos=None):
        base = max(self.wait_min,
                   min(self.wait_max,
                       self.wait_base + self.wait_k * (100.0 / max(1, speed))))
        span = None
        if last_pos is not None and target_pos is not None:
            span = abs(int(target_pos) - int(last_pos)) / 255.0
            base *= (span + self.wait_move_extra)
        rospy.loginfo("[grip] wait %.3fs (speed=%d, span=%s)",
                      base, speed, f"{span:.2f}" if span is not None else "NA")
        rospy.sleep(base)

    # ---------- motion helpers ----------
    def plan_execute_cartesian_to(self, target_pose):
        """從目前末端姿態，笛卡兒一路走到 target_pose（單一 waypoint）"""
        self.group.set_start_state_to_current_state()

        from geometry_msgs.msg import Pose
        if isinstance(target_pose, PoseStamped):
            wp = target_pose.pose
        else:
            wp = target_pose

        waypoints = [wp]
        plan, fraction = self.group.compute_cartesian_path(
            waypoints,
            self.eef_step,
            True
        )
        rospy.loginfo("[p2p] cartesian fraction=%.3f", fraction)
        if fraction < 0.99:
            rospy.logwarn("[p2p] Cartesian incomplete, fraction=%.2f", fraction)
            return False
        pts = len(plan.joint_trajectory.points) if hasattr(plan, "joint_trajectory") else -1
        rospy.loginfo("[p2p] executing cartesian (pts=%d)", pts)
        ok = self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.loginfo("[p2p] cartesian exec=%s", "OK" if ok else "FAIL")
        return ok

    def make_pose_stamped_from_xyz_rpy(self, xyz, rpy_deg):
        x, y, z = xyz
        r_deg, p_deg, y_deg = rpy_deg
        qx, qy, qz, qw = quaternion_from_euler(
            math.radians(r_deg),
            math.radians(p_deg),
            math.radians(y_deg)
        )
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def get_ee_z_axis_in_base(self, pose_stamped):
        """從 PoseStamped 的 quaternion 算出末端 Z 軸在 base 座標下的方向向量"""
        q = pose_stamped.pose.orientation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])  # 4x4
        z_axis = M[0:3, 2]  # 第三欄是 Z 軸
        norm = np.linalg.norm(z_axis)
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0])
        return z_axis / norm

    # ---------- main logic ----------
    def run_once(self):
        # 1) 用物體位置 + 姿態先算出 ee_z 方向
        ps_tmp = self.make_pose_stamped_from_xyz_rpy(self.target_xyz, self.target_rpy_deg)
        ee_z = self.get_ee_z_axis_in_base(ps_tmp)

        rospy.loginfo("[p2p] object_xyz = (%.3f, %.3f, %.3f)", *self.target_xyz)
        rospy.loginfo("[p2p] ee_z in base = [%.3f, %.3f, %.3f]", ee_z[0], ee_z[1], ee_z[2])

        # 2) 用 tcp_offset 把「物體位置」轉成「法蘭應該到的位置」
        grasp_xyz = np.array(self.target_xyz) - ee_z * self.tcp_offset
        ps_grasp = self.make_pose_stamped_from_xyz_rpy(grasp_xyz.tolist(), self.target_rpy_deg)

        gx, gy, gz = grasp_xyz
        rospy.loginfo("[p2p] grasp_xyz(flange) @ base (%.3f, %.3f, %.3f)", gx, gy, gz)

        # 3) 沿 -ee_z 方向拉出 Pre-Grasp
        approach_vec = -ee_z
        pre_xyz = grasp_xyz + approach_vec * self.approach_dist
        ps_pre = self.make_pose_stamped_from_xyz_rpy(pre_xyz.tolist(), self.target_rpy_deg)

        rospy.loginfo("[p2p] pre-grasp @ base (%.3f, %.3f, %.3f)",
                      pre_xyz[0], pre_xyz[1], pre_xyz[2])

        # 4) 先用 go() 走到 pre-grasp（關節空間規劃）
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(ps_pre)
        ok = self.group.go(wait=True)
        rospy.loginfo("[p2p] go(pre-grasp) = %s", "OK" if ok else "FAIL")
        self.group.stop()
        self.group.clear_pose_targets()
        if not ok:
            rospy.logerr("[p2p] failed to reach pre-grasp, abort")
            return

        # 5) 再用笛卡兒路徑從 pre-grasp 前進到 grasp
        if not self.plan_execute_cartesian_to(ps_grasp):
            rospy.logerr("[p2p] cartesian to grasp failed, abort")
            return

        # 6) 夾爪關閉
        if self.g:
            target_pos = 180
            rospy.loginfo("[grip] close -> %d (spd=%d force=%d)",
                          target_pos, self.grip_speed, self.grip_force)
            try:
                self.g.move_and_wait_for_pos(position=target_pos,
                                             speed=self.grip_speed,
                                             force=self.grip_force)
            except Exception as e:
                rospy.logwarn("[grip] close error: %s", e)
            self.grip_wait(self.grip_speed, self._last_grip_pos, target_pos)
            self._last_grip_pos = target_pos
        else:
            rospy.logwarn("[grip] not connected, skip gripping")

        # 7) 統一往 base_link +Z 方向抬升，而不是沿著夾爪 Z 軸
        retreat_xyz = [
            grasp_xyz[0],
            grasp_xyz[1],
            grasp_xyz[2] + self.retreat_up_height
        ]
        ps_retreat_up = self.make_pose_stamped_from_xyz_rpy(
            retreat_xyz,
            self.target_rpy_deg   # 姿態保持抓取時的姿態
        )

        rospy.loginfo("[p2p] retreat(up) @ base (%.3f, %.3f, %.3f)",
                      retreat_xyz[0], retreat_xyz[1], retreat_xyz[2])

        if not self.plan_execute_cartesian_to(ps_retreat_up):
            rospy.logwarn("[p2p] retreat(up) cartesian failed")

        # 8) 直接移動到固定放置位置與姿態
        ps_final = self.make_pose_stamped_from_xyz_rpy(self.final_xyz, self.final_rpy_deg)
        rospy.loginfo("[p2p] retreat(final) @ base (%.3f, %.3f, %.3f)",
                      self.final_xyz[0], self.final_xyz[1], self.final_xyz[2])

        if not self.plan_execute_cartesian_to(ps_final):
            rospy.logwarn("[p2p] retreat(final) cartesian failed")

        # 9) 放開物體
        if self.g:
            rospy.loginfo("[grip] open -> 0")
            try:
                self.g.move_and_wait_for_pos(position=0,
                                             speed=self.grip_speed,
                                             force=self.grip_force)
            except Exception as e:
                rospy.logwarn("[grip] open error: %s", e)
            self.grip_wait(self.grip_speed, self._last_grip_pos, 0)
            self._last_grip_pos = 0
        else:
            rospy.logwarn("[grip] not connected, skip opening")

        # 10) 放開後再往上抬一小段，避免碰到剛放下的物體
        post_up_xyz = [
            self.final_xyz[0],
            self.final_xyz[1],
            self.final_xyz[2] + self.post_open_up_height
        ]
        ps_post_up = self.make_pose_stamped_from_xyz_rpy(post_up_xyz, self.final_rpy_deg)

        rospy.loginfo("[p2p] post-open up @ base (%.3f, %.3f, %.3f)",
                      post_up_xyz[0], post_up_xyz[1], post_up_xyz[2])

        if not self.plan_execute_cartesian_to(ps_post_up):
            rospy.logwarn("[p2p] post-open up cartesian failed")

        rospy.loginfo("[p2p] DONE")


def main():
    rospy.init_node("pose_to_pick")
    app = PoseToPick()
    rospy.sleep(1.0)
    app.run_once()


if __name__ == "__main__":
    main()
