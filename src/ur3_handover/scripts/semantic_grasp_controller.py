#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Grasp Controller (AnyGrasp 6D Pose 接收版)
"""

import os, sys, math, numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_from_euler
from moveit_commander import MoveGroupCommander, roscpp_initialize
import tf2_ros, tf2_geometry_msgs

from robotiq_gripper import RobotiqGripper

class SemanticGraspController:
    def __init__(self):
        # ---- 1. 參數 ----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.move_group  = rospy.get_param("~move_group", "manipulator")

        self.tcp_offset    = float(rospy.get_param("~tcp_offset",    0.18))
        self.grasp_depth   = float(rospy.get_param("~grasp_depth",   0.05))
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))

        self.retreat_up_height = float(rospy.get_param("~retreat_up_height", 0.14))  # 待機點高度

        self.eef_step  = float(rospy.get_param("~eef_step",  0.02))
        self.vel_scale = float(rospy.get_param("~vel_scale", 0.10))
        self.acc_scale = float(rospy.get_param("~acc_scale", 0.10))

        # 固定放置接觸點（法蘭位置）
        self.final_xyz = [0.2401, 0.1751, 0.185]

        # 夾爪
        self.grip_ip    = rospy.get_param("~gripper_ip",   "192.168.86.7")
        self.grip_port  = int(rospy.get_param("~gripper_port", 63352))
        self.grip_speed = 100
        self.grip_force = 80

        # ---- 2. 初始化 ----
        roscpp_initialize([])
        self.group = MoveGroupCommander(self.move_group)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.group.set_num_planning_attempts(5)
        self.group.set_planning_time(10.0)

        self.g = RobotiqGripper()
        self.init_gripper()

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # 訂閱 AnyGrasp 姿態
        self.pose_sub = rospy.Subscriber(
            "/anygrasp/target_pose", PoseStamped, self.cb_anygrasp_pose, queue_size=1)
        # 隨時回待機點的指令（另開終端機 rostopic pub /semantic_grasp/go_home std_msgs/String "go" -1）
        rospy.Subscriber("/semantic_grasp/go_home", String, self._cb_go_home)

        rospy.loginfo("[p2p] 啟動完成，等待目標姿態...")
        rospy.loginfo("[p2p] 隨時回待機點: rostopic pub /semantic_grasp/go_home std_msgs/String 'go' -1")

    # ------------------------------------------------------------------ #
    #  ROS callbacks                                                       #
    # ------------------------------------------------------------------ #
    def cb_anygrasp_pose(self, msg: PoseStamped):
        rospy.loginfo(f"[p2p] 收到目標！Frame: {msg.header.frame_id}")
        try:
            T = self.tfbuf.lookup_transform(
                self.base_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            ps_base = tf2_geometry_msgs.do_transform_pose(msg, T)
        except Exception as e:
            rospy.logerr(f"TF 失敗: {e}"); return

        # 座標軸對齊：旋轉 Y 軸 -90°，讓 UR3 Z 軸對齊 AnyGrasp X 軸
        q_orig = [ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                  ps_base.pose.orientation.z, ps_base.pose.orientation.w]
        q_to_ur3 = quaternion_from_euler(0, -math.pi/2, 0)
        q_final  = quaternion_multiply(q_orig, q_to_ur3)
        ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
            ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_final

        # 安全檢查：夾爪指向上方 → 翻轉防撞桌
        ee_z = self.get_ee_z_axis_in_base(ps_base)
        if ee_z[2] > 0:
            rospy.logwarn("偵測到倒立姿態，自動修正方向...")
            q_fix = quaternion_from_euler(math.pi, 0, 0)
            q_safe = quaternion_multiply(q_final, q_fix)
            ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
                ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_safe

        self.run_once(ps_base)

    def _cb_go_home(self, msg):
        rospy.loginfo("[p2p] 收到回待機點指令...")
        self.go_to_ready_pose()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _normalize_start_state(self):
        """將超出 [-2π, 2π] 的關節值 wrap 回範圍內，避免 MoveIt 拒絕規劃"""
        state = self.group.get_current_state()
        positions = list(state.joint_state.position)
        changed = False
        for i, val in enumerate(positions):
            if val > math.pi * 2:
                positions[i] = val - math.pi * 2
                changed = True
            elif val < -math.pi * 2:
                positions[i] = val + math.pi * 2
                changed = True
        if changed:
            state.joint_state.position = positions
            self.group.set_start_state(state)
            rospy.logwarn("[p2p] 關節超出界限，已自動 normalize 起始狀態")
        else:
            self.group.set_start_state_to_current_state()

    def plan_execute_cartesian_to(self, target_pose):
        self._normalize_start_state()
        wp = target_pose.pose if isinstance(target_pose, PoseStamped) else target_pose
        plan, fraction = self.group.compute_cartesian_path([wp], self.eef_step, True)
        if fraction < 0.99:
            rospy.logwarn(f"笛卡兒路徑不全: {fraction:.2f}")
            return False
        return self.group.execute(plan, wait=True)

    def joint_plan_execute(self, ps_target, label="目標點"):
        """關節規劃 + 執行，含錯誤 log"""
        self._normalize_start_state()
        self.group.set_pose_target(ps_target)
        success, plan, _, error_code = self.group.plan()
        if not success:
            rospy.logerr(f"[p2p] {label} 規劃失敗: {error_code}")
            return False
        ok = self.group.execute(plan, wait=True)
        self.group.stop(); self.group.clear_pose_targets()
        if not ok:
            rospy.logerr(f"[p2p] {label} 執行失敗")
        return ok

    def get_ee_z_axis_in_base(self, pose_stamped):
        q = pose_stamped.pose.orientation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def make_pose_stamped(self, xyz, orientation):
        """從 xyz + 四元數 Quaternion 物件建立 PoseStamped"""
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation = orientation
        return ps

    def make_pose_stamped_from_xyz_rpy(self, xyz, rpy_deg):
        qx, qy, qz, qw = quaternion_from_euler(
            math.radians(rpy_deg[0]), math.radians(rpy_deg[1]), math.radians(rpy_deg[2]))
        ps = PoseStamped(); ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
        return ps

    def init_gripper(self):
        try:
            self.g.connect(self.grip_ip, self.grip_port); self.g.activate()
            rospy.sleep(1.0); self.g.move_and_wait_for_pos(0, 100, 80)
        except: self.g = None

    # ------------------------------------------------------------------ #
    #  待機點（放置區正上方）                                               #
    # ------------------------------------------------------------------ #
    def go_to_ready_pose(self):
        ready_xyz = [self.final_xyz[0], self.final_xyz[1],
                     self.final_xyz[2] + self.retreat_up_height]
        ps_ready = self.make_pose_stamped_from_xyz_rpy(ready_xyz, [180.0, 0.0, 0.0])
        rospy.loginfo(f"[p2p] 前往待機點 {ready_xyz} ...")
        if self.joint_plan_execute(ps_ready, "待機點"):
            rospy.loginfo("[p2p] ✅ 已到達待機點，等待下一次任務")
        else:
            rospy.logerr("[p2p] 無法到達待機點，請手動確認安全")

    def _prompt_ready(self):
        """任務結束（無論成功/失敗）都詢問是否回待機點"""
        try:
            print("\n==================================================")
            ans = input("[p2p] 任務結束。[Enter] 回待機點 | [n] 不移動：")
            print("==================================================\n")
            if ans.lower() != 'n':
                self.go_to_ready_pose()
        except EOFError:
            self.go_to_ready_pose()

    # ------------------------------------------------------------------ #
    #  核心流程                                                             #
    # ------------------------------------------------------------------ #
    def run_once(self, ps_target):
        try:
            self._grasp_and_place(ps_target)
        finally:
            self._prompt_ready()

    def _grasp_and_place(self, ps_target):
        # ── 幾何計算 ──────────────────────────────────────────────────── #
        target_xyz = np.array([ps_target.pose.position.x,
                                ps_target.pose.position.y,
                                ps_target.pose.position.z])
        ee_z = self.get_ee_z_axis_in_base(ps_target)  # 抓取方向軸

        rospy.loginfo("[p2p] object_surface = (%.3f, %.3f, %.3f)", *target_xyz)

        # 物體表面往內 grasp_depth → 再退 tcp_offset = 法蘭位置
        grasp_xyz = target_xyz + ee_z * self.grasp_depth - ee_z * self.tcp_offset
        # Pre-Grasp = 法蘭位置沿抓取軸退 approach_dist
        pre_xyz   = grasp_xyz - ee_z * self.approach_dist

        ps_grasp = self.make_pose_stamped(grasp_xyz, ps_target.pose.orientation)
        ps_pre   = self.make_pose_stamped(pre_xyz,   ps_target.pose.orientation)

        # 放置：pre_place 沿同一抓取軸退 approach_dist（保持抓取姿態，不強制垂直）
        final_xyz     = np.array(self.final_xyz)
        pre_place_xyz = final_xyz - ee_z * self.approach_dist
        ps_place      = self.make_pose_stamped(final_xyz,     ps_target.pose.orientation)
        ps_pre_place  = self.make_pose_stamped(pre_place_xyz, ps_target.pose.orientation)

        # ── 動作 A：Pre-Grasp（規劃 → 確認迴圈）────────────────────────── #
        while True:
            self._normalize_start_state()
            self.group.set_pose_target(ps_pre)
            success, plan, _, error_code = self.group.plan()
            if not success:
                rospy.logerr(f"[p2p] Pre-Grasp 規劃失敗: {error_code}")
                return

            try:
                print("\n==================================================")
                ans = input("⚠️ [安全鎖] 軌跡已顯示在 RViz！\n"
                            "  [Enter] 執行  |  [r] 重新規劃  |  [n] 取消：")
                print("==================================================\n")
            except EOFError:
                ans = ""

            if ans.lower() == 'n':
                return
            elif ans.lower() == 'r':
                rospy.loginfo("[p2p] 重新規劃同一抓取姿態...")
                continue
            else:
                break

        if not self.group.execute(plan, wait=True):
            rospy.logerr("[p2p] Pre-Grasp 執行失敗"); return
        self.group.stop(); self.group.clear_pose_targets()

        # ── 動作 B：Cartesian 前進至 Grasp ──────────────────────────────── #
        rospy.loginfo("[p2p] 執行直線抓取...")
        if not self.plan_execute_cartesian_to(ps_grasp): return

        # ── 動作 C：夾緊 ────────────────────────────────────────────────── #
        if self.g: self.g.move_and_wait_for_pos(180, self.grip_speed, self.grip_force)
        rospy.loginfo("[p2p] 夾爪夾緊")

        try:
            input("[p2p] 已夾取，按 [Enter] 繼續放置，[n] 取消：")
        except EOFError:
            pass

        # ── 動作 D：Cartesian 垂直抬升 10 ──────────────────────────────── #
        rospy.loginfo("[p2p] 垂直抬升 10...")
        lift_pose = self.make_pose_stamped(
            [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.1,
            ps_grasp.pose.orientation)
        if not self.plan_execute_cartesian_to(lift_pose): return

        # ── 動作 E：關節規劃 → Pre-Place（保持抓取姿態）─────────────────── #
        rospy.loginfo("[p2p] 關節規劃前往放置前置點...")
        if not self.joint_plan_execute(ps_pre_place, "放置前置點"): return

        # ── 動作 F：Cartesian 沿抓取軸前進至放置接觸點 ──────────────────── #
        rospy.loginfo("[p2p] 沿抓取方向降下至放置點...")
        if not self.plan_execute_cartesian_to(ps_place): return

        # ── 動作 G：張開夾爪 ────────────────────────────────────────────── #
        if self.g: self.g.move_and_wait_for_pos(0, self.grip_speed, self.grip_force)
        rospy.loginfo("[p2p] 夾爪張開，物體已放置")

        # ── 動作 H：Cartesian 後退至 Pre-Place ──────────────────────────── #
        rospy.loginfo("[p2p] 退出放置點...")
        self.plan_execute_cartesian_to(ps_pre_place)  # 失敗不中止，繼續到 finally

        rospy.loginfo("[p2p] 🎉 任務完成！")


if __name__ == "__main__":
    rospy.init_node("semantic_grasp_controller")
    app = SemanticGraspController(); rospy.spin()
