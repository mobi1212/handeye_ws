#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Grasp Controller (AnyGrasp 6D Pose 接收版)
完全移植 PoseToPick (YOLO) 邏輯版本
"""

import os, sys, math, numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_from_euler
from moveit_commander import MoveGroupCommander, roscpp_initialize
import tf2_ros, tf2_geometry_msgs

from robotiq_gripper import RobotiqGripper

class SemanticGraspController:
    def __init__(self):
        # ---- 1. 參數 (完全對齊 YOLO 版本) ----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.move_group  = rospy.get_param("~move_group", "manipulator")

        self.tcp_offset    = float(rospy.get_param("~tcp_offset", 0.18))
        self.grasp_depth   = float(rospy.get_param("~grasp_depth", 0.04)) 
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))
        
        self.retreat_up_height   = float(rospy.get_param("~retreat_up_height", 0.15)) 
        self.post_open_up_height = float(rospy.get_param("~post_open_up_height", 0.1)) 

        self.eef_step    = float(rospy.get_param("~eef_step",   0.02))
        self.vel_scale   = float(rospy.get_param("~vel_scale",  0.10))
        self.acc_scale   = float(rospy.get_param("~acc_scale",  0.10))

        # 固定放置位置
        self.final_xyz     = [0.2, 0.1, 0.185]
        self.final_rpy_deg = [180.0, 0.0, 0.0]

        # 夾爪
        self.grip_ip     = rospy.get_param("~gripper_ip",   "192.168.86.7")
        self.grip_port   = int(rospy.get_param("~gripper_port", 63352))
        self.grip_speed  = 100
        self.grip_force  = 80

        # ---- 2. 初始化 ----
        roscpp_initialize([])
        self.group = MoveGroupCommander(self.move_group)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)

        self.g = RobotiqGripper()
        self.init_gripper()

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # 訂閱 AnyGrasp
        self.pose_sub = rospy.Subscriber("/anygrasp/target_pose", PoseStamped, self.cb_anygrasp_pose, queue_size=1)
        rospy.loginfo("[p2p] AnyGrasp 邏輯啟動，等待目標姿態...")

    # ---------- 姿態處理：AnyGrasp -> UR3 ----------
    def cb_anygrasp_pose(self, msg: PoseStamped):
        rospy.loginfo(f"[p2p] 收到目標！Frame: {msg.header.frame_id}")
        try:
            T = self.tfbuf.lookup_transform(self.base_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            ps_base = tf2_geometry_msgs.do_transform_pose(msg, T)
        except Exception as e:
            rospy.logerr(f"TF 失敗: {e}"); return

        # 🧙‍♂️ 座標軸對齊 (讓 UR3 的 Z 軸對齊 AnyGrasp 的 X 軸)
        q_orig = [ps_base.pose.orientation.x, ps_base.pose.orientation.y, 
                  ps_base.pose.orientation.z, ps_base.pose.orientation.w]
        
        # 只要做這步：旋轉 Y 軸 -90 度，完美貼合 AnyGrasp 的姿態
        q_to_ur3 = quaternion_from_euler(0, -math.pi/2, 0) 
        q_final = quaternion_multiply(q_orig, q_to_ur3)
        
        ps_base.pose.orientation.x, ps_base.pose.orientation.y, ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_final

        # 🛡️ 安全檢查：如果夾爪「指向上方」，則再翻轉一次，防止撞桌
        ee_z = self.get_ee_z_axis_in_base(ps_base)
        if ee_z[2] > 0: # Z 軸向上
            rospy.logwarn("偵測到倒立姿態，自動修正方向...")
            # 這裡的翻轉是為了救命(把由下往上抓，變成由上往下抓)，保留不動
            q_fix = quaternion_from_euler(math.pi, 0, 0)
            q_final_safe = quaternion_multiply(q_final, q_fix)
            ps_base.pose.orientation.x, ps_base.pose.orientation.y, ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_final_safe

        # 進入原本的 YOLO 邏輯
        self.run_once(ps_base)

    # ---------- Helpers (完全移植自你的代碼) ----------
    def plan_execute_cartesian_to(self, target_pose):
        self.group.set_start_state_to_current_state()
        wp = target_pose.pose if isinstance(target_pose, PoseStamped) else target_pose
        plan, fraction = self.group.compute_cartesian_path([wp], self.eef_step, True)
        if fraction < 0.99:
            rospy.logwarn(f"笛卡兒路徑不全: {fraction:.2f}")
            return False
        return self.group.execute(plan, wait=True)

    def get_ee_z_axis_in_base(self, pose_stamped):
        q = pose_stamped.pose.orientation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def make_pose_stamped_from_xyz_rpy(self, xyz, rpy_deg):
        qx, qy, qz, qw = quaternion_from_euler(math.radians(rpy_deg[0]), math.radians(rpy_deg[1]), math.radians(rpy_deg[2]))
        ps = PoseStamped(); ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
        return ps

    def init_gripper(self):
        try:
            self.g.connect(self.grip_ip, self.grip_port); self.g.activate()
            rospy.sleep(1.0); self.g.move_and_wait_for_pos(0, 100, 80)
        except: self.g = None

    # ---------- 核心流程 (完全對齊 PoseToPick) ----------
    def run_once(self, ps_target):
        # 1. 取得目標位置與 ee_z 方向
        target_xyz = [ps_target.pose.position.x, ps_target.pose.position.y, ps_target.pose.position.z]
        ee_z = self.get_ee_z_axis_in_base(ps_target)

        rospy.loginfo("[p2p] object_surface = (%.3f, %.3f, %.3f)", *target_xyz)

        # 2. 應用抓取深度
        real_target_xyz = np.array(target_xyz) + (ee_z * self.grasp_depth)
        
        # 安全檢查 (撞桌保護)
        # TABLE_HEIGHT = 0.005 
        # if real_target_xyz[2] < TABLE_HEIGHT:
        #     rospy.logwarn(f"[Safety] 修正深度防止撞桌 (Z={real_target_xyz[2]:.3f})")
        #     real_target_xyz[2] = TABLE_HEIGHT

        # 3. 算出法蘭目標 (Grasp Pose)
        grasp_xyz = real_target_xyz - ee_z * self.tcp_offset
        ps_grasp = PoseStamped()
        ps_grasp.header.frame_id = self.base_frame
        ps_grasp.pose.position.x, ps_grasp.pose.position.y, ps_grasp.pose.position.z = grasp_xyz
        ps_grasp.pose.orientation = ps_target.pose.orientation

        # 4. 算出 Pre-Grasp (同樣用你的邏輯：grasp_xyz - ee_z * approach_dist)
        pre_xyz = grasp_xyz - ee_z * self.approach_dist
        ps_pre = PoseStamped()
        ps_pre.header.frame_id = self.base_frame
        ps_pre.pose.position.x, ps_pre.pose.position.y, ps_pre.pose.position.z = pre_xyz
        ps_pre.pose.orientation = ps_target.pose.orientation

        # --- 執行手臂動作 (加入你的手動確認邏輯) ---
        
        # 動作 A: Go to Pre-Grasp (關節規劃)
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(ps_pre)
        
        success, plan, planning_time, error_code = self.group.plan()

        if not success:
            rospy.logerr(f"[p2p] 規劃失敗，錯誤代碼: {error_code}")
            return

        # 🛑 安全確認鎖 (你的最愛)
        try:
            print("\n==================================================")
            ans = input("⚠️ [安全鎖] 軌跡已顯示在 RViz！確認安全請按 [Enter]，取消請按 [n]：")
            print("==================================================\n")
            if ans.lower() == 'n': return
        except EOFError: pass

        self.group.execute(plan, wait=True)
        self.group.stop(); self.group.clear_pose_targets()

        # 動作 B: 直線前進至 Grasp
        rospy.loginfo("[p2p] 執行直線抓取...")
        if not self.plan_execute_cartesian_to(ps_grasp): return

        # 動作 C: 夾緊
        if self.g: self.g.move_and_wait_for_pos(180, self.grip_speed, self.grip_force)

        try: input("[p2p] 已夾取，按 Enter 抬升放置...")
        except EOFError: pass

        # 動作 D: 抬升 (沿用你的 Z+ 抬升邏輯)
        retreat_xyz = [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + self.retreat_up_height]
        ps_retreat_up = PoseStamped()
        ps_retreat_up.header.frame_id = self.base_frame
        ps_retreat_up.pose.position.x, ps_retreat_up.pose.position.y, ps_retreat_up.pose.position.z = retreat_xyz
        ps_retreat_up.pose.orientation = ps_grasp.pose.orientation

        if not self.plan_execute_cartesian_to(ps_retreat_up): return

        # 動作 E: 放置流程 (Place)
        ps_place = self.make_pose_stamped_from_xyz_rpy(self.final_xyz, self.final_rpy_deg)
        ee_z_place = self.get_ee_z_axis_in_base(ps_place)
        pre_place_xyz = np.array(self.final_xyz) - ee_z_place * self.approach_dist
        ps_pre_place = self.make_pose_stamped_from_xyz_rpy(pre_place_xyz.tolist(), self.final_rpy_deg)

        self.group.set_pose_target(ps_pre_place)
        if not self.group.go(wait=True): return
        
        if not self.plan_execute_cartesian_to(ps_place): return

        if self.g: self.g.move_and_wait_for_pos(0, self.grip_speed, self.grip_force)

        post_up_xyz = np.array(self.final_xyz) - ee_z_place * self.post_open_up_height
        ps_post_up = self.make_pose_stamped_from_xyz_rpy(post_up_xyz.tolist(), self.final_rpy_deg)
        self.plan_execute_cartesian_to(ps_post_up)
        rospy.loginfo("[p2p] 🎉 任務完成！")

if __name__ == "__main__":
    rospy.init_node("semantic_grasp_controller")
    app = SemanticGraspController(); rospy.spin()