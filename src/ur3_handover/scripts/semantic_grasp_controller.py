#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Grasp Controller (AnyGrasp 6D Pose 接收版)
"""

import json
import os, sys, math, numpy as np
import traceback
# 確保從 scripts 目錄直接載入，避免 catkin relay exec() 的 import 問題
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rospy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import String
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_from_euler, quaternion_from_matrix
from moveit_commander import MoveGroupCommander, roscpp_initialize
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import tf2_ros, tf2_geometry_msgs

from robotiq_gripper import RobotiqGripper

class SemanticGraspController:
    def __init__(self):
        # ---- 1. 參數 ----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.move_group  = rospy.get_param("~move_group", "manipulator")
        self.use_grasp_metadata = bool(rospy.get_param("~use_grasp_metadata", True))
        self.prefer_camera_facing_side = bool(rospy.get_param("~prefer_camera_facing_side", True))
        self.use_server_depth_for_offset = bool(
            rospy.get_param("~use_server_depth_for_offset", False))
        self.prefer_nearest_ik = bool(rospy.get_param("~prefer_nearest_ik", True))

        self.tcp_offset    = float(rospy.get_param("~tcp_offset",    0.16))
        self.grasp_depth   = float(rospy.get_param("~grasp_depth",   0.05))
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.05))

        self.retreat_up_height = float(rospy.get_param("~retreat_up_height", 0.14))  # 待機點高度

        self.eef_step  = float(rospy.get_param("~eef_step",  0.02))
        self.vel_scale = float(rospy.get_param("~vel_scale", 0.10))
        self.acc_scale = float(rospy.get_param("~acc_scale", 0.10))

        # 固定交接點（法蘭位置）
        self.handover_xyz = rospy.get_param("~handover_xyz", [0.1606, 0.2881, 0.3154])
        # 力矩偵測參數
        self.handover_force_threshold = float(rospy.get_param("~handover_force_threshold", 10.0))
        self.handover_timeout = float(rospy.get_param("~handover_timeout", 30.0))
        # wrench 即時值
        self._wrench_force = np.zeros(3)

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
        self.group.set_planner_id("RRTConnect")
        self.group.set_num_planning_attempts(1)  # 由 joint_plan_execute 控制多次
        self.group.set_planning_time(5.0)        # 每次 5 秒，跑 3 次取最短

        self.g = RobotiqGripper()
        self.init_gripper()

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)
        self.ik_service_name = rospy.get_param("~ik_service", "/compute_ik")
        self.ik_srv = None
        if self.prefer_nearest_ik:
            try:
                rospy.wait_for_service(self.ik_service_name, timeout=2.0)
                self.ik_srv = rospy.ServiceProxy(self.ik_service_name, GetPositionIK)
                rospy.loginfo("[p2p] 啟用最近 IK 分支規劃: %s", self.ik_service_name)
            except (rospy.ROSException, rospy.ROSInterruptException):
                rospy.logwarn("[p2p] 找不到 IK service %s，退回 pose target 規劃",
                              self.ik_service_name)

        self.target_pose_topic = rospy.get_param("~target_pose_topic", "/anygrasp/target_pose")
        self.target_grasp_topic = rospy.get_param("~target_grasp_topic", "/anygrasp/target_grasp")

        # 訂閱 AnyGrasp 姿態 / 完整 grasp metadata
        if self.use_grasp_metadata:
            self.grasp_sub = rospy.Subscriber(
                self.target_grasp_topic, String, self.cb_anygrasp_grasp, queue_size=1)
            rospy.loginfo("[p2p] 使用完整 grasp metadata: %s", self.target_grasp_topic)
        else:
            self.pose_sub = rospy.Subscriber(
                self.target_pose_topic, PoseStamped, self.cb_anygrasp_pose, queue_size=1)
            rospy.loginfo("[p2p] 使用 legacy PoseStamped: %s", self.target_pose_topic)
        # 隨時回待機點的指令
        rospy.Subscriber("/semantic_grasp/go_home", String, self._cb_go_home)
        # TCP 力矩
        rospy.Subscriber("/wrench", WrenchStamped, self._wrench_cb)

        rospy.loginfo("[p2p] 啟動完成，等待目標姿態...")
        rospy.loginfo("[p2p] 隨時回待機點: rostopic pub /semantic_grasp/go_home std_msgs/String 'go' -1")

    # ------------------------------------------------------------------ #
    #  ROS callbacks                                                       #
    # ------------------------------------------------------------------ #
    def cb_anygrasp_grasp(self, msg: String):
        try:
            grasp = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logerr("[p2p] grasp metadata JSON 解析失敗: %s", msg.data)
            return

        required_keys = ("translation", "rotation")
        missing = [key for key in required_keys if key not in grasp]
        if missing:
            rospy.logerr("[p2p] grasp metadata 缺少欄位: %s", ", ".join(missing))
            return

        try:
            tvec = np.array(grasp["translation"], dtype=float)
            rot_mat = np.array(grasp["rotation"], dtype=float).reshape(3, 3)
        except (ValueError, TypeError) as e:
            rospy.logerr("[p2p] grasp metadata 數值格式錯誤: %s", e)
            return

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = grasp.get("frame_id", "camera_color_optical_frame")
        stamp_sec = float(grasp.get("stamp", 0.0))
        pose_msg.header.stamp = rospy.Time.from_sec(stamp_sec) if stamp_sec > 0.0 else rospy.Time.now()
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
        T = np.eye(4)
        T[:3, :3] = rot_mat
        q = quaternion_from_matrix(T)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        rospy.loginfo(
            "[p2p] 收到完整 grasp metadata！score=%.4f width=%.3f depth=%.3f",
            float(grasp.get("score", 0.0)),
            float(grasp.get("width", 0.0)),
            float(grasp.get("depth", self.grasp_depth)))
        self._process_target_pose(pose_msg, grasp)

    def cb_anygrasp_pose(self, msg: PoseStamped):
        rospy.loginfo(f"[p2p] 收到目標！Frame: {msg.header.frame_id}")
        self._process_target_pose(msg, None)

    def _process_target_pose(self, msg: PoseStamped, grasp_meta=None):
        if grasp_meta is None:
            rospy.loginfo("[p2p] 使用 legacy pose 模式，grasp_depth=%.3f", self.grasp_depth)
        try:
            T = self.tfbuf.lookup_transform(
                self.base_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            ps_base = tf2_geometry_msgs.do_transform_pose(msg, T)
        except Exception as e:
            rospy.logerr(f"TF 失敗: {e}"); return

        camera_pos = np.array([
            T.transform.translation.x,
            T.transform.translation.y,
            T.transform.translation.z,
        ], dtype=float)
        object_surface = np.array([
            ps_base.pose.position.x,
            ps_base.pose.position.y,
            ps_base.pose.position.z,
        ], dtype=float)

        # 座標軸對齊：預設維持舊版 Y 軸 -90°，必要時才啟用相機側選邊
        q_orig = [ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                  ps_base.pose.orientation.z, ps_base.pose.orientation.w]
        if self.prefer_camera_facing_side:
            q_final, choice_info = self.select_camera_facing_orientation(
                q_orig, object_surface, camera_pos)
        else:
            q_final = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi/2, 0))
            choice_info = None

        # --- 診斷：印出 TF 後、q_to_ur3 前的原始接近方向（相機座標系 X 軸）---
        from tf.transformations import quaternion_matrix as qmat
        M_orig = qmat([q_orig[0], q_orig[1], q_orig[2], q_orig[3]])
        approach_cam = M_orig[:3, 0]  # AnyGrasp X 軸 = 接近方向（in base_link after TF）
        rospy.loginfo(f"[p2p] AnyGrasp 接近方向(base) = ({approach_cam[0]:.3f}, {approach_cam[1]:.3f}, {approach_cam[2]:.3f})")
        rospy.loginfo(
            "[p2p] 相機位置(base) = (%.3f, %.3f, %.3f)",
            *camera_pos)
        if choice_info is not None:
            rospy.loginfo(
                "[p2p] camera->object = (%.3f, %.3f, %.3f)",
                *choice_info["camera_to_object"])
            rospy.loginfo(
                "[p2p] 映射候選分數: Y-90=%.3f, Y+90=%.3f，採用 %s",
                choice_info["score_neg90"],
                choice_info["score_pos90"],
                choice_info["label"])
        ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
            ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_final

        # 安全檢查：夾爪明顯朝上（> 0.7）才翻轉，避免誤判側面抓取
        ee_z = self.get_ee_z_axis_in_base(ps_base)
        rospy.loginfo(f"[p2p] 接收到的 ee_z = ({ee_z[0]:.3f}, {ee_z[1]:.3f}, {ee_z[2]:.3f})")
        if ee_z[2] > 0.7:
            rospy.logwarn(f"偵測到明顯倒立姿態 (ee_z[2]={ee_z[2]:.3f})，自動修正方向...")
            q_fix = quaternion_from_euler(math.pi, 0, 0)
            q_safe = quaternion_multiply(q_final, q_fix)
            ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
                ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_safe

        server_depth = None
        if grasp_meta is not None:
            try:
                server_depth = float(grasp_meta.get("depth", self.grasp_depth))
            except (TypeError, ValueError):
                rospy.logwarn("[p2p] grasp metadata depth 非法，忽略 server depth")
                server_depth = None
            if server_depth is not None and server_depth <= 0.0:
                rospy.logwarn("[p2p] grasp metadata depth=%.3f 無效，忽略 server depth",
                              server_depth)
                server_depth = None
            if server_depth is not None:
                rospy.loginfo("[p2p] 收到 server grasp depth = %.3f", server_depth)

        final_insert_depth = self.resolve_final_insert_depth(server_depth)
        self.run_once(
            ps_base,
            final_insert_depth=final_insert_depth,
            server_depth=server_depth)

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

    def _plan_length(self, plan):
        """計算關節空間總位移（越小越短）"""
        pts = plan.joint_trajectory.points
        if len(pts) < 2:
            return float('inf')
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            total += sum(abs(j2 - j1) for j1, j2 in zip(a.positions, b.positions))
        return total

    def solve_nearest_ik(self, ps_target):
        if self.ik_srv is None:
            return None

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.move_group
        req.ik_request.pose_stamped = ps_target
        req.ik_request.robot_state = self.group.get_current_state()
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.2)

        ee_link = self.group.get_end_effector_link()
        if ee_link:
            req.ik_request.ik_link_name = ee_link

        try:
            resp = self.ik_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn("[p2p] IK service 呼叫失敗，退回 pose target 規劃: %s", e)
            return None

        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            rospy.logwarn("[p2p] IK 求解失敗(error=%s)，退回 pose target 規劃",
                          resp.error_code.val)
            return None

        joint_map = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
        active_joints = self.group.get_active_joints()
        try:
            joint_target = [joint_map[name] for name in active_joints]
        except KeyError as e:
            rospy.logwarn("[p2p] IK 解缺少關節 %s，退回 pose target 規劃", e)
            return None

        rospy.loginfo("[p2p] 採用最近 IK 分支作為 joint target")
        return joint_target

    def joint_plan_execute(self, ps_target, label="目標點", n_attempts=3, execute=True):
        """多次規劃取最短關節路徑，可選擇只規劃不執行（供安全確認用）"""
        self._normalize_start_state()
        joint_target = self.solve_nearest_ik(ps_target) if self.prefer_nearest_ik else None
        try:
            if joint_target is not None:
                self.group.set_joint_value_target(joint_target)
            else:
                self.group.set_pose_target(ps_target)
        except Exception as e:
            if joint_target is not None:
                rospy.logwarn("[p2p] %s 設定 joint target 失敗，退回 pose target 規劃: %s",
                              label, e)
                self.group.clear_pose_targets()
                self.group.set_pose_target(ps_target)
            else:
                raise
        best_plan, best_len = None, float('inf')
        for i in range(n_attempts):
            success, plan, _, error_code = self.group.plan()
            if not success:
                rospy.logwarn(f"[p2p] {label} 第 {i+1} 次規劃失敗: {error_code}")
                continue
            length = self._plan_length(plan)
            if length < best_len:
                best_len, best_plan = length, plan
        self.group.clear_pose_targets()

        if best_plan is None:
            rospy.logerr(f"[p2p] {label} 規劃失敗（{n_attempts} 次均無解）")
            return False, None

        rospy.loginfo(f"[p2p] {label} 最短路徑: {best_len:.3f} rad（{n_attempts} 次中選出）")
        if not execute:
            return True, best_plan

        ok = self.group.execute(best_plan, wait=True)
        self.group.stop()
        if not ok:
            rospy.logerr(f"[p2p] {label} 執行失敗")
        return ok, best_plan

    def get_ee_z_axis_in_base(self, pose_stamped):
        q = pose_stamped.pose.orientation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def ee_z_from_quaternion(self, quat_xyzw):
        M = quaternion_matrix(quat_xyzw)
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def resolve_final_insert_depth(self, server_depth):
        """決定 grasp reference pose 轉成機器人 final grasp pose 時的前進補償量。"""
        if self.use_server_depth_for_offset and server_depth is not None:
            rospy.loginfo("[p2p] final grasp offset 使用 server depth = %.3f", server_depth)
            return float(server_depth)

        if server_depth is not None:
            rospy.loginfo(
                "[p2p] final grasp offset 使用本地 grasp_depth = %.3f（server depth %.3f 僅記錄）",
                self.grasp_depth, server_depth)
        else:
            rospy.loginfo("[p2p] final grasp offset 使用本地 grasp_depth = %.3f", self.grasp_depth)
        return self.grasp_depth

    def build_robot_grasp_poses(self, grasp_ref_pose, final_insert_depth):
        """
        AnyGrasp 回傳的是 grasp reference pose。
        先轉成 UR3 final grasp pose，再由 final grasp 沿接近軸退回 pre-grasp。
        """
        grasp_ref_xyz = np.array([
            grasp_ref_pose.pose.position.x,
            grasp_ref_pose.pose.position.y,
            grasp_ref_pose.pose.position.z
        ], dtype=float)
        ee_z = self.get_ee_z_axis_in_base(grasp_ref_pose)

        # final grasp pose: 由 grasp reference pose 沿接近軸補償插入量，再扣掉 tool0 到夾爪工作點的固定偏移
        final_grasp_xyz = grasp_ref_xyz + ee_z * final_insert_depth - ee_z * self.tcp_offset
        pregrasp_xyz = final_grasp_xyz - ee_z * self.approach_dist

        ps_grasp = self.make_pose_stamped(final_grasp_xyz, grasp_ref_pose.pose.orientation)
        ps_pre = self.make_pose_stamped(pregrasp_xyz, grasp_ref_pose.pose.orientation)

        return ps_grasp, ps_pre, {
            "grasp_ref_xyz": grasp_ref_xyz,
            "final_grasp_xyz": final_grasp_xyz,
            "pregrasp_xyz": pregrasp_xyz,
            "ee_z": ee_z,
            "final_insert_depth": float(final_insert_depth),
        }

    def select_camera_facing_orientation(self, q_orig, object_surface, camera_pos):
        camera_to_object = object_surface - camera_pos
        norm = np.linalg.norm(camera_to_object)
        if norm < 1e-9:
            rospy.logwarn("[p2p] camera_to_object 長度過小，退回舊版 Y-90 映射")
            q_default = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi / 2, 0))
            return q_default, {
                "camera_to_object": np.array([0.0, 0.0, 0.0]),
                "score_neg90": 0.0,
                "score_pos90": float("-inf"),
                "label": "Y-90 (fallback)",
            }

        camera_to_object = camera_to_object / norm
        candidates = []
        for y_deg in (-90.0, 90.0):
            q_map = quaternion_multiply(q_orig, quaternion_from_euler(0, math.radians(y_deg), 0))
            ee_z = self.ee_z_from_quaternion(q_map)
            score = float(np.dot(ee_z, camera_to_object))
            candidates.append({
                "label": f"Y{y_deg:+.0f}",
                "quat": q_map,
                "ee_z": ee_z,
                "score": score,
            })

        best = max(candidates, key=lambda item: item["score"])
        return best["quat"], {
            "camera_to_object": camera_to_object,
            "score_neg90": candidates[0]["score"],
            "score_pos90": candidates[1]["score"],
            "label": best["label"],
        }

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

    def _wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        self._wrench_force = np.array([f.x, f.y, f.z])

    def _measure_wrench_baseline(self, duration=0.5):
        samples = []
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)
        while rospy.Time.now().to_sec() - t0 < duration:
            samples.append(np.linalg.norm(self._wrench_force))
            rate.sleep()
        baseline = float(np.mean(samples)) if samples else 0.0
        rospy.loginfo("[p2p] wrench baseline = %.2f N (%d samples)", baseline, len(samples))
        return baseline

    def _wait_for_handover(self, baseline):
        rospy.loginfo("[p2p] 等待人手接取（threshold=%.1f N, timeout=%.0fs）...",
                      self.handover_force_threshold, self.handover_timeout)
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)
        CONFIRM_COUNT = 1  # 連續 2 次即觸發
        confirm = 0
        while rospy.Time.now().to_sec() - t0 < self.handover_timeout:
            delta = abs(np.linalg.norm(self._wrench_force) - baseline)
            if delta > self.handover_force_threshold:
                confirm += 1
                if confirm >= CONFIRM_COUNT:
                    rospy.loginfo("[p2p] 偵測到持續拉力 delta=%.2f N", delta)
                    return True
            else:
                confirm = 0
            rate.sleep()
        rospy.logwarn("[p2p] 交接逾時（%.0fs），自動鬆手", self.handover_timeout)
        return False

    def init_gripper(self):
        try:
            self.g.connect(self.grip_ip, self.grip_port); self.g.activate()
            rospy.sleep(1.0); self.g.move_and_wait_for_pos(0, 100, 80)
        except: self.g = None

    # ------------------------------------------------------------------ #
    #  待機點（放置區正上方）                                               #
    # ------------------------------------------------------------------ #
    def go_to_ready_pose(self):
        ready_xyz = list(self.handover_xyz)
        ps_ready = self.make_pose_stamped_from_xyz_rpy(ready_xyz, [180.0, 0.0, 0.0])
        rospy.loginfo(f"[p2p] 前往待機點 {ready_xyz} ...")
        ok, _ = self.joint_plan_execute(ps_ready, "待機點")
        if ok:
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
    def run_once(self, ps_target, final_insert_depth=None, server_depth=None):
        try:
            self._grasp_and_place(
                ps_target,
                final_insert_depth=final_insert_depth,
                server_depth=server_depth)
        except Exception as e:
            rospy.logerr("[p2p] 抓取流程未捕捉例外: %s", e)
            rospy.logerr("%s", traceback.format_exc())
        finally:
            self._prompt_ready()

    def _grasp_and_place(self, ps_target, final_insert_depth=None, server_depth=None):
        final_insert_depth = (
            self.grasp_depth if final_insert_depth is None else float(final_insert_depth))
        ps_grasp, ps_pre, grasp_info = self.build_robot_grasp_poses(
            ps_target, final_insert_depth)
        ee_z = grasp_info["ee_z"]
        grasp_xyz = grasp_info["final_grasp_xyz"]

        rospy.loginfo("[p2p] grasp_ref_xyz   = (%.3f, %.3f, %.3f)", *grasp_info["grasp_ref_xyz"])
        rospy.loginfo("[p2p] ee_z            = (%.3f, %.3f, %.3f)", *ee_z)
        if server_depth is not None:
            rospy.loginfo("[p2p] server_depth    = %.3f", server_depth)
        rospy.loginfo(
            "[p2p] final_insert_depth = %.3f, tcp_offset = %.3f, approach_dist = %.3f",
            grasp_info["final_insert_depth"], self.tcp_offset, self.approach_dist)
        rospy.loginfo("[p2p] final_grasp_xyz = (%.3f, %.3f, %.3f)", *grasp_info["final_grasp_xyz"])
        rospy.loginfo("[p2p] pregrasp_xyz    = (%.3f, %.3f, %.3f)", *grasp_info["pregrasp_xyz"])

        # 交接點（保持抓取姿態）
        ps_handover = self.make_pose_stamped(self.handover_xyz, ps_target.pose.orientation)

        # ── 動作 A：Pre-Grasp（規劃 → 確認迴圈）────────────────────────── #
        while True:
            success, plan = self.joint_plan_execute(ps_pre, "Pre-Grasp", execute=False)
            if not success:
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
        self.group.stop()

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

        # ── 動作 D：Cartesian 垂直抬升 10cm ─────────────────────────────── #
        rospy.loginfo("[p2p] 垂直抬升 10cm...")
        lift_pose = self.make_pose_stamped(
            [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.1],
            ps_grasp.pose.orientation)
        if not self.plan_execute_cartesian_to(lift_pose): return

        # ── 動作 E：關節規劃 → 交接點（保持抓取姿態）───────────────────── #
        rospy.loginfo("[p2p] 移動至交接點 %s ...", self.handover_xyz)
        ok, _ = self.joint_plan_execute(ps_handover, "交接點")
        if not ok: return

        # ── 動作 F：記錄 baseline → 等人拉 → 鬆手 ───────────────────────── #
        rospy.loginfo("[p2p] 到達交接點，等待穩定...")
        rospy.sleep(1.0)  # 等手臂震動消散
        baseline = self._measure_wrench_baseline(duration=1.0)
        self._wait_for_handover(baseline)

        # ── 動作 G：張開夾爪 ────────────────────────────────────────────── #
        if self.g: self.g.move_and_wait_for_pos(0, self.grip_speed, self.grip_force)
        rospy.loginfo("[p2p] 夾爪張開，物體已交接")

        rospy.loginfo("[p2p] 🎉 交接完成！")


if __name__ == "__main__":
    rospy.init_node("semantic_grasp_controller")
    app = SemanticGraspController(); rospy.spin()
