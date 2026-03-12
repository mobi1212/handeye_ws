#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose-to-Pick (含 TCP offset 修正 + 手動姿態選擇版本)

動作順序：
  1) 收到 YOLO 座標 -> 等待使用者輸入抓取姿態 (1/2/3)
  2) 物體位置 + 姿態 -> 算出 ee_z
  3) 用 tcp_offset 修正出「法蘭應到達的 grasp_xyz」
  4) 沿 -ee_z 方向拉出 pre-grasp
  5) go(pre-grasp) 關節空間
  6) Cartesian 從 pre-grasp 前進到 grasp
  7) 夾緊
  8) 統一往 base_link +Z 方向抬升（retreat up）
  9) 移動到固定放置姿勢 final pose
 10) 開夾
 11) 再往上抬一小段（避免撞到物體）

⚠️ 已加入：
  - 訂閱 /yolo/target_pixel (geometry_msgs/Point)
  - 利用 depth + intrinsics + TF 將像素 (u,v) 轉成 base_link XYZ
  - 更新 self.target_xyz 後呼叫 run_once() 執行原本夾取流程
"""

import os, sys, math, numpy as np
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

import rospy
from geometry_msgs.msg import PoseStamped, Point
from tf.transformations import quaternion_from_euler, quaternion_matrix
from moveit_commander import MoveGroupCommander, roscpp_initialize

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros, tf2_geometry_msgs

from robotiq_gripper import RobotiqGripper


class PoseToPick:
    def __init__(self):
        # ---- 參數 ----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.move_group  = rospy.get_param("~move_group", "manipulator")

        # 使用者輸入：物體位置 + 夾爪姿態（RPY, deg）
        # ⚠️ target_xyz 現在會被 YOLO + depth 自動覆蓋，但預設值仍保留
        self.target_xyz      = rospy.get_param("~target_xyz",      [0.4, 0.0, 0.15])
        # 預設姿態 (會被手動輸入覆蓋)
        self.target_rpy_deg  = rospy.get_param("~target_rpy_deg",  [180.0, 0.0, 0.0])

        # 相機與深度相關參數
        self.cam_frame   = rospy.get_param("~cam_frame",  "camera_color_optical_frame")
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic",
                                           "/camera/aligned_depth_to_color/image_raw")
        self.cinfo_topic = rospy.get_param("~cinfo_topic", "/camera/color/camera_info")

        # 深度＆內參
        self.bridge = CvBridge()
        self.depth = None
        self.depth_encoding = None
        self.fx = self.fy = self.cx = self.cy = None

        # TCP 偏移（夾爪指尖中心到 MoveIt end-effector 的距離, m）
        self.tcp_offset = float(rospy.get_param("~tcp_offset", 0.163))
        self.grasp_depth = float(rospy.get_param("~grasp_depth", 0.03))#抓取深度：指尖要「刺入」物體表面多深 (單位: m)
        # pre-grasp 距離（沿 -ee_z 方向），retreat_dist 現在僅保留參數，不再使用
        self.approach_dist = float(rospy.get_param("~approach_dist", 0.1))  # m
        self.retreat_dist  = float(rospy.get_param("~retreat_dist",  0.10))  # m（未使用）

        # 統一往上抬升的高度（grasp 之後）
        self.retreat_up_height    = float(rospy.get_param("~retreat_up_height", 0.15))  # m
        # 放開之後再往上抬升一小段
        self.post_open_up_height  = float(rospy.get_param("~post_open_up_height", 0.1))  # m

        self.eef_step    = float(rospy.get_param("~eef_step",   0.02))
        self.vel_scale   = float(rospy.get_param("~vel_scale",  0.10))
        self.acc_scale   = float(rospy.get_param("~acc_scale",  0.10))

        # 固定放置位置與姿態（final drop pose）
        self.final_xyz     = [0.2, 0.1, 0.165]
        self.final_rpy_deg = [180.0, 0.0, 0.0]

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
        rospy.loginfo("[p2p] initial target_xyz(object) = (%.3f, %.3f, %.3f)", *self.target_xyz)
        rospy.loginfo("[p2p] target_rpy_deg           = (%.1f, %.1f, %.1f)", *self.target_rpy_deg)
        rospy.loginfo("[p2p] tcp_offset=%.3f approach=%.3f retreat_up=%.3f post_open_up=%.3f",
                      self.tcp_offset, self.approach_dist,
                      self.retreat_up_height, self.post_open_up_height)

        # ---- Gripper ----
        self.g = RobotiqGripper()
        self._last_grip_pos = None
        self.init_gripper()

        # ---- TF ----
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # ---- Camera / depth 訂閱 ----
        rospy.Subscriber(self.depth_topic, Image, self.cb_depth, queue_size=1)
        rospy.Subscriber(self.cinfo_topic, CameraInfo, self.cb_cinfo, queue_size=1)

        rospy.loginfo("[p2p] waiting CameraInfo on %s ...", self.cinfo_topic)
        rospy.wait_for_message(self.cinfo_topic, CameraInfo)
        rospy.loginfo("[p2p] CameraInfo ready")

        # ---- 訂閱 YOLO 像素 ----
        self.yolo_sub = rospy.Subscriber(
            "/yolo/target_pixel",
            Point,
            self.cb_yolo_pixel,
            queue_size=1
        )
        rospy.loginfo("[p2p] Subscribed to /yolo/target_pixel, 等待偵測點觸發夾取流程。")

    # ---------- depth & intrinsics callbacks ----------
    def cb_depth(self, msg):
        self.depth_encoding = msg.encoding.lower()
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def cb_cinfo(self, msg):
        K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        rospy.loginfo_once(
            f"[p2p] intrinsics fx={self.fx:.3f} fy={self.fy:.3f} cx={self.cx:.3f} cy={self.cy:.3f}"
        )

    # ---------- YOLO callback：像素 -> base -> 呼叫 run_once ----------
    def cb_yolo_pixel(self, msg: Point):
        u = int(msg.x)
        v = int(msg.y)
        rospy.loginfo("[p2p] 收到 YOLO 像素點 u=%d, v=%d", u, v)

        # 確認深度與內參已就緒
        if self.depth is None or self.fx is None:
            rospy.logwarn("[p2p] depth 或 intrinsics 尚未就緒，略過這次點擊")
            return

        try:
            x, y, z = self.pixel_to_base(u, v)
        except Exception as e:
            rospy.logerr("[p2p] pixel_to_base 失敗: %s", e)
            return

        rospy.loginfo("[p2p] pixel(%d,%d) -> base_link XYZ=(%.3f, %.3f, %.3f)", u, v, x, y, z)

        # 將 YOLO 點轉換結果當作「物體中心」，覆蓋 target_xyz
        self.target_xyz = [x, y, z]

        # ----------------------------------------------------
        # [NEW] 手動輸入選擇姿態
        # ----------------------------------------------------
        print("\n=== 請選擇抓取姿態 ===")
        print("1: 垂直向下 (Pitch=0)")
        print("2: 斜抓 30度 (Pitch=-30, 向前傾)")
        print("3: 斜抓 60度 (Pitch=-60, 向前傾)")
        user_input = input("請輸入 (1/2/3) 並按 Enter: ").strip()

        if user_input == '1':
            self.target_rpy_deg = [180.0, 0.0, 0.0]
            rospy.loginfo("[p2p] 選擇姿態: 垂直向下")
        elif user_input == '2':
            self.target_rpy_deg = [180.0, -30.0, 0.0]
            rospy.loginfo("[p2p] 選擇姿態: 斜抓 30度")
        elif user_input == '3':
            self.target_rpy_deg = [180.0, -60.0, 0.0]
            rospy.loginfo("[p2p] 選擇姿態: 斜抓 60度")
        else:
            self.target_rpy_deg = [180.0, 0.0, 0.0] # 預設垂直
            rospy.logwarn("[p2p] 輸入無效，預設使用垂直向下")
        # ----------------------------------------------------

        # 直接執行原本的 Pose-to-Pick 管線
        self.run_once()

    # ---------- pixel -> base_link ----------
    def pixel_to_base(self, u, v):
        if self.depth is None:
            raise RuntimeError("No depth image received yet")

        # 注意：index 是 [row=v, col=u]
        d_raw = self.depth[v, u]
        if d_raw is None or np.isnan(float(d_raw)) or float(d_raw) <= 0:
            raise RuntimeError(f"Depth invalid at pixel ({u},{v}), raw={d_raw}")

        # uint16/mm → m；float32 則預設為 m
        if self.depth.dtype != np.float32:
            Z = float(d_raw) * 0.001
        else:
            Z = float(d_raw)

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        ps_cam = PoseStamped()
        ps_cam.header.stamp = rospy.Time(0)
        ps_cam.header.frame_id = self.cam_frame
        ps_cam.pose.position.x = X
        ps_cam.pose.position.y = Y
        ps_cam.pose.position.z = Z
        ps_cam.pose.orientation.w = 1.0

        T = self.tfbuf.lookup_transform(
            self.base_frame,
            self.cam_frame,
            rospy.Time(0),
            rospy.Duration(1.0)
        )
        ps_base = tf2_geometry_msgs.do_transform_pose(ps_cam, T)
        pos = ps_base.pose.position
        return pos.x, pos.y, pos.z

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
        # 1) 用物體位置 + 姿態先算出 ee_z 方向 (這是夾爪的指向)
        ps_tmp = self.make_pose_stamped_from_xyz_rpy(self.target_xyz, self.target_rpy_deg)
        ee_z = self.get_ee_z_axis_in_base(ps_tmp)

        rospy.loginfo("[p2p] object_surface = (%.3f, %.3f, %.3f)", *self.target_xyz)
        
        # ==========================================
        # [NEW] 應用抓取深度 (Grasp Depth)
        # ==========================================
        # 原理：沿著夾爪指向 (ee_z) 繼續前進 grasp_depth 的距離
        # 這樣可以確保指尖包覆物體，而不只是摸到表面
        
        real_target_xyz = np.array(self.target_xyz) + (ee_z * self.grasp_depth)
        
        # 安全檢查 (撞桌保護)：
        # 如果深入後 Z 值太低 (例如小於桌子高度 0.0)，強制定在安全高度
        TABLE_HEIGHT = 0.005 # 假設桌子表面是 0，留 5mm 安全裕度
        if real_target_xyz[2] < TABLE_HEIGHT:
            rospy.logwarn(f"[Safety] 抓取深度過深 (Z={real_target_xyz[2]:.3f})，可能撞桌！已修正至安全高度。")
            real_target_xyz[2] = TABLE_HEIGHT

        rospy.loginfo(f"[p2p] deep_target (center) = ({real_target_xyz[0]:.3f}, {real_target_xyz[1]:.3f}, {real_target_xyz[2]:.3f})")
        # ==========================================

        # 2) 用 tcp_offset 把「修正後的深點」轉成「法蘭應該到的位置」
        # 注意：這裡要用 real_target_xyz 來算
        grasp_xyz = real_target_xyz - ee_z * self.tcp_offset
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


        try:
            _ = input("[p2p] 已到抓取點，按空白+Enter 繼續：")
        except EOFError:
            pass


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

        # 8) 放置：先 go 到 pre-place（跟 pre-grasp 一樣的概念）
        ps_place = self.make_pose_stamped_from_xyz_rpy(self.final_xyz, self.final_rpy_deg)

        # 這裡用「工具的 -Z 方向」當作靠近方向（跟抓取時一致）
        ee_z_place = self.get_ee_z_axis_in_base(ps_place)   # ee z-axis in base
        pre_place_xyz = np.array(self.final_xyz) + (-ee_z_place) * self.approach_dist
        ps_pre_place = self.make_pose_stamped_from_xyz_rpy(pre_place_xyz.tolist(), self.final_rpy_deg)

        rospy.loginfo("[p2p] pre-place @ base (%.3f, %.3f, %.3f)",
                      pre_place_xyz[0], pre_place_xyz[1], pre_place_xyz[2])

        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(ps_pre_place)
        ok = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.loginfo("[p2p] go(pre-place) = %s", "OK" if ok else "FAIL")
        if not ok:
            rospy.logerr("[p2p] failed to reach pre-place -> KEEP HOLDING, abort")
            return

        # 9) 直線下放到 place（短距離 Cartesian，成功率高）
        rospy.loginfo("[p2p] place(cartesian) @ base (%.3f, %.3f, %.3f)",
                      self.final_xyz[0], self.final_xyz[1], self.final_xyz[2])

        if not self.plan_execute_cartesian_to(ps_place):
            rospy.logerr("[p2p] place cartesian failed -> KEEP HOLDING, abort")
            return

        # 10) 開夾（只在 place 成功後）
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

        # 11) post-open up：直線抬起（短距離 Cartesian）
        post_up_xyz = np.array(self.final_xyz) - ee_z_place * self.post_open_up_height
        ps_post_up = self.make_pose_stamped_from_xyz_rpy(post_up_xyz.tolist(), self.final_rpy_deg)


        rospy.loginfo("[p2p] post-open up @ base (%.3f, %.3f, %.3f)",
                      post_up_xyz[0], post_up_xyz[1], post_up_xyz[2])

        if not self.plan_execute_cartesian_to(ps_post_up):
            rospy.logwarn("[p2p] post-open up cartesian failed")


def main():
    rospy.init_node("pose_to_pick")
    app = PoseToPick()
    rospy.loginfo("[p2p] Node ready. 等待 /yolo/target_pixel 觸發夾取流程 (Ctrl+C 離開)。")
    rospy.spin()


if __name__ == "__main__":
    main()