#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Click-to-Pick (OpenCV pixel -> P_cam(optical) -> tf2 -> base)
動作順序：Approach(上方) → 笛卡兒下降 → 夾緊 → 笛卡兒抬升 → 開夾
手臂用 MoveIt；夾爪用 Robotiq URCap socket (string/TCP)
"""
import os, sys
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from robotiq_gripper import RobotiqGripper
import math, numpy as np
import rospy, cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf2_ros, tf2_geometry_msgs
from tf.transformations import quaternion_from_euler
from moveit_commander import MoveGroupCommander, roscpp_initialize

WIN = "Click-To-Pick (press q to quit)"

class ClickToPickCV:
    def __init__(self):
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.depth_encoding = None
        self.fx = self.fy = self.cx = self.cy = None
        self.logged_K = False
        self.logged_depth_enc = False

        # ---- params（與舊流程一致命名）----
        self.base_frame  = rospy.get_param("~base_frame", "base_link")
        self.cam_frame   = rospy.get_param("~cam_frame",  "camera_color_optical_frame")
        self.move_group  = rospy.get_param("~move_group", "manipulator")
        # 跟第一支一樣的預設姿態：(0, 180, 0)
        self.ee_rpy_deg  = tuple(rospy.get_param("~ee_rpy_deg", [0.0, 180.0, 0.0]))

        self.approach_z  = float(rospy.get_param("~approach_z", 0.05))  # 上方 m
        self.lift_z      = float(rospy.get_param("~lift_z",     0.13))  # 抬升 m
        self.eef_step    = float(rospy.get_param("~eef_step",   0.01))  # 笛卡兒步長 m
        self.vel_scale   = float(rospy.get_param("~vel_scale",  0.10))
        self.acc_scale   = float(rospy.get_param("~acc_scale",  0.10))

        # 🔸 點到的深度 z，要往上補多少才當作「物體高度」
        self.pick_offset_z = float(rospy.get_param("~pick_offset_z", 0.15))

        # 夾爪 TCP
        self.grip_ip     = rospy.get_param("~gripper_ip",   "192.168.86.7")
        self.grip_port   = int(rospy.get_param("~gripper_port", 63352))
        self.grip_speed  = int(rospy.get_param("~gripper_speed", 100))  # 0~255
        self.grip_force  = int(rospy.get_param("~gripper_force", 80))   # 0~255
        # 無回授等待（與速度/行程有關的簡化模型）
        self.wait_min    = float(rospy.get_param("~wait_min",  0.2))
        self.wait_max    = float(rospy.get_param("~wait_max",  1.2))
        self.wait_base   = float(rospy.get_param("~wait_base", 0.15))
        self.wait_k      = float(rospy.get_param("~wait_k",    0.9))
        self.wait_move_extra = float(rospy.get_param("~wait_move_extra", 0.2))

        # ---- tf2 ----
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # ---- MoveIt ----
        roscpp_initialize([])
        self.group = MoveGroupCommander(self.move_group)
        # 不指定 end_effector_link，沿用 MoveIt SRDF 的預設 tip link（跟第一支一樣）
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)

        # ---- subscribers ----
        rospy.Subscriber(rospy.get_param("~color_topic", "/camera/color/image_raw"),
                         Image, self.cb_color, queue_size=1)
        rospy.Subscriber(rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw"),
                         Image, self.cb_depth, queue_size=1)
        rospy.Subscriber(rospy.get_param("~cinfo_topic", "/camera/color/camera_info"),
                         CameraInfo, self.cb_cinfo, queue_size=1)

        # ---- UI ----
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)

        # ---- startup logs ----
        rospy.loginfo("[c2p] base=%s cam=%s group=%s", self.base_frame, self.cam_frame, self.move_group)
        rospy.loginfo("[c2p] approach_z=%.3f lift_z=%.3f pick_offset_z=%.3f eef_step=%.3f vel=%.2f acc=%.2f",
                      self.approach_z, self.lift_z, self.pick_offset_z, self.eef_step, self.vel_scale, self.acc_scale)
        rospy.loginfo("[c2p] ee_rpy_deg = (%.1f, %.1f, %.1f)", *self.ee_rpy_deg)
        rospy.loginfo("[c2p] waiting CameraInfo...")
        rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        rospy.loginfo("[c2p] CameraInfo ready")

        try:
            from moveit_commander import RobotCommander
            rc = RobotCommander()
            links = rc.get_link_names(self.move_group)
            rospy.loginfo("[c2p] move_group '%s' links: %s", self.move_group, ", ".join(links))
        except Exception as e:
            rospy.logwarn("[c2p] RobotCommander check failed: %s", e)

        self.wait_tf_ok(self.base_frame, self.cam_frame)

        # ---- Gripper ----
        self.g = RobotiqGripper()
        try:
            rospy.loginfo("[grip] connecting %s:%d ...", self.grip_ip, self.grip_port)
            self.g.connect(self.grip_ip, self.grip_port)
            self.g.activate()
            rospy.loginfo("[grip] activated")
            rospy.sleep(2.0)
            self.g.move_and_wait_for_pos(position=0, speed=self.grip_speed, force=self.grip_force)
            self._last_grip_pos = 0
            rospy.loginfo("[grip] opened to 0")
        except Exception as e:
            rospy.logwarn("[grip] connect/activate failed: %s", e)
            self.g = None
            self._last_grip_pos = None

    # ---------- helpers ----------
    def wait_tf_ok(self, target, source, timeout=5.0):
        t0 = rospy.Time.now().to_sec()
        while (rospy.Time.now().to_sec() - t0) < timeout and not rospy.is_shutdown():
            try:
                self.tfbuf.lookup_transform(target, source, rospy.Time(0), rospy.Duration(0.2))
                rospy.loginfo("[c2p] TF ok: %s <- %s", target, source)
                return True
            except Exception:
                rospy.logwarn_throttle(1.0, "[c2p] wait TF: %s <- %s", target, source)
        rospy.logerr("[c2p] TF not available: %s <- %s", target, source)
        return False

    def grip_wait(self, speed, last_pos=None, target_pos=None):
        base = max(self.wait_min, min(self.wait_max, self.wait_base + self.wait_k * (100.0 / max(1, speed))))
        span = None
        if last_pos is not None and target_pos is not None:
            span = abs(int(target_pos) - int(last_pos)) / 255.0
            base *= (span + self.wait_move_extra)
        rospy.loginfo("[grip] wait %.3fs (speed=%d, span=%s)", base, speed, f"{span:.2f}" if span is not None else "NA")
        rospy.sleep(base)

    # 用 ee_rpy_deg（跟第一支同邏輯）
    def make_down_quat(self, yaw_rad=0.0):
        r_deg, p_deg, y_deg = self.ee_rpy_deg
        return quaternion_from_euler(
            math.radians(r_deg),
            math.radians(p_deg),
            yaw_rad + math.radians(y_deg)
        )

    # 確保丟進 compute_cartesian_path 的是 Pose，不是 PoseStamped
    def plan_execute_cartesian_to(self, target_pose):
        self.group.set_start_state_to_current_state()

        from geometry_msgs.msg import Pose
        if isinstance(target_pose, PoseStamped):
            wp = target_pose.pose
        else:
            wp = target_pose

        waypoints = [wp]
        plan, fraction = self.group.compute_cartesian_path(waypoints, self.eef_step, True)
        rospy.loginfo("[c2p] cartesian fraction=%.3f", fraction)
        if fraction < 0.99:
            rospy.logwarn("[c2p] Cartesian incomplete, fraction=%.2f", fraction)
            return False
        pts = len(plan.joint_trajectory.points) if hasattr(plan, "joint_trajectory") else -1
        rospy.loginfo("[c2p] executing cartesian (pts=%d)", pts)
        ok = self.group.execute(plan, wait=True)
        self.group.stop(); self.group.clear_pose_targets()
        rospy.loginfo("[c2p] cartesian exec=%s", "OK" if ok else "FAIL")
        return ok

    def make_pose_stamped(self, x, y, z, yaw_rad=0.0):
        qx, qy, qz, qw = self.make_down_quat(yaw_rad)
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        return ps

    # ---------- callbacks ----------
    def cb_color(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        disp = self.color.copy()
        cv2.putText(disp, "Click a pixel (approach -> descend -> grip -> lift)", (18,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow(WIN, disp); cv2.waitKey(1)

    def cb_depth(self, msg):
        self.depth_encoding = msg.encoding.lower()
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        if not self.logged_depth_enc:
            rospy.loginfo("[c2p] depth encoding=%s dtype=%s", self.depth_encoding, str(self.depth.dtype))
            self.logged_depth_enc = True

    def cb_cinfo(self, msg):
        K = np.array(msg.K, dtype=np.float64).reshape(3,3)
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        if not self.logged_K:
            rospy.loginfo("[c2p] K fx=%.3f fy=%.3f cx=%.3f cy=%.3f", self.fx, self.fy, self.cx, self.cy)
            self.logged_K = True

    # ---------- interaction ----------
    def on_mouse(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.color is None or self.depth is None or self.fx is None:
            rospy.logwarn("[c2p] waiting images/depth/K...")
            return

        rospy.loginfo("[c2p] click pixel=(%d,%d)", x, y)
        d_raw = self.depth[y, x]
        if d_raw is None or np.isnan(float(d_raw)) or float(d_raw) <= 0:
            rospy.logwarn("[c2p] invalid depth at (%d,%d)", x, y)
            return
        rospy.loginfo("[c2p] depth raw=%s (encoding=%s, dtype=%s)", str(d_raw), self.depth_encoding, str(self.depth.dtype))

        Z = float(d_raw) * 0.001 if self.depth.dtype != np.float32 else float(d_raw)
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy
        rospy.loginfo("[c2p] camXYZ(optical)=(%.3f, %.3f, %.3f) m", X, Y, Z)

        ps_cam = PoseStamped()
        ps_cam.header.stamp = rospy.Time(0)
        ps_cam.header.frame_id = self.cam_frame
        ps_cam.pose.position.x, ps_cam.pose.position.y, ps_cam.pose.position.z = X, Y, Z
        ps_cam.pose.orientation.w = 1.0
        try:
            rospy.loginfo("[c2p] TF lookup: %s <- %s", self.base_frame, self.cam_frame)
            T = self.tfbuf.lookup_transform(self.base_frame, self.cam_frame, rospy.Time(0), rospy.Duration(1.0))
            ps_base = tf2_geometry_msgs.do_transform_pose(ps_cam, T)
        except Exception as e:
            rospy.logerr("[c2p] TF failed: %s", e); return

        yaw = 0.0
        qx,qy,qz,qw = self.make_down_quat(yaw)
        ps_base.pose.orientation.x, ps_base.pose.orientation.y = qx, qy
        ps_base.pose.orientation.z, ps_base.pose.orientation.w = qz, qw

        px, py, pz = ps_base.pose.position.x, ps_base.pose.position.y, ps_base.pose.position.z
        rospy.loginfo("[c2p] base raw target=(%.3f, %.3f, %.3f)", px, py, pz)

        # 🔸 把點擊出來的 z 往上補一點，才是真正抓取高度（跟你手動 +0.15 一樣意思）
        obj_z = pz + self.pick_offset_z
        rospy.loginfo("[c2p] obj_z = pz + pick_offset_z = %.3f + %.3f = %.3f",
                      pz, self.pick_offset_z, obj_z)

        # ----- Step1: Approach（先到物體上方）-----
        p_approach = self.make_pose_stamped(px, py, obj_z + self.approach_z, yaw)
        rospy.loginfo("[c2p] Approach target=(%.3f, %.3f, %.3f)",
                      px, py, obj_z + self.approach_z)
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(p_approach)
        ok = self.group.go(wait=True)
        rospy.loginfo("[c2p] approach go()=%s", "OK" if ok else "FAIL")
        self.group.stop(); self.group.clear_pose_targets()
        if not ok:
            return

        # ----- Step2: 笛卡兒下降到 obj_z -----
        p_pick = self.make_pose_stamped(px, py, obj_z, yaw)
        if not self.plan_execute_cartesian_to(p_pick):
            rospy.logwarn("[c2p] descend failed")
            return

        # ----- Grip: close -----
        if self.g:
            target_pos = 180
            rospy.loginfo("[grip] close -> %d (spd=%d force=%d)", target_pos, self.grip_speed, self.grip_force)
            try:
                self.g.move_and_wait_for_pos(position=target_pos, speed=self.grip_speed, force=self.grip_force)
            except Exception as e:
                rospy.logwarn("[grip] close error: %s", e)
            self.grip_wait(self.grip_speed, self._last_grip_pos, target_pos)
            self._last_grip_pos = target_pos
        else:
            rospy.logwarn("[grip] not connected, skip gripping")

        # ----- Step3: 笛卡兒抬升 -----
        p_lift = self.make_pose_stamped(px, py, obj_z + self.lift_z, yaw)
        if not self.plan_execute_cartesian_to(p_lift):
            rospy.logwarn("[c2p] lift failed")
            return

        # 自動放開
        if self.g:
            rospy.loginfo("[grip] open -> 0")
            try:
                self.g.move_and_wait_for_pos(position=0, speed=self.grip_speed, force=self.grip_force)
            except Exception as e:
                rospy.logwarn("[grip] open error: %s", e)
            self.grip_wait(self.grip_speed, self._last_grip_pos, 0)
            self._last_grip_pos = 0

        rospy.loginfo("[c2p] DONE")

def main():
    rospy.init_node("click_to_pick_cv")
    app = ClickToPickCV()

    rospy.loginfo("Ready. Click on the image; press 'q' to quit.")

    rate = rospy.Rate(60)  # 60 FPS GUI loop
    while not rospy.is_shutdown():

        if app.color is not None:
            disp = app.color.copy()
            cv2.putText(disp, "Click a pixel (approach -> descend -> grip -> lift)",
                        (18,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
