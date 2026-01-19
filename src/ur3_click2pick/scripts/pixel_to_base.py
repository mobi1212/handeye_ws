#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pixel_to_base.py
輸入像素點 -> 轉成 camera optical frame XYZ -> 再轉成 base_link 座標
只負責座標轉換，不做 MoveIt 或夾爪。
"""

import rospy
import numpy as np
import tf2_ros, tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

class PixelToBase:
    def __init__(self):
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_encoding = None

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.cam_frame  = rospy.get_param("~cam_frame",  "camera_color_optical_frame")

        # --- subscribers ---
        rospy.Subscriber("/camera/color/image_raw", Image, self.cb_color, queue_size=1)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",
                         Image, self.cb_depth, queue_size=1)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cb_cinfo, queue_size=1)

        # --- TF ---
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        rospy.loginfo("[pixel2base] waiting CameraInfo...")
        rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        rospy.loginfo("[pixel2base] CameraInfo ready")

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------
    def cb_color(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def cb_depth(self, msg):
        self.depth_encoding = msg.encoding.lower()
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def cb_cinfo(self, msg):
        K = np.array(msg.K).reshape(3,3)
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        rospy.loginfo_once = getattr(rospy, "loginfo_once", print)
        rospy.loginfo_once(f"[pixel2base] fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f}")

    # ----------------------------------------------------------------------
    # 核心功能：pixel -> base_link
    # ----------------------------------------------------------------------
    def convert(self, u, v):
        if self.depth is None:
            raise RuntimeError("No depth image received yet")

        d_raw = self.depth[v, u]
        if d_raw <= 0 or np.isnan(float(d_raw)):
            raise RuntimeError(f"Depth invalid at pixel ({u},{v})")

        # depth 單位：uint16 是毫米，float32 則是公尺
        if self.depth.dtype != np.float32:
            Z = float(d_raw) * 0.001
        else:
            Z = float(d_raw)

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # --- optical frame PoseStamped ---
        ps_cam = PoseStamped()
        ps_cam.header.stamp = rospy.Time(0)
        ps_cam.header.frame_id = self.cam_frame
        ps_cam.pose.position.x = X
        ps_cam.pose.position.y = Y
        ps_cam.pose.position.z = Z
        ps_cam.pose.orientation.w = 1.0

        # --- transform to base_link ---
        T = self.tfbuf.lookup_transform(
            self.base_frame,
            self.cam_frame,
            rospy.Time(0),
            rospy.Duration(1.0)
        )
        ps_base = tf2_geometry_msgs.do_transform_pose(ps_cam, T)
        pos = ps_base.pose.position
        return pos.x, pos.y, pos.z

# ----------------------------------------------------------------------
# CLI 入口點（可輸入像素點）
# ----------------------------------------------------------------------
def main():
    rospy.init_node("pixel_to_base")
    p2b = PixelToBase()

    import sys
    if len(sys.argv) == 3:
        u = int(sys.argv[1])
        v = int(sys.argv[2])
        rospy.sleep(0.5)
        x, y, z = p2b.convert(u, v)
        print(f"[pixel2base] pixel({u},{v}) -> base_link XYZ = ({x:.3f}, {y:.3f}, {z:.3f})")
        return

    rospy.loginfo("Usage: rosrun handeye_test pixel_to_base.py <u> <v>")

if __name__ == "__main__":
    main()
