#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs2

class ClickToPointDebug:
    def __init__(self):
        rospy.init_node("click_to_point_debug_node")
        self.bridge = CvBridge()

        self.latest_color = None
        self.latest_depth = None
        self.clicked_pixel = None

        # 從參數讀 base / camera frame，對應 launch 中的設定
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_cb)

        self.intrinsics = None

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 10, 0.1
        )
        ts.registerCallback(self.image_cb)

    def camera_info_cb(self, msg):
        if self.intrinsics is None:
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = msg.width
            self.intrinsics.height = msg.height
            self.intrinsics.ppx = msg.K[2]
            self.intrinsics.ppy = msg.K[5]
            self.intrinsics.fx = msg.K[0]
            self.intrinsics.fy = msg.K[4]
            self.intrinsics.model = rs2.distortion.none
            self.intrinsics.coeffs = [0, 0, 0, 0, 0]
            rospy.loginfo("Camera intrinsics received")

    def image_cb(self, color_msg, depth_msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

    def process_click(self):
        if self.clicked_pixel is None or self.intrinsics is None or self.latest_depth is None:
            return

        x, y = self.clicked_pixel
        depth = self.latest_depth[y, x] * 0.001  # mm -> m

        # 這裡原本的語法錯誤修正成正常的 if 區塊
        if depth == 0:
            rospy.logwarn("Depth = 0 at clicked point.")
            self.clicked_pixel = None
            return

        # 像素 + 深度 -> 相機座標
        camera_xyz = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)

        point_cam = PoseStamped()
        point_cam.header.stamp = rospy.Time.now()
        point_cam.header.frame_id = self.camera_frame  # 用參數指定的相機 frame
        point_cam.pose.position.x = camera_xyz[0]
        point_cam.pose.position.y = camera_xyz[1]
        point_cam.pose.position.z = camera_xyz[2]
        point_cam.pose.orientation.w = 1.0

        try:
            # 從相機 frame 轉到 base frame，同樣用參數
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            point_base = tf2_geometry_msgs.do_transform_pose(point_cam, tf)

            rospy.loginfo("Clicked pixel: (%d, %d)", x, y)
            rospy.loginfo("→ Camera XYZ: (%.3f, %.3f, %.3f)", *camera_xyz)
            rospy.loginfo("→ Base XYZ: (%.3f, %.3f, %.3f)",
                          point_base.pose.position.x,
                          point_base.pose.position.y,
                          point_base.pose.position.z)
        except Exception as e:
            rospy.logerr("TF transform failed: {}".format(e))

        self.clicked_pixel = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.clicked_pixel = (x, y)

if __name__ == "__main__":
    node = ClickToPointDebug()
    cv2.namedWindow("Color")
    cv2.setMouseCallback("Color", mouse_callback, param=node)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if node.latest_color is not None:
            cv2.imshow("Color", node.latest_color)
        node.process_click()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        rate.sleep()

    cv2.destroyAllWindows()
