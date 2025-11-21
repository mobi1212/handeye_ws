#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import cv2
import message_filters
import geometry_msgs.msg
import moveit_commander

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import pyrealsense2 as rs2
from tf.transformations import quaternion_from_euler

from robotiq_gripper import RobotiqGripper

class ClickToPick:
    def __init__(self):
        rospy.init_node("click_to_pick_node")
        moveit_commander.roscpp_initialize([])
        self.bridge = CvBridge()

        # 初始化 MoveIt
        self.arm = moveit_commander.MoveGroupCommander("manipulator")
        self.arm.set_planning_time(5)

        # 初始化 Robotiq TCP 夾爪
        self.gripper = RobotiqGripper()
        # gripper_ip = "192.168.86.7"
        # self.gripper.connect(gripper_ip, 63352)
        # 使用參數指定夾爪連線資訊
        gripper_ip = rospy.get_param('~gripper_host', '192.168.86.7')
        gripper_port = rospy.get_param('~gripper_port', 63352)
        rospy.loginfo("Connecting to gripper via TCP...")
        self.gripper.connect(gripper_ip, gripper_port)
        self.gripper.activate()

        # 初始化 TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 讀取座標系與相機 Topic 參數
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        # 原本預設 camera_frame 使用 camera_link
        # self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        # 與 handeye 標定一致，改用 camera_color_optical_frame 為預設
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')

        color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        camera_info_topic = rospy.get_param('~camera_info', '/camera/color/camera_info')

        # 訂閱影像
        self.latest_color = None
        self.latest_depth = None
        self.intrinsics = None
        self.clicked_pixel = None
        self.approach_height = 0.20  # 上方補償高度
        self.min_pick_z = 0.15       # 最低夾取 Z 高度

        # 原本直接訂閱固定的 Realsense topic
        # self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        # self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        # self.info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_cb)

        # 改為使用參數指定的 topic
        self.color_sub = message_filters.Subscriber(color_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_cb)

        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
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

    def move_gripper(self, open=True):
        if open:
            rospy.loginfo("Opening gripper...")
            self.gripper.move_and_wait_for_pos(self.gripper.get_open_position(), speed=128, force=128)
        else:
            rospy.loginfo("Closing gripper...")
            # self.gripper.move_and_wait_for_pos(self.gripper.get_closed_position(), speed=128, force=30)
            self.gripper.move_and_wait_for_pos(position=150, speed=128, force=30)

    def move_to_pose(self, pose):
        self.arm.set_pose_target(pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    def offset_pose_z(self, pose, dz):
        offset = geometry_msgs.msg.Pose()
        offset.position.x = pose.position.x
        offset.position.y = pose.position.y
        offset.position.z = pose.position.z + dz
        offset.orientation = pose.orientation
        return offset

    def process_click(self):
        if self.clicked_pixel is None or self.intrinsics is None or self.latest_depth is None:
            return

        x, y = self.clicked_pixel
        depth = self.latest_depth[y, x] * 0.001
        if depth == 0:
            rospy.logwarn("Depth = 0 at clicked point.")
            self.clicked_pixel = None
            return

        camera_xyz = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        point_cam = PoseStamped()
        # 原本硬寫 camera_link
        # point_cam.header.frame_id = "camera_link"
        # 改為使用參數指定的 camera_frame（預設 camera_color_optical_frame）
        point_cam.header.frame_id = self.camera_frame
        point_cam.pose.position.x = camera_xyz[0]
        point_cam.pose.position.y = camera_xyz[1]
        point_cam.pose.position.z = camera_xyz[2]
        point_cam.pose.orientation.w = 1.0

        try:
            # camera → base
            # 使用 base_frame / camera_frame 參數查詢 TF
            # tf = self.tf_buffer.lookup_transform("base_link", "camera_link", rospy.Time(0), rospy.Duration(1.0))
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0))
            point_base = tf2_geometry_msgs.do_transform_pose(point_cam, tf)
            base_pose = point_base.pose

            # 強制夾爪朝下姿態
            q = quaternion_from_euler(np.pi, 0, 0)
            base_pose.orientation.x = q[0]
            base_pose.orientation.y = q[1]
            base_pose.orientation.z = q[2]
            base_pose.orientation.w = q[3]

            # 上方姿態
            pose_above = self.offset_pose_z(base_pose, self.approach_height)

            # 安全限制最低 Z
            safe_pose = geometry_msgs.msg.Pose()
            safe_pose.position.x = base_pose.position.x
            safe_pose.position.y = base_pose.position.y
            safe_pose.position.z = max(base_pose.position.z, self.min_pick_z)
            safe_pose.orientation = base_pose.orientation

            # 抓取流程
            self.move_gripper(open=True)
            self.move_to_pose(pose_above)
            self.move_to_pose(safe_pose)
            self.move_gripper(open=False)
            self.move_to_pose(pose_above)

            # 移動到放置位置
            # 原本使用硬編碼的放置位置
            # place_pose = geometry_msgs.msg.Pose()
            # place_pose.position.x = 0.20630931738472915
            # place_pose.position.y = 0.14777829532821804
            # place_pose.position.z = 0.20655337742772715
            # place_pose.orientation.x = -0.9979443524730374
            # place_pose.orientation.y = 0.056801437612788844
            # place_pose.orientation.z = -0.029643570790565982
            # place_pose.orientation.w = 0.001387358308193802

            # 改為從參數讀取放置位姿，若未設定則退回上述預設值
            place_pose = geometry_msgs.msg.Pose()
            place_pose.position.x = rospy.get_param('~place_pose/x', 0.20630931738472915)
            place_pose.position.y = rospy.get_param('~place_pose/y', 0.14777829532821804)
            place_pose.position.z = rospy.get_param('~place_pose/z', 0.20655337742772715)
            place_pose.orientation.x = rospy.get_param('~place_pose/qx', -0.9979443524730374)
            place_pose.orientation.y = rospy.get_param('~place_pose/qy', 0.056801437612788844)
            place_pose.orientation.z = rospy.get_param('~place_pose/qz', -0.029643570790565982)
            place_pose.orientation.w = rospy.get_param('~place_pose/qw', 0.001387358308193802)

            self.move_to_pose(place_pose)
            self.move_gripper(open=True)
            rospy.loginfo("物品已放置至指定位置")

        except Exception as e:
            rospy.logerr("TF transform or motion failed: {}".format(e))

        self.clicked_pixel = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.clicked_pixel = (x, y)

if __name__ == "__main__":
    node = ClickToPick()
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
