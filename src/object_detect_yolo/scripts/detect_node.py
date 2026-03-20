#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import torch
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber


class YoloClickNode:
    def __init__(self):
        # ---- 模型 ----
        default_path = "/home/weilun/handeye_ws/src/object_detect_yolo/src/best.pt"
        model_path = rospy.get_param("~model_path", default_path)

        rospy.loginfo(f"[YOLO] loading model from: {model_path}")
        device = "cpu"
        self.model = YOLO(model_path).to(device)
        rospy.loginfo(f"[YOLO] model loaded on device: {device}")

        # ---- Image / Depth ----
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.depth_valid = False  # 確保深度同步正確

        # 同步訂閱
        color_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        depth_topic = "/camera/aligned_depth_to_color/image_raw"

        color_sub = Subscriber(color_topic, Image)
        depth_sub = Subscriber(depth_topic, Image)

        ats = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.03
        )
        ats.registerCallback(self.synced_callback)

        # ---- 滑鼠點擊座標 ----
        self.mouse_click = None  # (x, y)

        # ---- YOLO 儲存值 ----
        self.current_center = None

        cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("YOLOv8 Detection", self.on_mouse)

        # ---- Publisher ----
        self.pub_target = rospy.Publisher("/yolo/target_pixel", Point, queue_size=1)

        rospy.loginfo("[YOLO] Ready.")
        rospy.loginfo("[YOLO] 持續偵測中")
        rospy.loginfo("[YOLO] 按 Enter → 發送像素")
        rospy.loginfo("[YOLO] 按 ESC → 離開")
        rospy.loginfo("[YOLO] 滑鼠左鍵 → 直接點擊發送像素（立即送）")

    # -------------------------------------------------------
    #  同步 Color + Depth（解決之前的誤差來源）
    # -------------------------------------------------------
    def synced_callback(self, color_msg, depth_msg):
        try:
            self.color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            self.depth_valid = True
        except Exception as e:
            rospy.logwarn(f"[SYNC] cv_bridge error: {e}")
            self.depth_valid = False

    # -------------------------------------------------------
    # 滑鼠事件：左鍵直接送像素（立即）
    # -------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click = (x, y)
            pt = Point(float(x), float(y), 0.0)
            self.pub_target.publish(pt)
            rospy.loginfo(f"[YOLO] 滑鼠點擊發送像素: ({x}, {y})")

    # -------------------------------------------------------
    # YOLO 偵測
    # -------------------------------------------------------
    def run_detection(self, frame):
        results = self.model(frame, verbose=False)
        result = results[0]

        boxes = result.boxes
        best_center = None

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))

            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            x1, y1, x2, y2 = xyxy[best_idx]
            conf = confs[best_idx]
            cls = int(clss[best_idx])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            best_center = (cx, cy)

            cls_name = self.model.names.get(cls, "unknown")
            rospy.loginfo_throttle(
                1.0,
                f"[YOLO] best class={cls_name}, conf={conf:.2f}, center=({cx},{cy})"
            )

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1i, y1i - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame, best_center

    # -------------------------------------------------------
    # 主迴圈
    # -------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            if self.color is None:
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow("YOLOv8 Detection", black)
                cv2.waitKey(1)
                rate.sleep()
                continue

            frame_copy = self.color.copy()
            frame_disp, center = self.run_detection(frame_copy)
            self.current_center = center

            cv2.imshow("YOLOv8 Detection", frame_disp)
            key = cv2.waitKey(1) & 0xFF

            # -------------------------------
            # Enter → 發布 YOLO 偵測像素
            # -------------------------------
            if key == 13:  # Enter
                if self.current_center is not None:
                    cx, cy = self.current_center
                    pt = Point(float(cx), float(cy), 0.0)
                    self.pub_target.publish(pt)
                    rospy.loginfo(f"[YOLO] Enter 發送像素: ({cx}, {cy})")
                else:
                    rospy.logwarn("[YOLO] 沒有偵測框，無法發送。")

            # -------------------------------
            # ESC → 離開
            # -------------------------------
            elif key == 27:
                rospy.loginfo("[YOLO] ESC pressed. Exit.")
                break

            rate.sleep()

        cv2.destroyAllWindows()


def main():
    rospy.init_node("yolo_click_node")
    node = YoloClickNode()
    node.spin()


if __name__ == "__main__":
    main()
