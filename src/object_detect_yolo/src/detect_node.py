#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# 轉換器
bridge = CvBridge()

# 載入 YOLOv8s 模型
model = YOLO('yolov8s.pt')  # 預訓練 COCO 模型

def image_callback(msg):
    # ROS Image → OpenCV
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # YOLOv8 推論
    results = model(cv_img, verbose=False)
    result = results[0]  # 只取第一張圖的結果

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # 位置、conf、類別 分開拿
        xyxy = boxes.xyxy.cpu().numpy()   # (N, 4) -> x1, y1, x2, y2
        confs = boxes.conf.cpu().numpy()  # (N,)
        clss  = boxes.cls.cpu().numpy()   # (N,)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = confs[i]
            cls  = clss[i]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            rospy.loginfo(
                f"Class {int(cls)}, conf {conf:.2f}, center pixel ({cx},{cy})"
            )

    # 畫框並顯示
    annotated = result.plot()  # BGR numpy array
    cv2.imshow("YOLOv8 Detection", annotated)
    cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node('yolo_detect_node')

    # 使用 USB CAM 影像 topic
    rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)

    # ---------- RealSense 相關暫時不使用 ----------
    # 若未來要用 RealSense 再把下面打開
    # rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    # ---------------------------------------------

    rospy.loginfo("YOLOv8 detect_node started. Waiting for /usb_cam/image_raw ...")
    rospy.spin()
