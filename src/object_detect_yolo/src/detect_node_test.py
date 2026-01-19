#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

bridge = CvBridge()
model = YOLO('yolov8s.pt')

last_frame = None  # 存最新一張影像


def image_callback(msg):
    global last_frame
    # RealSense color Image -> OpenCV BGR
    last_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def run_detection(frame):
    """對傳入的 frame 做 YOLO 偵測並回傳標註後的影像"""
    results = model(frame, verbose=False)
    result = results[0]

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()   # (N, 4): x1, y1, x2, y2
        confs = boxes.conf.cpu().numpy()  # (N,)
        clss  = boxes.cls.cpu().numpy()   # (N,)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = confs[i]
            cls  = int(clss[i])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            rospy.loginfo(
                f"[YOLO] class={cls} ({model.names.get(cls, 'unknown')}), "
                f"conf={conf:.2f}, center_pixel=({cx}, {cy})"
            )

    annotated = result.plot()  # BGR numpy array
    return annotated


if __name__ == "__main__":
    rospy.init_node('yolo_detect_node')

    # 用 RealSense color topic
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

    rospy.loginfo(
        "YOLOv8 detect_node started. Waiting for /camera/color/image_raw ..."
    )

    rate = rospy.Rate(10)  # 10Hz loop

    while not rospy.is_shutdown():
        if last_frame is not None:
            # 顯示原始畫面
            cv2.imshow("RealSense Color", last_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('c'):
            # 按下 c：對當前畫面做偵測
            if last_frame is not None:
                frame_copy = last_frame.copy()
                annotated = run_detection(frame_copy)
                cv2.imshow("YOLOv8 Detection (snapshot)", annotated)
                cv2.waitKey(1)

        elif key == ord('q'):
            # 按下 q：退出
            rospy.loginfo("Quit YOLO detect_node.")
            break

        rate.sleep()

    cv2.destroyAllWindows()
