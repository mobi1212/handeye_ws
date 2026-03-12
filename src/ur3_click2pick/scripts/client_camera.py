#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import zmq
import time
import zlib
import pickle
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_matrix
from ultralytics import YOLO

class AnyGraspROSClient:
    def __init__(self):
        rospy.init_node('anygrasp_ros_client', anonymous=True)
        self.bridge = CvBridge()
        
        # --- 參數設定 ---
        self.server_addr = "tcp://0.tcp.jp.ngrok.io:17429" # ⚠️ 請更新 Ngrok 網址
        self.model_path = "/home/weilun/handeye_ws/src/object_detect_yolo/src/best.pt"
        
        # --- 1. 初始化 YOLO ---
        print(f"🧠 載入 YOLO 模型: {self.model_path}")
        self.yolo_model = YOLO(self.model_path).to("cpu")
        
        # --- 2. 初始化 ZMQ ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        print(f"🔌 連線至 AnyGrasp Server: {self.server_addr}")
        self.socket.connect(self.server_addr)
        
        # --- 3. ROS 發佈與訂閱 ---
        self.pose_pub = rospy.Publisher('/anygrasp/target_pose', PoseStamped, queue_size=1)
        
        # 訂閱影像與深度 (由 rs_camera.launch 提供)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        
        # 暫存區
        self.cv_color = None
        self.cv_depth = None
        
        print("✅ ROS 節點已啟動，等待影像輸入...")
        print("👉 按下 [s] 發送至 AI 大腦，按下 [q] 退出。")

    def color_callback(self, data):
        try:
            self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def depth_callback(self, data):
        try:
            # 轉換為 16-bit uint (RealSense 原生格式)
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown():
            if self.cv_color is None:
                continue

            # 複製一份影像用於顯示
            display_img = self.cv_color.copy()
            
            # --- YOLO 偵測 ---
            results = self.yolo_model(display_img, verbose=False)
            boxes = results[0].boxes
            best_bbox = None
            cls_name = "None"

            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[best_idx])
                best_bbox = [x1, y1, x2, y2]
                cls_name = self.yolo_model.names.get(int(boxes.cls[best_idx]), "obj")
                
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"{cls_name} {confs[best_idx]:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("AnyGrasp Client (ROS Mode)", display_img)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break
            
            if key & 0xFF == ord('s'):
                if best_bbox is None or self.cv_depth is None:
                    print("⚠️ 無法獲取目標或深度資訊。")
                    continue
                
                self.process_anygrasp(self.cv_color, self.cv_depth, best_bbox, cls_name)

        cv2.destroyAllWindows()

    def process_anygrasp(self, color, depth, bbox, name):
        print(f"\n📤 正在發送目標 [{name}] 至桌機運算...")
        start_t = time.time()

        # 1. 極限壓縮
        _, encoded = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        payload = {'color_jpg': encoded, 'depth': depth, 'bbox': bbox}
        
        # 2. 發送
        compressed = zlib.compress(pickle.dumps(payload))
        self.socket.send(compressed)
        
        # 3. 接收
        result = self.socket.recv_pyobj()
        print(f"⏱️ 耗時: {time.time() - start_t:.2f}s")

        if result['status'] == 'success':
            print(f"🎯 獲得 6D 座標，分數: {result['score']:.4f}")
            
            # --- 4. 發佈 ROS Pose ---
            tvec = np.array(result['translation'])
            rot_mat = np.array(result['rotation'])
            
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "camera_link"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
            
            T = np.eye(4); T[:3, :3] = rot_mat
            q = quaternion_from_matrix(T)
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = q
            
            self.pose_pub.publish(pose_msg)
            print("🚀 座標已發射！請在 RViz 查看彩色座標軸。")

            # --- 5. AR 預覽繪製 (確認邏輯) ---
            self.draw_ar_gripper(color, result)

    def draw_ar_gripper(self, img, res):
        t, r = np.array(res['translation']), np.array(res['rotation'])
        rvec, _ = cv2.Rodrigues(r)
        K = np.array([[617.183, 0.0, 319.639], [0.0, 617.122, 241.404], [0.0, 0.0, 1.0]])
        w, d = res['width'], res['depth']
        g3d = np.array([[-d-0.06,0,0],[-d,0,0],[-d,-w/2,0],[-d,w/2,0],[0,-w/2,0],[0,w/2,0]], dtype=np.float32)
        pts, _ = cv2.projectPoints(g3d, rvec, t, K, np.zeros(4))
        p = np.int32(pts).reshape(-1, 2)
        cv2.line(img, tuple(p[0]), tuple(p[1]), (255,0,0), 3)
        cv2.line(img, tuple(p[2]), tuple(p[3]), (0,255,0), 3)
        cv2.line(img, tuple(p[2]), tuple(p[4]), (0,255,0), 3)
        cv2.line(img, tuple(p[3]), tuple(p[5]), (0,255,0), 3)
        cv2.imshow("AR Preview (Press any key)", img)
        cv2.waitKey(0)
        cv2.destroyWindow("AR Preview (Press any key)")

if __name__ == "__main__":
    client = AnyGraspROSClient()
    client.run()