#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import numpy as np
import cv2
import zmq
import time
import zlib
import pickle
import moveit_commander
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_matrix
from ultralytics import YOLO

ZMQ_RECV_TIMEOUT_MS = 30000  # 30 秒無回應則放棄

class AnyGraspROSClient:
    def __init__(self):
        rospy.init_node('anygrasp_ros_client', anonymous=True)
        self.bridge = CvBridge()
        
        # --- 參數設定 ---
        self.server_addr = "tcp://0.tcp.jp.ngrok.io:17721" # ⚠️ 請更新 Ngrok 網址
        self.model_path = "/home/weilun/handeye_ws/src/ur3_click2pick/weights/best.pt"
        
        # --- 1. 初始化 YOLO ---
        print(f"🧠 載入 YOLO 模型: {self.model_path}")
        # 增加信心門檻至 0.25，避免亂抓
        self.yolo_model = YOLO(self.model_path).to("cpu")
        
        # --- 2. 初始化 ZMQ ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)  # 接收逾時
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_RECV_TIMEOUT_MS)  # 發送逾時
        self.socket.setsockopt(zmq.LINGER, 0)                       # 關閉時不等待未送出的訊息
        print(f"🔌 連線至 AnyGrasp Server: {self.server_addr}")
        self.socket.connect(self.server_addr)
        
        # --- 3. ROS 發佈與訂閱 ---
        self.pose_pub = rospy.Publisher('/anygrasp/target_pose', PoseStamped, queue_size=1)
        
        # 訂閱影像與深度
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        
        # 暫存區
        self.cv_color = None
        self.cv_depth = None
        self.manual_bbox = None # 🆕 用於儲存手動框選的 Bbox

        # --- 相機內參 (從 /camera/color/camera_info 動態取得) ---
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.scene = moveit_commander.PlanningSceneInterface()
        self.add_virtual_table()
        
        print("✅ ROS 節點已啟動，等待影像輸入...")
        print("-" * 50)
        print("👉 [s] : 發送目前偵測目標")
        print("👉 [r] : ⚠️ YOLO 偵測不到時，按此鍵手動框選物體")
        print("👉 [c] : 清除手動框選，回到 YOLO 自動偵測")
        print("👉 [q] : 退出程式")
        print("-" * 50)

    def camera_info_callback(self, msg):
        if self.fx is None:  # 只需取一次
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            print(f"📷 相機內參已載入: fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}")
            self.camera_info_sub.unregister()  # 取得後取消訂閱

    def color_callback(self, data):
        try:
            self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def add_virtual_table(self):
        rospy.loginfo("⏳ 正在建立虛擬桌面安全防線...")
        rospy.sleep(2)
        table_name = "safety_table"
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_link"
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.05
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box(table_name, table_pose, size=(1.5, 1.5, 0.01))
        rospy.loginfo("✅ 虛擬桌面防線已就位。")

    def depth_callback(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown():
            if self.cv_color is None:
                continue

            display_img = self.cv_color.copy()
            
            # --- 優先權 1：手動框選模式 ---
            if self.manual_bbox is not None:
                x1, y1, x2, y2 = self.manual_bbox
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(display_img, "MODE: MANUAL ROI", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                best_bbox = self.manual_bbox
                best_mask = None # 手動模式不提供 Mask
                cls_name = "manual_item"
            
            # --- 優先權 2：YOLO 自動偵測模式 ---
            else:
                results = self.yolo_model(display_img, verbose=False, conf=0.2)
                boxes = results[0].boxes
                best_bbox = None
                best_mask = None
                cls_name = "None"

                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.cpu().numpy()
                    best_idx = int(np.argmax(confs))
                    x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[best_idx])
                    best_bbox = [x1, y1, x2, y2]
                    cls_name = self.yolo_model.names.get(int(boxes.cls[best_idx]), "obj")
                    if results[0].masks is not None:
                        best_mask = results[0].masks.data[best_idx].cpu().numpy()
                    display_img = results[0].plot()

            cv2.imshow("AnyGrasp Client (ROS Mode)", display_img)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break
            
            # 🆕 [r] 觸發手動框選 (OpenCV 內建選擇器)
            if key & 0xFF == ord('r'):
                print("\n🖱️ 請在彈出的視窗中框選物體，完成按 [Enter]，取消按 [c]")
                roi = cv2.selectROI("Select Target", self.cv_color, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select Target")
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w, h = map(int, roi)
                    self.manual_bbox = [x, y, x+w, y+h]
                    print(f"✅ 已鎖定手動範圍: {self.manual_bbox}")

            # 🆕 [c] 清除手動模式，回到自動偵測
            if key & 0xFF == ord('c'):
                self.manual_bbox = None
                print("🔄 已切換回 YOLO 自動偵測模式。")

            # --- 當按下 [s] 發送時 ---
            if key & 0xFF == ord('s'):
                if best_bbox is None:
                    print("⚠️ 目前畫面上沒有目標物！請手動框選 [r]")
                    continue

                if self.fx is None:
                    print("⚠️ 相機內參尚未接收，請稍候...")
                    continue

                # 如果沒有 Mask (手動模式)，就直接用原始深度圖發送
                if best_mask is None:
                    print("🛠️ 使用 Bbox 區域發送 (無 Mask 模式)...")
                    clean_depth = self.cv_depth
                else:
                    print("🎯 啟動 SVD 桌面擬合去背...")
                    mask_resized = cv2.resize(best_mask, (self.cv_depth.shape[1], self.cv_depth.shape[0]))
                    obj_mask = (mask_resized > 0.5).astype(np.uint8)

                    # fx, fy = 617.183, 617.122  # 原本硬編碼
                    # cx, cy = 319.639, 241.404  # 原本硬編碼
                    fx, fy = self.fx, self.fy
                    cx, cy = self.cx, self.cy
                    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
                    mask_inner = cv2.dilate(obj_mask, kernel_inner, iterations=1)
                    mask_outer = cv2.dilate(obj_mask, kernel_outer, iterations=1)
                    moat_mask = cv2.subtract(mask_inner, obj_mask)
                    table_donut_mask = cv2.subtract(mask_outer, mask_inner)

                    v_donut, u_donut = np.where((table_donut_mask > 0) & (self.cv_depth > 0))
                    if len(v_donut) > 10:
                        Z_donut = self.cv_depth[v_donut, u_donut].astype(np.float64)
                        X_donut = (u_donut - cx) * Z_donut / fx
                        Y_donut = (v_donut - cy) * Z_donut / fy
                        points_3d = np.stack((X_donut, Y_donut, Z_donut), axis=-1)

                        centroid = np.mean(points_3d, axis=0)
                        centered = points_3d - centroid
                        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                        normal = Vt[-1]
                        a, b, c = normal
                        d = -np.dot(normal, centroid)

                        v_moat, u_moat = np.where(moat_mask > 0)
                        denom = a * (u_moat - cx) / fx + b * (v_moat - cy) / fy + c
                        denom = np.where(denom == 0, 1e-6, denom)
                        Z_filled = -d / denom
                        
                        clean_depth = self.cv_depth.copy()
                        clean_depth[v_moat, u_moat] = np.clip(Z_filled, 0, 65535).astype(np.uint16)
                    else:
                        clean_depth = self.cv_depth

                self.process_anygrasp(self.cv_color, clean_depth, best_bbox, cls_name)

        cv2.destroyAllWindows()

    def process_anygrasp(self, color, depth, bbox, name):
        print(f"\n📤 正在發送目標 [{name}] 至大腦...")
        start_t = time.time()
        _, encoded = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        payload = {'color_jpg': encoded, 'depth': depth, 'bbox': bbox}
        compressed = zlib.compress(pickle.dumps(payload))
        try:
            self.socket.send(compressed)
            result = self.socket.recv_pyobj()
        except zmq.Again:
            print(f"⏰ ZMQ 逾時 ({ZMQ_RECV_TIMEOUT_MS//1000}s)：Server 無回應，請確認連線")
            # REQ socket 逾時後狀態損毀，需重建
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_RECV_TIMEOUT_MS)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(self.server_addr)
            return
        print(f"⏱️ 運算耗時: {time.time() - start_t:.2f}s")

        if result['status'] == 'success':
            print(f"🎯 獲得 6D 座標，分數: {result['score']:.4f}")
            tvec = np.array(result['translation'])
            rot_mat = np.array(result['rotation'])
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "camera_color_optical_frame"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
            T = np.eye(4); T[:3, :3] = rot_mat
            q = quaternion_from_matrix(T)
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = q
            self.pose_pub.publish(pose_msg)
            print("🚀 座標已發佈至 /anygrasp/target_pose")
            self.draw_ar_gripper(color, result)

    def draw_ar_gripper(self, img, res):
        t, r = np.array(res['translation']), np.array(res['rotation'])
        rvec, _ = cv2.Rodrigues(r)
        # K = np.array([[617.183, 0.0, 319.639], [0.0, 617.122, 241.404], [0.0, 0.0, 1.0]])  # 原本硬編碼
        K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
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
    moveit_commander.roscpp_initialize(sys.argv)
    client = AnyGraspROSClient()
    client.run()