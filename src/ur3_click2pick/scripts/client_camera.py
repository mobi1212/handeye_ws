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


class AnyGraspROSClient:
    def __init__(self):
        rospy.init_node('anygrasp_ros_client', anonymous=True)
        self.bridge = CvBridge()
        
        # --- 參數設定 ---
        self.server_addr = "tcp://0.tcp.jp.ngrok.io:17721" # ⚠️ 請更新 Ngrok 網址
        self.model_path = "/home/weilun/handeye_ws/src/ur3_click2pick/weights/best.pt"
        
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
        # 🆕 發佈過濾情報 — 給 MoveIt Octomap 用的「動態去除目標物」深度圖
        self.filtered_depth_pub = rospy.Publisher('/camera/filtered_depth', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=1)
        self.depth_header = None
        self.cached_camera_info = None
        
        # 訂閱影像與深度 (由 rs_camera.launch 提供)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.info_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.camera_info_callback)
        
        # 暫存區與隱形斗篷遮罩
        self.cv_color = None
        self.cv_depth = None
        self.cloak_mask = None  # 🆕 用來記錄目標物位置的 2D 遮罩，全時動態套用

        self.scene = moveit_commander.PlanningSceneInterface()
        # 呼叫加入桌子的函式
        self.add_virtual_table()
        
        print("✅ ROS 節點已啟動，等待影像輸入...")
        print("👉 按下 [s] 發送至 AI 大腦，按下 [q] 退出。")

    def color_callback(self, data):
        try:
            self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def add_virtual_table(self):
        rospy.loginfo("⏳ 正在建立虛擬桌面安全防線...")
        rospy.sleep(2)  # 給 MoveIt 一點時間反應
        
        table_name = "safety_table"
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_link" # 參考手臂底座
        
        # 設定桌子位置 (中心點)
        table_pose.pose.position.x = 0.0
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.015  # 🚀 中心提升至 Z=-0.015m，徹底吞噬桌面雜訊
        table_pose.pose.orientation.w = 1.0
        
        # 設定桌子尺寸 (長, 寬, 厚度)
        # 厚度 6cm: 頂面 Z=+0.015m 底面 Z=-0.045m
        self.scene.add_box(table_name, table_pose, size=(1.5, 1.5, 0.06))
        
        rospy.loginfo("✅ 虛擬桌面安全防線已就位 (Self-Filtering 模式)。")

    def camera_info_callback(self, data):
        self.cached_camera_info = data

    def depth_callback(self, data):
        try:
            self.depth_header = data.header
            # 轉換為 16-bit uint (RealSense 原生格式)
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            # 🚀 移除了凍結影像的判斷式，現在每一幀都是最新的！
        except CvBridgeError as e:
            print(e)

    def publish_depth_to_octomap(self):
        """每一幀持續發佈即時深度圖 + CameraInfo 給 MoveIt Octomap"""
        if self.cv_depth is not None and self.depth_header is not None and self.cached_camera_info is not None:
            # 1. 永遠拿取最新的即時深度圖
            live_depth = self.cv_depth.copy()
            
            # 2. 如果已經鎖定目標 (按過 s)，就在最新畫面上蓋上隱形斗篷
            if self.cloak_mask is not None:
                live_depth[self.cloak_mask > 0] = 0
                
            # 3. 發佈給 Octomap 引擎 (桌面由 safety_table Self-Filtering 自動剔除)
            depth_msg = self.bridge.cv2_to_imgmsg(live_depth, "16UC1")
            depth_msg.header = self.depth_header
            self.filtered_depth_pub.publish(depth_msg)
            
            # 同步發佈 CameraInfo
            info_msg = self.cached_camera_info
            info_msg.header = self.depth_header
            self.camera_info_pub.publish(info_msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.cv_color is None:
                rate.sleep()
                continue

            # 🔄 每一幀都發佈即時深度圖給 MoveIt Octomap
            self.publish_depth_to_octomap()

            # 複製一份影像用於顯示
            display_img = self.cv_color.copy()
            
            # --- YOLO 偵測 ---
            results = self.yolo_model(display_img, verbose=False)
            boxes = results[0].boxes
            best_bbox = None
            best_mask = None
            cls_name = "None"

            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs)) # 找出信心度最高的目標
                
                # 取得 Bounding Box
                x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[best_idx])
                best_bbox = [x1, y1, x2, y2]
                cls_name = self.yolo_model.names.get(int(boxes.cls[best_idx]), "obj")
                
                # 取得 Mask
                if results[0].masks is not None:
                    best_mask = results[0].masks.data[best_idx].cpu().numpy()
                
                # 繪製畫面
                display_img = results[0].plot()

            cv2.imshow("AnyGrasp Client (ROS Mode)", display_img)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break
            
            # --- 當按下 [s] 發送時 ---
            if key & 0xFF == ord('s'):
                if best_bbox is None:
                    print("⚠️ 無法獲取目標 (Bounding Box)！")
                    continue
                
                if best_mask is None:
                    print("⚠️ 警告：模型沒有輸出遮罩 (Mask)！將傳送原始深度圖...")
                    clean_depth = self.cv_depth
                else:
                    print("🎯 啟動「邊緣斷開 + 強制保留桌面」魔法...")
                    
                    # 1. 取得二值化遮罩
                    mask_resized = cv2.resize(best_mask, (self.cv_depth.shape[1], self.cv_depth.shape[0]))
                    obj_mask = (mask_resized > 0.5).astype(np.uint8)

                    # ========== SVD 3D 桌面平面擬合修復 ==========
                    fx, fy = 617.183, 617.122
                    cx, cy = 319.639, 241.404

                    # 2. 建立遮罩：內圈與外圈
                    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
                    mask_inner = cv2.dilate(obj_mask, kernel_inner, iterations=1)
                    mask_outer = cv2.dilate(obj_mask, kernel_outer, iterations=1)

                    moat_mask = cv2.subtract(mask_inner, obj_mask)
                    table_donut_mask = cv2.subtract(mask_outer, mask_inner)

                    # --- 視覺化預覽 ---
                    debug_img = self.cv_color.copy()
                    debug_img[obj_mask > 0] = debug_img[obj_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    debug_img[moat_mask > 0] = debug_img[moat_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
                    debug_img[table_donut_mask > 0] = debug_img[table_donut_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
                    cv2.imwrite("/home/weilun/handeye_ws/donut_preview.jpg", debug_img)
                    print("📸 已儲存甜甜圈遮罩預覽圖至: /home/weilun/handeye_ws/donut_preview.jpg")

                    # 3. 提取 3D 點
                    v_donut, u_donut = np.where((table_donut_mask > 0) & (self.cv_depth > 0))
                    Z_donut = self.cv_depth[v_donut, u_donut].astype(np.float64)
                    X_donut = (u_donut - cx) * Z_donut / fx
                    Y_donut = (v_donut - cy) * Z_donut / fy
                    points_3d = np.stack((X_donut, Y_donut, Z_donut), axis=-1)

                    # 4. SVD 平面擬合
                    centroid = np.mean(points_3d, axis=0)
                    centered = points_3d - centroid
                    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                    normal = Vt[-1]
                    a, b, c = normal
                    d = -np.dot(normal, centroid)

                    # 5. 射線投影回填 (AnyGrasp 大腦專用)
                    v_moat, u_moat = np.where(moat_mask > 0)
                    if len(v_moat) > 0:
                        denom = a * (u_moat - cx) / fx + b * (v_moat - cy) / fy + c
                        denom = np.where(denom == 0, 1e-6, denom)
                        Z_filled = -d / denom
                        Z_filled = np.clip(Z_filled, 0, 65535)

                        clean_depth = self.cv_depth.copy()
                        clean_depth[v_moat, u_moat] = Z_filled.astype(np.uint16)
                    else:
                        clean_depth = self.cv_depth.copy()
                    # ====================================================

                    # 🚀 記錄隱形斗篷遮罩 (全時動態套用)
                    self.cloak_mask = cv2.bitwise_or(obj_mask, moat_mask)
                    print("🛡️ 已啟動全時隱形斗篷！Octomap 將持續動態更新，但目標物區域永久隱形。")

                    # 🆕 生成 YOLO 實體防撞方塊給 MoveIt
                    v_obj, u_obj = np.where((obj_mask > 0) & (self.cv_depth > 0))
                    if len(v_obj) > 0:
                        Z_obj = self.cv_depth[v_obj, u_obj].astype(np.float64) / 1000.0
                        X_obj = (u_obj - cx) * Z_obj / fx
                        Y_obj = (v_obj - cy) * Z_obj / fy
                        
                        min_x, max_x = np.percentile(X_obj, 5), np.percentile(X_obj, 95)
                        min_y, max_y = np.percentile(Y_obj, 5), np.percentile(Y_obj, 95)
                        min_z, max_z = np.percentile(Z_obj, 5), np.percentile(Z_obj, 95)
                        
                        box_size = (max_x - min_x + 0.02, max_y - min_y + 0.02, max_z - min_z + 0.02)
                        
                        box_pose = PoseStamped()
                        box_pose.header.frame_id = "camera_color_optical_frame"
                        box_pose.pose.position.x = (max_x + min_x) / 2
                        box_pose.pose.position.y = (max_y + min_y) / 2
                        box_pose.pose.position.z = (max_z + min_z) / 2
                        box_pose.pose.orientation.w = 1.0
                        
                        self.scene.add_box("target_box", box_pose, size=box_size)
                        print(f"📦 發佈 target_box 防撞方塊, 尺寸: {box_size[0]:.3f}x{box_size[1]:.3f}x{box_size[2]:.3f}m")

                # 發送去背後的 clean_depth 給大腦
                self.process_anygrasp(self.cv_color, clean_depth, best_bbox, cls_name)

            rate.sleep()

        cv2.destroyAllWindows()

    def process_anygrasp(self, color, depth, bbox, name):
        print(f"\n📤 正在發送目標 [{name}] 至桌機運算...")
        start_t = time.time()

        _, encoded = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        payload = {'color_jpg': encoded, 'depth': depth, 'bbox': bbox}
        
        compressed = zlib.compress(pickle.dumps(payload))
        self.socket.send(compressed)
        
        result = self.socket.recv_pyobj()
        print(f"⏱️ 耗時: {time.time() - start_t:.2f}s")

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
            print("🚀 座標已發射！請在 RViz 查看彩色座標軸。")

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
    moveit_commander.roscpp_initialize(sys.argv)
    client = AnyGraspROSClient()
    client.run()