#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ctypes
ctypes.CDLL("/lib/x86_64-linux-gnu/libffi.so.7", mode=ctypes.RTLD_GLOBAL)
import sys
import os
import json
import threading
import rospy
import numpy as np
import cv2
import zmq
import time
import zlib
import pickle
try:
    import moveit_commander
    MOVEIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  moveit_commander 無法載入（無手臂模式）: {e}")
    MOVEIT_AVAILABLE = False
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
# cv_bridge 在 conda 環境下有 libffi 衝突，改用手動轉換
# from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_matrix

ZMQ_RECV_TIMEOUT_MS = 60000  # 30 秒無回應則放棄

class AnyGraspROSClient:
    def __init__(self):
        rospy.init_node('anygrasp_ros_client', anonymous=True)
        self.use_moveit = bool(rospy.get_param('~use_moveit', False))

        # --- 參數設定 ---
        # self.server_addr = "tcp://140.124.181.184:5555"
        self.server_addr = "tcp://0.tcp.jp.ngrok.io:11541" # ⚠️ 請更新 Ngrok 網址

        # --- 1. 初始化 ZMQ ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_RECV_TIMEOUT_MS)
        self.socket.setsockopt(zmq.LINGER, 0)
        print(f"🔌 連線至 AnyGrasp Server: {self.server_addr}")
        self.socket.connect(self.server_addr)

        # --- 2. ROS 發佈與訂閱 ---
        self.pose_pub = rospy.Publisher('/anygrasp/target_pose', PoseStamped, queue_size=1)
        self.grasp_pub = rospy.Publisher('/anygrasp/target_grasp', String, queue_size=1)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)

        # 暫存區
        self.cv_color = None
        self.cv_depth = None
        self.manual_bbox = None

        # --- 3. 目標模式 ---
        self.vlm_target = None
        self.last_vlm_target = None   # 上一次輸入的物件名稱
        self.last_vlm_result = None   # 上一次完整 VLM 結果，供 [s] 只重算 AnyGrasp

        # --- 相機內參 ---
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        # --- 發送狀態 ---
        self.sending = False        # 背景 thread 正在發送中
        self.send_done = False      # 本次目標已發送完畢
        self.send_count = 0         # 累計發送次數（供顯示）
        self.ar_overlay_img = None  # 抓取結果 AR 疊加圖（靜態，直到下一次結果覆蓋）

        if MOVEIT_AVAILABLE and self.use_moveit:
            self.scene = moveit_commander.PlanningSceneInterface()
            self.add_virtual_table()
        elif MOVEIT_AVAILABLE:
            print("ℹ️  已跳過 MoveIt 初始化（~use_moveit:=false），可直接做無手臂測試。")

        print("✅ ROS 節點已啟動，等待影像輸入...")
        print("-" * 50)
        print("👉 [v] : VLM 模式（直接送目標名稱到遠端 server）")
        print("👉 [r] : 手動框選物體")
        print("👉 [s] : 重新發送目前目標給 AnyGrasp（重算抓取姿態）")
        print("👉 [c] : 清除 / 重置")
        print("👉 [q] : 退出")
        print("-" * 50)

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            print(f"📷 相機內參已載入: fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}")
            self.camera_info_sub.unregister()

    def color_callback(self, data):
        try:
            img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            if data.encoding == "rgb8":
                img = img[:, :, ::-1]
            self.cv_color = np.ascontiguousarray(img)
        except Exception as e:
            print(f"color_callback error: {e}")

    def depth_callback(self, data):
        try:
            self.cv_depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)
        except Exception as e:
            print(f"depth_callback error: {e}")

    def add_virtual_table(self):
        rospy.loginfo("⏳ 正在建立虛擬桌面安全防線...")
        rospy.sleep(2)
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_link"
        table_pose.pose.position.z = -0.05
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box("safety_table", table_pose, size=(1.5, 1.5, 0.01))
        rospy.loginfo("✅ 虛擬桌面防線已就位。")

    def run(self):
        while not rospy.is_shutdown():
            if self.cv_color is None:
                continue

            display_img = self.cv_color.copy()
            best_bbox = None
            best_mask = None
            cls_name = "None"
            current_mode = None
            can_reuse_vlm = False

            # --- 模式 1：手動框選 ---
            if self.manual_bbox is not None:
                x1, y1, x2, y2 = self.manual_bbox
                best_bbox = self.manual_bbox
                best_mask = None
                cls_name = "manual_item"
                current_mode = "roi"

            # --- 模式 2：VLM 直接送到遠端 server ---
            elif self.vlm_target:
                cls_name = self.vlm_target
                current_mode = "vlm"
                if self.last_vlm_result and self.last_vlm_result.get("object_name") == self.vlm_target:
                    best_bbox = self.last_vlm_result.get("bbox")
                    cached_mask = self.last_vlm_result.get("mask")
                    if cached_mask is not None:
                        best_mask = cached_mask
                    can_reuse_vlm = True
                status_text = f"VLM target: '{self.vlm_target}'"
                if self.sending:
                    status_text += " (sending...)"
                elif self.send_done:
                    if can_reuse_vlm:
                        status_text += f" (sent x{self.send_count}) [s]=regrasp"
                    else:
                        status_text += f" (sent x{self.send_count})"
                else:
                    status_text += " (ready)"
                cv2.putText(display_img, status_text,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            # --- 無目標：顯示待機畫面 ---
            else:
                cv2.putText(display_img, "[v] VLM  [r] Manual ROI",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # --- 遮罩半透明疊加 ---
            if best_mask is not None:
                h, w = display_img.shape[:2]
                m = cv2.resize(best_mask.astype(np.float32), (w, h)) > 0.5
                overlay = display_img.copy()
                overlay[m] = [0, 255, 255]  # 黃色
                display_img = cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0)

            if best_bbox is not None:
                x1, y1, x2, y2 = best_bbox
                box_color = (0, 200, 0) if self.send_done else (0, 255, 0)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, 2)
                label = cls_name
                if self.send_done and current_mode == "vlm" and can_reuse_vlm:
                    label += f" (sent x{self.send_count}) [s]=regrasp"
                elif self.send_done:
                    label += f" (sent x{self.send_count})"
                cv2.putText(display_img, label, (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # --- 首次自動發送（目標就緒且尚未發送過）---
            if (current_mode is not None and self.fx is not None
                    and not self.sending and not self.send_done
                    and self.cv_depth is not None):
                self._trigger_send(current_mode, best_mask, best_bbox, cls_name)

            # --- 狀態提示（左側即時畫面）---
            h_img, w_img = display_img.shape[:2]
            if self.sending:
                cv2.putText(display_img, "SENDING...", (10, h_img - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv2.putText(display_img, "LIVE + MASK", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # --- 右側：靜態抓取姿態（有結果前顯示佔位）---
            if self.ar_overlay_img is not None:
                right_panel = cv2.resize(self.ar_overlay_img,
                                         (w_img, h_img), interpolation=cv2.INTER_LINEAR)
                cv2.putText(right_panel, f"GRASP POSE (x{self.send_count})",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                right_panel = np.zeros((h_img, w_img, 3), dtype=np.uint8)
                cv2.putText(right_panel, "GRASP POSE", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(right_panel, "waiting for result...",
                            (10, h_img // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            combined = np.hstack([display_img, right_panel])
            cv2.imshow("AnyGrasp Client (ROS Mode)", combined)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

            # [v] VLM+SAM 模式
            if key & 0xFF == ord('v'):
                hint = f"（上次：{self.last_vlm_target}，直接 Enter 沿用）" if self.last_vlm_target else "（英文，例如：bottle）"
                prompt = input(f"\n🔍 請輸入目標物件 {hint}：").strip()
                if not prompt and self.last_vlm_target:
                    prompt = self.last_vlm_target
                if prompt:
                    self.last_vlm_target = prompt
                    self.vlm_target = prompt
                    self.manual_bbox = None
                    self.send_done = False
                    self.send_count = 0
                    self.ar_overlay_img = None
                    print(f"⚡ 已鎖定 VLM 目標 '{prompt}'，準備直接送往遠端 server...")

            # [r] 手動框選
            if key & 0xFF == ord('r'):
                print("\n🖱️ 請在彈出的視窗中框選物體，完成按 [Enter]，取消按 [c]")
                roi = cv2.selectROI("Select Target", self.cv_color, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select Target")
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w, h = map(int, roi)
                    self.manual_bbox = [x, y, x+w, y+h]
                    self.vlm_target = None
                    self.send_done = False
                    self.send_count = 0
                    self.ar_overlay_img = None
                    print(f"✅ 已鎖定手動範圍: {self.manual_bbox}，自動發送中...")

            # [s] 重新發送
            if key & 0xFF == ord('s'):
                if self.sending:
                    print("⚠️ 上一次發送尚未完成，請稍候")
                elif self.fx is None or self.cv_depth is None:
                    print("⚠️ 相機資料尚未就緒")
                elif self.vlm_target and self.last_vlm_result and self.last_vlm_result.get("object_name") == self.vlm_target:
                    print(f"🔄 沿用上一次 VLM 結果，對 [{cls_name}] 只重算 AnyGrasp 姿態...")
                    self._trigger_reuse_vlm()
                elif best_bbox is not None:
                    print(f"🔄 重新發送 [{cls_name}] 給 AnyGrasp 重算姿態...")
                    self._trigger_send(current_mode, best_mask, best_bbox, cls_name)
                else:
                    print("⚠️ 目前沒有可重送的目標，請先用 [v] 或 [r] 選取")

            # [c] 清除 / 重置
            if key & 0xFF == ord('c'):
                self.manual_bbox = None
                self.vlm_target = None
                self.last_vlm_result = None
                self.send_done = False
                self.send_count = 0
                self.ar_overlay_img = None
                print("🔄 已重置，請用 [v] 或 [r] 選取新目標。")

        cv2.destroyAllWindows()

    def _trigger_send(self, mode, best_mask, bbox, name):
        """拍快照並啟動背景發送 thread"""
        color_snap = self.cv_color.copy()
        depth_snap = self.cv_depth.copy()
        mask_snap = best_mask.copy() if best_mask is not None else None
        self.sending = True
        threading.Thread(
            target=self._send_worker,
            args=(mode, color_snap, depth_snap, mask_snap,
                  list(bbox) if bbox is not None else None, name),
            daemon=True
        ).start()

    def _trigger_reuse_vlm(self):
        """沿用上一次完整 VLM 結果，只重算 AnyGrasp 姿態"""
        cached = self.last_vlm_result
        if not cached:
            print("⚠️ 尚無可重用的 VLM 結果")
            return
        color_snap = self.cv_color.copy()
        depth_snap = self.cv_depth.copy()
        self.sending = True
        threading.Thread(
            target=self._send_worker_reuse_vlm,
            args=(color_snap, depth_snap, cached),
            daemon=True
        ).start()

    def _send_worker(self, mode, color, depth_raw, best_mask, bbox, name):
        """背景執行緒：依模式發送到遠端 server"""
        try:
            if mode == "vlm":
                print(f"🔍 直接送 VLM 請求到遠端 server：'{name}'")
                self.process_anygrasp(color, depth_raw, None, name, None, mode="vlm")
            else:
                print("🛠️ 使用 ROI 模式發送到遠端 server...")
                self.process_anygrasp(color, depth_raw, bbox, name, best_mask, mode="roi")
        finally:
            self.sending = False
            self.send_done = True
            self.send_count += 1

    def _send_worker_reuse_vlm(self, color, depth_raw, cached):
        try:
            object_name = cached.get("object_name", self.vlm_target or "unknown")
            print(f"♻️ 使用 cached VLM 結果重算 [{object_name}] 的 AnyGrasp 姿態...")
            self.process_anygrasp(
                color,
                depth_raw,
                cached.get("bbox"),
                object_name,
                cached.get("mask"),
                mode="reuse_vlm",
                object_shape=cached.get("object_shape"),
                target_grids=cached.get("target_grids"),
                reasoning=cached.get("reasoning", ""),
                estimated_com_grid=cached.get("estimated_com_grid", "N/A"),
                svd_mask=cached.get("svd_mask"),
            )
        finally:
            self.sending = False
            self.send_done = True
            self.send_count += 1

    def process_anygrasp(
        self, color, depth, bbox, name, best_mask=None, mode="legacy",
        object_shape=None, target_grids=None, reasoning="", estimated_com_grid="N/A",
        svd_mask=None,
    ):
        print(f"\n📤 正在發送目標 [{name}] 至 server（mode={mode}）...")
        start_t = time.time()
        _, encoded = cv2.imencode('.jpg', color, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        payload = {
            'mode': mode,
            'object_name': name,
            'color_jpg': encoded,
            'depth': depth,
            'intrinsics': {'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy}
        }
        if bbox is not None:
            payload['bbox'] = bbox
        if object_shape:
            payload['object_shape'] = object_shape
        if target_grids:
            payload['target_grids'] = target_grids
        if reasoning:
            payload['reasoning'] = reasoning
        if estimated_com_grid:
            payload['estimated_com_grid'] = estimated_com_grid

        if best_mask is not None:
            mask_resized = cv2.resize(
                best_mask.astype(np.uint8) * 255,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            ok, mask_encoded = cv2.imencode('.png', mask_resized)
            if ok:
                payload['mask'] = mask_encoded
                print(f"🎭 已附帶 target mask，像素數: {int((mask_resized > 127).sum())}")
        if svd_mask is not None:
            svd_mask_resized = cv2.resize(
                svd_mask.astype(np.uint8) * 255,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            ok, svd_mask_encoded = cv2.imencode('.png', svd_mask_resized)
            if ok:
                payload['svd_mask'] = svd_mask_encoded

        compressed = zlib.compress(pickle.dumps(payload))
        try:
            self.socket.send(compressed)
            result = self.socket.recv_pyobj()
        except zmq.Again:
            print(f"⏰ ZMQ 逾時 ({ZMQ_RECV_TIMEOUT_MS//1000}s)：Server 無回應，請確認連線")
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
            if result.get('target_grids'):
                print(f"   target_grids: {result.get('target_grids')}")
            if result.get('object_shape'):
                print(f"   object_shape: {result.get('object_shape')}")
            if result.get('mode') == 'vlm':
                cached_mask = None
                cached_svd_mask = None
                if result.get('mask') is not None:
                    mask_img = cv2.imdecode(result['mask'], cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        cached_mask = (mask_img > 127).astype(np.float32)
                if result.get('svd_mask') is not None:
                    svd_mask_img = cv2.imdecode(result['svd_mask'], cv2.IMREAD_GRAYSCALE)
                    if svd_mask_img is not None:
                        cached_svd_mask = (svd_mask_img > 127).astype(np.float32)
                self.last_vlm_result = {
                    "object_name": name,
                    "bbox": result.get('bbox'),
                    "object_shape": result.get('object_shape'),
                    "target_grids": result.get('target_grids', []),
                    "reasoning": result.get('reasoning', ''),
                    "estimated_com_grid": result.get('estimated_com_grid', 'N/A'),
                    "mask": cached_mask,
                    "svd_mask": cached_svd_mask if cached_svd_mask is not None else cached_mask,
                }
                print("💾 已快取本次 VLM 結果，之後可按 [s] 只重算 AnyGrasp。")
            tvec = np.array(result['translation'])
            rot_mat = np.array(result['rotation'])
            stamp = rospy.Time.now()

            grasp_meta = {
                "frame_id": "camera_color_optical_frame",
                "stamp": stamp.to_sec(),
                "translation": tvec.tolist(),
                "rotation": rot_mat.tolist(),
                "score": float(result.get('score', 0.0)),
                "width": float(result.get('width', 0.0)),
                "depth": float(result.get('depth', 0.0)),
            }
            self.grasp_pub.publish(json.dumps(grasp_meta))

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "camera_color_optical_frame"
            pose_msg.header.stamp = stamp
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
            T = np.eye(4); T[:3, :3] = rot_mat
            q = quaternion_from_matrix(T)
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            self.pose_pub.publish(pose_msg)
            print("🚀 座標已發佈至 /anygrasp/target_pose")
            self.draw_ar_gripper(color, result)

    def draw_ar_gripper(self, img, res):
        """將 AR 夾爪畫在圖上，存入 self.ar_overlay_img 供主循環顯示 5 秒"""
        t, r = np.array(res['translation']), np.array(res['rotation'])
        rvec, _ = cv2.Rodrigues(r)
        K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        w, d = res['width'], res['depth']
        g3d = np.array([[-d-0.06,0,0],[-d,0,0],[-d,-w/2,0],[-d,w/2,0],[0,-w/2,0],[0,w/2,0]], dtype=np.float32)
        pts, _ = cv2.projectPoints(g3d, rvec, t, K, np.zeros(4))
        p = np.int32(pts).reshape(-1, 2)
        drawn = img.copy()
        cv2.line(drawn, tuple(p[0]), tuple(p[1]), (255,0,0), 3)
        cv2.line(drawn, tuple(p[2]), tuple(p[3]), (0,255,0), 3)
        cv2.line(drawn, tuple(p[2]), tuple(p[4]), (0,255,0), 3)
        cv2.line(drawn, tuple(p[3]), tuple(p[5]), (0,255,0), 3)
        self.ar_overlay_img = drawn  # 靜態保留，直到下一次結果覆蓋

if __name__ == "__main__":
    use_moveit = False
    try:
        rospy.init_node
        use_moveit = bool(rospy.get_param('~use_moveit', False))
    except Exception:
        use_moveit = False
    if MOVEIT_AVAILABLE and use_moveit:
        moveit_commander.roscpp_initialize(sys.argv)
    client = AnyGraspROSClient()
    client.run()
