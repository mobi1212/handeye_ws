#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
semantic_brain_node.py — UR3 單臂語義抓取視覺前處理節點

改自模擬環境 brain.py，適配 UR3 單臂實體部署。
功能：OWL-v2 物件偵測 → SAM v1 分割 → SoM 網格 → Gemini 抓取區域推理

ROS 介面：
  訂閱: /camera/color/image_raw (sensor_msgs/Image)
        /system/trigger_llm     (std_msgs/String, JSON: {"object_name": "..."})
  發布: /system/llm_done        (std_msgs/String, JSON: {"status": "done"/"fail", ...})

輸出檔案: /tmp/semantic_brain/target_mask.png (供 client_camera.py 讀取)
"""

import sys
import os

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

import cv2
import json
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage
from transformers import pipeline, SamModel, SamProcessor
from google import genai
import warnings

warnings.filterwarnings('ignore')

# --- API Key：優先讀環境變數，備援讀 .env 檔 ---
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))
_api_key = os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise EnvironmentError("找不到 GOOGLE_API_KEY，請確認 handeye_ws/.env 存在")
_genai_client = genai.Client(api_key=_api_key)

# --- UR3 單臂 Gemini Prompt ---
UR3_SINGLE_ARM_PROMPT = """
你是一個機器人視覺分析專家，協助單臂 UR3 機械臂抓取物件。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示 UR3 機械臂基座位置、桌面高度，以及目標物件的擺放狀態。
【圖片 2】物件特寫網格圖：目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。
每個格子的中心都有白色標籤顯示其代號（如 A1、B2）。

【任務背景】
UR3 機械臂從上方接近桌面物件進行抓取，夾爪為平行夾爪。
需要選擇最穩定的抓取區域。

【約束條件】

約束一：高度選擇原則
Y1 是物件頂部，Y5 是物件底部靠近桌面。
- Y1：物件頂部，適合較高物體的上方抓取
- Y2、Y3：最佳抓取區間，手臂工作空間充裕，強烈推薦優先選擇
- Y4：可以使用，手臂需要稍微向下延伸，但仍在合理範圍內
- Y5：盡量避免，非常靠近桌面，手臂難以到達且容易碰撞桌面

約束二：抓取區域需連續且集中
選擇的網格應彼此相鄰，形成有效的夾取面。
至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束三：穩定性
選擇物件較寬或較厚的部位，避免邊角或細薄處。
考慮平行夾爪的夾取方向（左右對稱為佳）。

約束四：從【圖片 1】判斷手臂可達性
觀察物件在桌面的實際位置，選擇 UR3 手臂容易到達的區域。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明你如何根據上述約束做出這個選擇"
}
"""


def imgmsg_to_numpy(msg):
    """將 ROS sensor_msgs/Image 轉為 numpy RGB array"""
    dtype_class = np.uint8
    channels = 3 if "rgb8" in msg.encoding or "bgr8" in msg.encoding else 1
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
        if "bgr8" in msg.encoding:
            img = img[:, :, ::-1]
    else:
        img = img.reshape((msg.height, msg.width))
    return img


def draw_som_grid(img_rgb, rows=5, cols=5):
    """
    Set-of-Mark 網格繪製
    在每個格子中心放代號，用半透明色塊交替標示格子
    回傳：標注後的影像、每個格子的絕對座標字典
    """
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]

    overlay = img_rgb.copy()
    alpha = 0.15

    colors = [
        (173, 216, 230),  # 淡藍
        (255, 200, 150),  # 淡橘
    ]

    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            grid_id = f"{col_labels[c]}{r + 1}"
            grid_dict[grid_id] = (x1, y1, x2, y2)

            color = colors[(r + c) % 2]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    result = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(result, (0, y), (w, y), (80, 80, 80), 1)
    for j in range(1, cols):
        x = int(j * cell_w)
        cv2.line(result, (x, 0), (x, h), (80, 80, 80), 1)

    cv2.rectangle(result, (0, 0), (w - 1, h - 1), (80, 80, 80), 2)

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            grid_id = f"{col_labels[c]}{r + 1}"
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(cell_w, cell_h) / 60.0
            thickness = max(1, int(font_scale * 2))

            (text_w, text_h), _ = cv2.getTextSize(grid_id, font, font_scale, thickness)
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2

            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness)

    return result, grid_dict


class SemanticBrainNode:
    def __init__(self):
        rospy.init_node('semantic_brain_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        rospy.loginfo(f"Loading models ({self.device})...")

        # OWL-v2 zero-shot object detector
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device
        )

        # SAM v1 segmentation
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # Gemini VLM for grid analysis
        self.gemini_model = _genai_client

        rospy.loginfo("All AI models loaded.")

        # 輸出目錄：/tmp 供程式讀取，workspace 根目錄供紀錄
        self.save_dir = "/tmp/semantic_brain"
        self.log_base_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'vlm_logs'
        )
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_base_dir, exist_ok=True)
        self.session_log_dir = None  # 每次處理時建立

        self.target_object = ""
        self.need_process = False
        self.latest_image = None
        self.is_processing = False

        # 訂閱相機影像（持續接收最新一幀）
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_buffer_callback)

        # 接收來自 client_camera 或其他節點的 trigger
        rospy.Subscriber("/system/trigger_llm", String, self.trigger_callback)

        # 發布完成訊號
        self.done_pub = rospy.Publisher("/system/llm_done", String, queue_size=1)

        rospy.loginfo("Brain node ready, waiting for /system/trigger_llm ...")

    def image_buffer_callback(self, msg):
        self.latest_image = msg

    def trigger_callback(self, msg):
        if self.is_processing:
            rospy.logwarn("AI is processing, ignoring duplicate trigger...")
            return

        try:
            data = json.loads(msg.data)
            self.target_object = data.get("object_name", "unknown")
        except json.JSONDecodeError:
            self.target_object = msg.data

        rospy.loginfo(f"Received trigger, target: {self.target_object}")
        self.need_process = True

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.need_process and self.latest_image is not None:
                self.need_process = False
                self.is_processing = True
                self.process(self.latest_image, self.target_object)
                self.is_processing = False
            rate.sleep()

    def _log_save(self, filename, img_bgr):
        """同時存到 /tmp 和 vlm_logs 時間戳資料夾"""
        cv2.imwrite(os.path.join(self.save_dir, filename), img_bgr)
        if self.session_log_dir:
            cv2.imwrite(os.path.join(self.session_log_dir, filename), img_bgr)

    def process(self, img_msg, object_name):
        rospy.loginfo(f"[1/4] OWL-v2 detecting '{object_name}'...")
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = object_name.replace(" ", "_")
            self.session_log_dir = os.path.join(
                self.log_base_dir, f"{timestamp}_{safe_name}"
            )
            os.makedirs(self.session_log_dir, exist_ok=True)

            # 只保留最近 10 次紀錄，超過從最舊的開始刪除
            import shutil
            existing = sorted([
                d for d in os.listdir(self.log_base_dir)
                if os.path.isdir(os.path.join(self.log_base_dir, d))
            ])
            while len(existing) > 10:
                shutil.rmtree(os.path.join(self.log_base_dir, existing.pop(0)))

            rospy.loginfo(f"   Log dir: {self.session_log_dir}")

            img_np = imgmsg_to_numpy(img_msg)
            h, w = img_np.shape[:2]
            img_pil = PILImage.fromarray(img_np)

            self._log_save("original_rgb.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            # --- OWL-v2 偵測 ---
            preds = self.detector(img_pil, candidate_labels=[object_name])
            if not preds:
                rospy.logerr(f"Object '{object_name}' not found")
                self.done_pub.publish(json.dumps({
                    "status": "fail", "reason": "object_not_found"
                }))
                return

            best_pred = max(preds, key=lambda x: x['score'])
            box = best_pred['box']
            x_min = int(box['xmin'])
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
            rospy.loginfo(f"   Detected '{object_name}', confidence: {best_pred['score']:.2f}")

            # --- 裁切物件區域 ---
            pad = 20
            c_xmin = max(0, x_min - pad)
            c_ymin = max(0, y_min - pad)
            c_xmax = min(w, x_max + pad)
            c_ymax = min(h, y_max + pad)
            cropped_img = img_np[c_ymin:c_ymax, c_xmin:c_xmax].copy()

            self._log_save("object_crop_raw.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # --- SoM 網格 ---
            rospy.loginfo("[2/4] Drawing Set-of-Mark grid...")
            grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

            # 轉換為全圖絕對座標（給 SAM 使用）
            grid_dict_absolute = {}
            for grid_id, (lx1, ly1, lx2, ly2) in grid_dict_local.items():
                grid_dict_absolute[grid_id] = [
                    c_xmin + lx1, c_ymin + ly1,
                    c_xmin + lx2, c_ymin + ly2
                ]

            grid_img_bgr = cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR)
            self._log_save("cropped_grid_for_vlm.png", grid_img_bgr)
            grid_img_path = os.path.join(self.save_dir, "cropped_grid_for_vlm.png")

            # --- Gemini 推理 ---
            rospy.loginfo("[3/4] Gemini analyzing grid...")
            gemini_local_img = PILImage.open(grid_img_path)

            for model_id in ['gemini-flash-latest', 'gemini-2.5-flash-lite']:
                try:
                    response = self.gemini_model.models.generate_content(
                        model=model_id,
                        contents=[UR3_SINGLE_ARM_PROMPT, img_pil, gemini_local_img]
                    )
                    rospy.loginfo(f"   Model: {model_id}")
                    break
                except Exception as e:
                    if '503' in str(e) or 'UNAVAILABLE' in str(e):
                        rospy.logwarn(f"   {model_id} 過載，切換備用...")
                        continue
                    raise

            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            vlm_result = json.loads(clean_json)

            target_grids = vlm_result.get('target_grids', [])
            rospy.loginfo(f"   Gemini selected grids: {target_grids}")
            rospy.loginfo(f"   Reasoning: {vlm_result.get('reasoning', '')}")

            if not target_grids:
                rospy.logerr("Gemini returned no valid grids")
                self.done_pub.publish(json.dumps({
                    "status": "fail", "reason": "no_grids"
                }))
                return

            # --- SAM v1 分割 ---
            rospy.loginfo("[4/4] SAM segmentation...")
            inputs = self.sam_processor(
                img_pil,
                input_boxes=[[[x_min, y_min, x_max, y_max]]],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs.original_sizes.cpu(),
                inputs.reshaped_input_sizes.cpu()
            )
            global_mask = masks[0][0][0].numpy()

            self._log_save("sam_global_mask_full.png", (global_mask * 255).astype(np.uint8))

            # --- 合成最終 mask（只保留 target_grids 區域內的 SAM mask）---
            final_mask = np.zeros_like(global_mask, dtype=bool)
            for grid_id in target_grids:
                if grid_id not in grid_dict_absolute:
                    rospy.logwarn(f"   Unknown grid id: {grid_id}")
                    continue
                gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]

            self._log_save("target_mask.png", (final_mask * 255).astype(np.uint8))

            # 同時存一張彩色視覺化：在原圖上疊加最終遮罩
            overlay = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[final_mask] = (0, 255, 100)
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            for grid_id in target_grids:
                if grid_id in grid_dict_absolute:
                    gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                    cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                    cv2.putText(overlay, grid_id, (gx1 + 4, gy1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            self._log_save("result_overlay.png", overlay)

            rospy.loginfo(f"   Logs saved: {self.session_log_dir}")

            # 存 Gemini 推理結果 JSON
            if self.session_log_dir:
                with open(os.path.join(self.session_log_dir, "gemini_result.json"), 'w', encoding='utf-8') as f:
                    json.dump(vlm_result, f, ensure_ascii=False, indent=2)

            rospy.loginfo("Processing complete!")
            self.done_pub.publish(json.dumps({
                "status": "done",
                "object_name": object_name,
                "target_grids": target_grids,
                "reasoning": vlm_result.get('reasoning', '')
            }))

        except Exception as e:
            rospy.logerr(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            self.done_pub.publish(json.dumps({
                "status": "fail", "reason": str(e)
            }))


if __name__ == '__main__':
    node = SemanticBrainNode()
    rospy.sleep(2)
    node.run()
