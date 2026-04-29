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
sys.path.insert(0, ros_path)

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
_api_keys = [k for k in [
    os.environ.get("GOOGLE_API_KEY"),
    os.environ.get("GOOGLE_API_KEY_2"),
    os.environ.get("GOOGLE_API_KEY_3"),
] if k]
if not _api_keys:
    raise EnvironmentError("找不到 GOOGLE_API_KEY，請確認 handeye_ws/.env 存在")
_genai_clients = [genai.Client(api_key=k) for k in _api_keys]
_client_index  = 0  # 目前使用的 client 索引

# --- UR3 單臂 Gemini Prompt ---
UR3_SINGLE_ARM_PROMPT = """
你是一個機器人視覺分析專家，協助單臂 UR3 機械臂抓取物件。

你將收到一張圖片：
【物件特寫網格圖】目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。
每個格子的中心都有白色標籤顯示其代號（如 A1、B2）。
物件邊界以紫色輪廓線標示。

【任務背景】
UR3 機械臂從上方接近桌面物件進行抓取，夾爪為平行夾爪。
需要選擇最穩定的抓取區域。

【約束條件】

約束一：優先選擇適合夾爪穩定接觸的部位
請優先選擇物體上「較容易被平行夾爪穩定夾住」的區域，而不是單純依照圖中高低位置選擇。
- 優先選擇較厚、較穩、較容易形成夾持面的部位
- 若物體有握持部位（如把柄、握把、中段），優先考慮該部位
- 避免只因為位置較高就選頂端，或只因為靠近底部就完全排除
- 但若區域非常靠近桌面、明顯會增加碰撞風險，仍應降低優先度

約束二：物件覆蓋率
只選擇物件實際佔據面積較大的格子（紫色輪廓線內）。
若某格子內幾乎沒有物件（大部分是背景），不應選擇。

約束三：抓取區域需連續且集中
選擇的網格應彼此相鄰，形成有效的夾取面。
至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束四：穩定性
選擇物件較寬或較厚的部位，避免邊角或細薄處。
考慮平行夾爪的夾取方向（左右對稱為佳）。

約束五：質心與抓取性平衡
根據你對目標物件的知識，推估其質心位置，但抓取點必須同時滿足「靠近質心」與「表面可抓取」兩個條件。
- 目標是選「最靠近質心的可抓取表面」，而非直接夾在質心位置
- 光滑金屬平面（如鎚頭側面、刀身）、圓弧面（如杯底）摩擦係數低，即使靠近質心也應避免
- 有紋路、有包覆或截面為圓柱形的表面（如把柄、握把）摩擦係數高，優先選擇
- 具體例子：鎚子應夾「把柄靠近鎚頭的頸部段」，絕對不可選鎚頭本體（光滑金屬，必滑落）；螺絲起子應夾握柄而非金屬桿
- 重量分佈均勻的物體（積木、書本）直接選幾何中心附近即可
- 若視覺難以判斷材質，以密度估算質心：金屬 > 陶瓷/玻璃 > 木頭/塑膠

約束六：球體特殊規則
若物件為球形（如球、橘子、蘋果），請選擇物件佔據面積 > 20% 的所有格子。
球體沒有固定抓取方向，提供最大覆蓋範圍讓機器人自行選擇最佳接觸點。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "estimated_com_grid": "估計質心所在格子代號",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明質心估計依據，以及如何根據上述約束做出這個選擇"
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

        # Gemini VLM for grid analysis（round-robin 多 key）
        self.gemini_clients = _genai_clients
        self.gemini_key_idx = 0

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
        rospy.loginfo(f"[1/5] OWL-v2 detecting '{object_name}'...")
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
            while len(existing) > 50:
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

            # --- SAM v1 分割（提前到 Gemini 之前）---
            rospy.loginfo("[2/5] SAM segmentation...")
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

            # --- 裁切物件區域 ---
            pad = 20
            c_xmin = max(0, x_min - pad)
            c_ymin = max(0, y_min - pad)
            c_xmax = min(w, x_max + pad)
            c_ymax = min(h, y_max + pad)
            cropped_img = img_np[c_ymin:c_ymax, c_xmin:c_xmax].copy()

            self._log_save("object_crop_raw.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # --- SoM 網格 + SAM 輪廓疊加 ---
            rospy.loginfo("[3/5] Drawing Set-of-Mark grid with SAM contour...")
            grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

            # 在格子圖上疊加 SAM mask 的輪廓線，讓 Gemini 看到物件邊界
            cropped_mask = global_mask[c_ymin:c_ymax, c_xmin:c_xmax].astype(np.uint8)
            contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(grid_img_rgb, contours, -1, (180, 0, 255), 2)  # 紫色輪廓

            # 轉換為全圖絕對座標
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
            rospy.loginfo("[4/5] Gemini analyzing grid...")
            gemini_local_img = PILImage.open(grid_img_path)

            n_clients = len(self.gemini_clients)
            response = None
            IS_RATE_LIMIT = lambda e: any(c in e for c in ['429', 'RESOURCE_EXHAUSTED'])
            IS_SERVER_ERR = lambda e: any(c in e for c in ['503', 'UNAVAILABLE'])
            for model_id in ['gemini-2.5-flash', 'gemini-2.5-flash-lite']:
                for attempt in range(n_clients):
                    client = self.gemini_clients[self.gemini_key_idx]
                    try:
                        response = client.models.generate_content(
                            model=model_id,
                            contents=[UR3_SINGLE_ARM_PROMPT, gemini_local_img]
                        )
                        usage = response.usage_metadata
                        rospy.loginfo(
                            f"   Model: {model_id}, Key: [{self.gemini_key_idx}] | "
                            f"tokens — input: {usage.prompt_token_count}, "
                            f"output: {usage.candidates_token_count}, "
                            f"total: {usage.total_token_count}"
                        )
                        break
                    except Exception as e:
                        err = str(e)
                        if IS_SERVER_ERR(err):
                            rospy.logwarn(f"   ⚠️  {model_id} 伺服器過載 (503)，直接切換模型...")
                            break  # 跳出 key loop，換下一個模型
                        elif IS_RATE_LIMIT(err):
                            next_idx = (self.gemini_key_idx + 1) % n_clients
                            rospy.logwarn(f"   ⚠️  Key [{self.gemini_key_idx}] 限額 (429)，切換至 Key [{next_idx}]...")
                            self.gemini_key_idx = next_idx
                            continue
                        raise
                if response is not None:
                    break
                rospy.logwarn(f"   🔄 {model_id} 無法使用，嘗試 fallback 模型...")
            if response is None:
                raise RuntimeError("所有模型均無法使用，請稍後再試")

            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            vlm_result = json.loads(clean_json)

            target_grids = vlm_result.get('target_grids', [])
            com_grid     = vlm_result.get('estimated_com_grid', 'N/A')
            object_shape = vlm_result.get('object_shape', 'box')
            rospy.loginfo(f"   Estimated CoM grid:  {com_grid}")
            rospy.loginfo(f"   Object shape: {object_shape}")
            rospy.loginfo(f"   Gemini selected grids: {target_grids}")
            rospy.loginfo(f"   Reasoning: {vlm_result.get('reasoning', '')}")

            if not target_grids:
                rospy.logerr("Gemini returned no valid grids")
                self.done_pub.publish(json.dumps({
                    "status": "fail", "reason": "no_grids"
                }))
                return

            # --- [5/5] 覆蓋率後處理：踢掉物件占比 < 20% 的格子 ---
            rospy.loginfo("[5/5] Coverage validation...")
            min_coverage = 0.20
            validated_grids = []
            for grid_id in target_grids:
                if grid_id not in grid_dict_absolute:
                    rospy.logwarn(f"   Unknown grid id: {grid_id}")
                    continue
                gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                cell_area = (gx2 - gx1) * (gy2 - gy1)
                obj_pixels = global_mask[gy1:gy2, gx1:gx2].sum()
                coverage = obj_pixels / cell_area if cell_area > 0 else 0
                rospy.loginfo(f"   {grid_id}: coverage = {coverage:.1%}")
                if coverage >= min_coverage:
                    validated_grids.append(grid_id)
                else:
                    rospy.logwarn(f"   {grid_id} 覆蓋率不足 ({coverage:.1%} < {min_coverage:.0%})，已排除")

            if not validated_grids:
                rospy.logwarn("所有格子覆蓋率不足，使用 Gemini 原始選擇")
                validated_grids = target_grids

            target_grids = validated_grids

            # --- 合成最終 mask（只保留 target_grids 區域內的 SAM mask）---
            final_mask = np.zeros_like(global_mask, dtype=bool)
            for grid_id in target_grids:
                if grid_id not in grid_dict_absolute:
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
                "object_shape": object_shape,
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
