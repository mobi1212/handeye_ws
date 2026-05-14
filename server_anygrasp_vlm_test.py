import argparse
import json
import os
import pickle
import shutil
import time
import warnings
import zlib
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import torch
import zmq
from PIL import Image as PILImage
from dotenv import load_dotenv
from google import genai
from graspnetAPI import GraspGroup
from gsnet import AnyGrasp
from transformers import SamModel, SamProcessor, pipeline as hf_pipeline

warnings.filterwarnings("ignore")


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
    "object_shape": "sphere | box | cylinder | irregular",
    "estimated_com_grid": "估計質心所在格子代號",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明質心估計依據，以及如何根據上述約束做出這個選擇"
}
"""


def draw_som_grid(img_rgb, rows=5, cols=5):
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]
    overlay = img_rgb.copy()
    colors = [(173, 216, 230), (255, 200, 150)]
    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            grid_id = f"{col_labels[c]}{r + 1}"
            grid_dict[grid_id] = (x1, y1, x2, y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[(r + c) % 2], -1)

    result = cv2.addWeighted(overlay, 0.15, img_rgb, 0.85, 0)

    for i in range(1, rows):
        cv2.line(result, (0, int(i * cell_h)), (w, int(i * cell_h)), (80, 80, 80), 1)
    for j in range(1, cols):
        cv2.line(result, (int(j * cell_w), 0), (int(j * cell_w), h), (80, 80, 80), 1)
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
            font_scale = min(cell_w, cell_h) / 60.0
            thickness = max(1, int(font_scale * 2))
            (text_w, text_h), _ = cv2.getTextSize(
                grid_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2
            cv2.putText(
                result, grid_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2
            )
            cv2.putText(
                result, grid_id, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness
            )

    return result, grid_dict


def safe_normalize(vec, eps=1e-6):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def decode_mask(mask_payload, image_shape):
    if mask_payload is None:
        return None
    if isinstance(mask_payload, np.ndarray):
        mask = mask_payload
    else:
        mask = cv2.imdecode(mask_payload, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def extract_target_points(points, bbox, mask_bool, fx, fy, cx, cy, img_w, img_h):
    if len(points) == 0:
        return points

    valid_depth = points[:, 2] > 1e-6
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32)

    proj_points = points[valid_depth]
    u = (fx * proj_points[:, 0] / proj_points[:, 2] + cx).astype(int)
    v = (fy * proj_points[:, 1] / proj_points[:, 2] + cy).astype(int)
    valid_proj = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    target_mask = np.ones(len(proj_points), dtype=bool)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        target_mask &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if mask_bool is not None:
        mask_match = np.zeros(len(proj_points), dtype=bool)
        valid_idx = np.where(valid_proj)[0]
        mask_match[valid_idx] = mask_bool[v[valid_idx], u[valid_idx]]
        target_mask &= mask_match
    return proj_points[target_mask]


def filter_grasps_by_geometry(gg, points, bbox, mask_bool, object_shape, fx, fy, cx, cy, img_w, img_h):
    target_pts = extract_target_points(points, bbox, mask_bool, fx, fy, cx, cy, img_w, img_h)
    if len(target_pts) < 30 or len(gg) == 0:
        print("   ⚠️ 目標點雲不足，跳過幾何過濾")
        return gg

    rotations = np.array([g.rotation_matrix for g in gg])
    translations = np.array([g.translation for g in gg])
    approach = rotations[:, :, 0]
    closing = rotations[:, :, 1]
    valid_mask = np.ones(len(gg), dtype=bool)

    if object_shape == "sphere":
        center = np.mean(target_pts, axis=0)
        to_center = center - translations
        norm = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
        dot = np.einsum("ij,ij->i", approach, to_center / norm)
        valid_mask &= dot > 0.85
    elif object_shape == "box":
        centered = target_pts - np.mean(target_pts, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        dominant_normal = safe_normalize(vt[-1])
        if dominant_normal is None:
            return gg
        dot = np.abs(np.einsum("ij,j->i", approach, dominant_normal))
        valid_mask &= dot > 0.80
    elif object_shape == "cylinder":
        centered = target_pts - np.mean(target_pts, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = safe_normalize(vt[0])
        if axis is None:
            return gg
        dot = np.abs(np.einsum("ij,j->i", closing, axis))
        valid_mask &= dot < 0.35
    else:
        print(f"   ℹ️ object_shape={object_shape}，跳過幾何過濾")
        return gg

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        print("   ⚠️ 幾何過濾後無候選，回退原始排序")
        return gg
    return gg[valid_indices]


def apply_svd_fill(depth_raw, mask_for_svd, fx, fy, cx, cy):
    if mask_for_svd is None:
        return depth_raw

    mask_resized = cv2.resize(mask_for_svd.astype(np.float32), (depth_raw.shape[1], depth_raw.shape[0]))
    obj_mask = (mask_resized > 0.5).astype(np.uint8)
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask_inner = cv2.dilate(obj_mask, kernel_inner, iterations=1)
    mask_outer = cv2.dilate(obj_mask, kernel_outer, iterations=1)
    moat_mask = cv2.subtract(mask_inner, obj_mask)
    table_donut_mask = cv2.subtract(mask_outer, mask_inner)

    v_donut, u_donut = np.where((table_donut_mask > 0) & (depth_raw > 0))
    if len(v_donut) <= 10:
        return depth_raw

    z_donut = depth_raw[v_donut, u_donut].astype(np.float64)
    x_donut = (u_donut - cx) * z_donut / fx
    y_donut = (v_donut - cy) * z_donut / fy
    points_3d = np.stack((x_donut, y_donut, z_donut), axis=-1)
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    a, b, c = normal
    d = -np.dot(normal, centroid)
    v_moat, u_moat = np.where(moat_mask > 0)
    denom = a * (u_moat - cx) / fx + b * (v_moat - cy) / fy + c
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    z_filled = -d / denom
    clean_depth = depth_raw.copy()
    clean_depth[v_moat, u_moat] = np.clip(z_filled, 0, 65535).astype(np.uint16)
    return clean_depth


class VLMServer:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.save_dir = "saved_results_vlm_test"
        self.vlm_log_dir = os.path.join(os.path.dirname(__file__), "vlm_logs_server_test")
        if cfgs.save:
            os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.vlm_log_dir, exist_ok=True)

        print("🚀 正在載入 AnyGrasp 模型...")
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        print("✅ AnyGrasp 載入完成")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 正在載入 VLM 模型 ({self.device}) ...")
        self.detector = hf_pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device,
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        api_keys = [k for k in [
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("GOOGLE_API_KEY_2"),
            os.environ.get("GOOGLE_API_KEY_3"),
        ] if k]
        self.gemini_clients = [genai.Client(api_key=k) for k in api_keys]
        self.gemini_key_idx = 0
        if self.gemini_clients:
            print(f"✅ Gemini clients ready: {len(self.gemini_clients)} keys")
        else:
            print("⚠️ 未找到 GOOGLE_API_KEY，VLM 模式將無法使用")
        print("✅ VLM 模型載入完成")

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{cfgs.port}")

        self.vis = None
        if cfgs.debug:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="AnyGrasp VLM Test", width=800, height=600)

    def make_session_dir(self, object_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = object_name.replace(" ", "_")
        session_dir = os.path.join(self.vlm_log_dir, f"{timestamp}_{safe_name}")
        os.makedirs(session_dir, exist_ok=True)
        existing = sorted(
            d for d in os.listdir(self.vlm_log_dir)
            if os.path.isdir(os.path.join(self.vlm_log_dir, d))
        )
        while len(existing) > 50:
            shutil.rmtree(os.path.join(self.vlm_log_dir, existing.pop(0)))
        return session_dir

    def save_image(self, session_dir, filename, img_bgr):
        if session_dir:
            cv2.imwrite(os.path.join(session_dir, filename), img_bgr)

    def run_gemini(self, grid_img_path):
        if not self.gemini_clients:
            raise RuntimeError("找不到 GOOGLE_API_KEY，無法執行 Gemini")

        image = PILImage.open(grid_img_path)
        n_clients = len(self.gemini_clients)
        response = None
        is_rate_limit = lambda e: any(code in e for code in ["429", "RESOURCE_EXHAUSTED"])
        is_server_err = lambda e: any(code in e for code in ["503", "UNAVAILABLE"])

        for model_id in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            for _ in range(n_clients):
                client = self.gemini_clients[self.gemini_key_idx]
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[UR3_SINGLE_ARM_PROMPT, image],
                    )
                    break
                except Exception as exc:
                    err = str(exc)
                    if is_server_err(err):
                        break
                    if is_rate_limit(err):
                        self.gemini_key_idx = (self.gemini_key_idx + 1) % n_clients
                        continue
                    raise
            if response is not None:
                break
        if response is None:
            raise RuntimeError("所有 Gemini 模型均無法使用")

        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json, strict=False)

    def run_vlm_pipeline(self, color_bgr, object_name):
        session_dir = self.make_session_dir(object_name)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(color_rgb)
        h, w = color_rgb.shape[:2]
        self.save_image(session_dir, "original_rgb.png", color_bgr)

        preds = self.detector(img_pil, candidate_labels=[object_name])
        if not preds:
            raise RuntimeError(f"Object '{object_name}' not found")
        best_pred = max(preds, key=lambda x: x["score"])
        box = best_pred["box"]
        x_min = int(box["xmin"])
        y_min = int(box["ymin"])
        x_max = int(box["xmax"])
        y_max = int(box["ymax"])
        bbox = [x_min, y_min, x_max, y_max]

        inputs = self.sam_processor(
            img_pil,
            input_boxes=[[[x_min, y_min, x_max, y_max]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs.original_sizes.cpu(),
            inputs.reshaped_input_sizes.cpu(),
        )
        global_mask = masks[0][0][0].numpy() > 0.0
        self.save_image(session_dir, "sam_global_mask_full.png", (global_mask * 255).astype(np.uint8))

        pad = 20
        c_xmin = max(0, x_min - pad)
        c_ymin = max(0, y_min - pad)
        c_xmax = min(w, x_max + pad)
        c_ymax = min(h, y_max + pad)
        cropped_img = color_rgb[c_ymin:c_ymax, c_xmin:c_xmax].copy()
        self.save_image(session_dir, "object_crop_raw.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

        grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)
        cropped_mask = global_mask[c_ymin:c_ymax, c_xmin:c_xmax].astype(np.uint8)
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(grid_img_rgb, contours, -1, (180, 0, 255), 2)

        grid_dict_absolute = {}
        for grid_id, (lx1, ly1, lx2, ly2) in grid_dict_local.items():
            grid_dict_absolute[grid_id] = [c_xmin + lx1, c_ymin + ly1, c_xmin + lx2, c_ymin + ly2]

        grid_img_bgr = cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR)
        grid_img_path = os.path.join(session_dir, "cropped_grid_for_vlm.png")
        cv2.imwrite(grid_img_path, grid_img_bgr)
        vlm_result = self.run_gemini(grid_img_path)

        target_grids = vlm_result.get("target_grids", [])
        object_shape = vlm_result.get("object_shape", "box")
        if not target_grids:
            raise RuntimeError("Gemini returned no valid grids")

        min_coverage = 0.20
        validated_grids = []
        for grid_id in target_grids:
            if grid_id not in grid_dict_absolute:
                continue
            gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
            cell_area = max((gx2 - gx1) * (gy2 - gy1), 1)
            obj_pixels = global_mask[gy1:gy2, gx1:gx2].sum()
            coverage = obj_pixels / cell_area
            if coverage >= min_coverage:
                validated_grids.append(grid_id)
        if not validated_grids:
            validated_grids = target_grids
        target_grids = validated_grids

        final_mask = np.zeros_like(global_mask, dtype=bool)
        for grid_id in target_grids:
            if grid_id not in grid_dict_absolute:
                continue
            gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
            final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]

        self.save_image(session_dir, "target_mask.png", (final_mask * 255).astype(np.uint8))
        overlay = color_bgr.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[final_mask] = (0, 255, 100)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        for grid_id in target_grids:
            if grid_id in grid_dict_absolute:
                gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                cv2.putText(overlay, grid_id, (gx1 + 4, gy1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        self.save_image(session_dir, "result_overlay.png", overlay)
        with open(os.path.join(session_dir, "gemini_result.json"), "w", encoding="utf-8") as handle:
            json.dump(vlm_result, handle, ensure_ascii=False, indent=2)

        return {
            "bbox": bbox,
            "global_mask": global_mask,
            "final_mask": final_mask,
            "object_shape": object_shape,
            "target_grids": target_grids,
            "reasoning": vlm_result.get("reasoning", ""),
            "estimated_com_grid": vlm_result.get("estimated_com_grid", "N/A"),
            "session_dir": session_dir,
        }

    def decode_color(self, payload):
        if "color_jpg" in payload:
            color_bgr = cv2.imdecode(payload["color_jpg"], cv2.IMREAD_COLOR)
            if color_bgr is None:
                raise RuntimeError("JPG decode failed")
            return color_bgr
        if "color" in payload:
            colors = payload["color"]
            return cv2.cvtColor((colors * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        raise RuntimeError("payload missing color image")

    def get_intrinsics(self, payload):
        intr = payload.get("intrinsics", {})
        return (
            float(intr.get("fx", 617.183)),
            float(intr.get("fy", 617.122)),
            float(intr.get("cx", 319.639)),
            float(intr.get("cy", 241.404)),
        )

    def build_point_cloud(self, depths, colors, mask_bool, bbox, fx, fy, cx, cy):
        h, w = depths.shape
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        points_z = depths / 1000.0
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        depth_valid = (points_z > 0.1) & (points_z < 1.5)
        point_mask = depth_valid.copy()
        if mask_bool is not None:
            point_mask &= mask_bool
        points = np.stack([points_x, points_y, points_z], axis=-1)[point_mask].astype(np.float32)
        colors_mask = colors[point_mask].astype(np.float32)

        densify_region = None
        if mask_bool is not None and np.any(mask_bool):
            ys, xs = np.where(mask_bool)
            densify_region = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        elif bbox is not None:
            densify_region = bbox

        if densify_region is not None:
            bx1, by1, bx2, by2 = densify_region
            bx1 = max(0, bx1)
            by1 = max(0, by1)
            bx2 = min(w, bx2)
            by2 = min(h, by2)
            depth_roi = depths[by1:by2, bx1:bx2].astype(np.float32)
            color_roi = colors[by1:by2, bx1:bx2]
            roi_h, roi_w = depth_roi.shape
            upsample = 2
            depth_up = cv2.resize(depth_roi, (roi_w * upsample, roi_h * upsample), interpolation=cv2.INTER_LINEAR)
            color_up = cv2.resize(color_roi, (roi_w * upsample, roi_h * upsample), interpolation=cv2.INTER_LINEAR)
            up_h, up_w = depth_up.shape
            u_grid, v_grid = np.meshgrid(np.arange(up_w), np.arange(up_h))
            u_orig = bx1 + u_grid / upsample
            v_orig = by1 + v_grid / upsample
            z_up = depth_up / 1000.0
            x_up = (u_orig - cx) / fx * z_up
            y_up = (v_orig - cy) / fy * z_up
            valid_up = (z_up > 0.1) & (z_up < 1.5)
            if mask_bool is not None:
                mask_roi = mask_bool[by1:by2, bx1:bx2].astype(np.uint8) * 255
                mask_up = cv2.resize(mask_roi, (roi_w * upsample, roi_h * upsample), interpolation=cv2.INTER_NEAREST) > 127
                valid_up &= mask_up
            extra_points = np.stack([x_up, y_up, z_up], axis=-1)[valid_up].astype(np.float32)
            extra_colors = color_up[valid_up].astype(np.float32)
            if len(extra_points) > 0:
                points = np.vstack([points, extra_points])
                colors_mask = np.vstack([colors_mask, extra_colors])
        return points, colors_mask

    def infer_anygrasp(self, color_bgr, depths, bbox, mask_bool, object_shape, fx, fy, cx, cy):
        colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        points, colors_mask = self.build_point_cloud(depths, colors, mask_bool, bbox, fx, fy, cx, cy)
        gg, cloud = self.anygrasp.get_grasp(
            points,
            colors_mask,
            lims=[-0.3, 0.3, -0.2, 0.4, 0.2, 0.8],
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )
        if len(gg) == 0:
            raise RuntimeError("No grasp detected")

        h, w = depths.shape
        if bbox is not None or mask_bool is not None:
            translations = gg.translations
            u = (fx * translations[:, 0] / translations[:, 2] + cx).astype(int)
            v = (fy * translations[:, 1] / translations[:, 2] + cy).astype(int)
            valid_candidates = np.ones(len(gg), dtype=bool)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                valid_candidates &= (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            if mask_bool is not None:
                in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                mask_candidates = np.zeros(len(gg), dtype=bool)
                valid_idx = np.where(in_image)[0]
                mask_candidates[valid_idx] = mask_bool[v[valid_idx], u[valid_idx]]
                valid_candidates &= mask_candidates
            valid_indices = np.where(valid_candidates)[0]
            if len(valid_indices) == 0:
                raise RuntimeError("ROI empty")
            gg = gg[valid_indices]

        gg = gg.nms().sort_by_score()
        gg = filter_grasps_by_geometry(gg, points, bbox, mask_bool, object_shape, fx, fy, cx, cy, w, h)
        gg = gg.sort_by_score()
        best_grasp = gg[0]
        result = {
            "status": "success",
            "score": float(best_grasp.score),
            "width": float(best_grasp.width),
            "depth": float(best_grasp.depth),
            "translation": best_grasp.translation.tolist(),
            "rotation": best_grasp.rotation_matrix.tolist(),
        }
        return result, cloud, gg

    def maybe_save_result(self, color_bgr, depths, result, cloud):
        if not self.cfgs.save:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(self.save_dir, timestamp)
        cv2.imwrite(f"{prefix}_rgb.jpg", color_bgr)
        np.save(f"{prefix}_depth.npy", depths)
        o3d.io.write_point_cloud(f"{prefix}_cloud.ply", cloud)
        with open(f"{prefix}_result.json", "w") as handle:
            json.dump(result, handle, indent=2)

    def maybe_update_vis(self, cloud, gg):
        if self.vis is None:
            return
        trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cloud = o3d.geometry.PointCloud(cloud)
        cloud.transform(trans_mat)
        gripper_geo = gg[0:1].to_open3d_geometry_list()
        gripper_geo[0].transform(trans_mat)
        self.vis.clear_geometries()
        self.vis.add_geometry(cloud, reset_bounding_box=True)
        self.vis.add_geometry(gripper_geo[0], reset_bounding_box=False)

    def handle_payload(self, payload):
        mode = payload.get("mode", "legacy")
        color_bgr = self.decode_color(payload)
        depths = payload["depth"]
        fx, fy, cx, cy = self.get_intrinsics(payload)

        bbox = payload.get("bbox")
        mask_bool = decode_mask(payload.get("mask"), depths.shape)
        object_shape = payload.get("object_shape", "box")
        target_grids = payload.get("target_grids", [])
        reasoning = payload.get("reasoning", "")
        estimated_com_grid = payload.get("estimated_com_grid", "N/A")
        session_dir = None

        if mode == "vlm":
            object_name = payload.get("object_name")
            if not object_name:
                raise RuntimeError("mode=vlm requires object_name")
            vlm_meta = self.run_vlm_pipeline(color_bgr, object_name)
            bbox = vlm_meta["bbox"]
            mask_bool = vlm_meta["final_mask"]
            object_shape = vlm_meta["object_shape"]
            target_grids = vlm_meta["target_grids"]
            reasoning = vlm_meta["reasoning"]
            estimated_com_grid = vlm_meta["estimated_com_grid"]
            session_dir = vlm_meta["session_dir"]
            depths = apply_svd_fill(depths, vlm_meta["global_mask"], fx, fy, cx, cy)
        elif mode == "roi":
            if bbox is None:
                raise RuntimeError("mode=roi requires bbox")
        else:
            if mask_bool is not None:
                depths = apply_svd_fill(depths, mask_bool.astype(np.float32), fx, fy, cx, cy)

        result, cloud, gg = self.infer_anygrasp(color_bgr, depths, bbox, mask_bool, object_shape, fx, fy, cx, cy)
        result.update({
            "mode": mode,
            "bbox": bbox,
            "object_shape": object_shape,
            "target_grids": target_grids,
            "reasoning": reasoning,
            "estimated_com_grid": estimated_com_grid,
            "session_dir": session_dir,
        })
        self.maybe_save_result(color_bgr, depths, result, cloud)
        self.maybe_update_vis(cloud, gg)
        return result

    def run(self):
        print(f"🎧 測試伺服器已上線，監聽 Port {self.cfgs.port} ...")
        try:
            while True:
                if self.vis is not None:
                    if not self.vis.poll_events():
                        break
                    self.vis.update_renderer()
                try:
                    compressed_data = self.socket.recv(flags=zmq.NOBLOCK)
                    payload = pickle.loads(zlib.decompress(compressed_data))
                except zmq.Again:
                    time.sleep(0.01)
                    continue

                print("\n📦 收到請求，開始處理...")
                try:
                    result = self.handle_payload(payload)
                except Exception as exc:
                    print(f"❌ 處理失敗: {exc}")
                    self.socket.send_pyobj({"status": "fail", "message": str(exc)})
                    continue

                self.socket.send_pyobj(result)
                print(f"✅ 完成，score={result['score']:.4f}")
        finally:
            if self.vis is not None:
                self.vis.destroy_window()
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(description="AnyGrasp VLM migration test server")
    parser.add_argument("--checkpoint_path", default="../checkpoint/checkpoint_detection.tar")
    parser.add_argument("--max_gripper_width", type=float, default=0.1)
    parser.add_argument("--gripper_height", type=float, default=0.03)
    parser.add_argument("--top_down_grasp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--port", type=int, default=5555)
    cfgs = parser.parse_args()
    server = VLMServer(cfgs)
    server.run()


if __name__ == "__main__":
    main()
