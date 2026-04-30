# Server 升級計劃書：整合 VLM Pipeline 至 server_anygrasp.py

**版本**: v1.0  
**日期**: 2026-04-21  
**狀態**: ⏸️ 擱置  
**目標**: 將 OWL-v2 + SAM + Gemini + SVD 桌面擬合全部整合進遠端 server_anygrasp.py，  
讓本地 client 只需一次 ZMQ 往返即可完成「語義抓取」。

---

## 一、背景說明

目前架構：
```
本地 brain_node.py  →  OWL-v2 + SAM + Gemini  →  /tmp/target_mask.png
本地 client.py      →  讀 mask  →  ZMQ  →  server_anygrasp.py  →  AnyGrasp
```

升級後架構：
```
本地 client.py  →  ZMQ {mode, color, depth, object_name}
                →  server_anygrasp.py：OWL-v2 + SAM + Gemini + SVD + AnyGrasp
                ←  ZMQ {pose + mask_png + grids}
```

---

## 二、環境準備

在 server 的 conda 環境（anygrasp 或目前跑 server_anygrasp.py 的環境）安裝：

```bash
pip install transformers accelerate
pip install google-genai
pip install Pillow python-dotenv
```

確認 PyTorch 已安裝（SAM 需要）：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

在 `server_anygrasp.py` 同目錄下建立 `.env` 檔：
```
GOOGLE_API_KEY=（填入 Gemini API Key）
GOOGLE_API_KEY_2=（第二個 key，可選）
GOOGLE_API_KEY_3=（第三個 key，可選）
```

---

## 三、完整改寫後的 server_anygrasp.py

**直接以下面的完整程式碼取代原本的 server_anygrasp.py。**

```python
import zmq
import numpy as np
import torch
import argparse
import time
import cv2
import zlib
import pickle
import sys
import os
import json
import warnings
from datetime import datetime
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import open3d as o3d
from dotenv import load_dotenv
from PIL import Image as PILImage
from transformers import pipeline as hf_pipeline, SamModel, SamProcessor
from google import genai

warnings.filterwarnings('ignore')

# ==========================================
# 0. API Key 初始化
# ==========================================
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
_api_keys = [k for k in [
    os.environ.get("GOOGLE_API_KEY"),
    os.environ.get("GOOGLE_API_KEY_2"),
    os.environ.get("GOOGLE_API_KEY_3"),
] if k]
if not _api_keys:
    print("⚠️  找不到 GOOGLE_API_KEY，VLM 模式將無法使用")
    _api_keys = []
_genai_clients = [genai.Client(api_key=k) for k in _api_keys]
_genai_key_idx = 0

# ==========================================
# 1. Gemini Prompt
# ==========================================
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

約束二：物件覆蓋率
網格圖上疊加了物件的紫色輪廓線，只選擇物件實際佔據面積較大的格子。
若某格子內幾乎沒有物件（大部分是背景），不應選擇。

約束三：抓取區域需連續且集中
選擇的網格應彼此相鄰，形成有效的夾取面。
至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束四：穩定性
選擇物件較寬或較厚的部位，避免邊角或細薄處。
考慮平行夾爪的夾取方向（左右對稱為佳）。

約束五：從【圖片 1】判斷手臂可達性
觀察物件在桌面的實際位置，選擇 UR3 手臂容易到達的區域。

約束六：質心與抓取性平衡
根據你對目標物件的知識，推估其質心位置，但抓取點必須同時滿足「靠近質心」與「表面可抓取」兩個條件。
- 目標是選「最靠近質心的可抓取表面」，而非直接夾在質心位置
- 光滑金屬平面（如鎚頭側面、刀身）、圓弧面（如杯底）摩擦係數低，即使靠近質心也應避免
- 有紋路、有包覆或截面為圓柱形的表面（如把柄、握把）摩擦係數高，優先選擇
- 具體例子：鎚子應夾「把柄靠近鎚頭的頸部段」，絕對不可選鎚頭本體（光滑金屬，必滑落）
- 重量分佈均勻的物體（積木、書本）直接選幾何中心附近即可
- 若視覺難以判斷材質，以密度估算質心：金屬 > 陶瓷/玻璃 > 木頭/塑膠

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "estimated_com_grid": "估計質心所在格子代號",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明質心估計依據，以及如何根據上述約束做出這個選擇"
}
"""

# ==========================================
# 2. AnyGrasp Model Initialization
# ==========================================
parser = argparse.ArgumentParser(description="AnyGrasp AI Inference Server")
parser.add_argument('--checkpoint_path', default='../checkpoint/checkpoint_detection.tar')
parser.add_argument('--max_gripper_width', type=float, default=0.1)
parser.add_argument('--gripper_height', type=float, default=0.03)
parser.add_argument('--top_down_grasp', action='store_true')
parser.add_argument('--debug', action='store_true', help="Enable Open3D visualization")
parser.add_argument('--save', action='store_true', help="Save results for later review")
cfgs = parser.parse_args()

SAVE_DIR = "saved_results"
if cfgs.save:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 已開啟自動存檔功能，資料將儲存於: {SAVE_DIR}/")

print("🚀 正在載入 AnyGrasp 模型...")
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
print("✅ AnyGrasp 載入完成！")

# ==========================================
# 3. VLM Models Initialization
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 正在載入 VLM 模型（{DEVICE}）...")

owl_detector = hf_pipeline(
    model="google/owlv2-base-patch16-ensemble",
    task="zero-shot-object-detection",
    device=DEVICE
)
print("   ✅ OWL-v2 載入完成")

sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print("   ✅ SAM v1 載入完成")
print("✅ 所有模型載入完成！")

# VLM logs 目錄
VLM_LOG_DIR = os.path.join(os.path.dirname(__file__), "vlm_logs")
os.makedirs(VLM_LOG_DIR, exist_ok=True)

# ==========================================
# 4. ZMQ Configuration
# ==========================================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

FX, FY = 617.183, 617.122
CX, CY = 319.639, 241.404
DEPTH_SCALE = 1000.0
WORKSPACE_LIMS = [-0.3, 0.3, -0.2, 0.4, 0.2, 0.8]

# ==========================================
# 5. Open3D Visualization (optional)
# ==========================================
vis = None
if cfgs.debug:
    print("🖥️  啟動 3D 預覽視窗...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AnyGrasp", width=800, height=600)

print("🎧 伺服器已上線，監聽 Port 5555...")

# ==========================================
# 6. Helper Functions
# ==========================================

def draw_som_grid(img_rgb, rows=5, cols=5):
    """Set-of-Mark 網格繪製，回傳標注影像和格子座標字典"""
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]
    overlay = img_rgb.copy()
    colors = [(173, 216, 230), (255, 200, 150)]
    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w); y1 = int(r * cell_h)
            x2 = int((c+1) * cell_w); y2 = int((r+1) * cell_h)
            grid_dict[f"{col_labels[c]}{r+1}"] = (x1, y1, x2, y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[(r+c)%2], -1)

    result = cv2.addWeighted(overlay, 0.15, img_rgb, 0.85, 0)

    for i in range(1, rows):
        cv2.line(result, (0, int(i*cell_h)), (w, int(i*cell_h)), (80,80,80), 1)
    for j in range(1, cols):
        cv2.line(result, (int(j*cell_w), 0), (int(j*cell_w), h), (80,80,80), 1)
    cv2.rectangle(result, (0,0), (w-1,h-1), (80,80,80), 2)

    for r in range(rows):
        for c in range(cols):
            x1 = int(c*cell_w); y1 = int(r*cell_h)
            x2 = int((c+1)*cell_w); y2 = int((r+1)*cell_h)
            grid_id = f"{col_labels[c]}{r+1}"
            cx_g = (x1+x2)//2; cy_g = (y1+y2)//2
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = min(cell_w, cell_h) / 60.0
            th = max(1, int(fs*2))
            (tw, tth), _ = cv2.getTextSize(grid_id, font, fs, th)
            tx = cx_g - tw//2; ty = cy_g + tth//2
            cv2.putText(result, grid_id, (tx,ty), font, fs, (0,0,0), th+2)
            cv2.putText(result, grid_id, (tx,ty), font, fs, (255,255,255), th)

    return result, grid_dict


def svd_table_fitting(depth_raw, obj_mask_bool, fx, fy, cx, cy):
    """
    SVD 桌面擬合：以物件遮罩外圍的甜甜圈取樣點雲，
    擬合桌面平面，填補物件下方的深度空洞。
    回傳 clean_depth（uint16）。
    """
    obj_mask = obj_mask_bool.astype(np.uint8)
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask_inner = cv2.dilate(obj_mask, kernel_inner, iterations=1)
    mask_outer = cv2.dilate(obj_mask, kernel_outer, iterations=1)
    moat_mask = cv2.subtract(mask_inner, obj_mask)
    table_donut_mask = cv2.subtract(mask_outer, mask_inner)

    v_donut, u_donut = np.where((table_donut_mask > 0) & (depth_raw > 0))
    if len(v_donut) > 10:
        Z = depth_raw[v_donut, u_donut].astype(np.float64)
        X = (u_donut - cx) * Z / fx
        Y = (v_donut - cy) * Z / fy
        pts = np.stack((X, Y, Z), axis=-1)
        centroid = np.mean(pts, axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        normal = Vt[-1]
        a, b, c_n = normal
        d = -np.dot(normal, centroid)
        v_m, u_m = np.where(moat_mask > 0)
        denom = a*(u_m-cx)/fx + b*(v_m-cy)/fy + c_n
        denom = np.where(denom == 0, 1e-6, denom)
        Z_filled = -d / denom
        clean_depth = depth_raw.copy()
        clean_depth[v_m, u_m] = np.clip(Z_filled, 0, 65535).astype(np.uint16)
        return clean_depth
    return depth_raw


def run_vlm_pipeline(color_bgr, object_name, depth_raw, fx, fy, cx, cy):
    """
    完整 VLM pipeline：OWL-v2 → SAM → SoM → Gemini → 覆蓋率驗證 → SVD → AnyGrasp
    
    回傳 dict：
      成功時：{
        "status": "success",
        "target_mask_png": bytes,   # target_mask（格子過濾後）PNG bytes
        "full_mask_png":   bytes,   # SAM 完整 mask PNG bytes（供 SVD 和顯示）
        "bbox":            [x1,y1,x2,y2],
        "target_grids":    [...],
        "gemini_reasoning": str,
        "clean_depth":     np.ndarray,   # SVD 填補後的深度圖（內部使用）
        "color_bgr":       np.ndarray,   # 原始 BGR 影像（供 AnyGrasp 使用）
      }
      失敗時：{"status": "fail", "reason": str}
    """
    global _genai_key_idx

    img_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb)
    h, w = img_rgb.shape[:2]

    # --- 建立 session log 目錄（保留最近 50 次）---
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = object_name.replace(" ", "_")
    session_dir = os.path.join(VLM_LOG_DIR, f"{timestamp}_{safe_name}")
    os.makedirs(session_dir, exist_ok=True)
    existing = sorted([d for d in os.listdir(VLM_LOG_DIR)
                       if os.path.isdir(os.path.join(VLM_LOG_DIR, d))])
    while len(existing) > 50:
        shutil.rmtree(os.path.join(VLM_LOG_DIR, existing.pop(0)))

    def _save(filename, img_bgr_or_gray):
        cv2.imwrite(os.path.join(session_dir, filename), img_bgr_or_gray)

    _save("original_rgb.png", color_bgr)

    # --- [1/5] OWL-v2 偵測 ---
    print(f"   [1/5] OWL-v2 detecting '{object_name}'...")
    preds = owl_detector(img_pil, candidate_labels=[object_name])
    if not preds:
        return {"status": "fail", "reason": f"object_not_found: '{object_name}'"}

    best_pred = max(preds, key=lambda x: x['score'])
    box = best_pred['box']
    x_min, y_min = int(box['xmin']), int(box['ymin'])
    x_max, y_max = int(box['xmax']), int(box['ymax'])
    print(f"      Detected confidence: {best_pred['score']:.2f}, bbox: [{x_min},{y_min},{x_max},{y_max}]")

    # --- [2/5] SAM 分割 ---
    print("   [2/5] SAM segmentation...")
    inputs = sam_processor(
        img_pil,
        input_boxes=[[[x_min, y_min, x_max, y_max]]],
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs.original_sizes.cpu(),
        inputs.reshaped_input_sizes.cpu()
    )
    global_mask = masks[0][0][0].numpy()  # bool array, shape (H, W)
    _save("sam_global_mask_full.png", (global_mask * 255).astype(np.uint8))

    # --- [3/5] 裁切 + SoM 網格 + SAM 輪廓 ---
    print("   [3/5] Drawing SoM grid with SAM contour...")
    pad = 20
    c_xmin = max(0, x_min - pad); c_ymin = max(0, y_min - pad)
    c_xmax = min(w, x_max + pad); c_ymax = min(h, y_max + pad)
    cropped_img = img_rgb[c_ymin:c_ymax, c_xmin:c_xmax].copy()
    _save("object_crop_raw.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

    grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

    # SAM 輪廓疊加（紫色）
    cropped_mask = global_mask[c_ymin:c_ymax, c_xmin:c_xmax].astype(np.uint8)
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grid_img_rgb, contours, -1, (180, 0, 255), 2)

    # 格子座標轉全圖絕對座標
    grid_dict_abs = {
        gid: [c_xmin+lx1, c_ymin+ly1, c_xmin+lx2, c_ymin+ly2]
        for gid, (lx1, ly1, lx2, ly2) in grid_dict_local.items()
    }

    grid_bgr = cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR)
    _save("cropped_grid_for_vlm.png", grid_bgr)

    # --- [4/5] Gemini 推理 ---
    print("   [4/5] Gemini analyzing grid...")
    if not _genai_clients:
        return {"status": "fail", "reason": "no_gemini_api_key"}

    grid_pil = PILImage.fromarray(grid_img_rgb)
    n_clients = len(_genai_clients)
    response = None
    for attempt in range(n_clients * 2):
        client = _genai_clients[_genai_key_idx]
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=[UR3_SINGLE_ARM_PROMPT, img_pil, grid_pil]
            )
            print(f"      Key index: {_genai_key_idx}")
            break
        except Exception as e:
            err = str(e)
            if any(c in err for c in ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE']):
                print(f"      Key {_genai_key_idx} 配額/過載，切換...")
                _genai_key_idx = (_genai_key_idx + 1) % n_clients
                continue
            return {"status": "fail", "reason": f"gemini_error: {e}"}

    if response is None:
        return {"status": "fail", "reason": "all_gemini_keys_exhausted"}

    try:
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        vlm_result = json.loads(clean_json)
    except Exception as e:
        return {"status": "fail", "reason": f"gemini_json_parse_error: {e}"}

    target_grids = vlm_result.get('target_grids', [])
    com_grid     = vlm_result.get('estimated_com_grid', 'N/A')
    reasoning    = vlm_result.get('reasoning', '')
    print(f"      CoM grid: {com_grid}, Selected: {target_grids}")

    if not target_grids:
        return {"status": "fail", "reason": "gemini_no_grids"}

    # --- [5/5] 覆蓋率驗證 ---
    print("   [5/5] Coverage validation...")
    min_coverage = 0.20
    validated = []
    for gid in target_grids:
        if gid not in grid_dict_abs:
            continue
        gx1, gy1, gx2, gy2 = grid_dict_abs[gid]
        cell_area = (gx2-gx1) * (gy2-gy1)
        obj_pixels = global_mask[gy1:gy2, gx1:gx2].sum()
        coverage = obj_pixels / cell_area if cell_area > 0 else 0
        print(f"      {gid}: coverage = {coverage:.1%}")
        if coverage >= min_coverage:
            validated.append(gid)
        else:
            print(f"      {gid} 覆蓋率不足，排除")
    if not validated:
        print("      所有格子覆蓋率不足，使用原始選擇")
        validated = target_grids
    target_grids = validated

    # 合成 target mask（只保留選定格子內的 SAM mask）
    final_mask = np.zeros_like(global_mask, dtype=bool)
    for gid in target_grids:
        if gid not in grid_dict_abs:
            continue
        gx1, gy1, gx2, gy2 = grid_dict_abs[gid]
        final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]

    _save("target_mask.png", (final_mask * 255).astype(np.uint8))

    # 儲存 Gemini JSON
    with open(os.path.join(session_dir, "gemini_result.json"), 'w', encoding='utf-8') as f:
        json.dump(vlm_result, f, ensure_ascii=False, indent=2)

    # 儲存結果疊加圖
    overlay = color_bgr.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[final_mask] = (0, 255, 100)
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    for gid in target_grids:
        if gid in grid_dict_abs:
            gx1, gy1, gx2, gy2 = grid_dict_abs[gid]
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (0,255,255), 2)
            cv2.putText(overlay, gid, (gx1+4, gy1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    _save("result_overlay.png", overlay)
    print(f"   ✅ VLM pipeline 完成，logs: {session_dir}")

    # 將 mask 編碼為 PNG bytes
    _, target_mask_buf = cv2.imencode('.png', (final_mask * 255).astype(np.uint8))
    _, full_mask_buf   = cv2.imencode('.png', (global_mask * 255).astype(np.uint8))

    # SVD 桌面擬合（使用完整 SAM mask）
    print("   SVD 桌面擬合...")
    clean_depth = svd_table_fitting(depth_raw, global_mask, fx, fy, cx, cy)

    return {
        "status":           "success",
        "target_mask_png":  target_mask_buf.tobytes(),
        "full_mask_png":    full_mask_buf.tobytes(),
        "bbox":             [x_min, y_min, x_max, y_max],
        "target_grids":     target_grids,
        "gemini_reasoning": reasoning,
        "clean_depth":      clean_depth,    # 內部用，不傳給 client
        "color_bgr":        color_bgr,      # 內部用，不傳給 client
    }


def run_anygrasp(color_bgr, depth_raw, bbox, fx=FX, fy=FY, cx=CX, cy=CY):
    """
    執行 AnyGrasp 推論，回傳最佳抓取姿態 dict。
    bbox 為 [x1,y1,x2,y2] 或 None（全場景）。
    """
    colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depths = depth_raw
    h, w = depths.shape
    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    points_z = depths / DEPTH_SCALE
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0.1) & (points_z < 1.5)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors_masked = colors[mask].astype(np.float32)

    # --- 目標區域點雲密度增加（深度圖插值法）---
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        bx1 = max(0, bx1); by1 = max(0, by1)
        bx2 = min(w, bx2); by2 = min(h, by2)
        depth_roi = depth_raw[by1:by2, bx1:bx2].astype(np.float32)
        color_roi = colors[by1:by2, bx1:bx2]
        roi_h, roi_w = depth_roi.shape
        UPSAMPLE = 2
        depth_up = cv2.resize(depth_roi, (roi_w*UPSAMPLE, roi_h*UPSAMPLE),
                              interpolation=cv2.INTER_LINEAR)
        color_up = cv2.resize(color_roi, (roi_w*UPSAMPLE, roi_h*UPSAMPLE),
                              interpolation=cv2.INTER_LINEAR)
        up_h, up_w = depth_up.shape
        u_grid, v_grid = np.meshgrid(np.arange(up_w), np.arange(up_h))
        u_orig = bx1 + u_grid / UPSAMPLE
        v_orig = by1 + v_grid / UPSAMPLE
        Z_up = depth_up / DEPTH_SCALE
        X_up = (u_orig - cx) / fx * Z_up
        Y_up = (v_orig - cy) / fy * Z_up
        valid_up = (Z_up > 0.1) & (Z_up < 1.5)
        extra_pts = np.stack([X_up, Y_up, Z_up], axis=-1)[valid_up].astype(np.float32)
        extra_col = color_up[valid_up].astype(np.float32)
        points = np.vstack([points, extra_pts])
        colors_masked = np.vstack([colors_masked, extra_col])
        print(f"   📐 目標區域點雲密度 x{UPSAMPLE}，新增 {len(extra_pts)} 點")

    print("   🧠 AnyGrasp 計算抓取點...")
    gg, cloud = anygrasp.get_grasp(
        points, colors_masked, lims=WORKSPACE_LIMS,
        apply_object_mask=True, dense_grasp=False, collision_detection=True
    )

    if len(gg) == 0:
        return {"status": "fail", "message": "No grasp detected"}

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        translations = gg.translations
        u = (fx * translations[:, 0] / translations[:, 2] + cx).astype(int)
        v = (fy * translations[:, 1] / translations[:, 2] + cy).astype(int)
        valid = np.where((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2))[0]
        if len(valid) == 0:
            return {"status": "fail", "message": "ROI empty"}
        gg = gg[valid]
        print(f"   🎯 bbox 過濾後剩餘 {len(gg)} 個候選點")

    gg = gg.nms().sort_by_score()
    best = gg[0]
    return {
        "status":      "success",
        "score":       float(best.score),
        "width":       float(best.width),
        "depth":       float(best.depth),
        "translation": best.translation.tolist(),
        "rotation":    best.rotation_matrix.tolist(),
        "cloud":       cloud,  # 內部用，不傳給 client
        "gg":          gg,     # 內部用，不傳給 client
    }


# ==========================================
# 7. Main Loop
# ==========================================
try:
    while True:
        if vis is not None:
            if not vis.poll_events():
                break
            vis.update_renderer()

        try:
            compressed_data = socket.recv(flags=zmq.NOBLOCK)
            payload = pickle.loads(zlib.decompress(compressed_data))
        except zmq.Again:
            time.sleep(0.01)
            continue

        print("\n📦 收到請求...")
        mode = payload.get("mode", "grasp")

        # ── VLM 模式 ──────────────────────────────────────────────────
        if mode == "vlm":
            print(f"🔍 VLM 模式，目標：'{payload.get('object_name','?')}'")
            color_bgr = cv2.imdecode(
                np.frombuffer(payload['color_jpg'], np.uint8), cv2.IMREAD_COLOR)
            depth_raw   = payload['depth']
            object_name = payload['object_name']
            intrinsics  = payload.get('camera_intrinsics', {})
            fx = intrinsics.get('fx', FX)
            fy = intrinsics.get('fy', FY)
            cx = intrinsics.get('cx', CX)
            cy = intrinsics.get('cy', CY)

            vlm_res = run_vlm_pipeline(color_bgr, object_name, depth_raw, fx, fy, cx, cy)

            if vlm_res["status"] != "success":
                result = vlm_res
            else:
                clean_depth = vlm_res.pop("clean_depth")
                vlm_color   = vlm_res.pop("color_bgr")
                bbox        = vlm_res["bbox"]

                grasp_res = run_anygrasp(vlm_color, clean_depth, bbox, fx, fy, cx, cy)

                if grasp_res["status"] != "success":
                    result = {"status": "fail", "reason": grasp_res["message"],
                              **{k: v for k, v in vlm_res.items()
                                 if k not in ("cloud", "gg")}}
                else:
                    cloud = grasp_res.pop("cloud", None)
                    gg    = grasp_res.pop("gg", None)
                    result = {**vlm_res, **grasp_res}

                    # Open3D 視覺化
                    if vis is not None and cloud is not None and gg is not None:
                        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                        cloud.transform(trans_mat)
                        geo = gg[0:1].to_open3d_geometry_list()
                        geo[0].transform(trans_mat)
                        vis.clear_geometries()
                        vis.add_geometry(cloud, reset_bounding_box=True)
                        vis.add_geometry(geo[0], reset_bounding_box=False)

            # 移除 numpy array 大型欄位再傳送（已轉成 PNG bytes）
            result.pop("cloud", None)
            result.pop("gg", None)
            print(f"   status: {result['status']}")

        # ── 手動/一般 AnyGrasp 模式 ───────────────────────────────────
        else:
            print("🤲 手動模式 (AnyGrasp only)")
            if 'color_jpg' in payload:
                color_bgr = cv2.imdecode(
                    np.frombuffer(payload['color_jpg'], np.uint8), cv2.IMREAD_COLOR)
            else:
                color_bgr = cv2.cvtColor(
                    (payload['color'] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            depth_raw = payload['depth']
            bbox      = payload.get('bbox', None)

            grasp_res = run_anygrasp(color_bgr, depth_raw, bbox)
            cloud = grasp_res.pop("cloud", None)
            gg    = grasp_res.pop("gg", None)
            result = grasp_res

            if vis is not None and cloud is not None and gg is not None:
                trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud.transform(trans_mat)
                geo = gg[0:1].to_open3d_geometry_list()
                geo[0].transform(trans_mat)
                vis.clear_geometries()
                vis.add_geometry(cloud, reset_bounding_box=True)
                vis.add_geometry(geo[0], reset_bounding_box=False)

            print(f"   status: {result['status']}, score: {result.get('score','N/A')}")

            # 存檔（原有邏輯保留）
            if cfgs.save and result.get("status") == "success":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = os.path.join(SAVE_DIR, ts)
                cv2.imwrite(f"{prefix}_rgb.jpg", color_bgr)
                np.save(f"{prefix}_depth.npy", depth_raw)
                with open(f"{prefix}_result.json", "w") as f:
                    json.dump({k: v for k, v in result.items()
                               if isinstance(v, (str, float, int, list))}, f, indent=4)
                print(f"💾 已存檔: {ts}_*")

        # 統一回傳：zlib + pickle（兩種模式都用這個格式）
        socket.send(zlib.compress(pickle.dumps(result)))
        print("✅ 回傳完成")

except KeyboardInterrupt:
    print("\n🛑 使用者中斷 (Ctrl+C)")
except Exception as e:
    print(f"\n❌ 非預期錯誤: {e}")
    import traceback; traceback.print_exc()
finally:
    if vis is not None:
        vis.destroy_window()
    socket.close()
    context.term()
    print("👋 程式已安全結束")
```

---

## 四、驗證步驟

### Step 1：套件安裝確認
```bash
python -c "from transformers import pipeline; print('transformers OK')"
python -c "from google import genai; print('google-genai OK')"
python -c "from PIL import Image; print('Pillow OK')"
```

### Step 2：獨立測試 VLM pipeline（不啟動 ZMQ）
在 server 目錄下建立測試腳本 `test_vlm.py`：
```python
import cv2, numpy as np
# 先 import server 的函式（把 server_anygrasp.py 最底部的 while True 用 if False 包起來）
# 或直接複製 run_vlm_pipeline 函式到這裡測試
color = cv2.imread("test_image.jpg")  # 放一張測試圖
result = run_vlm_pipeline(color, "bottle", np.zeros((480,640), dtype=np.uint16),
                          617.183, 617.122, 319.639, 241.404)
print(result["status"], result.get("target_grids"))
```

### Step 3：完整端對端測試
啟動 server：
```bash
python3 server_anygrasp.py --debug
```
確認 log 顯示：
```
✅ AnyGrasp 載入完成！
✅ OWL-v2 載入完成
✅ SAM v1 載入完成
✅ 所有模型載入完成！
🎧 伺服器已上線，監聽 Port 5555...
```

---

## 五、回報清單

完成後請確認以下項目並回報：

- [ ] 套件安裝成功（transformers, google-genai, Pillow, python-dotenv）
- [ ] server 啟動後三個模型都顯示「載入完成」
- [ ] `.env` 已建立，GOOGLE_API_KEY 有填入
- [ ] `vlm_logs/` 目錄會在 server 目錄下自動建立
- [ ] 手動模式（mode=grasp）回傳格式與原本相同（zlib+pickle，client 需同步修改接收方式）
- [ ] 若測試 VLM 模式，請回報 `target_grids` 輸出是否合理

---

## 六、注意事項

1. **response 格式變更**：原本 server 用 `send_pyobj`（未壓縮），改版後改為 `zlib+pickle`。本地 client 需要同步修改接收方式，**兩邊需同時部署**才能正常運作。

2. **首次啟動較慢**：OWL-v2 和 SAM 第一次執行會自動下載 HuggingFace 模型（約 400MB），需要網路連線。

3. **Gemini timeout**：若 API key 配額耗盡，最多會嘗試每個 key 兩次，再回傳 fail。

4. **VLM 模式不傳 `clean_depth` 和 `color_bgr`** 給 client（只傳 PNG bytes），減少傳輸量。

5. **`cloud` 和 `gg` 物件不可序列化**，已在傳送前 pop 掉。
