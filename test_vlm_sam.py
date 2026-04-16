#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vlm_sam.py — VLM+SAM 互動測試工具（即時預覽 + 按鍵觸發）

操作方式：
  [v]       設定目標物件名稱（英文）
  [s]       對當前畫面執行 OWL-v2 → SAM → Gemini 分析
  [q]       離開

執行：
  conda activate anygrasp
  python3 test_vlm_sam.py                    # RealSense 即時畫面
  python3 test_vlm_sam.py --image test.jpg   # 靜態圖片
  python3 test_vlm_sam.py --target bottle    # 預設目標名稱
"""

import os
import sys
import json
import argparse
import time
import threading
import cv2
import numpy as np
from PIL import Image as PILImage

OUTPUT_DIR = "/tmp/test_vlm_sam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 模型（延遲載入，只在第一次 [s] 時初始化）
# ─────────────────────────────────────────────
_detector = None
_sam_model = None
_sam_processor = None
_gemini_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_models():
    global _detector, _sam_model, _sam_processor, _gemini_model
    import torch
    from transformers import pipeline, SamModel, SamProcessor
    from google import genai
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("找不到 GOOGLE_API_KEY，請確認 handeye_ws/.env 存在")

    dev = get_device()
    print(f"[模型] 載入中（{dev}）...")

    if _detector is None:
        _detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=dev
        )
        print("[模型] OWL-v2 就緒")

    if _sam_model is None:
        _sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(dev)
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        print("[模型] SAM v1 就緒")

    if _gemini_model is None:
        _gemini_model = genai.Client(api_key=api_key)
        print("[模型] Gemini 就緒")


# ─────────────────────────────────────────────
# SoM 網格
# ─────────────────────────────────────────────

def draw_som_grid(img_rgb, rows=5, cols=5):
    h, w = img_rgb.shape[:2]
    cell_w, cell_h = w / cols, h / rows
    col_labels = [chr(65 + i) for i in range(cols)]
    overlay = img_rgb.copy()
    colors = [(173, 216, 230), (255, 200, 150)]
    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            gid = f"{col_labels[c]}{r + 1}"
            grid_dict[gid] = (x1, y1, x2, y2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colors[(r + c) % 2], -1)

    result = cv2.addWeighted(overlay, 0.15, img_rgb, 0.85, 0)
    for i in range(1, rows):
        cv2.line(result, (0, int(i * cell_h)), (w, int(i * cell_h)), (80, 80, 80), 1)
    for j in range(1, cols):
        cv2.line(result, (int(j * cell_w), 0), (int(j * cell_w), h), (80, 80, 80), 1)
    cv2.rectangle(result, (0, 0), (w - 1, h - 1), (80, 80, 80), 2)

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            gid = f"{col_labels[c]}{r + 1}"
            cx_g, cy_g = (x1 + x2) // 2, (y1 + y2) // 2
            fs = min(cell_w, cell_h) / 60.0
            th = max(1, int(fs * 2))
            (tw, tth), _ = cv2.getTextSize(gid, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx, ty = cx_g - tw // 2, cy_g + tth // 2
            cv2.putText(result, gid, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
            cv2.putText(result, gid, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)

    return result, grid_dict


# ─────────────────────────────────────────────
# VLM+SAM 管線
# ─────────────────────────────────────────────

UR3_PROMPT = """
你是一個機器人視覺分析專家，協助單臂 UR3 機械臂抓取物件。

【圖片 1】全局場景圖。
【圖片 2】物件特寫網格圖：5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。

選擇 UR3 夾爪最佳抓取區域：
- 優先 Y2-Y3（中段），避免 Y5（靠桌面）
- 選 2-4 個相鄰格子，形成有效夾取面
- 選物件較寬/厚的部位

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明選擇理由"
}
"""


def run_pipeline(img_bgr, target, result_holder):
    """背景執行緒：跑完整 VLM+SAM 管線，結果存進 result_holder dict"""
    import torch

    try:
        load_models()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        img_pil = PILImage.fromarray(img_rgb)

        # ── OWL-v2 偵測 ──
        result_holder['status'] = 'owlv2'
        print(f"\n[1/4] OWL-v2 偵測「{target}」...")
        preds = _detector(img_pil, candidate_labels=[target])
        if not preds:
            result_holder['status'] = 'fail'
            result_holder['error'] = f"找不到「{target}」"
            print(f"[1/4] ❌ {result_holder['error']}")
            return

        best = max(preds, key=lambda x: x['score'])
        b = best['box']
        x1, y1, x2, y2 = int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])
        conf = best['score']
        print(f"[1/4] ✅ 信心度 {conf:.3f}，bbox ({x1},{y1})-({x2},{y2})")

        # ── SAM 分割 ──
        result_holder['status'] = 'sam'
        print("[2/4] SAM 分割...")
        dev = get_device()
        inputs = _sam_processor(
            img_pil,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            outputs = _sam_model(**inputs)
        masks = _sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs.original_sizes.cpu(),
            inputs.reshaped_input_sizes.cpu()
        )
        global_mask = masks[0][0][0].numpy()
        print(f"[2/4] ✅ mask 像素數：{int(global_mask.sum())}")

        # ── SoM 網格 ──
        result_holder['status'] = 'som'
        print("[3/4] 繪製 SoM 網格...")
        pad = 20
        c_x1, c_y1 = max(0, x1 - pad), max(0, y1 - pad)
        c_x2, c_y2 = min(w, x2 + pad), min(h, y2 + pad)
        cropped = img_rgb[c_y1:c_y2, c_x1:c_x2].copy()
        grid_img, grid_local = draw_som_grid(cropped)

        grid_abs = {}
        for gid, (lx1, ly1, lx2, ly2) in grid_local.items():
            grid_abs[gid] = [c_x1 + lx1, c_y1 + ly1, c_x1 + lx2, c_y1 + ly2]

        grid_path = os.path.join(OUTPUT_DIR, "som_grid.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

        # ── Gemini 推理 ──
        result_holder['status'] = 'gemini'
        print("[4/4] Gemini 分析...")
        grid_pil = PILImage.open(grid_path)
        # 模型優先順序：gemini-flash-latest → 2.5-flash-lite（前者過載時自動切換）
        for model_id in ['gemini-flash-latest', 'gemini-2.5-flash-lite']:
            try:
                response = _gemini_model.models.generate_content(
                    model=model_id,
                    contents=[UR3_PROMPT, img_pil, grid_pil]
                )
                print(f"[4/4] 使用模型：{model_id}")
                break
            except Exception as e:
                if '503' in str(e) or 'UNAVAILABLE' in str(e):
                    print(f"[4/4] {model_id} 過載，切換備用模型...")
                    continue
                raise
        raw = response.text.replace("```json", "").replace("```", "").strip()
        vlm = json.loads(raw)
        target_grids = vlm.get("target_grids", [])
        print(f"[4/4] ✅ 抓取格子：{target_grids}")
        print(f"       理由：{vlm.get('reasoning','')}")

        # ── 合成最終 mask ──
        final_mask = np.zeros_like(global_mask, dtype=bool)
        for gid in target_grids:
            if gid in grid_abs:
                gx1, gy1, gx2, gy2 = grid_abs[gid]
                final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]

        cv2.imwrite(os.path.join(OUTPUT_DIR, "target_mask.png"),
                    (final_mask * 255).astype(np.uint8))

        # ── 結果視覺化（三格拼接）──
        # 左：原圖 + bbox
        vis_raw = img_bgr.copy()
        cv2.rectangle(vis_raw, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(vis_raw, f"{target} {conf:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        # 中：SAM 全域 mask overlay
        vis_sam = img_bgr.copy()
        overlay_sam = vis_sam.copy()
        overlay_sam[global_mask > 0] = (0, 80, 255)
        vis_sam = cv2.addWeighted(vis_sam, 0.55, overlay_sam, 0.45, 0)
        cv2.putText(vis_sam, "SAM mask", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)

        # 右：最終抓取 mask + 網格覆蓋
        vis_final = img_bgr.copy()
        overlay_f = vis_final.copy()
        overlay_f[final_mask] = (0, 0, 255)
        vis_final = cv2.addWeighted(vis_final, 0.5, overlay_f, 0.5, 0)
        # 標出選取的格子代號
        for gid in target_grids:
            if gid in grid_abs:
                gx1, gy1, gx2, gy2 = grid_abs[gid]
                cv2.rectangle(vis_final, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                cv2.putText(vis_final, gid, (gx1 + 4, gy2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis_final, f"grids: {target_grids}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 確保三張圖高度一致
        th = max(vis_raw.shape[0], vis_sam.shape[0], vis_final.shape[0])

        def pad_height(img, target_h):
            if img.shape[0] < target_h:
                pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                return np.vstack([img, pad])
            return img

        panel = np.hstack([
            pad_height(vis_raw, th),
            pad_height(vis_sam, th),
            pad_height(vis_final, th),
        ])

        result_path = os.path.join(OUTPUT_DIR, "result_panel.png")
        cv2.imwrite(result_path, panel)

        result_holder['status'] = 'done'
        result_holder['panel'] = panel
        result_holder['bbox'] = [x1, y1, x2, y2]
        result_holder['target_grids'] = target_grids
        result_holder['reasoning'] = vlm.get('reasoning', '')

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_holder['status'] = 'fail'
        result_holder['error'] = str(e)


# ─────────────────────────────────────────────
# 相機
# ─────────────────────────────────────────────

def open_realsense():
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("📷 RealSense 啟動")
    return pipeline


def grab_realsense(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data())


# ─────────────────────────────────────────────
# 主迴圈
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  help="靜態圖片路徑（不加則用 RealSense）")
    parser.add_argument("--target", default="", help="目標物件名稱（英文）")
    args = parser.parse_args()

    # 確認 API Key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ 請先設定：export GOOGLE_API_KEY='your_key'")
        sys.exit(1)

    static_mode = args.image is not None
    target = args.target

    # 靜態圖片模式
    if static_mode:
        static_img = cv2.imread(args.image)
        if static_img is None:
            print(f"❌ 無法讀取圖片：{args.image}")
            sys.exit(1)

    # RealSense 模式
    rs_pipeline = None
    if not static_mode:
        try:
            rs_pipeline = open_realsense()
        except Exception as e:
            print(f"❌ RealSense 開啟失敗：{e}")
            sys.exit(1)

    WIN_PREVIEW = "VLM+SAM 測試  [v] 設定目標  [s] 送出  [q] 離開"
    WIN_RESULT  = "分析結果  [任意鍵關閉]"
    cv2.namedWindow(WIN_PREVIEW, cv2.WINDOW_NORMAL)

    processing = False
    result_holder = {}
    frozen_frame = None   # 送出時凍結的那一幀

    PROCESSING_LABELS = {
        'owlv2':  'OWL-v2 偵測中...',
        'sam':    'SAM 分割中...',
        'som':    '繪製 SoM 網格...',
        'gemini': 'Gemini 推理中...',
    }

    print("\n操作說明：")
    print("  [v] 設定目標物件名稱")
    print("  [s] 對當前畫面送出 VLM+SAM 分析")
    print("  [q] 離開")
    if target:
        print(f"  目前目標：{target}")
    else:
        print("  ⚠️  尚未設定目標，請先按 [v]")

    while True:
        # 取得畫面
        if static_mode:
            frame = static_img.copy()
        else:
            frame = grab_realsense(rs_pipeline)
            if frame is None:
                continue

        display = frame.copy()

        # 狀態列
        bar_h = 50
        bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)

        if processing:
            stage = result_holder.get('status', '')
            label = PROCESSING_LABELS.get(stage, '處理中...')
            # 閃爍效果
            color = (0, 200, 255) if int(time.time() * 2) % 2 == 0 else (0, 150, 200)
            cv2.putText(bar, label, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # 凍結畫面覆蓋提示
            display = frozen_frame.copy() if frozen_frame is not None else display
            cv2.putText(display, "PROCESSING - FROZEN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            tgt_text = f"目標：{target}" if target else "目標：未設定（按 [v]）"
            tgt_color = (0, 255, 120) if target else (0, 100, 255)
            cv2.putText(bar, tgt_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, tgt_color, 2)
            hint = "[v] 設定目標   [s] 送出分析   [q] 離開"
            cv2.putText(bar, hint, (display.shape[1] - 480, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        combined = np.vstack([display, bar])
        cv2.imshow(WIN_PREVIEW, combined)

        # 檢查背景執行緒完成
        if processing and result_holder.get('status') in ('done', 'fail'):
            processing = False
            if result_holder['status'] == 'done':
                panel = result_holder['panel']
                print(f"\n✅ 完成！抓取格子：{result_holder['target_grids']}")
                print(f"   理由：{result_holder['reasoning']}")
                print(f"   結果圖：{OUTPUT_DIR}/result_panel.png")
                # 顯示結果視窗
                cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)
                # 縮放到合理大小（最寬 1800px）
                max_w = 1800
                if panel.shape[1] > max_w:
                    scale = max_w / panel.shape[1]
                    panel_show = cv2.resize(panel, (max_w, int(panel.shape[0] * scale)))
                else:
                    panel_show = panel
                cv2.imshow(WIN_RESULT, panel_show)
            else:
                print(f"\n❌ 失敗：{result_holder.get('error','unknown')}")

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('v'):
            cv2.destroyWindow(WIN_RESULT) if cv2.getWindowProperty(WIN_RESULT, cv2.WND_PROP_VISIBLE) >= 1 else None
            t = input("\n🔍 目標物件名稱（英文）：").strip()
            if t:
                target = t
                print(f"✅ 目標設定為：{target}")

        elif key == ord('s'):
            if not target:
                print("⚠️  請先按 [v] 設定目標")
            elif processing:
                print("⚠️  正在處理中，請稍候...")
            else:
                frozen_frame = frame.copy()
                result_holder.clear()
                result_holder['status'] = 'owlv2'
                processing = True
                t = threading.Thread(
                    target=run_pipeline,
                    args=(frozen_frame.copy(), target, result_holder),
                    daemon=True
                )
                t.start()
                print(f"\n▶ 開始分析「{target}」...")

        # 按任意鍵關閉結果視窗
        elif key != 255:
            try:
                if cv2.getWindowProperty(WIN_RESULT, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(WIN_RESULT)
            except Exception:
                pass

    cv2.destroyAllWindows()
    if rs_pipeline:
        rs_pipeline.stop()


if __name__ == "__main__":
    main()
