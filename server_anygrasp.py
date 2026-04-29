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
from datetime import datetime
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import open3d as o3d  # 移到最上方統一載入，方便存檔使用


def _safe_normalize(vec, eps=1e-6):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def _decode_mask(mask_payload, image_shape):
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


def _extract_target_points(points, bbox, mask_bool, fx, fy, cx, cy):
    if len(points) == 0:
        return points

    valid_depth = points[:, 2] > 1e-6
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32)

    proj_points = points[valid_depth]
    u = (fx * proj_points[:, 0] / proj_points[:, 2] + cx).astype(int)
    v = (fy * proj_points[:, 1] / proj_points[:, 2] + cy).astype(int)
    valid_proj = (u >= 0) & (u < int(cx * 2 + 1)) & (v >= 0) & (v < int(cy * 2 + 1))

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


def filter_grasps_by_geometry(gg, points, bbox, mask_bool, object_shape, fx, fy, cx, cy):
    """
    根據物件形狀標籤對 AnyGrasp 候選做幾何過濾。
    若過濾後沒有候選，則回退到原始候選。
    """
    target_pts = _extract_target_points(points, bbox, mask_bool, fx, fy, cx, cy)
    if len(target_pts) < 30 or len(gg) == 0:
        print("   ⚠️ 目標點雲不足，跳過幾何過濾")
        return gg

    rotations = np.array([g.rotation_matrix for g in gg])
    translations = np.array([g.translation for g in gg])
    approach = rotations[:, :, 0]  # X 軸 = 接近方向
    closing = rotations[:, :, 1]   # Y 軸 = 閉合方向
    valid_mask = np.ones(len(gg), dtype=bool)

    if object_shape == "sphere":
        center = np.mean(target_pts, axis=0)
        to_center = center - translations
        norm = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
        to_center_unit = to_center / norm
        dot = np.einsum("ij,ij->i", approach, to_center_unit)
        valid_mask &= (dot > 0.85)
        print(f"   🔵 sphere 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    elif object_shape == "box":
        centered = target_pts - np.mean(target_pts, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        dominant_normal = _safe_normalize(vt[-1])
        if dominant_normal is None:
            print("   ⚠️ box 法向量估計失敗，跳過幾何過濾")
            return gg
        dot = np.abs(np.einsum("ij,j->i", approach, dominant_normal))
        valid_mask &= (dot > 0.80)
        print(f"   📦 box 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    elif object_shape == "cylinder":
        centered = target_pts - np.mean(target_pts, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = _safe_normalize(vt[0])  # 第一主成分 = 圓柱主軸
        if axis is None:
            print("   ⚠️ cylinder 主軸估計失敗，跳過幾何過濾")
            return gg
        dot = np.abs(np.einsum("ij,j->i", closing, axis))
        valid_mask &= (dot < 0.35)
        print(f"   🥤 cylinder 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    else:
        print(f"   ℹ️ 未知形狀 '{object_shape}'，跳過幾何過濾")
        return gg

    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        print("   ⚠️ 幾何過濾後無候選，回退原始排序")
        return gg

    return gg[valid_indices]


# ==========================================
# 1. Model Initialization
# ==========================================
parser = argparse.ArgumentParser(description="AnyGrasp AI Inference Server")
parser.add_argument('--checkpoint_path', default='../checkpoint/checkpoint_detection.tar')
parser.add_argument('--max_gripper_width', type=float, default=0.1)
parser.add_argument('--gripper_height', type=float, default=0.03)
parser.add_argument('--top_down_grasp', action='store_true')
parser.add_argument('--debug', action='store_true', help="Enable Open3D visualization")
parser.add_argument('--save', action='store_true', help="Save results for later review")  # ✨ 新增存檔開關
cfgs = parser.parse_args()

# ✨ 建立存檔資料夾
SAVE_DIR = "saved_results"
if cfgs.save:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 已開啟自動存檔功能，資料將儲存於: {SAVE_DIR}/")

print("🚀 正在載入 AnyGrasp AI 模型...")
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()
print("✅ 模型載入完成！")

# ==========================================
# 2. ZeroMQ & Camera Parameter Configuration
# ==========================================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# 相機內參 (Camera Intrinsics)
FX, FY = 617.183, 617.122
CX, CY = 319.639, 241.404
DEPTH_SCALE = 1000.0
WORKSPACE_LIMS = [-0.3, 0.3, -0.2, 0.4, 0.2, 0.8]

# ==========================================
# 3. Visualization Engine (Open3D)
# ==========================================
vis = None
if cfgs.debug:
    print("🖥️  啟動非阻塞 3D 預覽視窗...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AnyGrasp", width=800, height=600)

print("🎧 伺服器已上線，監聽 Port 5555... (按下 Ctrl+C 或關閉視窗結束)")

# ==========================================
# 4. Main Inference Loop
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

        print("\n📦 收到影像包裹，開始運算...")

        if 'color_jpg' in payload:
            color_bgr = cv2.imdecode(payload['color_jpg'], cv2.IMREAD_COLOR)
            if color_bgr is None:
                print("❌ 錯誤：JPG 解碼失敗")
                socket.send_pyobj({"status": "fail", "message": "Decode error"})
                continue
            colors = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            colors = payload['color']
            color_bgr = cv2.cvtColor((colors * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # 為了存檔用

        depths = payload['depth']
        bbox = payload.get('bbox', None)
        mask_bool = _decode_mask(payload.get('mask', None), depths.shape)
        object_shape = payload.get('object_shape', 'box')
        print(f"🧩 物件形狀提示: {object_shape}")
        if mask_bool is not None:
            print(f"🎭 已收到 target mask，覆蓋像素數: {int(mask_bool.sum())}")

        h, w = depths.shape
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        points_z = depths / DEPTH_SCALE
        points_x = (xmap - CX) / FX * points_z
        points_y = (ymap - CY) / FY * points_z

        depth_valid = (points_z > 0.1) & (points_z < 1.5)
        point_mask = depth_valid.copy()
        if mask_bool is not None:
            point_mask &= mask_bool

        points = np.stack([points_x, points_y, points_z], axis=-1)[point_mask].astype(np.float32)
        colors_mask = colors[point_mask].astype(np.float32)

        # --- 目標區域點雲密度增加（深度圖插值法）---
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

            UPSAMPLE = 2
            depth_up = cv2.resize(depth_roi, (roi_w * UPSAMPLE, roi_h * UPSAMPLE),
                                  interpolation=cv2.INTER_LINEAR)
            color_up = cv2.resize(color_roi, (roi_w * UPSAMPLE, roi_h * UPSAMPLE),
                                  interpolation=cv2.INTER_LINEAR)

            up_h, up_w = depth_up.shape
            u_grid, v_grid = np.meshgrid(np.arange(up_w), np.arange(up_h))

            # sub-pixel 對應回原圖的浮點座標
            u_orig = bx1 + u_grid / UPSAMPLE
            v_orig = by1 + v_grid / UPSAMPLE

            Z_up = depth_up / DEPTH_SCALE
            X_up = (u_orig - CX) / FX * Z_up
            Y_up = (v_orig - CY) / FY * Z_up

            valid_up = (Z_up > 0.1) & (Z_up < 1.5)
            if mask_bool is not None:
                mask_roi = mask_bool[by1:by2, bx1:bx2].astype(np.uint8) * 255
                mask_up = cv2.resize(mask_roi, (roi_w * UPSAMPLE, roi_h * UPSAMPLE),
                                     interpolation=cv2.INTER_NEAREST) > 127
                valid_up &= mask_up
            extra_points = np.stack([X_up, Y_up, Z_up], axis=-1)[valid_up].astype(np.float32)
            extra_colors = color_up[valid_up].astype(np.float32)

            if len(extra_points) > 0:
                points = np.vstack([points, extra_points])
                colors_mask = np.vstack([colors_mask, extra_colors])
                print(f"📐 目標區域點雲密度 x{UPSAMPLE}，新增 {len(extra_points)} 點")

        print("🧠 計算全場景抓取點 (包含防撞偵測)...")
        gg, cloud = anygrasp.get_grasp(
            points, colors_mask, lims=WORKSPACE_LIMS,
            apply_object_mask=True, dense_grasp=False, collision_detection=True
        )

        if len(gg) == 0:
            print("⚠️ 找不到任何有效的抓取點。")
            socket.send_pyobj({"status": "fail", "message": "No grasp detected"})
            continue

        if bbox is not None or mask_bool is not None:
            translations = gg.translations
            u = (FX * translations[:, 0] / translations[:, 2] + CX).astype(int)
            v = (FY * translations[:, 1] / translations[:, 2] + CY).astype(int)

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
                print("⚠️ 指定區域內無安全抓取點！")
                socket.send_pyobj({"status": "fail", "message": "ROI empty"})
                continue
            gg = gg[valid_indices]
            print(f"🎯 已鎖定目標區域，剩餘 {len(gg)} 個候選點。")

        gg = gg.nms().sort_by_score()
        gg = filter_grasps_by_geometry(gg, points, bbox, mask_bool, object_shape, FX, FY, CX, CY)
        gg = gg.sort_by_score()
        best_grasp = gg[0]

        result = {
            "status": "success",
            "score": float(best_grasp.score),
            "width": float(best_grasp.width),
            "depth": float(best_grasp.depth),
            "translation": best_grasp.translation.tolist(),
            "rotation": best_grasp.rotation_matrix.tolist()
        }

        socket.send_pyobj(result)
        print(f"✅ 運算完成！最高分 {result['score']:.4f}，座標已回傳。")

        # ✨ 資料儲存邏輯 (黑盒子)
        if cfgs.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = os.path.join(SAVE_DIR, timestamp)

            # 1. 存彩色圖
            cv2.imwrite(f"{prefix}_rgb.jpg", color_bgr)
            # 2. 存深度圖 (存成 .npy 檔以保留真實數值)
            np.save(f"{prefix}_depth.npy", depths)
            # 3. 存點雲檔 (.ply)
            o3d.io.write_point_cloud(f"{prefix}_cloud.ply", cloud)
            # 4. 存抓取結果 (.json)
            with open(f"{prefix}_result.json", "w") as f:
                json.dump(result, f, indent=4)
            print(f"💾 本次運算資料已儲存為: {timestamp}_*")

        if vis is not None:
            trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            cloud.transform(trans_mat)
            gripper_geo = gg[0:1].to_open3d_geometry_list()
            gripper_geo[0].transform(trans_mat)

            vis.clear_geometries()
            vis.add_geometry(cloud, reset_bounding_box=True)
            vis.add_geometry(gripper_geo[0], reset_bounding_box=False)
            print("🔄 3D 視窗已更新。")

except KeyboardInterrupt:
    print("\n🛑 伺服器由使用者手動中斷 (Ctrl+C)。")
except Exception as e:
    print(f"\n❌ 發生非預期錯誤: {e}")
finally:
    if vis is not None:
        vis.destroy_window()
    socket.close()
    context.term()
    print("👋 程式已安全結束。")
