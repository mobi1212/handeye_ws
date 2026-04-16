# 系統架構與 VLM+SAM 整合計劃書

**版本**: v3.0  
**更新日期**: 2026-04-16  
**狀態**: VLM+SAM 本機驗證完成，待整合至完整管線

---

## 1. 現有架構（整合前）

### 1.1 完整硬體拓撲

```
筆電（Ubuntu 20.04 + ROS Noetic）          遠端 AI Server（GPU）
┌──────────────────────────────────┐        ┌─────────────────────┐
│  RealSense D435                  │        │  server_anygrasp.py │
│       ↓ ROS topics               │        │                     │
│  client_camera.py                │◄──────►│  AnyGrasp           │
│    YOLO 偵測                     │  ZMQ   │  6D 抓取姿態估計     │
│    SVD 桌面擬合                  │  TCP   │                     │
│       ↓ /anygrasp/target_pose    │        └─────────────────────┘
│  semantic_grasp_controller.py    │
│    TF 轉換 / MoveIt              │
│       ↓                          │
│  UR3 機械臂（192.168.86.7）      │
└──────────────────────────────────┘
```

### 1.2 管線 A：語義抓取（主要流程）

```
RealSense D435
  ├─ /camera/color/image_raw
  └─ /camera/aligned_depth_to_color/image_raw
           │
           ▼
  client_camera.py  [conda: anygrasp / Python 3.8]
    ├─ [模式1] YOLO (best.pt) → bbox + mask
    ├─ [模式2] 手動框選 [r] → bbox only
    └─ SVD 桌面擬合（有 mask 時）→ clean_depth
           │ ZMQ REQ (zlib+pickle)
           ▼
  server_anygrasp.py  [AI Server / GPU]
    └─ AnyGrasp → 6D 抓取姿態
           │ /anygrasp/target_pose (PoseStamped)
           ▼
  semantic_grasp_controller.py
    ├─ TF: camera_color_optical_frame → base_link
    ├─ 姿態對齊: X軸旋轉 Y-90° → UR3 Z軸
    └─ MoveIt: Pre-Grasp → 前進 → 夾取 → 抬升 → 放置
```

### 1.3 管線 B：傳統 Click-to-Pick（備用）

```
click_to_pick_cv.py → YOLO 偵測像素 → pixel_to_base.py → pose_to_pick.py
```

### 1.4 關鍵參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `tcp_offset` | 0.18 m | 夾爪指尖到 MoveIt EEF 距離 |
| `grasp_depth` | 0.04 m | 插入物體表面深度 |
| `approach_dist` | 0.05 m | Pre-Grasp 退後距離 |
| `retreat_up_height` | 0.15 m | 抓取後垂直抬升 |
| ZMQ timeout | 30 秒 | 逾時後自動重建 REQ socket |

---

## 2. 新功能：VLM+SAM 語義辨識

### 2.1 驗證結果（2026-04-16）

本機已成功跑通完整管線：
- **OWL-v2**：PS4 controller 信心度 0.695，bbox 正確
- **SAM v1**：分割出 18,869 像素，mask 精準
- **SoM 網格**：5×5 網格正確疊加在物件上
- **Gemini**（gemini-flash-latest）：正確回傳 JSON 格式的抓取格子

### 2.2 VLM+SAM 子管線

```
使用者輸入文字（例如 "ps4 controller"）
           │ [v] 鍵
           ▼
  client_camera.py 發布 /system/trigger_llm
           │ {"object_name": "ps4 controller"}
           ▼
  semantic_brain_node.py  [conda: grasp-py310 / Python 3.10]
    │
    ├─ [1/4] OWL-v2 (google/owlv2-base-patch16-ensemble)
    │         zero-shot 偵測 → bbox (不需訓練)
    │
    ├─ [2/4] SAM v1 (facebook/sam-vit-base)
    │         bbox → 高解析度 global mask
    │
    ├─ [3/4] draw_som_grid()
    │         裁切物件 → 疊加 5×5 SoM 網格
    │
    └─ [4/4] Gemini (gemini-flash-latest, 備援 gemini-2.5-flash-lite)
              原始場景圖 + 網格圖 → JSON {target_grids: ["B2","C2",...]}
                         │
                         ▼
              最終 mask = SAM global_mask ∩ target_grids 區域
              儲存 /tmp/semantic_brain/target_mask.png
           │ /system/llm_done {"status":"done"}
           ▼
  client_camera.py 讀取 target_mask.png
    └─ 走既有 SVD 桌面擬合 → ZMQ → AnyGrasp → MoveIt（不需修改後段）
```

### 2.3 整合後的三種偵測模式

```
client_camera.py 偵測模式優先權：

優先權 1：手動框選 [r]
  bbox only，無 mask，SVD 跳過

優先權 2：VLM+SAM 模式 [v]
  輸入文字 → brain node 處理
  → 讀取 target_mask.png
  → 高解析度 SAM mask + SVD 桌面擬合

優先權 3：YOLO 自動模式（預設）
  best.pt 固定類別偵測
  → YOLO mask（低解析）+ SVD 桌面擬合
```

---

## 3. 整合後完整架構

```
筆電                                        遠端 AI Server
┌──────────────────────────────────────┐   ┌─────────────────────┐
│                                      │   │  server_anygrasp.py │
│  RealSense D435                      │   │                     │
│    /camera/color/image_raw           │   │  AnyGrasp GPU 推論  │
│    /camera/aligned_depth_to_color    │   │                     │
│           │                          │   └──────────┬──────────┘
│           ▼                          │              │ ZMQ REP
│  ┌────────────────────────────────┐  │              │
│  │ client_camera.py               │  │              │
│  │  [anygrasp env / Python 3.8]   │◄─┼──────────────┘
│  │                                │  │
│  │  模式1: YOLO                   │  │
│  │  模式2: 手動框選 [r]           │  │
│  │  模式3: VLM+SAM [v] ←─────┐  │  │
│  │         讀 target_mask.png  │  │  │
│  └──────────────┬──────────────┘  │  │
│                 │ /system/trigger  │  │
│                 ▼                  │  │
│  ┌────────────────────────────────┐│  │
│  │ semantic_brain_node.py         ││  │
│  │  [grasp-py310 env / Python 3.10││  │
│  │                                ││  │
│  │  OWL-v2 → SAM → SoM → Gemini  ││  │
│  │  → /tmp/semantic_brain/        ││  │
│  │    target_mask.png  ───────────┘│  │
│  │  → /system/llm_done            │  │
│  └────────────────────────────────┘  │
│                 │ /anygrasp/target_pose│
│                 ▼                    │
│  semantic_grasp_controller.py        │
│   [anygrasp env / Python 3.8]        │
│    TF 轉換 → MoveIt → UR3           │
└──────────────────────────────────────┘
```

---

## 4. 環境對應

| 腳本 | conda 環境 | Python | 說明 |
|------|-----------|--------|------|
| `client_camera.py` | `anygrasp` | 3.8 | YOLO + ZMQ + ROS |
| `semantic_grasp_controller.py` | `anygrasp` | 3.8 | MoveIt + TF |
| **`semantic_brain_node.py`** | **`grasp-py310`** | **3.10** | **OWL-v2 + SAM + Gemini** |
| `test_vlm_sam.py` | `grasp-py310` | 3.10 | 獨立測試工具 |
| `server_anygrasp.py` | AI Server | 3.10 | 遠端 GPU 推論 |

> `semantic_brain_node.py` 透過 `sys.path` 手動載入 `/opt/ros/noetic`，不依賴 conda 環境內的 ROS 安裝。

---

## 5. 啟動順序（整合後完整管線）

| # | 終端機 | 指令 | 環境 |
|---|--------|------|------|
| 1 | T1 | `roslaunch realsense2_camera rs_camera.launch align_depth:=true` | system |
| 2 | T2 | `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7` | system |
| 3 | T3 | `roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true` | system |
| 4 | T4 | `roslaunch easy_handeye publish.launch ...` | system |
| 5 | T5 | `python3 server_anygrasp.py --debug`（遠端） | AI Server |
| 6 | **T6** | **`conda activate grasp-py310`**<br>**`rosrun ur3_click2pick semantic_brain_node.py`** | **grasp-py310** |
| 7 | T7 | `conda activate anygrasp`<br>`rosrun ur3_click2pick semantic_grasp_controller.py` | anygrasp |
| 8 | T8 | `conda activate anygrasp`<br>`python3 client_camera.py` | anygrasp |

---

## 6. 操作流程（使用者視角）

### 模式 A：YOLO 自動（現有功能，無變化）
```
啟動後預設進入此模式
按 [s] → 自動送出信心度最高的物件 → 機械臂抓取
```

### 模式 B：手動框選（現有功能，無變化）
```
按 [r] → 框選物件
按 [s] → 送出 → 機械臂抓取
```

### 模式 C：VLM+SAM（新功能）
```
按 [v] → 輸入目標名稱（英文，例如 "ps4 controller"）
        → brain node 處理（OWL-v2 + SAM + Gemini）~10-30秒
        → 畫面出現黃色框 + "VLM: ps4 controller"
按 [s] → 送出高精度 mask → SVD → AnyGrasp → 機械臂抓取
按 [c] → 清除，回到 YOLO 模式
```

---

## 7. 待辦事項

| 項目 | 狀態 | 說明 |
|------|------|------|
| OWL-v2 + SAM + Gemini 本機驗證 | ✅ 完成 | test_vlm_sam.py 測試通過 |
| semantic_brain_node.py 建立 | ✅ 完成 | 單臂 UR3 版本 |
| client_camera.py 整合 [v] 模式 | ✅ 完成 | ROS topic 介面 |
| grasp-py310 環境建立 | ✅ 完成 | Python 3.10 + google-genai |
| **ROS 整合端對端測試** | ⬜ 待做 | 需要同時啟動 brain node + client |
| **實體手臂抓取測試** | ⬜ 待做 | 接上 UR3 跑完整管線 |

---

*v3.0 — VLM+SAM 驗證完成版*
