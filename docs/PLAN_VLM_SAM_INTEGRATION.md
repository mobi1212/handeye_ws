# 系統架構與 VLM+SAM 整合計劃書

**版本**: v5.0
**更新日期**: 2026-04-22
**狀態**: ✅ 已完成

---

## 1. 整體架構

### 1.1 硬體拓撲

```
筆電（Ubuntu 20.04 + ROS Noetic）              遠端 AI Server（GPU）
┌──────────────────────────────────────┐        ┌─────────────────────┐
│  RealSense D435                      │        │  server_anygrasp.py │
│    /camera/color/image_raw           │        │                     │
│    /camera/aligned_depth_to_color    │◄──────►│  AnyGrasp           │
│           │                          │  ZMQ   │  6D 抓取姿態估計     │
│           ▼                          │  TCP   │                     │
│  client_camera.py  [anygrasp/py3.8]  │        └─────────────────────┘
│    模式1: 手動框選 [r]               │
│    模式2: VLM+SAM [v]  ←──────┐    │
│    SVD 桌面擬合                │    │
│           │ /anygrasp/target_pose    │
│           ▼                   │    │
│  semantic_grasp_controller.py │    │
│    TF 轉換 / MoveIt           │    │
│           ↓                   │    │
│  UR3 機械臂（192.168.86.7）   │    │
│                                │    │
│  brain_node.py  [grasp-py310]  │    │
│    OWL-v2 → SAM → SoM → Gemini─────┘
└──────────────────────────────────────┘
```

### 1.2 主要管線：語義抓取

```
使用者輸入 [v] + 物件名稱
           │ /system/trigger_llm {"object_name": "hammer"}
           ▼
  brain_node.py
    [1/5] OWL-v2  → bbox
    [2/5] SAM v1  → global_mask（提前到 Gemini 之前）
    [3/5] 裁切 + 5×5 SoM 網格 + SAM 輪廓疊加（紫色）
    [4/5] Gemini  → target_grids（看得到物件實際邊界）
    [5/5] 覆蓋率驗證（< 20% 的格子自動踢掉）
           │ /system/llm_done + /tmp/semantic_brain/target_mask.png
           ▼
  client_camera.py
    讀取 target_mask.png → SVD 桌面擬合 → ZMQ → AnyGrasp Server
           │ /anygrasp/target_pose (PoseStamped)
           ▼
  semantic_grasp_controller.py
    TF: camera_color_optical_frame → base_link
    姿態對齊: AnyGrasp X 軸 × Y -90° → UR3 Z 軸
    MoveIt: Pre-Grasp → Cartesian 前進 → 夾取 → Z+抬升 → 放置
```

---

## 2. 各腳本現況

### 2.1 brain_node.py（原 semantic_brain_node.py）

| 功能 | 說明 |
|------|------|
| OWL-v2 | zero-shot 偵測，不需訓練 |
| SAM v1 | `facebook/sam-vit-base`，bbox → global mask |
| SoM 網格 | 5×5，疊加 SAM 輪廓線（紫色）供 Gemini 判斷覆蓋率 |
| Gemini | `gemini-2.5-flash-preview-04-17`，雙 API key round-robin |
| 覆蓋率驗證 | Gemini 選完後過濾 < 20% 的格子 |
| 日誌 | `vlm_logs/` 保留最近 10 次，包含 JSON 推理結果 |

**Gemini 約束（6條）：**
1. 高度選擇：Y2/Y3 優先，避開 Y5（靠桌面）
2. 物件覆蓋率：只選物件像素占格子面積大的格子
3. 連續且集中：至少 2 個相鄰格子
4. 穩定性：選較寬/較厚部位，左右對稱
5. 手臂可達性：從全局圖判斷 UR3 能否到達
6. 質心與抓取性平衡：選「最靠近質心的可抓取表面」，避開光滑金屬面

**API Key 設定（`handeye_ws/.env`）：**
```
GOOGLE_API_KEY=第一個key
GOOGLE_API_KEY_2=第二個key（選填）
```

### 2.2 client_camera.py

| 功能 | 說明 |
|------|------|
| 雙面板顯示 | 左：即時相機 + mask 覆蓋，右：靜態抓取姿態 AR |
| VLM 觸發 | `[v]` 輸入物件名稱；Enter 沿用上次輸入 |
| 手動框選 | `[r]` ROI 框選 |
| 重發送 | `[s]` 同一 mask 重新請 AnyGrasp 計算 |
| SVD 桌面擬合 | 使用 SAM mask 做甜甜圈取樣，填補遮蔽深度空洞 |
| ZMQ | REQ-REP，30 秒逾時後自動重建 socket |

### 2.3 semantic_grasp_controller.py

**抓取動作序列：**
```
A. 關節規劃 → Pre-Grasp     ← [Enter]/[r]/[n] 安全確認
B. Cartesian 直線前進 → Grasp
C. 夾緊                      ← [Enter] 確認繼續
D. Cartesian 垂直 Z+ 抬升 5cm
E. 關節規劃 → Pre-Place（保持抓取姿態）
F. Cartesian 前進 → Place
G. 張開夾爪
H. Cartesian 後退 → Pre-Place
   → 詢問是否回待機點
```

**關鍵參數：**

| 參數 | 值 | 說明 |
|------|-----|------|
| `tcp_offset` | 0.18 m | 夾爪指尖到 MoveIt EEF 距離 |
| `grasp_depth` | 0.05 m | 插入物體表面深度 |
| `approach_dist` | 0.05 m | Pre-Grasp 退後距離 |
| `retreat_up_height` | 0.14 m | 待機點高於放置點的高度 |
| `final_xyz` | `[0.2401, 0.1751, 0.185]` | 固定放置座標 |
| `vel_scale` / `acc_scale` | 0.10 | 速度/加速度 10% |

**其他機制：**
- `_normalize_start_state()`：自動 wrap 超出 ±2π 的關節值
- `/semantic_grasp/go_home` topic：隨時回待機點
- MoveIt：`RRTConnect`，5 次嘗試，10 秒規劃時間

### 2.4 TF 架構（已修正 TF loop）

```
base_link → camera_color_optical_frame  （easy_handeye static TF）
camera_color_optical_frame → ...        （RealSense 內部，publish_tf=false 關閉覆蓋）
```

**修正項目：**
- `publish.launch`：移除 `camera_color_optical_frame → camera_link` 橋接節點
- `rs_camera.launch`：`publish_tf` 預設改為 `false`

---

## 3. 環境對應

| 腳本 | conda 環境 | Python | 說明 |
|------|-----------|--------|------|
| `client_camera.py` | `anygrasp` | 3.8 | ZMQ + ROS + SVD |
| `semantic_grasp_controller.py` | `anygrasp` | 3.8 | MoveIt + TF + 夾爪 |
| `brain_node.py` | `grasp-py310` | 3.10 | OWL-v2 + SAM + Gemini |
| `server_anygrasp.py` | AI Server | 3.10 | 遠端 GPU 推論 |

> `brain_node.py` 透過 `sys.path` 手動載入 `/opt/ros/noetic`，不依賴 conda 內的 ROS。

---

## 4. 已完成 / 待優化

| 項目 | 狀態 | 說明 |
|------|------|------|
| OWL-v2 + SAM + Gemini 整合 | ✅ | brain_node.py 運作正常 |
| SAM 提前 + 覆蓋率驗證 | ✅ | 防止 Gemini 選到空格子 |
| TF loop 修正 | ✅ | Cartesian 路徑規劃恢復正常 |
| 雙 API key round-robin | ✅ | 最多三個 key，配額耗盡自動切換 |
| 關節超出界限自動 normalize | ✅ | wrist_3 > 2π 不再卡住 |
| 抓取後垂直抬升 | ✅ | Z+0.05m 取代沿抓取軸後退 |
| 套件改名 ur3_click2pick → ur3_handover | ✅ | 反映最終 human handover 目標 |
| tmux 一鍵啟動腳本 | ✅ | start_grasp.sh / start_calibration.sh |
| 目標區域點雲密度增加 | ✅ | server_anygrasp.py，深度圖插值 2x，見下方說明 |
| AnyGrasp 受 mask 約束 | ⬜ | 目前只傳 bbox，AnyGrasp 可能選到 mask 外 |
| VLM pipeline 移至遠端 server | ⬜ | 計劃書：SERVER_UPGRADE_PLAN.md，待桌機端實作 |
| 格子更細 | ⬜ | 目前 5×5，可視表現再評估是否改 7×7 |

---

## 5. 點雲密度增加說明（2026-04-22）

### 背景
AnyGrasp 對桌面物體有時生成過於傾斜的抓取姿態（45-60°）。
討論過的方向（接近角過濾、CoM 加權、修改幾何）均有侷限：
- 接近角過濾依賴桌面法向量，水杯等需側面抓取的物體會誤殺
- 修改輸入幾何會讓 AnyGrasp 基於假幾何計算夾爪深度和寬度，不可靠
- 增加點雲密度不直接減少傾斜，但增加目標區域候選數量，提升選到好姿態的機率

### 做法（深度圖插值法）
在 `server_anygrasp.py` 點雲建立後、AnyGrasp 推論前：
1. 取 bbox 範圍內的深度圖和 RGB
2. 用 `INTER_LINEAR` 插值放大 `UPSAMPLE` 倍（預設 2x）
3. 用 sub-pixel 座標反投影到 3D（座標：`bx1 + u'/UPSAMPLE`）
4. 新點疊加在原有點雲上，原點保留

### 抓取姿態傾斜問題的根本結論
- 修改 AnyGrasp 輸入讓它「考慮重量分佈」不實際，需重新訓練模型才能真正理解物理資訊
- CoM 加權（用 Gemini 的 `estimated_com_grid` 反投影 3D）可作為後處理評分，未來可加

---

## 6. 架構演進方向

```
現況（pick-and-place）
  桌面物體 → AnyGrasp 抓取 → 固定座標放置

近期目標
  server_anygrasp.py 整合 VLM pipeline（見 SERVER_UPGRADE_PLAN.md）

長期目標（human handover）
  偵測人手位置 → 動態計算交接點 → 交接姿態調整 → 感知人手接取後放開
```

---

*v5.0 — 2026-04-22，點雲密度增加 + 架構演進記錄*
