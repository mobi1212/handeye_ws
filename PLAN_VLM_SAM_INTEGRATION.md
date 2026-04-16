# 執行計劃書：VLM + SAM 整合至語義抓取管線（實體 UR3 部署版）

**文件版本**: v2.0  
**建立日期**: 2026-04-15  
**更新日期**: 2026-04-16  
**狀態**: 根據模擬環境實測結果更新

---

## 1. 背景與動機

### 現況痛點

目前 Pipeline A（語義抓取）的視覺辨識層仰賴 **YOLO**，有以下限制：

| 問題 | 影響 |
|------|------|
| YOLO 只能辨識訓練過的固定類別（`best.pt`） | 遇到新物體就失效，需重新訓練 |
| YOLO 的 instance mask 解析度低（壓縮過） | SVD 桌面擬合的邊界點噪點多，深度填補不精確 |
| 無法用自然語言指定目標（「幫我拿紅色的杯子」） | 使用者只能靠信心值最高的物體，無法主動選擇 |

### v1.0 計劃的技術選型誤差（已修正）

v1.0 計劃書假設使用 Grounding DINO + SAM2 + Gemini 做 bbox 偵測。  
模擬環境實測（`brain.py`）驗證後，**實際架構如下**：

| 元件 | v1.0 假設 | v2.0 實際（已驗證） |
|------|-----------|---------------------|
| 物件偵測 | Grounding DINO | **OWL-v2**（`google/owlv2-base-patch16-ensemble`，HuggingFace） |
| 分割 | SAM2 | **SAM v1**（`facebook/sam-vit-base`，HuggingFace） |
| Gemini 角色 | 直接偵測 bbox | **分析 SoM 網格圖，決定最佳抓取區域** |
| 架構 | 插入 client_camera.py | **獨立 ROS 節點** (`semantic_brain_node`) |
| 雙臂/單臂 | 不涉及 | 模擬是雙臂，**部署需改為 UR3 單臂** |

---

## 2. 技術選型（已驗證版本）

### 2.1 物件偵測：OWL-v2（Zero-Shot）

| 項目 | 說明 |
|------|------|
| 模型 | `google/owlv2-base-patch16-ensemble` |
| 輸入 | PIL Image + 候選物件名稱清單（文字） |
| 輸出 | bbox `[xmin, ymin, xmax, ymax]` + 信心分數 |
| 依賴套件 | `transformers 4.44.2`（已在模擬環境確認） |
| 運算位置 | 本機（CPU 或 GPU） |
| 為何選它 | 無需訓練即可辨識任意新物體；模型已在模擬環境跑通 |

### 2.2 分割：SAM v1（Segment Anything）

| 項目 | 說明 |
|------|------|
| 模型 | `facebook/sam-vit-base` |
| 輸入 | PIL Image + bbox prompt `[[[xmin, ymin, xmax, ymax]]]` |
| 輸出 | Binary mask（shape = H×W，bool） |
| 依賴套件 | `transformers 4.44.2`（SamModel + SamProcessor） |
| 運算位置 | 本機（CUDA 優先，CPU 備援） |

> **注意**：模擬環境的 `conda list` 顯示 `sam-2 1.0` 也已安裝，但 `brain.py` 最終採用 SAM v1（透過 transformers），因其 API 更簡單且已驗證穩定。

### 2.3 VLM 推理：Google Gemini API

| 項目 | 說明 |
|------|------|
| 模型 | `gemini-flash-latest`（`google-generativeai 0.8.6` 已確認） |
| 輸入 | 全景圖（PIL）+ SoM 網格圖（PNG）+ system prompt |
| 輸出 | JSON：`{object_name, left_grids, right_grids, reasoning}` |
| 角色 | **不做 bbox 偵測**，只分析 SoM 網格決定抓取的格子位置 |
| API Key | **環境變數讀取，嚴禁寫死在程式碼中**（見 Section 4） |

### 2.4 SoM（Set-of-Mark）網格

`brain.py` 中已實作 `draw_som_grid()`：
- 將 OWL-v2 偵測到的物件裁切後，疊加 5×5 網格（X 軸 A-E，Y 軸 1-5）
- 棋盤式半透明底色 + 白字黑邊格子代號
- 傳給 Gemini 分析，由 Gemini 決定最佳抓取格子（例如 `["B2", "C2"]`）

---

## 3. 整合架構設計

### 3.1 新舊流程對比

```
【現有流程】
使用者按 [s]
    └─> YOLO 偵測 → bbox + 低解析度 mask
        └─> SVD 桌面擬合 → AnyGrasp Server → 6D pose → MoveIt

【新流程（v2.0）】
使用者輸入目標文字（例如：「bottle」）→ ROS topic 觸發
    └─> OWL-v2 偵測 → bbox（zero-shot，無需訓練）
        └─> SAM v1 → 高解析度 mask
            └─> 裁切 + SoM 網格繪製
                └─> Gemini → 分析網格 → 最佳抓取區域格子代號
                    └─> 遮罩合成（只保留目標格子的 SAM mask）
                        └─> mask 存檔 / ROS topic 通知
                            └─> client_camera.py 讀取 mask
                                └─> SVD 桌面擬合 → AnyGrasp Server → 6D pose → MoveIt
```

### 3.2 ROS 節點架構

```
                        ┌─────────────────────────────┐
                        │   semantic_brain_node.py     │
/camera/color/image_raw │   (改自 brain.py)            │
──────────────────────► │                              │
                        │  OWL-v2 → bbox               │
/system/trigger_llm     │  SAM v1 → global mask        │
──────────────────────► │  draw_som_grid()             │
  {"object_name": ...,  │  Gemini → grid selection     │
   "mode": "single"}    │  save left_mask.png          │
                        │                              │
                        └─────────────┬───────────────┘
                                      │ /system/llm_done
                                      │ {"status":"done", ...}
                                      ▼
                        ┌─────────────────────────────┐
                        │     client_camera.py         │
                        │  收到 done 後讀取 mask 檔    │
                        │  SVD → AnyGrasp Server       │
                        └─────────────────────────────┘
```

### 3.3 雙臂 → 單臂改動（重要）

`brain.py` 原始設計為雙臂空中交接任務（左手 + 右手格子）。  
UR3 是**單臂**，需要以下調整：

| 差異點 | brain.py（雙臂模擬） | UR3 部署目標 |
|--------|---------------------|-------------|
| Gemini prompt | `VISION_SYSTEM_PROMPT`（左右分工） | 改用 `LEFT_ONLY_PROMPT` 邏輯，只選最佳單一抓取區 |
| `mode` 參數 | `"dual"` / `"left_only"` | 固定使用 `"single"`，只輸出 `target_grids` |
| 輸出 JSON | `left_grids`, `right_grids` | 只用 `target_grids`（取代 `left_grids`） |
| mask 檔案 | `left_mask.png` + `right_mask.png` | 只需 `target_mask.png` |
| save_dir | `/home/rvl/ros_ws/...` | 改為 `/home/weilun/handeye_ws/` 下的路徑 |

---

## 4. 實作任務清單

### Task 0：確認環境版本相容性（估計 0.5 天）

模擬環境（`message.txt`）已確認的關鍵套件版本：

```
Python          3.10.20
torch           2.4.1+cu124
torchvision     0.19.1+cu124
transformers    4.44.2          ← OWL-v2 + SAM v1
google-generativeai  0.8.6      ← Gemini API
sam-2           1.0             ← 已安裝（備用）
lang-sam        0.2.1           ← 備用（含 GroundingDINO+SAM）
supervision     0.27.0.post2
numpy           2.2.6
opencv-python   4.13.0.92
```

實體部署機器（AI Server 端，目前跑 `server_anygrasp.py`）需確認上述套件版本一致，尤其是 `transformers >= 4.44`（OWL-v2 需要）。

**驗收標準**：

```bash
python3 -c "
from transformers import pipeline, SamModel, SamProcessor
import google.generativeai as genai
print('OWL-v2 + SAM v1 + Gemini SDK 均可 import')
"
```

---

### Task 1：建立 `semantic_brain_node.py`（改自 brain.py）（估計 1 天）

**檔案位置**：`src/ur3_click2pick/scripts/semantic_brain_node.py`

#### 1a. API Key 安全性修正（必做）

`brain.py` 第 35 行有 **hardcode API key**，部署時必須移除：

```python
# ❌ brain.py 現有（禁止）：
MY_GEMINI_KEY = "AIzaSy..."
genai.configure(api_key=MY_GEMINI_KEY)

# ✅ 改為環境變數讀取：
import os
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("請設定環境變數 GOOGLE_API_KEY（不要寫進程式碼）")
genai.configure(api_key=api_key)
```

設定方式：

```bash
# 方式 A（推薦）：寫入 ~/.bashrc
echo 'export GOOGLE_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

# 方式 B：.env 檔（已在 .gitignore）
echo 'GOOGLE_API_KEY=your_key_here' > /home/weilun/handeye_ws/.env
```

#### 1b. 單臂 Gemini Prompt 設計

移除雙臂邏輯，針對 UR3 單臂寫新的 prompt：

```python
UR3_SINGLE_ARM_PROMPT = """
你是一個機器人視覺分析專家，協助單臂 UR3 機械臂抓取物件。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示 UR3 機械臂基座位置、桌面高度，以及目標物件的擺放狀態。
【圖片 2】物件特寫網格圖：目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。

【任務背景】
UR3 機械臂從上方或側面接近桌面物件進行抓取，夾爪為平行夾爪。

【約束條件】
1. 高度選擇：Y1 是頂部，Y5 靠近桌面（手臂難以到達）。
   優先選擇 Y2-Y3，避免 Y5（碰桌風險）。
2. 夾取面積：選擇 2-4 個相鄰格子，形成有效的夾取面。
3. 穩定性：選擇物件較寬/較厚的部位，避免邊角。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "target_grids": ["網格代號", ...],
    "reasoning": "說明選擇理由"
}
"""
```

#### 1c. 單臂流程改動（對照 brain.py）

| brain.py 原始 | semantic_brain_node.py 改動 |
|--------------|----------------------------|
| `mode = "dual"` / `"left_only"` | `mode = "single"`（固定） |
| `left_grids`, `right_grids` | `target_grids` |
| `save_final_mask(left_grids, "left_mask.png")` | `save_final_mask(target_grids, "target_mask.png")` |
| `save_dir = "/home/rvl/ros_ws/..."` | `save_dir = "/tmp/semantic_brain/"` |
| `/system/trigger_llm` payload `mode` | 維持相同 topic，只去掉雙臂 mode |

#### 1d. ROS 介面規格

```
訂閱：
  /camera/color/image_raw     (sensor_msgs/Image)    ← 持續接收最新幀
  /system/trigger_llm         (std_msgs/String)       ← JSON: {"object_name": "bottle"}

發布：
  /system/llm_done            (std_msgs/String)       ← JSON: {"status": "done"/"fail", ...}

輸出檔案：
  /tmp/semantic_brain/original_rgb.png
  /tmp/semantic_brain/object_crop_raw.png
  /tmp/semantic_brain/cropped_grid_for_vlm.png
  /tmp/semantic_brain/sam_global_mask_full.png
  /tmp/semantic_brain/target_mask.png               ← client_camera.py 讀取此檔
```

---

### Task 2：修改 `client_camera.py`（估計 1 天）

**整合方式**：新增模式 `[v]`，觸發後發送 trigger 給 brain node，等待 done 訊號，再讀取 `target_mask.png` 走後續流程。

#### 2a. 新增 ROS Publisher/Subscriber（在 `__init__`）

```python
# 在 __init__ 中加入：
from std_msgs.msg import String
self.brain_trigger_pub = rospy.Publisher("/system/trigger_llm", String, queue_size=1)
rospy.Subscriber("/system/llm_done", String, self._brain_done_callback)
self.brain_result = None      # 等待 brain node 回傳
self.vlm_target = None        # 使用者輸入的目標物名稱
self.vlm_mask_path = "/tmp/semantic_brain/target_mask.png"
```

```python
def _brain_done_callback(self, msg):
    import json
    self.brain_result = json.loads(msg.data)
```

#### 2b. 新增鍵盤控制 `[v]`

```python
# 在 run() 的按鍵處理區塊新增：
if key & 0xFF == ord('v'):
    target = input("\n🔍 請輸入目標物件（英文，例如：bottle）：").strip()
    if target:
        self.vlm_target = target
        self.brain_result = None
        import json
        payload = json.dumps({"object_name": target})
        self.brain_trigger_pub.publish(payload)
        print(f"⚡ 已發送 trigger，等待 brain node 處理 '{target}'...")
```

#### 2c. 在模式優先權邏輯插入 VLM mask 模式

```python
# --- 優先權 2：VLM+SAM 模式（新增）---
elif self.vlm_target and self.brain_result and self.brain_result.get("status") == "done":
    import cv2 as _cv2
    mask_img = _cv2.imread(self.vlm_mask_path, _cv2.IMREAD_GRAYSCALE)
    if mask_img is not None:
        best_mask = (mask_img > 127).astype(np.float32)
        # 從 mask 計算 bbox
        ys, xs = np.where(mask_img > 127)
        if len(xs) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            best_bbox = [x1, y1, x2, y2]
            cls_name = self.vlm_target
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(display_img, f"VLM: {cls_name}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            best_bbox, best_mask, cls_name = None, None, "None"
    else:
        best_bbox, best_mask, cls_name = None, None, "None"

# --- 優先權 3：YOLO 自動偵測（原有，不變）---
else:
    ...
```

#### 2d. 按 `[c]` 清除也清除 VLM 狀態

```python
if key & 0xFF == ord('c'):
    self.manual_bbox = None
    self.vlm_target = None
    self.brain_result = None
    print("🔄 已切換回 YOLO 自動偵測模式。")
```

---

### Task 3：SAM mask 與 SVD 相容性確認（估計 0.5 天）

SAM v1 輸出的 mask 為 `np.ndarray(bool)`，需與現有 float mask 邏輯相容。  
Task 2c 中已將 mask 讀取為 float32（`astype(np.float32)`），直接相容現有 SVD 邏輯。

**驗收標準**：以 SAM mask 跑 SVD 時，`donut_mask` 點數 > 10，填補深度數值與物體周圍桌面相近，無明顯跳變。

---

### Task 4：端對端整合測試（估計 1 天）

#### 啟動順序（在現有管線基礎上新增步驟 6.5）

| # | 功能 | 指令 |
|---|------|------|
| 1 | 相機 | `roslaunch realsense2_camera rs_camera.launch align_depth:=true` |
| 2 | 手臂 | `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7` |
| 3 | MoveIt | `roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true` |
| 4 | TF 外參 | *(同 CLAUDE.md)* |
| 5 | AnyGrasp Server | `python3 server_anygrasp.py --debug` |
| **6** | **Brain Node（新增）** | **`rosrun ur3_click2pick semantic_brain_node.py`** |
| 7 | 機械臂控制器 | `rosrun ur3_click2pick semantic_grasp_controller.py` |
| 8 | 視覺前端 | `python3 client_camera.py` |

---

## 5. 套件依賴彙整（實體部署環境）

以下套件需在 **AI Server 端**（跑 brain node 的機器）確認安裝：

| 套件 | 版本（模擬驗證） | 用途 |
|------|----------------|------|
| `transformers` | `4.44.2` | OWL-v2 物件偵測 + SAM v1 分割 |
| `google-generativeai` | `0.8.6` | Gemini API（SoM 網格推理） |
| `torch` | `2.4.1+cu124` | 深度學習推論 |
| `torchvision` | `0.19.1+cu124` | 影像預處理 |
| `opencv-python` | `4.13.0.92` | 影像處理 |
| `Pillow` | `12.1.1` | PIL Image 轉換 |
| `numpy` | `2.2.6` | 陣列運算 |

**不需下載額外 checkpoint**：OWL-v2 與 SAM v1 模型均透過 HuggingFace 自動下載。

---

## 6. 資料流圖（v2.0）

```
使用者按 [v] 輸入目標文字（例如 "bottle"）
    │
    ▼
client_camera.py
    └─ 發布 /system/trigger_llm
               │
               ▼
    semantic_brain_node.py
        ├─ OWL-v2：zero-shot 偵測 "bottle" → bbox
        ├─ SAM v1：bbox → 高解析度 global mask
        ├─ draw_som_grid()：裁切 + SoM 5×5 網格
        ├─ Gemini：全景圖 + 網格圖 → target_grids（JSON）
        └─ 合成 target_mask.png（grid 範圍內的 SAM mask）
               │
               ▼ /system/llm_done {"status":"done"}
    client_camera.py
        └─ 讀取 /tmp/semantic_brain/target_mask.png
               │
               ▼（與 YOLO/手動模式匯流）
    SVD 桌面擬合（用 SAM mask，解析度更高）
        └─> clean_depth（填補深度空洞）
               │
               ▼
    ZMQ → AnyGrasp Server（GPU 端）
        └─> 6D 抓取姿態
               │
               ▼
    /anygrasp/target_pose (PoseStamped)
               │
               ▼
    semantic_grasp_controller.py（不需修改）
        └─> MoveIt 執行抓取
```

---

## 7. 驗收標準

| # | 測試情境 | 預期行為 |
|---|----------|----------|
| 1 | 按 `[v]` 輸入 `"bottle"`，桌上有瓶子 | OWL-v2 偵測到 bbox，SAM 產生精確 mask，Gemini 選定網格，AnyGrasp 回傳 6D pose，機械臂成功抓取 |
| 2 | 輸入 YOLO 訓練集中不存在的物件名稱 | OWL-v2 仍能偵測，確認 zero-shot 能力 |
| 3 | 輸入桌上不存在的物件 | `/system/llm_done` 回傳 `{"status":"fail","reason":"object_not_found"}`，機械臂不動 |
| 4 | 按 `[c]` 清除後 | 自動回到 YOLO 模式，原有功能正常 |
| 5 | SAM mask 品質 | SVD 填補後深度值與周邊桌面連續（無跳變），`donut_mask` 點數 > 10 |

---

## 8. 已知風險與對策

| 風險 | 可能性 | 對策 |
|------|--------|------|
| brain.py API key 硬編碼意外 commit | **高（已存在）** | **Task 1a 必做**：改為環境變數；`git diff --staged` 確認後再 PR |
| OWL-v2 / SAM v1 首次執行需從 HuggingFace 下載模型 | 中 | 部署前先在有網路環境 pre-download；或設定 `TRANSFORMERS_OFFLINE=1` 後本機 cache |
| Gemini API 在機器人現場無網路 | 中 | VLM 模式降級提示，保留 YOLO 與手動 ROI 作為備援 |
| Gemini 回傳 JSON 格式不符預期 | 中 | 加入 robust JSON parsing（`response.text.replace("```json","")...`，brain.py 已實作） |
| brain.py `save_dir` 路徑硬編碼為模擬機器路徑 | **高（已存在）** | Task 1c 必改為 `/tmp/semantic_brain/` 並加 `os.makedirs` |
| OWL-v2 推論速度（CPU 模式）| 中 | 配合現有 `[s]` 單次觸發設計，不做連續偵測，可接受 3-5 秒延遲 |
| 雙臂 Gemini prompt 誤導單臂決策 | **高（若直接用 brain.py）** | Task 1b 必須替換為 `UR3_SINGLE_ARM_PROMPT` |

---

## 9. 與現有系統的邊界

| 元件 | 是否需修改 |
|------|-----------|
| `semantic_grasp_controller.py` | **不需修改**（完全獨立，從 `/anygrasp/target_pose` 接收） |
| `server_anygrasp.py` | **不需修改**（ZMQ 介面不變） |
| `client_camera.py` | **需修改**（Task 2，新增模式 `[v]`） |
| `semantic_brain_node.py` | **新增**（改自 brain.py，Task 1） |

---

*文件結束 — v2.0 基於模擬環境 brain.py 實測架構更新*
