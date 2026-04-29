# VLM 形狀感知抓取品質提升計畫書

**版本**: v1.0
**日期**: 2026-04-24

---

## 1. 目標

利用 VLM（Gemini）辨識物體形狀，並根據不同幾何特性調整抓取策略：
- **球體**：朝球心方向抓取
- **盒子/平面物體**：沿表面法向量方向抓取
- **圓柱體**：與主軸垂直方向抓取
- **不規則物體**：提供 top-N 候選姿態給 VLM 評分選擇

---

## 2. 三種方案比較

| 方案 | 概念 | 優點 | 缺點 |
|------|------|------|------|
| **A. VLM 前置幾何約束** | VLM 辨識形狀 → 幾何規則過濾 AnyGrasp 輸出 | 低延遲、不用再呼叫 API | 依賴硬編碼規則，彈性差 |
| **B. VLM 後置評分** | AnyGrasp top-N 姿態 → 投影回影像 → VLM 選最佳 | 最靈活、VLM 看得到實際抓取姿態 | 多一次 API 呼叫，延遲 +3~5 秒 |
| **C. 混合方案** | 先用形狀規則粗篩 → 剩餘候選送 VLM 精選 | 兼顧速度和品質 | 實作複雜度最高 |

### 建議路線：Phase 1（方案 A）→ Phase 2（方案 B）

先做簡單有效的形狀幾何規則，驗證有改善後再加 VLM 後置評分。

---

## 3. Phase 1：形狀感知幾何過濾（方案 A）

### 3.1 修改 Gemini Prompt（brain_node.py）

在現有 prompt 的輸出格式中加入 `object_shape` 欄位（目前已存在但未在 prompt 中明確要求）：

```json
{
    "object_name": "物件英文名稱",
    "object_shape": "sphere | cylinder | box | irregular",
    "estimated_com_grid": "估計質心所在格子代號",
    "target_grids": ["網格代號", ...],
    "reasoning": "..."
}
```

新增 prompt 約束：

```
約束七：物體形狀辨識
請根據物件外觀判斷其幾何形狀類別，用於下游抓取策略：
- "sphere"：球形或近似球形（球、橘子、蘋果、圓球玩具）
- "cylinder"：圓柱形，有明確主軸（水瓶、杯子、罐子、筆）
- "box"：盒狀或平板狀，有明確平面（書本、積木、盒子、手機）
- "irregular"：不規則形狀（工具、玩偶、複雜零件）
若不確定，選 "irregular"。
```

### 3.2 傳遞形狀資訊（brain_node.py → client_camera.py → server_anygrasp.py）

**目前流程**：brain_node 已在 `/system/llm_done` 中發布 `object_shape`（第 491 行），但 client 沒有使用它。

**修改 client_camera.py**：
1. 訂閱 `/system/llm_done` 時解析 `object_shape`
2. 將 `object_shape` 加入 ZMQ payload 送到 server

**修改 server_anygrasp.py**：
1. 從 payload 讀取 `object_shape`
2. AnyGrasp 推論後，根據形狀套用不同的姿態篩選策略

### 3.3 形狀感知姿態篩選邏輯（server_anygrasp.py）

在 AnyGrasp 輸出 GraspGroup 後，根據 `object_shape` 對候選姿態進行評分/過濾：

#### 球體（sphere）
```python
# 球心 = 目標點雲的幾何中心
center = target_points.mean(axis=0)
for grasp in grasp_group:
    # 計算抓取接近方向與「指向球心」方向的對齊度
    approach = grasp.rotation_matrix[:, 0]  # AnyGrasp X 軸 = 接近方向
    to_center = center - grasp.translation
    to_center_norm = to_center / np.linalg.norm(to_center)
    alignment = np.dot(approach, to_center_norm)
    grasp.score *= (0.5 + 0.5 * alignment)  # 對齊球心的加分
```

#### 圓柱體（cylinder）
```python
# 用 PCA 估計主軸
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(target_points)
main_axis = pca.components_[0]  # 第一主成分 = 主軸方向

for grasp in grasp_group:
    approach = grasp.rotation_matrix[:, 0]
    # 接近方向應與主軸垂直（點積越小越好）
    perpendicularity = 1.0 - abs(np.dot(approach, main_axis))
    grasp.score *= (0.5 + 0.5 * perpendicularity)
```

#### 盒子（box）
```python
# 用 PCA 估計表面法向量（第三主成分 = 最薄方向 ≈ 法向量）
pca = PCA(n_components=3)
pca.fit(target_points)
normal = pca.components_[2]  # 最小特徵值方向

for grasp in grasp_group:
    approach = grasp.rotation_matrix[:, 0]
    # 接近方向應與法向量對齊
    alignment = abs(np.dot(approach, normal))
    grasp.score *= (0.5 + 0.5 * alignment)
```

#### 不規則（irregular）
不做額外過濾，使用 AnyGrasp 原始評分。

### 3.4 重新排序與選取

```python
# 依修正後的分數排序
grasp_group = grasp_group.sort_by_score()
# 取 top-1（或後續 Phase 2 取 top-N）
best_grasp = grasp_group[0]
```

---

## 4. Phase 2：VLM 後置評分（方案 B）

### 4.1 概念

AnyGrasp 輸出 top-5 候選姿態 → 投影到 RGB 影像上畫箭頭標號 → 送回 Gemini 選擇最佳。

### 4.2 投影與視覺化（server_anygrasp.py 新增函式）

```python
def project_grasps_to_image(grasps, color_img, fx, fy, cx, cy, top_n=5):
    """將 top-N 抓取姿態投影到 RGB 影像，繪製接近方向箭頭 + 編號"""
    vis_img = color_img.copy()
    for i, g in enumerate(grasps[:top_n]):
        # 抓取位置投影到像素
        x3d, y3d, z3d = g.translation
        u = int(fx * x3d / z3d + cx)
        v = int(fy * y3d / z3d + cy)

        # 接近方向箭頭（30 像素長）
        approach = g.rotation_matrix[:, 0]
        tip_3d = g.translation + approach * 0.03
        u2 = int(fx * tip_3d[0] / tip_3d[2] + cx)
        v2 = int(fy * tip_3d[1] / tip_3d[2] + cy)

        color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255)][i]
        cv2.arrowedLine(vis_img, (u, v), (u2, v2), color, 2, tipLength=0.3)
        cv2.putText(vis_img, f"#{i+1}", (u+5, v-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis_img
```

### 4.3 Gemini 選擇 Prompt

```
你是機器人抓取專家。圖片中標示了 5 個候選抓取姿態（#1~#5），箭頭方向為夾爪接近方向。
物體形狀為：{object_shape}

請根據以下原則選擇最佳抓取：
1. 接觸點穩定：夾爪能穩固接觸物體表面
2. 接近方向合理：不會碰撞桌面或其他物體
3. 形狀適配：
   - 球體：從側面接近，避免從正上方
   - 圓柱：垂直於主軸方向接近
   - 盒子：沿最大平面的法向量接近
   - 不規則：選夾持面最寬的位置

輸出格式（純 JSON）：
{
    "selected": 1,
    "reasoning": "..."
}
```

### 4.4 通訊方案

兩個選項：

**選項 A：Server 端直接呼叫 Gemini**
- Server 已有影像和姿態，直接投影 + 呼叫 API
- 需要在 server 端設定 API key
- 優點：不用額外 ZMQ 來回
- 缺點：server 端需要 google-genai 依賴

**選項 B：姿態回傳 Client，Client 呼叫 Gemini**
- Server 回傳 top-5 姿態（含分數）→ Client 投影 → Client 呼叫 brain_node
- 優點：API key 集中管理
- 缺點：多一次 ZMQ 來回 + ROS topic 通訊

**建議**：選項 A 較簡單，Server 端直接整合。

---

## 5. 涉及檔案與修改範圍

| 檔案 | Phase 1 修改 | Phase 2 修改 |
|------|-------------|-------------|
| `brain_node.py` | 約束七 + prompt 要求 `object_shape` | 無 |
| `client_camera.py` | 解析 `object_shape`，加入 ZMQ payload | 接收 top-N 資訊（選項 B 才需要）|
| `server_anygrasp.py` | 形狀感知篩選邏輯（PCA + 評分） | 投影視覺化 + Gemini 呼叫（選項 A）|
| `semantic_grasp_controller.py` | 無（姿態已在 server 端篩選完） | 無 |

---

## 6. 依賴

| 依賴 | Phase | 說明 |
|------|-------|------|
| `scikit-learn` (PCA) | Phase 1 | Server 端，用於估計圓柱主軸/盒子法向量 |
| `google-genai` | Phase 2 | Server 端才需要（選項 A）|

---

## 7. 風險與緩解

| 風險 | 影響 | 緩解方案 |
|------|------|----------|
| VLM 形狀辨識錯誤 | 套用錯誤幾何規則 | `irregular` 作為 fallback，不做額外過濾 |
| PCA 主軸估計不穩定 | 圓柱/盒子篩選方向錯誤 | 點雲數量 < 100 時跳過形狀篩選 |
| Phase 2 API 延遲 | 整體流程 +3~5 秒 | 限制 top-5，使用 flash-lite 加速 |
| 形狀篩選後候選為 0 | 沒有可用抓取姿態 | 分數乘法而非硬過濾，保證至少有一個候選 |

---

## 8. 實驗設計

### 8.1 對比組

| 組別 | 說明 |
|------|------|
| Baseline | AnyGrasp 原始 top-1，無形狀篩選 |
| Phase 1 | 形狀幾何規則篩選後 top-1 |
| Phase 2 | 形狀篩選 + VLM 後置選擇 |

### 8.2 測試物件（每種形狀至少 2 個）

| 形狀 | 物件 | 測試重點 |
|------|------|----------|
| 球體 | 網球、橘子 | 是否朝球心方向抓取 |
| 圓柱 | 水瓶、馬克杯 | 是否垂直主軸抓取 |
| 盒子 | 紙盒、書本 | 是否沿法向量接近 |
| 不規則 | 鎚子、螺絲起子 | 是否選到握柄而非金屬部位 |

### 8.3 評估指標

- **抓取成功率**（%）：每物件 10 次，成功抬離桌面 = 成功
- **接近方向合理性**（人工標註）：1~5 分
- **API 回應延遲**（秒）

---

## 9. 實施順序

```
Phase 1（預估 2~3 小時）
  Step 1: brain_node.py — 新增約束七，prompt 明確要求 object_shape
  Step 2: client_camera.py — 解析 llm_done 中的 object_shape，加入 ZMQ payload
  Step 3: server_anygrasp.py — 新增 shape_aware_rerank() 函式
  Step 4: 測試 4 種形狀各 3 次

Phase 2（預估 6~8 小時）
  Step 5: server_anygrasp.py — 新增 project_grasps_to_image()
  Step 6: server_anygrasp.py — 整合 Gemini API 呼叫
  Step 7: 設計 grasp selection prompt
  Step 8: 對比實驗（Baseline vs Phase 1 vs Phase 2）
```

---

*v1.0 — 2026-04-24*
