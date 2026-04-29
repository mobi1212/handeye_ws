# 抓取姿態傾斜問題解決方案

**版本**: v1.0  
**日期**: 2026-04-24  
**目標**: 解決 AnyGrasp 生成過度傾斜抓取姿態的問題，採用「VLM 單次推論 + 幾何擬合 + 向量內積過濾」架構

---

## 一、核心思路

```
Gemini（已有）輸出 object_shape 標籤
    ↓
server_anygrasp.py 對目標點雲做幾何擬合
    ↓
用向量內積對 AnyGrasp 候選做後處理過濾
    ↓
選出符合物理語義的抓取姿態
```

**不新增 VLM 呼叫**，只在現有 Gemini 輸出中加一個欄位，後續全部純數學運算。

---

## 二、各物體的幾何策略

| 形狀類別 | 代表物體 | 幾何擬合 | 過濾條件 |
|---------|---------|---------|---------|
| `sphere` | 球、圓形水果 | RANSAC 擬合球心 | 夾爪接近向量需指向球心，內積 > 0.9 |
| `box` | 盒子、書本、積木 | SVD 計算表面法向量 | 夾爪接近向量需平行法向量，內積絕對值 > 0.85 |
| `cylinder` | 水杯、瓶子、把手、鎚柄 | 擬合圓柱主軸向量 | 夾爪閉合方向需垂直主軸，內積絕對值 < 0.3 |

> **AnyGrasp 的向量對應**（rotation matrix R）：
> - 接近方向（approach）= `R[:, 0]`（X 軸，夾爪插入方向）
> - 閉合方向（closing）= `R[:, 1]`（Y 軸，手指開合方向）

---

## 三、需要改動的檔案

### 3.1 brain_node.py — 加 `object_shape` 輸出

**改動**：Gemini prompt 輸出格式加一個欄位

```python
# 舊輸出格式（第 98-104 行）：
{
    "object_name": "...",
    "estimated_com_grid": "...",
    "target_grids": [...],
    "reasoning": "..."
}

# 新輸出格式：
{
    "object_name": "...",
    "estimated_com_grid": "...",
    "object_shape": "sphere | box | cylinder",   # ← 新增
    "target_grids": [...],
    "reasoning": "..."
}
```

**prompt 約束新增說明**（加在約束六之後）：

```
約束七：幾何形狀分類
根據目標物件的外觀，從以下三類中選擇最接近的形狀標籤：
- "sphere"：球體或主要呈圓弧面（球、蘋果、橘子）
- "box"：平面為主的長方體或板狀物（盒子、書、積木、鎚頭）
- "cylinder"：圓柱體或條狀把手（水杯、瓶子、鎚柄、螺絲起子握柄）
若物件有多個部位（如鎚子），以「目標抓取部位」的形狀為準，而非整個物件。
```

**在 `process()` 中讀取並傳遞**（第 402 行附近）：
```python
object_shape = vlm_result.get('object_shape', 'box')  # 預設 box
```

**在 `/system/llm_done` 的 publish 加入**（第 470-475 行）：
```python
self.done_pub.publish(json.dumps({
    "status": "done",
    "object_name": object_name,
    "object_shape": object_shape,    # ← 新增
    "target_grids": target_grids,
    "reasoning": vlm_result.get('reasoning', '')
}))
```

---

### 3.2 client_camera.py — 讀取 shape 並加入 ZMQ payload

**`_brain_done_callback`（第 89 行）**：
```python
# 已有 self.brain_result = json.loads(msg.data)
# 新增：
self.object_shape = self.brain_result.get("object_shape", "box")
```

**`process_anygrasp()`（第 356 行）**：
```python
payload = {
    'color_jpg': encoded,
    'depth': depth,
    'bbox': bbox,
    'object_shape': getattr(self, 'object_shape', 'box'),   # ← 新增
}
```

**`[c]` 清除邏輯**：
```python
self.object_shape = 'box'   # 重置為預設值
```

---

### 3.3 server_anygrasp.py — 幾何擬合 + 向量內積過濾（核心）

在現有架構中，插入位置：`gg.nms().sort_by_score()` 之後，`best_grasp = gg[0]` 之前。

#### 完整後處理函式

```python
def filter_grasps_by_geometry(gg, points, colors_mask, bbox, object_shape,
                               fx=FX, fy=FY, cx=CX, cy=CY):
    """
    根據物件形狀標籤，對 AnyGrasp 候選做幾何過濾。
    回傳過濾後的 GraspGroup，若全部被過濾則回傳原始 gg。
    """
    import open3d as o3d

    # 1. 取出目標區域的點雲
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        trans = gg.translations
        u = (fx * points[:, 0] / points[:, 2] + cx).astype(int)
        v = (fy * points[:, 1] / points[:, 2] + cy).astype(int)
        in_bbox = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        target_pts = points[in_bbox]
    else:
        target_pts = points

    if len(target_pts) < 10:
        print("   ⚠️ 目標點雲點數不足，跳過幾何過濾")
        return gg

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(target_pts)

    rotations = np.array([g.rotation_matrix for g in gg])   # (N, 3, 3)
    approach  = rotations[:, :, 0]   # X 軸 = 接近方向
    closing   = rotations[:, :, 1]   # Y 軸 = 閉合方向

    valid_mask = np.ones(len(gg), dtype=bool)

    # ── Sphere ────────────────────────────────────────────────────────────
    if object_shape == 'sphere':
        center = np.mean(target_pts, axis=0)
        translations = np.array([g.translation for g in gg])
        to_center = center - translations                      # 每個候選指向球心的向量
        norm = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
        to_center_unit = to_center / norm
        dot = np.einsum('ij,ij->i', approach, to_center_unit)
        valid_mask &= (dot > 0.85)
        print(f"   🔵 sphere 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    # ── Box / Plane ────────────────────────────────────────────────────────
    elif object_shape == 'box':
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        normals = np.asarray(pcd.normals)
        if len(normals) == 0:
            return gg
        # 取主要法向量（最常見方向）
        dominant_normal = normals[np.argmax(np.linalg.norm(normals, axis=1))]
        dominant_normal /= np.linalg.norm(dominant_normal) + 1e-6
        dot = np.abs(np.einsum('ij,j->i', approach, dominant_normal))
        valid_mask &= (dot > 0.80)
        print(f"   📦 box 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    # ── Cylinder / Handle ─────────────────────────────────────────────────
    elif object_shape == 'cylinder':
        # 用 PCA 取主軸（點雲最長方向 = 圓柱軸）
        centered = target_pts - np.mean(target_pts, axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis = Vt[0]   # 第一主成分 = 圓柱主軸
        axis /= np.linalg.norm(axis) + 1e-6
        # 閉合方向需垂直於主軸（內積接近 0）
        dot = np.abs(np.einsum('ij,j->i', closing, axis))
        valid_mask &= (dot < 0.35)
        print(f"   🥤 cylinder 過濾：{valid_mask.sum()}/{len(gg)} 候選通過")

    # 若全部被過濾，回退到原始排序
    if valid_mask.sum() == 0:
        print("   ⚠️ 幾何過濾後無候選，使用原始最高分")
        return gg

    return gg[np.where(valid_mask)[0]]
```

#### 插入位置（主迴圈）

```python
gg = gg.nms().sort_by_score()

# ← 在這裡插入
object_shape = payload.get('object_shape', 'box')
gg = filter_grasps_by_geometry(gg, points, colors_mask, bbox, object_shape)
gg = gg.sort_by_score()   # 過濾後重新排序

best_grasp = gg[0]
```

---

## 四、資料流（改動後）

```
brain_node.py
  Gemini → object_shape + target_grids + ...
  /system/llm_done → { object_shape, target_grids, ... }
      ↓
client_camera.py
  讀取 object_shape，加入 ZMQ payload
  ZMQ → { color, depth, bbox, object_shape }
      ↓
server_anygrasp.py
  點雲建構 + bbox 密度增加（已有）
  AnyGrasp 全場景推論
  bbox 空間過濾（已有）
  幾何擬合 + 向量內積過濾（新增）
  → best_grasp
```

---

## 五、已有 vs 新增

| 項目 | 狀態 | 位置 |
|------|------|------|
| VLM 單次推論（Gemini） | ✅ 已有 | brain_node.py |
| SAM mask 生成 | ✅ 已有 | brain_node.py |
| 點雲建構 | ✅ 已有 | server_anygrasp.py |
| bbox 空間過濾 | ✅ 已有 | server_anygrasp.py |
| 目標區域點雲密度增加 | ✅ 已有 | server_anygrasp.py |
| MoveIt 執行 | ✅ 已有 | semantic_grasp_controller.py |
| `object_shape` 輸出 | ⬜ 新增 | brain_node.py（Gemini prompt） |
| `object_shape` 傳遞 | ⬜ 新增 | client_camera.py（ZMQ payload） |
| 幾何擬合 + 向量內積過濾 | ⬜ 新增 | server_anygrasp.py |

---

## 六、注意事項

1. **AnyGrasp rotation 對應**：目前用 `R[:, 0]` 作接近方向（X 軸），實際執行前需用現有 log 或 debug 視覺化確認這個對應是否正確。

2. **球體 RANSAC**：本方案用點雲幾何質心代替 RANSAC 球心（更快更穩），若需精確球心可加入 `pyransac3d` 套件。

3. **過濾閾值可調整**：sphere 0.85、box 0.80、cylinder 0.35 是初始值，實測後根據結果調整。

4. **server upgrade 整合**：待 SERVER_UPGRADE_PLAN.md 實作完成後，`object_shape` 可直接在 server 端從 Gemini 取得，不需要透過 client 傳遞。
