# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 完整啟動管線（語義抓取）

依序在不同終端機執行，每個步驟都要等前一個就緒：

| # | 功能 | 指令 |
|---|------|------|
| 1 | 相機 | `roslaunch realsense2_camera rs_camera.launch align_depth:=true` |
| 2 | 手臂 | `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7` |
| 3 | MoveIt | `roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true` |
| 4 | TF 外參 | `roslaunch easy_handeye publish.launch eye_on_hand:=false namespace_prefix:=ur3_realsense_handeyecalibration_eye_on_base robot_base_frame:=base_link tracking_base_frame:=camera_color_optical_frame calibration_file:=$HOME/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml` |
| 5 | AI 大腦（遠端） | `python3 server_anygrasp.py --debug` |
| 6 | 機械臂控制器 | `rosrun ur3_handover semantic_grasp_controller.py` |
| 7 | 視覺前端 | `python3 client_camera.py` |

**前置確認**
- 筆電與 UR3 (`192.168.86.7`) 互 Ping 通
- Ngrok 已啟動：`ngrok tcp 5555`，並更新 `client_camera.py:26` 的 server 地址
- 外參檔存在：`~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`

## 其他常用指令

```bash
# 建置工作區
cd /home/weilun/handeye_ws && catkin_make
source /home/weilun/handeye_ws/devel/setup.bash

# 傳統 Click-to-Pick 流程 (YOLO → 像素座標 → 機械臂)
roslaunch ur3_handover click_to_pick_cv.launch
```

## 整體架構

系統分為兩條獨立管線，共用同一台機械臂：

### 管線 A：語義抓取（主要流程）

```
RealSense D435
    │  /camera/color/image_raw
    │  /camera/aligned_depth_to_color/image_raw
    ▼
client_camera.py          (ur3_handover/scripts/)
  ├─ YOLO 物體偵測 (weights/best.pt)
  ├─ SVD 桌面擬合：填補物體遮蔽造成的深度空洞
  └─ ZMQ REQ → 遠端 AnyGrasp Server (Ngrok TCP)
                              │ 回傳 6D 抓取姿態
    ▼  ROS topic: /anygrasp/target_pose (PoseStamped)
semantic_grasp_controller.py  (ur3_handover/scripts/)
  ├─ TF 轉換：camera_color_optical_frame → base_link
  ├─ 姿態對齊：AnyGrasp X 軸旋轉 Y -90° → UR3 Z 軸
  ├─ 安全鎖：互動式 input() 確認後才執行
  └─ MoveIt：Pre-Grasp → Cartesian 前進 → 夾取 → 抬升 → 放置
```

### 管線 B：傳統 Click-to-Pick

```
click_to_pick_cv.py  →  YOLO 偵測像素 → pixel_to_base.py (TF 投影)
                     →  pose_to_pick.py (MoveIt 執行)
```

`pose_to_pick.py` 的 `PoseToPick` class 是完整可用的抓取執行器，`semantic_grasp_controller.py` 的邏輯從它移植而來。

### 遠端 AnyGrasp Server 側 (anygrasp_sdk/integration/)

- `grasp_detector.py`：封裝 AnyGrasp 推論，輸入點雲，輸出 6D 抓取姿態
- `main_controller.py`：**草稿，未完成**，`execute_grasp()` 是空 `pass`
- `takephoto.py`：獨立腳本，用 pyrealsense2 直接拍照並印出相機內參

## 關鍵設計細節

### ZMQ 通訊
- Protocol：REQ-REP，Client 送壓縮 pickle (`zlib + pickle`)，內含 `{color_jpg, depth, bbox}`
- Server 地址硬編碼在 `client_camera.py:26`，每次 Ngrok 重啟須手動更新
- 逾時設定：`ZMQ_RECV_TIMEOUT_MS = 30000`（30 秒），逾時後 REQ socket 必須重建才能繼續

### 相機內參
- 動態訂閱 `/camera/color/camera_info`，取得一次後 unregister
- 內參在 `self.fx/fy/cx/cy` 就緒前按 `[s]` 會被攔截提示等待
- 原本硬編碼值（`fx=617.183, fy=617.122, cx=319.639, cy=241.404`）以 `# 原本硬編碼` 保留在旁

### SVD 桌面擬合
僅在 YOLO 有提供 mask 時啟用（手動框選模式跳過）：
1. 對物體 mask 做兩次膨脹，形成「甜甜圈」取樣環
2. 外環（`table_donut_mask`）取桌面點雲，SVD 擬合平面法向量
3. 內環（`moat_mask`）用平面方程式反算深度，填補遮蔽空洞

### 抓取流程關鍵參數（semantic_grasp_controller.py）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `tcp_offset` | 0.18 m | 夾爪指尖到 MoveIt EEF 距離 |
| `grasp_depth` | 0.04 m | 物體表面往內插入深度 |
| `approach_dist` | 0.05 m | Pre-Grasp 退後距離 |
| `retreat_up_height` | 0.15 m | 抓取後垂直抬升 |
| `final_xyz` | `[0.2, 0.1, 0.185]` | 固定放置座標（硬編碼） |

安全桌面檢查（`TABLE_HEIGHT`）目前被 `#` 註解，虛擬桌面碰撞物件設在 `base_link` Z=-0.05m。

### Hand-Eye 校正
- 使用 `easy_handeye` 套件（`src/easy_handeye/`），類型為 eye-on-base
- 外參檔：`~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`
- `publish_handeye_tf.py` 讀取 YAML 並廣播 static TF（`base_link` → `camera_color_optical_frame`）

### RealSense D435 深度有效範圍
- **0~20 cm**：盲區，完全不可用
- **30~60 cm**：黃金甜蜜點，誤差 < 1mm，適合抓取
- **80~100 cm**：誤差約 1~2%
- **150 cm+**：僅適合避障，不適合抓取
