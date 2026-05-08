
---

# 🚀 UR3 × RealSense × AnyGrasp：6D 全自動語意抓取操作指南

> **🤖 系統架構：**
> * **大腦 (桌機端)**：`server_anygrasp.py` — 接收影像，運算全空間 6D 抓取姿態
> * **AI 視覺 (筆電端)**：`brain_node.py` — OWL-v2 + SAM + Gemini，語義物件分割
> * **眼睛 (筆電端)**：`client_camera.py` — 相機畫面、VLM 觸發、SVD 桌面擬合、ZMQ 傳送
> * **交接感知 (筆電端)**：`handover_perception.py` — MediaPipe Hands、手心 3D、handover RViz markers
> * **肌肉 (ROS 端)**：`semantic_grasp_controller.py` — TF 轉換、MoveIt 規劃執行、夾爪控制
>
> **⚠️ 前提條件**：手眼標定已完成，外參 YAML 位於 `~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`
> **⚙️ handover 參數檔**：`src/ur3_handover/config/handover_params.yaml`，現場要改交接區域或力矩門檻時請優先改這個檔案

---

## 零、啟動腳本狀態

目前 `start_grasp.sh` 已停用，原因是 tmux 自動啟動時偶發 ROS master / MoveIt 啟動時序不一致。抓取流程目前以手動分開啟動各節點為準；`start_calibration.sh` 可照常使用。

| 腳本 | 用途 | 指令 |
|------|------|------|
| `start_grasp.sh` | 抓取系統一鍵啟動 | 已停用，勿使用 |
| `start_calibration.sh` | 手眼標定管線（4 pane） | `cd ~/handeye_ws && ./start_calibration.sh` |

**前置（腳本啟動前需手動完成）：**
1. `sudo nmcli con up "UR3"` — 啟動機器人網卡
2. `ngrok tcp 5555` — 啟動 Ngrok，並更新 `client_camera.py:26` server 地址
3. AI Server 端先啟動 `server_anygrasp.py`
4. 若要調整交接區域或力矩放手門檻，先編輯 `src/ur3_handover/config/handover_params.yaml`

> 以下手動步驟供需要單獨啟動某節點時參考。

---

## 一、網段與連線準備

**1. 啟動機器人網卡：**

```bash
sudo nmcli con up "UR3"
```

確認 UR3 (`192.168.86.7`)、筆電、AnyGrasp 桌機互相 ping 得通。

**2. 啟動 Ngrok 通道（AnyGrasp Server 在遠端時）：**

```bash
ngrok tcp 5555
```

啟動後將 `0.tcp.jp.ngrok.io:XXXXX` 更新到 `client_camera.py` 第 26 行的 server 地址。

---

## 二、手眼標定（每次到現場先做）

> 標定完成後外參 YAML 會自動存到 `~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`，之後啟動系統直接讀取即可。
>
> **一鍵啟動：** `./start_calibration.sh`（自動開啟相機、手臂、MoveIt、標定 GUI）

**手動啟動（前置：相機和手臂需先啟動 T1、T2、T3）：**
**[T1] 相機：**
```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

**[T2] 手臂驅動：**
```bash
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7
```

**[T3] MoveIt：**
```bash
roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true
```

**[標定終端機] 啟動標定節點：**
```bash
roslaunch easy_handeye ur3_eye_to_hand_calibration.launch robot_ip:=192.168.86.7
```

確認：
- ArUco marker ID `582`，尺寸 `0.04m`（4cm）
- 相機畫面中能看到 ArUco marker 被偵測到（有框線）

**標定 GUI 操作步驟：**

1. 將 ArUco marker用夾爪抓好
2. 在 easy_handeye GUI 中點確認初始姿態後按 **`Next Pose`** → 手臂自動移動到新姿態
3. 確認 marker 在相機中清晰可見後，點 **`Take Sample`** 記錄一筆
4. 重複步驟 2~3，**至少取 15~20 筆**，姿態越多樣越好（不同角度、不同高度）
5. 取樣完成後點 **`Compute`** 計算外參跟點 **`Save`** 儲存

**驗證標定結果：**
```bash
cat ~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml
```

---

## 三、系統核心啟動（標定完成後，依序開）

> `start_grasp.sh` 目前停用；以下手動流程為正式做法。

**手動啟動（各節點分開開時參考）：**

**[T4] RViz（安全確認用）：**
```bash
rosrun rviz rviz -d /home/weilun/handeye_ws/anygrasp_debug.rviz
```

確認 MotionPlanning 面板已開啟，勾選 `Show Trail`。

**[T5] 手眼外參 TF：**
```bash
roslaunch easy_handeye publish.launch \
  eye_on_hand:=false \
  namespace_prefix:=ur3_realsense_handeyecalibration_eye_on_base \
  robot_base_frame:=base_link \
  tracking_base_frame:=camera_color_optical_frame \
  calibration_file:=$HOME/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml
```

---

## 三、啟動 AI 抓取管線

**[T6] AnyGrasp Server（遠端桌機）：**（基本上會先開好）
```bash
conda activate anygrasp
cd ~/anygrasp_sever/src/anygrasp_sdk/grasp_detection
python3 server_anygrasp.py --debug --save
```

確認顯示：`伺服器已上線...`

**[T7] AI 大腦節點（brain_node）：**
```bash
conda activate grasp-py310
rosrun ur3_handover brain_node.py
```

確認顯示：`Brain node ready, waiting for /system/trigger_llm ...`

**[T8] 機械臂控制節點：**
```bash
conda activate anygrasp
cd ~/handeye_ws
rosrun ur3_handover semantic_grasp_controller.py
```

確認顯示：`啟動完成，等待目標姿態...`

**[T9] 視覺前端：**
```bash
conda activate anygrasp
cd ~/handeye_ws/src/ur3_handover/scripts
python3 client_camera.py
```

相機畫面彈出，左側即時畫面，右側抓取姿態預覽。

**[T10] 交接感知節點（handover_perception）：**
```bash
cd ~/handeye_ws
source devel/setup.bash
rosrun ur3_handover handover_perception.py
```

> `handover_perception.py` 請直接用系統 Python 執行，不要在 conda 環境下啟動。
> `semantic_grasp_controller.py` 目前改回純力矩釋放，不再需要手動送 `release` 指令。

---

## 四、client_camera.py 按鍵

| 鍵 | 功能 |
|----|------|
| `v` | VLM 模式：輸入目標物件名稱（英文）；直接 Enter 沿用上次輸入 |
| `r` | 手動框選 ROI |
| `s` | 重新發送當前 mask 給 AnyGrasp 重算姿態 |
| `c` | 清除目標，重置狀態 |
| `q` | 離開 |

---

## 五、實戰操作 SOP

### 模式 A：VLM 語義抓取（主要流程）

1. 確認雙側畫面已出現（左：即時相機，右：等待抓取結果）
2. 按 **`v`** → 輸入目標物件名稱（例如：`hammer`）；若上次相同直接按 Enter
3. 等待 AI 處理（約 10~30 秒）→ 左側出現**黃色遮罩**表示分割完成
4. 系統自動發送 → 右側出現**抓取姿態預覽**（綠線表示抓取軸）
5. 轉頭看 **RViz** 確認軌跡安全
6. 切到 `semantic_grasp_controller.py` 終端機 → 看到 `⚠️ [安全鎖]` 提示
7. 確認安全按 **`Enter`** 執行；不確定按 `r` 重規劃；取消按 `n`
8. 夾緊後按 **`Enter`** 繼續放置
9. 放置完成後按 **`Enter`** 回待機點（或按 `n` 停在原地）

### 模式 B：手動框選

1. 按 **`r`** → 框選物件，按 Enter 確認
2. 後續流程同步驟 4 開始

---

## 六、緊急操作

**隨時回待機點（不需要透過 client 畫面）：**
```bash
rostopic pub /semantic_grasp/go_home std_msgs/String "go" -1
```

**UR3 緊急停止：**
直接按示教器上的紅色 E-Stop 按鈕。

---

## 七、常見錯誤排查

| 錯誤現象 | 原因 | 解法 |
|----------|------|------|
| `wrist_3_joint outside bounds` | wrist_3 轉過頭超出 ±2π | 程式自動 normalize，重新觸發規劃即可 |
| Cartesian 路徑不全 `fraction < 0.99` | 路徑受碰撞物件阻擋 | 按 `r` 重規劃，或手動移動手臂到更好起始位置 |
| `TF 失敗` | easy_handeye TF 未啟動 | 確認 T5 的 publish.launch 有在跑 |
| ZMQ 逾時（30s） | Ngrok 斷線或 server 未啟動 | 重啟 Ngrok 並更新 client_camera.py server 地址 |
| `Object not found` | OWL-v2 偵測不到目標 | 換更精確描述，或確認物件在畫面中清晰可見 |
| 右側畫面一直顯示 `waiting for result...` | ZMQ 未回應 | 確認 server_anygrasp.py 有在運行且 Ngrok 通道正常 |
| 夾爪夾不深 | `grasp_depth` 過小 或 `tcp_offset` 有偏差 | 調整 `semantic_grasp_controller.py` 中對應參數 |

---

## 八、RealSense D435 深度有效範圍

| 距離 | 狀態 | 說明 |
|------|------|------|
| 0 ~ 20 cm | 盲區 | 完全不可用 |
| 30 ~ 60 cm | 黃金甜蜜點 | 誤差 < 1mm，最適合抓取 |
| 80 ~ 100 cm | 開始飄移 | 誤差約 1~2% |
| 150 cm 以上 | 僅供避障 | 不適合抓取 |

---
