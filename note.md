# UR3 語義抓取系統 — 操作快速參考

## 啟動順序

> 每個步驟等上一個就緒後再執行

| # | 功能 | 指令 |
|---|------|------|
| 1 | 相機 | `roslaunch realsense2_camera rs_camera.launch align_depth:=true` |
| 2 | 手臂驅動 | `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7` |
| 3 | MoveIt | `roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true` |
| 4 | TF 外參 | `roslaunch easy_handeye publish.launch eye_on_hand:=false namespace_prefix:=ur3_realsense_handeyecalibration_eye_on_base robot_base_frame:=base_link tracking_base_frame:=camera_color_optical_frame calibration_file:=$HOME/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml` |
| 5 | AnyGrasp Server（遠端） | `python3 server_anygrasp.py --debug` |
| 6 | AI 大腦 | `conda activate grasp-py310 && rosrun ur3_handover brain_node.py` |
| 7 | 機械臂控制器 | `conda activate anygrasp && rosrun ur3_handover semantic_grasp_controller.py` |
| 8 | 視覺前端 | `conda activate anygrasp && python3 client_camera.py` |

**前置確認**
- UR3 IP `192.168.86.7` 可 ping 通
- Ngrok 已啟動：`ngrok tcp 5555`，並更新 `client_camera.py` 第 26 行的 server 地址
- 外參檔存在：`~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`

---

## client_camera.py 按鍵

| 鍵 | 功能 |
|----|------|
| `v` | VLM 模式：輸入目標物件名稱（英文）；直接 Enter 沿用上次輸入 |
| `r` | 手動框選 ROI |
| `s` | 重新發送當前 mask 給 AnyGrasp 重算姿態 |
| `c` | 清除目標，重置狀態 |
| `q` | 離開 |

---

## 抓取流程關鍵參數

> 檔案：`src/ur3_handover/scripts/semantic_grasp_controller.py`

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `tcp_offset` | 0.18 m | 夾爪指尖到 MoveIt EEF 距離 |
| `grasp_depth` | 0.05 m | 物體表面往內插入深度 |
| `approach_dist` | 0.05 m | Pre-Grasp 退後距離 |
| `retreat_up_height` | 0.14 m | 待機點高於放置點的高度 |
| `final_xyz` | `[0.2401, 0.1751, 0.185]` | 固定放置座標（法蘭位置） |
| `vel_scale` | 0.10 | 速度比例（10%） |
| `acc_scale` | 0.10 | 加速度比例（10%） |

---

## 抓取動作序列

```
A. 關節規劃 → Pre-Grasp（接近點）   ← [Enter] 確認 / [r] 重規劃 / [n] 取消
B. Cartesian 直線前進 → Grasp
C. 夾緊                              ← [Enter] 確認繼續
D. Cartesian 垂直抬升 5 cm
E. 關節規劃 → Pre-Place
F. Cartesian 直線前進 → Place
G. 張開夾爪
H. Cartesian 後退 → Pre-Place
   → 詢問是否回待機點
```

---

## 常見錯誤

| 錯誤 | 原因 | 解法 |
|------|------|------|
| `wrist_3_joint outside bounds` | wrist_3 轉過頭超出 ±2π | 程式會自動 normalize，重新規劃即可 |
| `Cartesian 路徑不全 (fraction < 0.99)` | 路徑受碰撞物件阻擋 | 用 [r] 重新規劃，或手動移動手臂到較好的起始位置 |
| `TF 失敗` | easy_handeye TF 未啟動 | 確認步驟 4 的 publish.launch 有跑 |
| ZMQ 逾時（30s） | Ngrok 斷線或 server 未啟動 | 重啟 Ngrok 並更新 client_camera.py 地址 |
| `Object not found` | OWL-v2 偵測不到目標 | 換更精確的描述，或物件移到較亮的位置 |
| 規劃失敗（`INVALID_GOAL`） | 軌跡時間戳過期 | 會自動重規劃，正常現象 |

---

## RealSense D435 深度有效範圍

| 距離 | 狀態 | 誤差 |
|------|------|------|
| 0 ~ 20 cm | 盲區，完全不可用 | — |
| 30 ~ 60 cm | 黃金甜蜜點，適合抓取 | < 1 mm |
| 80 ~ 100 cm | 開始飄移 | 1 ~ 2% |
| 150 cm 以上 | 僅適合避障 | — |

---

## 常用指令

```bash
# 建置
cd /home/weilun/handeye_ws && catkin_make
source /home/weilun/handeye_ws/devel/setup.bash

# 讀取手臂目前位置
python3 -c "
import rospy; from moveit_commander import MoveGroupCommander, roscpp_initialize
roscpp_initialize([]); rospy.init_node('tmp', anonymous=True)
g = MoveGroupCommander('manipulator')
p = g.get_current_pose().pose
print(f'xyz = [{p.position.x:.4f}, {p.position.y:.4f}, {p.position.z:.4f}]')
"

# 回待機點（任何時候）
rostopic pub /semantic_grasp/go_home std_msgs/String "go" -1
```
