# UR3 語義抓取系統 — 操作快速參考

> 這份文件是現場速查卡。完整操作說明請看 `hackmd.md`；
> repo / submodule 使用方式請看 `docs/REPO_SETUP_AND_SUBMODULE_GUIDE.md`。

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
| 9 | Handover 感知 | `rosrun ur3_handover handover_perception.py` |

**前置確認**
- UR3 IP `192.168.86.7` 可 ping 通
- Ngrok 已啟動：`ngrok tcp 5555`，並更新 `client_camera.py` 第 26 行的 server 地址
- 外參檔存在：`~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`
- `handover_perception.py` 請直接用系統 Python 跑，不要在 conda 環境下啟動
- handover 區域、手心偏移與力矩釋放門檻統一改 `src/ur3_handover/config/handover_params.yaml`
- `start_grasp.sh` 目前已停用，抓取流程改成手動分開啟動各節點

```bash
# 手動啟動節點前若要套用 handover 參數
rosparam load /home/weilun/handeye_ws/src/ur3_handover/config/handover_params.yaml
```

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

## 參數改哪裡

- handover 區域、掌心偏移、handedness 穩定化、力矩釋放門檻：
  `src/ur3_handover/config/handover_params.yaml`
- 抓取深度、approach 距離、固定放置點、速度比例：
  `src/ur3_handover/scripts/semantic_grasp_controller.py`
- AnyGrasp server 地址：
  `src/ur3_handover/scripts/client_camera.py`

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

需要完整 SOP、校正流程、實戰步驟、深度範圍說明時，直接看 `hackmd.md`。
