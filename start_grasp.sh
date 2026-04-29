#!/bin/bash
# UR3 語義抓取系統 — 一鍵啟動腳本
# 用法：./start_grasp.sh
#
# ─── 操作方式 ────────────────────────────────────────────
#  切換 window：  滑鼠點底部 [ROS] / [AI] tab
#  切換 pane：    滑鼠點擊
#  全螢幕 pane：  Ctrl+b, z（再按還原）
#  關閉單一 pane：Ctrl+b, x → 按 y
#  關閉整個系統： 滑鼠點右下角 [ ✕ 關閉系統 ]  或  Ctrl+b, Q
# ─────────────────────────────────────────────────────────

if ! command -v tmux &>/dev/null; then
    echo "tmux 未安裝，請先執行：sudo apt-get install -y tmux"
    exit 1
fi

SESSION="grasp"
WS="$HOME/handeye_ws"
CONDA_INIT="source $HOME/miniconda3/etc/profile.d/conda.sh"

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' 已在運行，直接連入..."
    tmux attach -t $SESSION
    exit 0
fi

# ── 全域設定（-g 套用到所有 window/pane）────────────────
tmux new-session -d -s $SESSION -n "ROS"

tmux set-option -g -t $SESSION mouse on
tmux set-option -g -t $SESSION pane-border-status top
tmux set-option -g -t $SESSION pane-border-format "#{?pane_active,#[fg=colour46 bold],#[fg=colour244]} #{@label} "
tmux set-option -g -t $SESSION allow-rename off
tmux set-option -g -t $SESSION automatic-rename off
tmux set-option -g -t $SESSION status-right "#[fg=colour196 bold bg=colour235] [ ✕ 關閉系統 ] #[default]"
tmux set-option -g -t $SESSION status-right-length 30

# 滑鼠點右下角 [ ✕ 關閉系統 ] 執行 kill-session
tmux bind-key -T root MouseDown1StatusRight \
    confirm-before -p "關閉整個抓取系統? (y/n)" kill-session

# Ctrl+b Q 也可以關閉
tmux bind-key -T prefix Q \
    confirm-before -p "關閉整個抓取系統? (y/n)" kill-session

# Ctrl+b Tab 切換 window
tmux bind-key -T prefix Tab next-window

# ── Window 0: ROS 基礎 ────────────────────────────────────
P_CAM=$(tmux display-message -t $SESSION:0.0 -p '#{pane_id}')
P_ARM=$(tmux split-window -h -t "$P_CAM" -P -F '#{pane_id}')
P_MOV=$(tmux split-window -v -t "$P_CAM" -P -F '#{pane_id}')
P_TF=$( tmux split-window -v -t "$P_ARM" -P -F '#{pane_id}')

tmux set-option -p -t "$P_CAM" @label "📷 相機"
tmux set-option -p -t "$P_ARM" @label "🦾 手臂驅動"
tmux set-option -p -t "$P_MOV" @label "🧠 MoveIt"
tmux set-option -p -t "$P_TF"  @label "📐 TF 外參"

tmux send-keys -t "$P_CAM" "cd $WS && source devel/setup.bash && roslaunch realsense2_camera rs_camera.launch align_depth:=true" Enter
tmux send-keys -t "$P_ARM" "cd $WS && source devel/setup.bash && roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7" Enter
tmux send-keys -t "$P_MOV" "cd $WS && source devel/setup.bash && sleep 5 && roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true" Enter
tmux send-keys -t "$P_TF"  "cd $WS && source devel/setup.bash && sleep 8 && roslaunch easy_handeye publish.launch eye_on_hand:=false namespace_prefix:=ur3_realsense_handeyecalibration_eye_on_base robot_base_frame:=base_link tracking_base_frame:=camera_color_optical_frame calibration_file:=\$HOME/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml" Enter

# ── Window 1: AI 管線 ─────────────────────────────────────
P_RVIZ=$(tmux new-window   -t $SESSION -n "AI" -P -F '#{pane_id}')
P_BRAIN=$(tmux split-window -h -t "$P_RVIZ"    -P -F '#{pane_id}')
P_CTRL=$( tmux split-window -v -t "$P_RVIZ"    -P -F '#{pane_id}')
P_CLI=$(  tmux split-window -v -t "$P_BRAIN"   -P -F '#{pane_id}')

tmux set-option -p -t "$P_RVIZ"  @label "👁  RViz"
tmux set-option -p -t "$P_BRAIN" @label "🤖 brain_node"
tmux set-option -p -t "$P_CTRL"  @label "🎮 controller"
tmux set-option -p -t "$P_CLI"   @label "📸 client_camera"

tmux send-keys -t "$P_RVIZ"  "cd $WS && source devel/setup.bash && sleep 10 && rosrun rviz rviz -d $WS/anygrasp_debug.rviz" Enter
tmux send-keys -t "$P_BRAIN" "cd $WS && source devel/setup.bash && sleep 10 && $CONDA_INIT && conda activate grasp-py310 && python3 src/ur3_handover/scripts/brain_node.py" Enter
tmux send-keys -t "$P_CTRL"  "cd $WS && source devel/setup.bash && sleep 12 && $CONDA_INIT && conda activate anygrasp && rosrun ur3_handover semantic_grasp_controller.py" Enter
tmux send-keys -t "$P_CLI"   "cd $WS/src/ur3_handover/scripts && source $WS/devel/setup.bash && sleep 15 && $CONDA_INIT && conda activate anygrasp && python3 client_camera.py" Enter

tmux select-window -t $SESSION:0
tmux attach -t $SESSION
