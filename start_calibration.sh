#!/bin/bash
# UR3 手眼標定 — 一鍵啟動腳本

if ! command -v tmux &>/dev/null; then
    echo "tmux 未安裝，請先執行：sudo apt-get install -y tmux"
    exit 1
fi

SESSION="calibration"
WS="$HOME/handeye_ws"

if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' 已在運行，直接連入..."
    tmux attach -t $SESSION
    exit 0
fi

# ── 全域設定 ──────────────────────────────────────────────
tmux new-session -d -s $SESSION -n "calibration"

tmux set-option -g -t $SESSION mouse on
tmux set-option -g -t $SESSION pane-border-status top
tmux set-option -g -t $SESSION pane-border-format "#{?pane_active,#[fg=colour46 bold],#[fg=colour244]} #{@label} "
tmux set-option -g -t $SESSION allow-rename off
tmux set-option -g -t $SESSION automatic-rename off
tmux set-option -g -t $SESSION status-right "#[fg=colour196 bold bg=colour235] [ ✕ 關閉標定 ] #[default]"
tmux set-option -g -t $SESSION status-right-length 25

tmux bind-key -T root MouseDown1StatusRight \
    confirm-before -p "關閉標定系統? (y/n)" kill-session
tmux bind-key -T prefix Q \
    confirm-before -p "關閉標定系統? (y/n)" kill-session

# ── Window 0: 標定服務 ────────────────────────────────────
P_CAM=$(tmux display-message -t $SESSION:0.0 -p '#{pane_id}')
P_ARM=$(tmux split-window -h -t "$P_CAM" -P -F '#{pane_id}')
P_MOV=$(tmux split-window -v -t "$P_CAM" -P -F '#{pane_id}')
P_CAL=$(tmux split-window -v -t "$P_ARM" -P -F '#{pane_id}')

tmux set-option -p -t "$P_CAM" @label "📷 相機"
tmux set-option -p -t "$P_ARM" @label "🦾 手臂驅動"
tmux set-option -p -t "$P_MOV" @label "🧠 MoveIt"
tmux set-option -p -t "$P_CAL" @label "📐 手眼標定"

tmux send-keys -t "$P_CAM" "cd $WS && source devel/setup.bash && roslaunch realsense2_camera rs_camera.launch align_depth:=true" Enter
tmux send-keys -t "$P_ARM" "cd $WS && source devel/setup.bash && roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7" Enter
tmux send-keys -t "$P_MOV" "cd $WS && source devel/setup.bash && sleep 5 && roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true" Enter
tmux send-keys -t "$P_CAL" "cd $WS && source devel/setup.bash && sleep 8 && roslaunch easy_handeye ur3_eye_to_hand_calibration.launch robot_ip:=192.168.86.7" Enter

tmux attach -t $SESSION
