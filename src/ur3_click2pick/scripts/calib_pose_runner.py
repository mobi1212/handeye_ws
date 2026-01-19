#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calib Pose Runner (for easy_handeye sampling)

用途：
  讀取資料夾內的 YAML 姿態（pose 或 joint），依序移動機械手臂到每個姿態。
  每到一個姿態會「停住等待」，直到你按下按鈕（鍵盤 Enter 或呼叫 ROS service）才會繼續下一個姿態。

建議用法：
  1) 啟動 ur_robot_driver + MoveIt
  2) 執行本節點，手臂會走到 pose_001.yaml
  3) 你在 easy_handeye 介面按 Take Sample
  4) 回到終端按 Enter（或呼叫 /calib_poses/next）到下一個姿態
"""

import os
import glob
import math
import threading
import re

import rospy
from geometry_msgs.msg import PoseStamped
from moveit_commander import MoveGroupCommander, roscpp_initialize
from tf.transformations import quaternion_from_euler

try:
    import yaml
except Exception as e:
    raise RuntimeError("需要 python3-yaml：sudo apt-get install python3-yaml") from e

try:
    from std_srvs.srv import Trigger, TriggerResponse
except Exception:
    Trigger = None


def _as_float_list(x, n=None):
    if x is None:
        return None
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"expected list, got {type(x)}")
    out = [float(v) for v in x]
    if n is not None and len(out) != n:
        raise ValueError(f"expected len={n}, got {len(out)}")
    return out


class CalibPoseRunner:
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.move_group = rospy.get_param("~move_group", "manipulator")

        # 讀取 YAML 的資料夾
        self.poses_dir = rospy.get_param("~poses_dir", "")
        if not self.poses_dir:
            # 預設：/home/weilun/handeye_ws/poses
            self.poses_dir = os.path.expanduser("~/handeye_ws/poses")


        # 互動模式：keyboard / service
        self.wait_mode = rospy.get_param("~wait_mode", "keyboard").strip().lower()
        # 若 wait_mode=service，service 名稱
        self.next_service_name = rospy.get_param("~next_service_name", "~next")

        self.vel_scale = float(rospy.get_param("~vel_scale", 0.10))
        self.acc_scale = float(rospy.get_param("~acc_scale", 0.10))

        self._next_event = threading.Event()

        roscpp_initialize([])
        self.group = MoveGroupCommander(self.move_group)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)

        rospy.loginfo("[calib_runner] base_frame=%s move_group=%s", self.base_frame, self.move_group)
        rospy.loginfo("[calib_runner] poses_dir=%s", self.poses_dir)
        rospy.loginfo("[calib_runner] wait_mode=%s", self.wait_mode)

        if self.wait_mode == "service":
            if Trigger is None:
                raise RuntimeError("wait_mode=service 需要 std_srvs，請確認 ROS 環境已 source")
            rospy.Service(self.next_service_name, Trigger, self._on_next)
            rospy.loginfo("[calib_runner] service ready: %s", rospy.resolve_name(self.next_service_name))

    def _on_next(self, req):
        self._next_event.set()
        return TriggerResponse(success=True, message="next pose")

    def _wait_user(self, hint=""):
        if self.wait_mode == "service":
            rospy.loginfo("[calib_runner] 等待 service 呼叫繼續%s", f" ({hint})" if hint else "")
            self._next_event.clear()
            while not rospy.is_shutdown():
                if self._next_event.wait(timeout=0.1):
                    return True
            return False

        # keyboard
        try:
            s = input(f"\n[calib_runner] 已到位 {hint}。按 Enter 繼續；輸入 q 離開；輸入 r 重做此姿態： ").strip().lower()
        except EOFError:
            # 沒有 tty 時，避免卡死：改成 sleep 等 ROS shutdown
            rospy.logwarn("[calib_runner] 無法讀取鍵盤輸入（可能在非互動環境），將等待 Ctrl+C 結束")
            while not rospy.is_shutdown():
                rospy.sleep(0.2)
            return False

        if s == "q":
            return False
        if s == "r":
            return "repeat"
        return True

    def _load_pose_files(self):
        if not os.path.isdir(self.poses_dir):
            raise RuntimeError(f"poses_dir 不存在：{self.poses_dir}")

        def _natural_key(path: str):
            name = os.path.basename(path)
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

        files = sorted(glob.glob(os.path.join(self.poses_dir, "*.yaml")) +
                    glob.glob(os.path.join(self.poses_dir, "*.yml")),
                    key=_natural_key)
        if not files:
            raise RuntimeError(f"poses_dir 沒有找到任何 .yaml：{self.poses_dir}")
        return files

    def _pose_from_yaml(self, d):
        # 支援三種格式：
        # A) 你現在這種： position: {x,y,z}, orientation: {x,y,z,w}
        # B) 外面包一層 pose: {position..., orientation...}
        # C) 我原本格式： xyz: [..], rpy_deg: [..] 或 quat: [..]

        frame_id = (d.get("frame_id") or self.base_frame).strip()

        # 先嘗試 A / B（geometry_msgs/Pose 風格）
        pd = d
        if isinstance(d.get("pose"), dict):
            pd = d["pose"]

        if isinstance(pd.get("position"), dict) and isinstance(pd.get("orientation"), dict):
            p = pd["position"]
            o = pd["orientation"]

            xyz = [float(p["x"]), float(p["y"]), float(p["z"])]
            q = [float(o["x"]), float(o["y"]), float(o["z"]), float(o["w"])]

            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
            return ps

        # 否則走原本格式（xyz + rpy_deg/quat）
        xyz = _as_float_list(d.get("xyz"), 3)

        quat = d.get("quat")
        rpy_deg = d.get("rpy_deg")

        if quat is not None:
            q = _as_float_list(quat, 4)
        else:
            rpy = _as_float_list(rpy_deg, 3)
            qx, qy, qz, qw = quaternion_from_euler(
                math.radians(rpy[0]), math.radians(rpy[1]), math.radians(rpy[2])
            )
            q = [qx, qy, qz, qw]

        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = xyz[0]
        ps.pose.position.y = xyz[1]
        ps.pose.position.z = xyz[2]
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        return ps


    def _go_pose(self, ps):
        self.group.set_start_state_to_current_state()
        self.group.set_pose_target(ps)
        ok = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        return bool(ok)

    def _go_joint(self, joints):
        self.group.set_start_state_to_current_state()
        ok = self.group.go(joints, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        return bool(ok)

    def run(self):
        files = self._load_pose_files()
        rospy.loginfo("[calib_runner] total poses=%d", len(files))

        i = 0
        while i < len(files) and not rospy.is_shutdown():
            f = files[i]
            with open(f, "r", encoding="utf-8") as fp:
                d = yaml.safe_load(fp) or {}

            name = d.get("name") or os.path.basename(f)
            typ = (d.get("type") or "pose").strip().lower()
            rospy.loginfo("\n[calib_runner] (%d/%d) %s  type=%s", i + 1, len(files), name, typ)

            ok = False
            if typ == "joint":
                joints = _as_float_list(d.get("joints"), 6)
                ok = self._go_joint(joints)
            else:
                ps = self._pose_from_yaml(d)
                ok = self._go_pose(ps)

            if not ok:
                rospy.logerr("[calib_runner] move FAIL: %s", name)
                # 失敗就停住，讓你決定要不要繼續
                res = self._wait_user(hint=f"{name}（移動失敗）")
                if res is False:
                    break
                if res == "repeat":
                    continue
                i += 1
                continue

            # 到位後等待：你可以去 easy_handeye 按 Take Sample
            res = self._wait_user(hint=name)
            if res is False:
                break
            if res == "repeat":
                continue
            i += 1

        rospy.loginfo("[calib_runner] DONE / EXIT")


def main():
    rospy.init_node("calib_pose_runner")
    app = CalibPoseRunner()
    rospy.sleep(0.5)
    app.run()


if __name__ == "__main__":
    main()
