#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handover perception node.

Detect a single hand from RGB-D, estimate a palm center in 3D, and publish a
compact JSON hand state for the controller to consume during handover.
"""

import json
import os
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  # Registers PointStamped transform support.
import yaml
from geometry_msgs.msg import PointStamped
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker


PALM_LANDMARK_IDS = (0, 5, 9, 13, 17)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


class HandoverPerceptionNode:
    def __init__(self):
        rospy.init_node("handover_perception", anonymous=False)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        ws_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        default_model_path = os.path.join(ws_root, "models", "hand_landmarker.task")
        default_config_path = os.path.join(ws_root, "src", "ur3_handover", "config", "handover_params.yaml")

        self.base_frame = self._get_param("~base_frame", None, "base_link")
        self.default_camera_frame = self._get_param(
            "~default_camera_frame", None, "camera_color_optical_frame")
        self.color_topic = self._get_param(
            "~color_topic", None, "/camera/color/image_raw")
        self.depth_topic = self._get_param(
            "~depth_topic", None, "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = self._get_param(
            "~camera_info_topic", None, "/camera/color/camera_info")
        self.hand_state_topic = self._get_param(
            "~hand_state_topic", "/semantic_handover/hand_state_topic",
            "/semantic_handover/hand_state")
        self.marker_topic = self._get_param(
            "~marker_topic", "/semantic_handover/marker_topic",
            "/semantic_handover/markers")
        self.process_hz = float(self._get_param(
            "~process_hz", "/semantic_handover/process_hz", 8.0))
        self.model_path = self._get_param("~model_path", None, default_model_path)
        self.depth_scale = float(self._get_param("~depth_scale", None, 1000.0))
        self.depth_sample_radius_px = int(self._get_param(
            "~depth_sample_radius_px", "/semantic_handover/depth_sample_radius_px", 6))
        self.min_valid_depth_m = float(self._get_param(
            "~min_valid_depth_m", "/semantic_handover/min_valid_depth_m", 0.15))
        self.max_valid_depth_m = float(self._get_param(
            "~max_valid_depth_m", "/semantic_handover/max_valid_depth_m", 1.20))
        self.stability_window = int(self._get_param(
            "~stability_window", "/semantic_handover/stability_window", 5))
        self.stability_pos_threshold = float(
            self._get_param(
                "~stability_pos_threshold",
                "/semantic_handover/stability_pos_threshold",
                0.03))
        self.marker_lifetime_sec = float(self._get_param("~marker_lifetime_sec", None, 0.5))
        self.landmark_marker_scale = float(self._get_param("~landmark_marker_scale", None, 0.014))
        self.palm_marker_scale = float(self._get_param("~palm_marker_scale", None, 0.035))
        self.skeleton_line_width = float(self._get_param("~skeleton_line_width", None, 0.008))
        self.handover_config_path = self._get_param(
            "~handover_config_path", None, default_config_path)
        self._handover_config_mtime = None

        default_zone_center = self._get_param(
            "~handover_zone_center_xyz",
            "/semantic_handover/handover_zone_center_xyz",
            [0.1606, 0.2881, 0.1554])
        default_zone_half_extents = self._get_param(
            "~handover_zone_half_extents_xyz",
            "/semantic_handover/handover_zone_half_extents_xyz",
            [0.10, 0.10, 0.08])
        self.zone_min = np.array(
            self._get_param(
                "~handover_zone_min_xyz",
                "/semantic_handover/handover_zone_min_xyz",
                (np.array(default_zone_center) - np.array(default_zone_half_extents)).tolist(),
            ),
            dtype=float,
        )
        self.zone_max = np.array(
            self._get_param(
                "~handover_zone_max_xyz",
                "/semantic_handover/handover_zone_max_xyz",
                (np.array(default_zone_center) + np.array(default_zone_half_extents)).tolist(),
            ),
            dtype=float,
        )
        self.handover_zone_visible = bool(self._get_param(
            "~handover_zone_visible", "/semantic_handover/handover_zone_visible", True))
        self.handover_zone_color_rgba = np.array(
            self._get_param(
                "~handover_zone_color_rgba",
                "/semantic_handover/handover_zone_color_rgba",
                [0.1, 0.5, 1.0, 0.3],
            ),
            dtype=float,
        )
        self.swap_handedness_labels = bool(self._get_param(
            "~swap_handedness_labels", "/semantic_handover/swap_handedness_labels", True))
        self.right_hand_normal_sign = float(self._get_param(
            "~right_hand_normal_sign", "/semantic_handover/right_hand_normal_sign", 1.0))
        self.left_hand_normal_sign = float(self._get_param(
            "~left_hand_normal_sign", "/semantic_handover/left_hand_normal_sign", -1.0))
        self.handedness_debounce_frames = int(self._get_param(
            "~handedness_debounce_frames", "/semantic_handover/handedness_debounce_frames", 3))
        self.palm_normal_ema_alpha = float(self._get_param(
            "~palm_normal_ema_alpha", "/semantic_handover/palm_normal_ema_alpha", 0.25))

        if not os.path.exists(self.model_path):
            rospy.logfatal("找不到 hand landmarker model: %s", self.model_path)
            raise FileNotFoundError(self.model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
        )
        self.detector = HandLandmarker.create_from_options(options)

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        self.hand_state_pub = rospy.Publisher(self.hand_state_topic, String, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=20)
        self.color_sub = rospy.Subscriber(self.color_topic, Image, self._color_cb, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)

        self.latest_color_rgb = None
        self.latest_color_stamp = None
        self.latest_color_frame = self.default_camera_frame
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_depth_frame = self.default_camera_frame

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.palm_history = deque(maxlen=max(1, self.stability_window))
        self._stable_handedness = "Unknown"
        self._pending_handedness = "Unknown"
        self._pending_handedness_count = 0
        self._smoothed_palm_normal = None
        self.processing = False
        self._reload_handover_config_if_needed(force=True)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1e-3, self.process_hz)), self._process_timer)

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "[handover] 啟動完成，發布 hand_state: %s, markers: %s，交接區 min=%s max=%s",
            self.hand_state_topic,
            self.marker_topic,
            self.zone_min.tolist(),
            self.zone_max.tolist(),
        )

    def _get_param(self, private_name, shared_name, default):
        if private_name and rospy.has_param(private_name):
            return rospy.get_param(private_name)
        if shared_name and rospy.has_param(shared_name):
            return rospy.get_param(shared_name)
        return default

    def _shutdown(self):
        try:
            self.detector.close()
        except Exception:
            pass

    def _load_handover_config(self):
        if not self.handover_config_path or not os.path.exists(self.handover_config_path):
            return None
        try:
            with open(self.handover_config_path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[handover] 讀取 handover config 失敗: %s", exc)
            return None
        data = payload.get("semantic_handover")
        return data if isinstance(data, dict) else None

    def _reload_handover_config_if_needed(self, force=False):
        if not self.handover_config_path or not os.path.exists(self.handover_config_path):
            return
        try:
            mtime = os.path.getmtime(self.handover_config_path)
        except OSError:
            return
        if not force and self._handover_config_mtime == mtime:
            return

        cfg = self._load_handover_config()
        if cfg is None:
            return

        center = np.array(cfg.get("handover_zone_center_xyz", [0.1606, 0.2881, 0.1554]), dtype=float)
        half_extents = np.array(cfg.get("handover_zone_half_extents_xyz", [0.10, 0.10, 0.08]), dtype=float)
        zone_min = np.array(cfg.get("handover_zone_min_xyz", (center - half_extents).tolist()), dtype=float)
        zone_max = np.array(cfg.get("handover_zone_max_xyz", (center + half_extents).tolist()), dtype=float)
        zone_visible = bool(cfg.get("handover_zone_visible", self.handover_zone_visible))
        zone_color_rgba = np.array(
            cfg.get("handover_zone_color_rgba", self.handover_zone_color_rgba.tolist()),
            dtype=float,
        )
        swap_handedness_labels = bool(
            cfg.get("swap_handedness_labels", self.swap_handedness_labels))
        right_hand_normal_sign = float(
            cfg.get("right_hand_normal_sign", self.right_hand_normal_sign))
        left_hand_normal_sign = float(
            cfg.get("left_hand_normal_sign", self.left_hand_normal_sign))
        handedness_debounce_frames = int(
            cfg.get("handedness_debounce_frames", self.handedness_debounce_frames))
        palm_normal_ema_alpha = float(
            cfg.get("palm_normal_ema_alpha", self.palm_normal_ema_alpha))

        self.zone_min = zone_min
        self.zone_max = zone_max
        self.handover_zone_visible = zone_visible
        self.handover_zone_color_rgba = zone_color_rgba
        self.swap_handedness_labels = swap_handedness_labels
        self.right_hand_normal_sign = right_hand_normal_sign
        self.left_hand_normal_sign = left_hand_normal_sign
        self.handedness_debounce_frames = max(1, handedness_debounce_frames)
        self.palm_normal_ema_alpha = float(np.clip(palm_normal_ema_alpha, 0.0, 1.0))
        self._handover_config_mtime = mtime
        rospy.loginfo(
            "[handover] 已重載 handover config: zone min=%s max=%s visible=%s color=%s swap_handedness=%s signs=(R:%s,L:%s) debounce=%d ema=%.2f",
            self.zone_min.tolist(),
            self.zone_max.tolist(),
            self.handover_zone_visible,
            self.handover_zone_color_rgba.tolist(),
            self.swap_handedness_labels,
            self.right_hand_normal_sign,
            self.left_hand_normal_sign,
            self.handedness_debounce_frames,
            self.palm_normal_ema_alpha,
        )

    def _camera_info_cb(self, msg: CameraInfo):
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

    def _color_cb(self, msg: Image):
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        except ValueError as e:
            rospy.logwarn_throttle(5.0, "[handover] color reshape 失敗: %s", e)
            return

        if msg.encoding == "rgb8":
            rgb = img
        elif msg.encoding == "bgr8":
            rgb = img[:, :, ::-1]
        else:
            rospy.logwarn_throttle(5.0, "[handover] 不支援的 color encoding: %s", msg.encoding)
            return

        self.latest_color_rgb = np.ascontiguousarray(rgb)
        self.latest_color_stamp = msg.header.stamp
        if msg.header.frame_id:
            self.latest_color_frame = msg.header.frame_id

    def _depth_cb(self, msg: Image):
        if msg.encoding not in ("16UC1", "mono16"):
            rospy.logwarn_throttle(5.0, "[handover] 不支援的 depth encoding: %s", msg.encoding)
            return

        try:
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        except ValueError as e:
            rospy.logwarn_throttle(5.0, "[handover] depth reshape 失敗: %s", e)
            return

        self.latest_depth = depth
        self.latest_depth_stamp = msg.header.stamp
        if msg.header.frame_id:
            self.latest_depth_frame = msg.header.frame_id

    def _process_timer(self, _event):
        self._reload_handover_config_if_needed()
        if self.processing:
            return
        if self.latest_color_rgb is None or self.latest_depth is None:
            return
        if self.fx is None or self.fy is None:
            return

        self.processing = True
        try:
            state, markers = self._build_hand_state()
            self.hand_state_pub.publish(json.dumps(state))
            self._publish_markers(markers)
        except Exception as e:
            rospy.logerr_throttle(2.0, "[handover] hand_state 建立失敗: %s", e)
        finally:
            self.processing = False

    def _build_hand_state(self):
        color_rgb = self.latest_color_rgb.copy()
        depth = self.latest_depth.copy()
        stamp = self.latest_color_stamp if self.latest_color_stamp is not None else rospy.Time.now()
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb))

        state = {
            "valid": False,
            "frame_id": self.base_frame,
            "stamp": stamp.to_sec(),
            "palm_center_3d": None,
            "palm_normal_3d": None,
            "stability": 0.0,
            "in_zone": False,
            "reason": "no_hand",
            "num_hands": len(result.hand_landmarks),
        }
        markers = self._build_base_markers(state)

        if len(result.hand_landmarks) == 0:
            self.palm_history.clear()
            self._smoothed_palm_normal = None
            return state, markers
        if len(result.hand_landmarks) > 1:
            state["reason"] = "multiple_hands"
            self.palm_history.clear()
            self._smoothed_palm_normal = None
            return state, self._build_base_markers(state)

        hand_landmarks = result.hand_landmarks[0]
        h, w = depth.shape[:2]

        palm_pixels = []
        for idx in PALM_LANDMARK_IDS:
            lm = hand_landmarks[idx]
            palm_pixels.append([lm.x * w, lm.y * h])
        palm_pixels = np.array(palm_pixels, dtype=np.float32)
        palm_center_px = np.mean(palm_pixels, axis=0)
        u = int(np.clip(round(float(palm_center_px[0])), 0, w - 1))
        v = int(np.clip(round(float(palm_center_px[1])), 0, h - 1))

        depth_m = self._sample_depth_m(depth, u, v)
        if depth_m is None:
            state["reason"] = "invalid_depth"
            self.palm_history.clear()
            self._smoothed_palm_normal = None
            return state, self._build_base_markers(state)

        point_cam = np.array([
            (u - self.cx) * depth_m / self.fx,
            (v - self.cy) * depth_m / self.fy,
            depth_m,
        ], dtype=float)
        point_base = self._transform_point_to_base(point_cam, stamp)
        if point_base is None:
            state["reason"] = "tf_unavailable"
            self.palm_history.clear()
            self._smoothed_palm_normal = None
            return state, self._build_base_markers(state)

        landmark_points_base = self._compute_landmark_points_in_base(
            hand_landmarks, depth, stamp)
        raw_handedness = (
            result.handedness[0][0].category_name
            if result.handedness and result.handedness[0] else "Unknown")
        handedness = self._stabilize_handedness(self._normalize_handedness(raw_handedness))
        palm_normal = self._compute_palm_normal(point_base, landmark_points_base, handedness)

        in_zone = bool(np.all(point_base >= self.zone_min) and np.all(point_base <= self.zone_max))
        if in_zone:
            self.palm_history.append(point_base)
            stability = self._compute_stability()
        else:
            self.palm_history.clear()
            stability = 0.0

        state.update({
            "valid": True,
            "palm_center_3d": point_base.tolist(),
            "palm_normal_3d": palm_normal.tolist() if palm_normal is not None else None,
            "stability": stability,
            "in_zone": in_zone,
            "reason": "ok" if in_zone else "out_of_zone",
            "palm_center_px": [u, v],
            "depth_m": depth_m,
            "handedness_raw": raw_handedness,
            "handedness": handedness,
        })
        return state, self._build_base_markers(
            state, landmark_points_base=landmark_points_base)

    def _sample_depth_m(self, depth, u, v):
        radius = max(0, self.depth_sample_radius_px)
        x1 = max(0, u - radius)
        x2 = min(depth.shape[1], u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(depth.shape[0], v + radius + 1)

        patch = depth[y1:y2, x1:x2]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        depth_m = float(np.median(valid) / self.depth_scale)
        if depth_m < self.min_valid_depth_m or depth_m > self.max_valid_depth_m:
            return None
        return depth_m

    def _transform_point_to_base(self, point_cam_xyz, stamp):
        point_msg = PointStamped()
        point_msg.header.frame_id = self.latest_depth_frame or self.latest_color_frame
        point_msg.header.stamp = stamp
        point_msg.point.x = float(point_cam_xyz[0])
        point_msg.point.y = float(point_cam_xyz[1])
        point_msg.point.z = float(point_cam_xyz[2])
        try:
            point_base = self.tfbuf.transform(
                point_msg, self.base_frame, rospy.Duration(0.2))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[handover] TF 轉換失敗: %s", e)
            return None

        return np.array([
            point_base.point.x,
            point_base.point.y,
            point_base.point.z,
        ], dtype=float)

    def _compute_stability(self):
        if not self.palm_history:
            return 0.0

        pts = np.array(self.palm_history, dtype=float)
        center = np.mean(pts, axis=0)
        max_dev = float(np.max(np.linalg.norm(pts - center, axis=1)))
        motion_score = max(0.0, 1.0 - max_dev / max(1e-6, self.stability_pos_threshold))
        count_score = min(1.0, len(self.palm_history) / float(max(1, self.stability_window)))
        return float(motion_score * count_score)

    def _compute_landmark_points_in_base(self, hand_landmarks, depth, stamp):
        h, w = depth.shape[:2]
        points = []
        for lm in hand_landmarks:
            u = int(np.clip(round(float(lm.x * w)), 0, w - 1))
            v = int(np.clip(round(float(lm.y * h)), 0, h - 1))
            depth_m = self._sample_depth_m(depth, u, v)
            if depth_m is None:
                points.append(None)
                continue

            point_cam = np.array([
                (u - self.cx) * depth_m / self.fx,
                (v - self.cy) * depth_m / self.fy,
                depth_m,
            ], dtype=float)
            point_base = self._transform_point_to_base(point_cam, stamp)
            if point_base is None:
                points.append(None)
                continue
            points.append(point_base)
        return points

    def _normalize_handedness(self, handedness):
        label = str(handedness).strip().capitalize()
        if label not in ("Left", "Right"):
            return "Unknown"
        if not self.swap_handedness_labels:
            return label
        return "Left" if label == "Right" else "Right"

    def _stabilize_handedness(self, handedness):
        if handedness not in ("Left", "Right"):
            self._pending_handedness = "Unknown"
            self._pending_handedness_count = 0
            return self._stable_handedness
        if handedness == self._stable_handedness:
            self._pending_handedness = handedness
            self._pending_handedness_count = 0
            return self._stable_handedness
        if handedness != self._pending_handedness:
            self._pending_handedness = handedness
            self._pending_handedness_count = 1
        else:
            self._pending_handedness_count += 1
        if self._pending_handedness_count >= self.handedness_debounce_frames:
            self._stable_handedness = handedness
            self._pending_handedness = handedness
            self._pending_handedness_count = 0
        return self._stable_handedness

    def _smooth_palm_normal(self, normal):
        if normal is None:
            self._smoothed_palm_normal = None
            return None
        current = np.array(normal, dtype=float)
        norm = np.linalg.norm(current)
        if norm < 1e-6:
            return None
        current = current / norm

        if self._smoothed_palm_normal is None:
            self._smoothed_palm_normal = current
            return current

        previous = np.array(self._smoothed_palm_normal, dtype=float)
        prev_norm = np.linalg.norm(previous)
        if prev_norm < 1e-6:
            self._smoothed_palm_normal = current
            return current
        previous = previous / prev_norm

        if float(np.dot(current, previous)) < 0.0:
            current = -current

        alpha = float(np.clip(self.palm_normal_ema_alpha, 0.0, 1.0))
        blended = (1.0 - alpha) * previous + alpha * current
        blend_norm = np.linalg.norm(blended)
        if blend_norm < 1e-6:
            blended = current
            blend_norm = np.linalg.norm(blended)
        blended = blended / max(blend_norm, 1e-6)
        self._smoothed_palm_normal = blended
        return blended

    def _compute_palm_normal(self, palm_center, landmark_points_base, handedness):
        required_ids = (0, 5, 9, 17)
        if landmark_points_base is None:
            return None
        if any(idx >= len(landmark_points_base) or landmark_points_base[idx] is None for idx in required_ids):
            return None

        wrist = np.array(landmark_points_base[0], dtype=float)
        index_mcp = np.array(landmark_points_base[5], dtype=float)
        middle_mcp = np.array(landmark_points_base[9], dtype=float)
        pinky_mcp = np.array(landmark_points_base[17], dtype=float)

        across = pinky_mcp - index_mcp
        forward = middle_mcp - wrist
        if np.linalg.norm(across) < 1e-6 or np.linalg.norm(forward) < 1e-6:
            return None

        normal = np.cross(across, forward)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        normal = normal / norm
        if handedness == "Right":
            sign = self.right_hand_normal_sign
        elif handedness == "Left":
            sign = self.left_hand_normal_sign
        else:
            sign = 1.0
        if sign < 0.0:
            normal = -normal
        return self._smooth_palm_normal(normal)

    def _publish_markers(self, markers):
        delete_all = Marker()
        delete_all.header.frame_id = self.base_frame
        delete_all.header.stamp = rospy.Time.now()
        delete_all.action = Marker.DELETEALL
        self.marker_pub.publish(delete_all)
        for marker in markers:
            self.marker_pub.publish(marker)

    def _build_base_markers(self, state, landmark_points_base=None):
        stamp = rospy.Time.now()
        markers = [self._make_status_text_marker(stamp, state)]

        if self.handover_zone_visible:
            markers.append(self._make_zone_marker(stamp))

        palm_center = state.get("palm_center_3d")
        if isinstance(palm_center, (list, tuple)) and len(palm_center) == 3:
            markers.append(self._make_palm_marker(stamp, palm_center, state))
            palm_normal = state.get("palm_normal_3d")
            if isinstance(palm_normal, (list, tuple)) and len(palm_normal) == 3:
                markers.append(self._make_palm_normal_marker(stamp, palm_center, palm_normal, state))

        if landmark_points_base:
            markers.append(self._make_landmark_marker(stamp, landmark_points_base, state))
            markers.append(self._make_skeleton_marker(stamp, landmark_points_base, state))

        return markers

    def _marker_template(self, stamp, marker_id, ns, marker_type):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(self.marker_lifetime_sec)
        return marker

    def _make_zone_marker(self, stamp):
        marker = self._marker_template(stamp, 1, "handover_zone", Marker.CUBE)
        center = (self.zone_min + self.zone_max) / 2.0
        size = self.zone_max - self.zone_min
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])
        marker.scale.x = float(size[0])
        marker.scale.y = float(size[1])
        marker.scale.z = float(size[2])
        color = np.array(self.handover_zone_color_rgba, dtype=float).flatten()
        if color.size < 4:
            color = np.array([0.1, 0.5, 1.0, 0.3], dtype=float)
        marker.color.r = float(np.clip(color[0], 0.0, 1.0))
        marker.color.g = float(np.clip(color[1], 0.0, 1.0))
        marker.color.b = float(np.clip(color[2], 0.0, 1.0))
        marker.color.a = float(np.clip(color[3], 0.0, 1.0))
        return marker

    def _make_palm_marker(self, stamp, palm_center, state):
        marker = self._marker_template(stamp, 2, "handover_palm", Marker.SPHERE)
        marker.pose.position.x = float(palm_center[0])
        marker.pose.position.y = float(palm_center[1])
        marker.pose.position.z = float(palm_center[2])
        marker.scale.x = self.palm_marker_scale
        marker.scale.y = self.palm_marker_scale
        marker.scale.z = self.palm_marker_scale
        stable = bool(state.get("in_zone", False)) and float(state.get("stability", 0.0)) > 0.0
        if stable:
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
        else:
            marker.color.r = 1.0
            marker.color.g = 0.7
            marker.color.b = 0.2
        marker.color.a = 0.95
        return marker

    def _make_palm_normal_marker(self, stamp, palm_center, palm_normal, state):
        marker = self._marker_template(stamp, 5, "handover_palm_normal", Marker.ARROW)
        start = np.array(palm_center, dtype=float)
        direction = np.array(palm_normal, dtype=float)
        length = 0.10
        end = start + direction * length

        p1 = PointStamped().point
        p1.x = float(start[0])
        p1.y = float(start[1])
        p1.z = float(start[2])
        p2 = PointStamped().point
        p2.x = float(end[0])
        p2.y = float(end[1])
        p2.z = float(end[2])
        marker.points = [p1, p2]
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.03
        if bool(state.get("in_zone", False)):
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 1.0
        else:
            marker.color.r = 0.7
            marker.color.g = 0.7
            marker.color.b = 1.0
        marker.color.a = 0.95
        return marker

    def _make_landmark_marker(self, stamp, landmark_points_base, state):
        marker = self._marker_template(stamp, 3, "handover_landmarks", Marker.SPHERE_LIST)
        marker.scale.x = self.landmark_marker_scale
        marker.scale.y = self.landmark_marker_scale
        marker.scale.z = self.landmark_marker_scale
        for point in landmark_points_base:
            if point is None:
                continue
            geom_point = PointStamped().point
            geom_point.x = float(point[0])
            geom_point.y = float(point[1])
            geom_point.z = float(point[2])
            marker.points.append(geom_point)
        if bool(state.get("in_zone", False)):
            marker.color.r = 0.1
            marker.color.g = 0.95
            marker.color.b = 0.2
        else:
            marker.color.r = 0.8
            marker.color.g = 0.8
            marker.color.b = 0.2
        marker.color.a = 0.9
        return marker

    def _make_skeleton_marker(self, stamp, landmark_points_base, state):
        marker = self._marker_template(stamp, 4, "handover_skeleton", Marker.LINE_LIST)
        marker.scale.x = self.skeleton_line_width
        if bool(state.get("in_zone", False)):
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.7
        else:
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.15
        marker.color.a = 0.95
        for a_idx, b_idx in HAND_CONNECTIONS:
            if a_idx >= len(landmark_points_base) or b_idx >= len(landmark_points_base):
                continue
            pa = landmark_points_base[a_idx]
            pb = landmark_points_base[b_idx]
            if pa is None or pb is None:
                continue
            p1 = PointStamped().point
            p1.x = float(pa[0])
            p1.y = float(pa[1])
            p1.z = float(pa[2])
            p2 = PointStamped().point
            p2.x = float(pb[0])
            p2.y = float(pb[1])
            p2.z = float(pb[2])
            marker.points.extend([p1, p2])
        return marker

    def _make_status_text_marker(self, stamp, state):
        marker = self._marker_template(stamp, 5, "handover_status", Marker.TEXT_VIEW_FACING)
        zone_center = (self.zone_min + self.zone_max) / 2.0
        marker.pose.position.x = float(zone_center[0])
        marker.pose.position.y = float(zone_center[1])
        marker.pose.position.z = float(self.zone_max[2] + 0.08)
        marker.scale.z = 0.035
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = (
            f"handover: {state.get('reason', 'unknown')}"
            f" | stable={float(state.get('stability', 0.0)):.2f}"
            f" | in_zone={bool(state.get('in_zone', False))}"
        )
        return marker


if __name__ == "__main__":
    HandoverPerceptionNode()
    rospy.spin()
