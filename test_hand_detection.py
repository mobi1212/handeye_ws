#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 MediaPipe HandLandmarker + RealSense D435
按 [q] 離開
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
import pyrealsense2 as rs

MODEL_PATH = "/home/weilun/handeye_ws/models/hand_landmarker.task"

# --- MediaPipe HandLandmarker ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)
detector = HandLandmarker.create_from_options(options)

# --- RealSense ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

pipeline.start(config)
print("RealSense started. Press [q] to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        # MediaPipe 要 RGB
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        display = color_img.copy()
        h, w = display.shape[:2]

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name  # Left/Right

            # 畫 21 個 landmarks
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (px, py), 3, (0, 255, 0), -1)

            # 畫連線
            connections = [
                (0,1),(1,2),(2,3),(3,4),       # thumb
                (0,5),(5,6),(6,7),(7,8),        # index
                (0,9),(9,10),(10,11),(11,12),   # middle
                (0,13),(13,14),(14,15),(15,16), # ring
                (0,17),(17,18),(18,19),(19,20), # pinky
                (5,9),(9,13),(13,17),           # palm
            ]
            for a, b in connections:
                pa = (int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h))
                pb = (int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h))
                cv2.line(display, pa, pb, (0, 200, 0), 1)

            # 手心中心（landmark 0=wrist, 9=middle_finger_mcp 的中點）
            wrist = hand_landmarks[0]
            mcp = hand_landmarks[9]
            cx = int((wrist.x + mcp.x) / 2 * w)
            cy = int((wrist.y + mcp.y) / 2 * h)

            # 深度
            depth_mm = depth_img[min(cy, h-1), min(cx, w-1)]
            depth_m = depth_mm / 1000.0

            cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)
            label = f"{handedness} d={depth_m:.2f}m"
            cv2.putText(display, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 狀態列
        n_hands = len(result.hand_landmarks)
        status = f"Hands: {n_hands}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Hand Detection Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    detector.close()
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")
