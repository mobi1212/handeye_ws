#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline grasp-side debugger.

Purpose:
- replay one AnyGrasp result without a robot
- transform the grasp pose from camera frame to base frame
- apply the same controller mapping used in semantic_grasp_controller.py
- compute ee_z / grasp_xyz / pre_xyz
- check whether pre-grasp lands on the camera side or the far side

Inputs:
- Either a saved server result json containing translation/rotation
- Or manual --translation + --rotation

Example:
  python3 offline_grasp_debug.py \
    --translation 0.12 0.03 0.42 \
    --rotation 1 0 0 0 1 0 0 0 1

  python3 offline_grasp_debug.py \
    --result-json /path/to/result.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml


DEFAULT_HAND_EYE = os.path.expanduser(
    "~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml"
)


def normalize(vec, eps=1e-9):
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("zero-length vector")
    return vec / norm


def quaternion_to_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=float)


def matrix_to_quaternion(m):
    t = np.trace(m)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return normalize(q)


def euler_y_matrix(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def load_hand_eye(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    t = data["transformation"]
    q = np.array([t["qx"], t["qy"], t["qz"], t["qw"]], dtype=float)
    trans = np.array([t["x"], t["y"], t["z"]], dtype=float)
    rot = quaternion_to_matrix(q)
    return rot, trans


def load_result(args):
    if args.result_json:
        with open(args.result_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        translation = np.array(data["translation"], dtype=float)
        rotation = np.array(data["rotation"], dtype=float).reshape(3, 3)
        return translation, rotation

    translation = np.array(args.translation, dtype=float)
    rotation = np.array(args.rotation, dtype=float).reshape(3, 3)
    return translation, rotation


def fmt(vec):
    return "(" + ", ".join(f"{v:.3f}" for v in vec) + ")"


def main():
    parser = argparse.ArgumentParser(description="Offline grasp-side debugger")
    parser.add_argument("--result-json", help="server saved result json with translation/rotation")
    parser.add_argument("--translation", nargs=3, type=float, help="manual translation x y z")
    parser.add_argument(
        "--rotation",
        nargs=9,
        type=float,
        help="manual 3x3 rotation matrix in row-major order",
    )
    parser.add_argument("--hand-eye", default=DEFAULT_HAND_EYE, help="easy_handeye yaml path")
    parser.add_argument("--tcp-offset", type=float, default=0.18)
    parser.add_argument("--grasp-depth", type=float, default=0.05)
    parser.add_argument("--approach-dist", type=float, default=0.05)
    parser.add_argument(
        "--q-to-ur3-y-deg",
        type=float,
        default=-90.0,
        help="current controller uses Y -90 deg",
    )
    args = parser.parse_args()

    if not args.result_json and (args.translation is None or args.rotation is None):
        parser.error("provide either --result-json or both --translation and --rotation")

    hand_eye_path = Path(args.hand_eye)
    if not hand_eye_path.exists():
        parser.error(f"hand-eye file not found: {hand_eye_path}")

    t_cam_obj, r_cam_obj = load_result(args)
    r_base_cam, t_base_cam = load_hand_eye(str(hand_eye_path))

    # Transform AnyGrasp result from camera to base.
    r_base_obj = r_base_cam @ r_cam_obj
    t_base_obj = r_base_cam @ t_cam_obj + t_base_cam

    # Match semantic_grasp_controller.py mapping.
    r_to_ur3 = euler_y_matrix(args.q_to_ur3_y_deg)
    r_final = r_base_obj @ r_to_ur3
    ee_z = normalize(r_final[:, 2])

    object_surface = t_base_obj
    camera_pos = t_base_cam
    camera_to_object = normalize(object_surface - camera_pos)
    object_to_camera = normalize(camera_pos - object_surface)

    grasp_xyz = object_surface + ee_z * args.grasp_depth - ee_z * args.tcp_offset
    pre_xyz = grasp_xyz - ee_z * args.approach_dist

    flipped_ee_z = -ee_z
    grasp_xyz_flip = object_surface + flipped_ee_z * args.grasp_depth - flipped_ee_z * args.tcp_offset
    pre_xyz_flip = grasp_xyz_flip - flipped_ee_z * args.approach_dist

    print("=== Offline Grasp Debug ===")
    print(f"hand_eye: {hand_eye_path}")
    print(f"camera_pos(base)      = {fmt(camera_pos)}")
    print(f"object_surface(base)  = {fmt(object_surface)}")
    print(f"camera_to_object(unit)= {fmt(camera_to_object)}")
    print(f"object_to_camera(unit)= {fmt(object_to_camera)}")
    print("")

    print("[Current controller mapping]")
    print(f"ee_z                  = {fmt(ee_z)}")
    print(f"dot(ee_z, camera->obj)= {float(np.dot(ee_z, camera_to_object)):.3f}")
    print(f"grasp_xyz             = {fmt(grasp_xyz)}")
    print(f"pre_xyz               = {fmt(pre_xyz)}")
    print(f"obj->pre(unit)        = {fmt(normalize(pre_xyz - object_surface))}")
    print("")

    print("[Flipped-side comparison]")
    print(f"ee_z_flipped          = {fmt(flipped_ee_z)}")
    print(f"dot(flip, camera->obj)= {float(np.dot(flipped_ee_z, camera_to_object)):.3f}")
    print(f"grasp_xyz_flipped     = {fmt(grasp_xyz_flip)}")
    print(f"pre_xyz_flipped       = {fmt(pre_xyz_flip)}")
    print(f"obj->pre_flip(unit)   = {fmt(normalize(pre_xyz_flip - object_surface))}")
    print("")

    current_score = float(np.dot(ee_z, camera_to_object))
    flipped_score = float(np.dot(flipped_ee_z, camera_to_object))

    if current_score >= flipped_score:
        print("Diagnosis: current ee_z is more aligned with camera->object.")
        print("If grasp still approaches from the far side, inspect offset formula usage and RViz pose frame.")
    else:
        print("Diagnosis: current ee_z points away from the camera-facing side.")
        print("Most likely issue: controller-side sign / axis mapping is flipped.")


if __name__ == "__main__":
    main()
