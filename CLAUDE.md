# CLAUDE.md

This file provides guidance to Claude Code and other coding agents working in
this repository.

## Documentation Map

- User-facing operation manual: `hackmd.md`
- On-site quick reference: `note.md`
- Repo/submodule setup: `docs/REPO_SETUP_AND_SUBMODULE_GUIDE.md`
- Handover development plan: `docs/UR3_HANDOVER_DEVELOPMENT_PLAN.md`

Do not duplicate long operator instructions here. If a task is about running
the system, refer to the documents above first.

## Repository Overview

This workspace centers on a UR3 semantic grasp and handover pipeline:

- `src/ur3_handover/scripts/client_camera.py`
  Camera UI, VLM trigger, mask handling, ZMQ client to AnyGrasp server
- `src/ur3_handover/scripts/semantic_grasp_controller.py`
  MoveIt execution, grasp state machine, release behavior
- `src/ur3_handover/scripts/handover_perception.py`
  MediaPipe hand perception, palm center / normal, handover markers
- `src/ur3_handover/config/handover_params.yaml`
  Field-tunable handover zone, offsets, release thresholds, handedness tuning

## Build And Runtime Baseline

```bash
cd /home/weilun/handeye_ws
catkin_make
source devel/setup.bash
```

Important runtime assumptions:

- Robot IP is `192.168.86.7`
- Hand-eye calibration file lives at
  `~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`
- `start_grasp.sh` is currently not the source of truth; manual node startup is
  the supported path
- `handover_perception.py` should be run with system Python, not the AnyGrasp
  conda environment

## External Dependencies

- `src/anygrasp_sdk` is an external Git submodule
- Initialize it with:

```bash
git submodule update --init --recursive
```

- If the submodule looks modified in the main repo, check both:

```bash
git status
git -C src/anygrasp_sdk status
```

For submodule workflow details, use
`docs/REPO_SETUP_AND_SUBMODULE_GUIDE.md`.

## Development Notes

- Handover tuning should go into `handover_params.yaml` whenever possible,
  rather than hardcoding site-specific values in scripts.
- `client_camera.py` contains an Ngrok/TCP server address that may need manual
  refresh depending on the current remote AnyGrasp server session.
- There is still legacy repository debt around `src/graspnetAPI` gitlink /
  submodule metadata inconsistency. Treat that area carefully and avoid cleanup
  unless the task is specifically about repository structure.
