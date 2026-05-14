# Project Status - 2026-05-14

## Summary

This workspace has moved beyond the original click-to-pick prototype and now
has a working UR3 semantic grasp pipeline plus a first usable handover path.

Current project state:

- Main operator flow is manual multi-node startup, not `start_grasp.sh`
- The baseline semantic grasp pipeline is usable
- Shape-aware AnyGrasp filtering is already merged into the main server
- Dynamic handover perception and force-triggered release are implemented
- A server-side VLM migration path exists as a test branch/script, not yet the
  default production server

## Completed

### 1. Core semantic grasp pipeline

The main grasp pipeline is in place:

- `src/ur3_handover/scripts/client_camera.py`
- `server_anygrasp.py`
- `src/ur3_handover/scripts/semantic_grasp_controller.py`

Implemented behavior:

- Camera-side target selection
- Remote AnyGrasp inference through ZMQ
- Grasp pose publication back into ROS
- MoveIt-based pre-grasp, approach, close, lift, and follow-up motion

### 2. Shape-aware grasp filtering

`server_anygrasp.py` already contains geometry-aware post-filtering based on
`object_shape`:

- `sphere`
- `box`
- `cylinder`

This means the original shape-aware grasp plan is partially landed in the main
runtime server, not just on paper.

### 3. Client-side workflow refactor

`src/ur3_handover/scripts/client_camera.py` has been refactored away from the
older "local brain node must finish first" interaction style.

Current direction:

- VLM target name can be sent directly to a remote server flow
- Manual ROI mode still exists
- Previous VLM result can be reused for regrasp (`[s]`)
- MoveIt initialization can be skipped with `~use_moveit:=false` for testing

### 4. Handover path

Handover is no longer only a plan. There is implemented runtime support in:

- `src/ur3_handover/scripts/handover_perception.py`
- `src/ur3_handover/scripts/semantic_grasp_controller.py`
- `src/ur3_handover/config/handover_params.yaml`

Implemented behavior:

- MediaPipe-based hand detection
- Palm center 3D estimation
- Handover zone filtering
- Stability gating before approach
- Dynamic handover target resolution
- Force-based release trigger
- Config-driven tuning reload

### 5. Documentation split

Documentation roles are now clearer:

- `hackmd.md`: operator manual
- `note.md`: on-site quick reference
- `CLAUDE.md`: repo map for coding agents
- `docs/REPO_SETUP_AND_SUBMODULE_GUIDE.md`: repo/submodule workflow

## In Progress

### 1. Server-side VLM migration

There are two test scripts related to moving the VLM pipeline to the server:

- `server_anygrasp_vlm_test.py`
- `src/ur3_handover/scripts/server_anygrasp_vlm_test.py`

Status:

- The `src/ur3_handover/scripts/` version is the more advanced one
- It includes server-side OWL-v2, SAM, Gemini, SVD fill, reuse-VLM flow, and
  richer timing/logging
- This is still not the default production server path
- The root-level copy looks like an older duplicate and should eventually be
  merged or removed

### 2. Runtime convergence

The repo currently contains two practical operating modes:

- Current baseline production-like flow:
  local `brain_node.py` + local `client_camera.py` + remote `server_anygrasp.py`
- Migration/test flow:
  direct VLM target into remote `server_anygrasp_vlm_test.py`

This needs one final decision so the repo has a single source of truth.

## Known Gaps

### 1. Duplicate server test scripts

There are two untracked VLM server test files with overlapping purpose.

Action needed:

- keep the `src/ur3_handover/scripts/` version as the likely main candidate
- remove or archive the root-level duplicate after verification

### 2. Documentation debt in plan files

Not every plan in `docs/` is still equally useful.

Current assessment:

- Safe to delete if history is not needed:
  `docs/PLAN_VLM_SAM_INTEGRATION.md`
- Keep for now:
  `docs/SERVER_UPGRADE_PLAN.md`
- Keep for now, but partially landed already:
  `docs/PLAN_SHAPE_AWARE_GRASP.md`
- Strong overlap with shape-aware work; consider archive/merge instead of
  keeping as an active plan:
  `docs/GRASP_TILT_SOLUTION.md`
- Keep:
  `docs/UR3_HANDOVER_DEVELOPMENT_PLAN.md`

### 3. Startup truth is manual, not scripted

`start_grasp.sh` is not the reliable source of truth. The real supported path
today is manual node startup as documented in `hackmd.md` and `note.md`.

## Recommended Next Steps

1. Choose the official runtime path:
   keep `brain_node.py` as the main path, or promote the server-side VLM test
   server into the main server.
2. Remove duplication:
   merge or delete the extra `server_anygrasp_vlm_test.py`.
3. Clean up docs:
   delete or archive completed plan files and keep only current references.
4. Validate end-to-end:
   run one confirmed on-robot semantic grasp test and one handover test using
   the current documented startup sequence.

## File Notes

- `remote_test_bundle/` and `remote_test_bundle.tar.gz` appear to be prepared
  remote validation assets for the server-side VLM migration.
- `client_camera.py` currently contains a hardcoded Ngrok/TCP address and still
  requires session-by-session refresh.
- `src/anygrasp_sdk` remains an external submodule and should be treated
  separately from main-repo edits.
