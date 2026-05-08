# Gripper Scale Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make SmolVLA, PI0, and PI0.5 automatically adapt gripper scaling based on checkpoint stats.

**Architecture:** Add a shared `inference_sdk.policy.gripper_scale` helper that infers whether a feature's last dimension is unit-scaled. Import that helper from ACT and the VLA engines, then gate gripper `/1000` and `*1000` conversions with per-engine state/action flags.

**Tech Stack:** Python, NumPy, PyTorch, existing policy engine modules.

---

### Task 1: Shared Helper

**Files:**
- Create: `inference_sdk/policy/gripper_scale.py`
- Modify: `inference_sdk/policy/act.py`

- [x] Write a temporary Python assertion that imports `feature_gripper_stats_are_unit_scaled` and verifies it returns `True` for `[0,1]` stats and `False` for robot-space stats; it should fail before the helper exists.
- [x] Create `inference_sdk/policy/gripper_scale.py` with `feature_gripper_stats_are_unit_scaled(stats, feature_name)` using ACT's existing heuristic.
- [x] Replace ACT's local `_feature_gripper_stats_are_unit_scaled` function with the shared helper import and update call sites.
- [x] Run the temporary Python assertion and ensure it passes.

### Task 2: VLA Engine Flags

**Files:**
- Modify: `inference_sdk/policy/smolvla.py`
- Modify: `inference_sdk/policy/pi0.py`
- Modify: `inference_sdk/policy/pi05.py`

- [x] Write temporary Python assertions that instantiate each engine with synthetic stats and no model load, then verify unit stats perform `/1000` and `*1000` while robot-space stats skip those conversions; these assertions should fail before implementation.
- [x] Add `_state_gripper_stats_unit_scaled` and `_action_gripper_stats_unit_scaled` fields to each engine.
- [x] Set those fields after stats load using `feature_gripper_stats_are_unit_scaled` for `observation.state` and `action`.
- [x] Gate the hard-coded `/1000` and `*1000` conversions on the new flags.
- [x] Add log messages matching ACT's gripper scale reporting.
- [x] Run the temporary Python assertions and ensure they pass.

### Task 3: Documentation and Verification

**Files:**
- Modify: `README.md`

- [x] Update the README gripper-scale note to say ACT, SmolVLA, PI0, and PI0.5 auto-adapt checkpoint stats while preserving robot-space SDK inputs/outputs.
- [x] Run `python -m compileall inference_sdk` to catch syntax/import errors.
- [x] Run the temporary Python assertions one final time.
