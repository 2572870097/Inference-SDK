# Gripper Scale Adaptation Design

## Goal

Make SmolVLA, PI0, and PI0.5 handle checkpoint gripper statistics stored either in normalized `[0, 1]` scale or robot-space scale, matching the behavior already present in ACT.

## Decision

Keep the public SDK contract unchanged: callers pass robot state/action in robot space, where Alicia-M style grippers use `[0, 1000]`. Policy engines will inspect the last dimension of `observation.state` and `action` stats. If the stats look unit-scaled, engines convert robot-space gripper values to/from `[0, 1]`; if stats are already robot-space, engines skip the extra conversion.

## Components

- Add a shared policy helper for gripper scale inference so ACT, SmolVLA, PI0, and PI0.5 do not duplicate the same heuristic.
- Update SmolVLA to set state/action gripper scale flags after stats load and gate the existing `/1000` and `*1000` conversions on those flags.
- Update PI0 and PI0.5 with the same flags while preserving their existing feature normalization flow.
- Update README to clarify that all four engines auto-adapt checkpoint gripper stats but still expose robot-space gripper values at the SDK boundary.

## Error Handling

If stats are missing, preserve the current conservative behavior and assume unit-scaled checkpoint stats. This keeps older checkpoints and Alicia-M demos compatible. Non-finite state/action values continue to be handled by existing callers and runtime validation.

## Testing

Use one-off Python assertions instead of adding a permanent test suite, because this repository currently has no tests directory. Verify the helper detects `[0, 1]` stats and robot-space stats, then verify representative SmolVLA/PI0/PI0.5 preprocessing/postprocessing methods scale only when the corresponding flag is enabled.
