"""Template control loop for the process-local async inference runtime.

This example intentionally leaves camera and robot I/O as small placeholder
functions. Replace them with your backend's real hardware integration.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from inference_sdk import AsyncInferenceConfig, get_global_async_runtime


def read_camera_images() -> Dict[str, np.ndarray]:
    """Return BGR images keyed by camera role, for example {'head': image}."""
    raise NotImplementedError("Connect this function to your camera pipeline.")


def read_robot_state() -> np.ndarray:
    """Return the current robot state vector."""
    raise NotImplementedError("Connect this function to your robot state reader.")


def send_robot_action(action: np.ndarray) -> None:
    """Send one action vector to the robot controller."""
    raise NotImplementedError("Connect this function to your robot action sender.")


def sleep_until_next_tick(start_time: float, fps: float) -> None:
    elapsed = time.monotonic() - start_time
    time.sleep(max(0.0, (1.0 / fps) - elapsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an async inference control-loop template.")
    parser.add_argument("--model-type", required=True, choices=["act", "smolvla", "pi0"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--aggregate-fn", default="weighted_average")
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = get_global_async_runtime()

    runtime.load_policy(
        algorithm_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        instruction=args.instruction,
        config=AsyncInferenceConfig(
            control_fps=args.fps,
            chunk_size_threshold=args.chunk_size_threshold,
            aggregate_fn_name=args.aggregate_fn,
        ),
    )

    if args.warmup:
        runtime.warmup(
            images=read_camera_images(),
            state=read_robot_state(),
            instruction=args.instruction,
        )

    runtime.start()

    try:
        while True:
            tick_start = time.monotonic()
            result = runtime.step(
                images=read_camera_images(),
                state=read_robot_state(),
                instruction=args.instruction,
            )
            send_robot_action(result.action)
            sleep_until_next_tick(tick_start, args.fps)
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
