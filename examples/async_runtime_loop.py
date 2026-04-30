"""Template control loop for the process-local async inference runtime.

This example intentionally leaves camera and robot I/O as small placeholder
functions. Replace them with your backend's real hardware integration.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np

from inference_sdk import AsyncInferenceConfig, SUPPORTED_MODEL_TYPES, get_global_async_runtime


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
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--aggregate-fn", default="weighted_average")
    parser.add_argument("--enable-rtc", action="store_true", help="Enable RTC for SmolVLA/PI0/PI0.5 policies.")
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["ZEROS", "ONES", "LINEAR", "EXP", "zeros", "ones", "linear", "exp"],
        default="LINEAR",
        help="RTC prefix attention schedule. Default: LINEAR.",
    )
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=0)
    parser.add_argument("--rtc-debug", action="store_true")
    parser.add_argument("--rtc-debug-maxlen", type=int, default=100)
    parser.add_argument(
        "--fallback-mode",
        choices=["hold", "repeat"],
        default="hold",
        help=(
            "Action to use when the async queue is empty. "
            "`hold` sends the current robot state, `repeat` repeats the last action."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the initial warmup action queue before entering the control loop.",
    )
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        default=True,
        help="Run one synchronous prediction before starting the control loop. Enabled by default.",
    )
    warmup_group.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip startup warmup. The first control ticks may use fallback actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_rtc and args.model_type not in {"smolvla", "pi0", "pi05"}:
        raise ValueError("--enable-rtc is only supported for SmolVLA, PI0 and PI0.5")

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
            fallback_mode=args.fallback_mode,
            enable_rtc=args.enable_rtc,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            rtc_debug=args.rtc_debug,
            rtc_debug_maxlen=args.rtc_debug_maxlen,
        ),
    )

    if args.warmup:
        runtime.warmup(
            images=read_camera_images(),
            state=read_robot_state(),
            instruction=args.instruction,
        )

    runtime.start()

    if args.warmup and not runtime.wait_until_ready(min_queue_size=1, timeout=args.startup_timeout):
        runtime.stop()
        raise RuntimeError("Async runtime did not produce an initial action before startup timeout.")

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
