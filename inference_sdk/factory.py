"""Factory helpers for creating inference engines."""

from __future__ import annotations

from typing import Optional

from .base import BaseInferenceEngine, SmoothingConfig
from .engines import (
    ACTInferenceEngine,
    PI0InferenceEngine,
    SmolVLAInferenceEngine,
)

SUPPORTED_MODEL_TYPES = ("act", "smolvla", "pi0")


def create_engine(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> BaseInferenceEngine:
    """
    Create an inference engine by model type.

    Args:
        model_type: "act", "smolvla", or "pi0"
        device: Requested torch device string
        smoothing_config: Optional smoothing configuration
        strict_device: If True, fail instead of silently falling back
    """
    if smoothing_config is None:
        smoothing_config = SmoothingConfig(
            enable_async_inference=True,
            aggregate_fn_name="latest_only",
        )

    normalized = str(model_type).lower()
    if normalized == "act":
        return ACTInferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    if normalized == "smolvla":
        return SmolVLAInferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    if normalized == "pi0":
        return PI0InferenceEngine(
            device=device,
            smoothing_config=smoothing_config,
            strict_device=strict_device,
        )
    raise ValueError(f"Unknown model type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}")


def create_inference_engine(
    model_type: str,
    device: str = "cuda:0",
    smoothing_config: Optional[SmoothingConfig] = None,
    strict_device: bool = False,
) -> BaseInferenceEngine:
    """Backward-compatible alias for create_engine()."""
    return create_engine(
        model_type=model_type,
        device=device,
        smoothing_config=smoothing_config,
        strict_device=strict_device,
    )


__all__ = ["SUPPORTED_MODEL_TYPES", "create_engine", "create_inference_engine"]
