"""Runtime helpers for optional local dependency and model search paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

MODEL_ROOT_ENV_KEYS = ("INFERENCE_SDK_MODEL_ROOTS", "INFERENCE_SDK_MODEL_ROOT")
SPARKMIND_PATH_ENV_KEYS = (
    "INFERENCE_SDK_SPARKMIND_PATH",
    "SPARKMIND_PATH",
    "SPARKMIND_ROOT",
)


def _normalize_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def iter_unique_paths(candidates: Iterable[Path | str | None]) -> Iterator[Path]:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = _normalize_path(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def iter_env_paths(env_keys: Sequence[str]) -> Iterator[Path]:
    raw_candidates: list[str] = []
    for key in env_keys:
        value = os.environ.get(key)
        if not value:
            continue
        raw_candidates.extend(part.strip() for part in value.split(os.pathsep) if part.strip())
    yield from iter_unique_paths(raw_candidates)


def configure_optional_import_paths(
    env_keys: Sequence[str] = SPARKMIND_PATH_ENV_KEYS,
) -> list[Path]:
    """Add optional dependency roots from environment variables to sys.path."""
    added_paths: list[Path] = []
    for candidate in iter_env_paths(env_keys):
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str in sys.path:
            continue
        sys.path.insert(0, candidate_str)
        added_paths.append(candidate)
    return added_paths


def iter_model_search_roots(checkpoint_path: Path | None = None) -> Iterator[Path]:
    """Yield generic model search roots without assuming a host repository name."""
    candidates: list[Path] = list(iter_env_paths(MODEL_ROOT_ENV_KEYS))

    if checkpoint_path is not None:
        normalized_checkpoint = _normalize_path(checkpoint_path)
        candidates.append(normalized_checkpoint)

        max_parent_depth = min(4, len(normalized_checkpoint.parents))
        for idx in range(max_parent_depth):
            candidates.append(normalized_checkpoint.parents[idx])

    candidates.extend(
        [
            Path.cwd() / "models",
            Path.cwd(),
        ]
    )

    yield from iter_unique_paths(candidates)
