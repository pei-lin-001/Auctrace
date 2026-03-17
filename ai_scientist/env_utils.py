from __future__ import annotations

import os

from ai_scientist.project_env import load_project_env

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def load_env() -> None:
    """Load repo-local .env into process environment (non-overriding)."""
    load_project_env()


def env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer, got: {raw!r}") from exc


def env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{key} must be a boolean (one of: {sorted(_TRUE_VALUES | _FALSE_VALUES)}), got: {raw!r}"
    )

