from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("ai-scientist")

RATE_LIMIT_MAX_CALLS_ENV_VAR = "AI_SCIENTIST_API_RATE_LIMIT_MAX_CALLS"
RATE_LIMIT_WINDOW_SECONDS_ENV_VAR = "AI_SCIENTIST_API_RATE_LIMIT_WINDOW_SECONDS"
RATE_LIMIT_WINDOW_MINUTES_ENV_VAR = "AI_SCIENTIST_API_RATE_LIMIT_WINDOW_MINUTES"
RATE_LIMIT_SCOPE_ENV_VAR = "AI_SCIENTIST_API_RATE_LIMIT_SCOPE"
RATE_LIMIT_STATE_PATH_ENV_VAR = "AI_SCIENTIST_API_RATE_LIMIT_STATE_PATH"
DEFAULT_STATE_FILE_NAME = "auctrace-api-rate-limit.json"


@dataclass(frozen=True)
class ApiRateLimitConfig:
    max_calls: int
    window_seconds: float
    scope: str
    state_path: Path


def api_rate_limit_config() -> ApiRateLimitConfig | None:
    raw_max_calls = os.environ.get(RATE_LIMIT_MAX_CALLS_ENV_VAR)
    if raw_max_calls is None or not raw_max_calls.strip():
        return None
    try:
        max_calls = int(raw_max_calls)
    except ValueError as exc:
        raise RuntimeError(
            f"{RATE_LIMIT_MAX_CALLS_ENV_VAR} must be an integer, got: {raw_max_calls}"
        ) from exc
    if max_calls <= 0:
        raise RuntimeError(
            f"{RATE_LIMIT_MAX_CALLS_ENV_VAR} must be positive, got: {raw_max_calls}"
        )

    window_seconds = _window_seconds_from_env()
    scope = (os.environ.get(RATE_LIMIT_SCOPE_ENV_VAR) or "global").strip() or "global"
    state_path = _state_path_from_env()
    return ApiRateLimitConfig(
        max_calls=max_calls,
        window_seconds=window_seconds,
        scope=scope,
        state_path=state_path,
    )


def wait_for_api_rate_limit(model: str) -> None:
    config = api_rate_limit_config()
    if config is None:
        return

    scope_key = _scope_key(config.scope, model)
    log_waits = True
    while True:
        now = time.time()
        with _locked_state_file(config.state_path) as handle:
            state = _load_state(handle)
            scopes = state.setdefault("scopes", {})
            timestamps = scopes.get(scope_key, [])
            if not isinstance(timestamps, list):
                timestamps = []
            timestamps = [
                float(timestamp)
                for timestamp in timestamps
                if now - float(timestamp) < config.window_seconds
            ]

            if len(timestamps) < config.max_calls:
                timestamps.append(now)
                scopes[scope_key] = timestamps
                _store_state(handle, state)
                return

            wait_seconds = max(0.0, timestamps[0] + config.window_seconds - now)
            scopes[scope_key] = timestamps
            _store_state(handle, state)

        if log_waits:
            logger.warning(
                "API rate limit active for scope=%s: reached %d calls in %.1fs; waiting %.1fs before calling model %s",
                scope_key,
                config.max_calls,
                config.window_seconds,
                wait_seconds,
                model,
            )
            log_waits = False
        time.sleep(wait_seconds if wait_seconds > 0 else 0.01)


def _window_seconds_from_env() -> float:
    raw_seconds = os.environ.get(RATE_LIMIT_WINDOW_SECONDS_ENV_VAR)
    raw_minutes = os.environ.get(RATE_LIMIT_WINDOW_MINUTES_ENV_VAR)
    if raw_seconds is not None and raw_seconds.strip():
        try:
            window_seconds = float(raw_seconds)
        except ValueError as exc:
            raise RuntimeError(
                f"{RATE_LIMIT_WINDOW_SECONDS_ENV_VAR} must be a number, got: {raw_seconds}"
            ) from exc
        if window_seconds <= 0:
            raise RuntimeError(
                f"{RATE_LIMIT_WINDOW_SECONDS_ENV_VAR} must be positive, got: {raw_seconds}"
            )
        return window_seconds
    if raw_minutes is not None and raw_minutes.strip():
        try:
            window_minutes = float(raw_minutes)
        except ValueError as exc:
            raise RuntimeError(
                f"{RATE_LIMIT_WINDOW_MINUTES_ENV_VAR} must be a number, got: {raw_minutes}"
            ) from exc
        if window_minutes <= 0:
            raise RuntimeError(
                f"{RATE_LIMIT_WINDOW_MINUTES_ENV_VAR} must be positive, got: {raw_minutes}"
            )
        return window_minutes * 60.0
    return 60.0


def _state_path_from_env() -> Path:
    configured = os.environ.get(RATE_LIMIT_STATE_PATH_ENV_VAR)
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / DEFAULT_STATE_FILE_NAME


def _scope_key(scope: str, model: str) -> str:
    normalized_scope = scope.strip().lower()
    if normalized_scope == "per_model":
        return f"model:{model.strip()}"
    return "global"


@contextmanager
def _locked_state_file(path: Path) -> Iterator[object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            yield handle
        finally:
            handle.flush()
            os.fsync(handle.fileno())
            _unlock_file(handle)


def _load_state(handle: object) -> dict:
    text = handle.read()  # type: ignore[attr-defined]
    if not text.strip():
        return {"scopes": {}}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"scopes": {}}
    if not isinstance(payload, dict):
        return {"scopes": {}}
    return payload


def _store_state(handle: object, payload: dict) -> None:
    handle.seek(0)  # type: ignore[attr-defined]
    handle.truncate()  # type: ignore[attr-defined]
    json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)  # type: ignore[arg-type]
    handle.flush()  # type: ignore[attr-defined]


if os.name == "nt":
    import msvcrt

    def _lock_file(handle: object) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]

    def _unlock_file(handle: object) -> None:
        handle.seek(0)  # type: ignore[attr-defined]
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

else:
    import fcntl

    def _lock_file(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

    def _unlock_file(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]

