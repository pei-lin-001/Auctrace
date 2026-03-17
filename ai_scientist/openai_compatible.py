from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx
import openai

from .project_env import load_project_env

OPENAI_COMPATIBLE_BASE_ENV_VARS = (
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
)
OPENAI_COMPATIBLE_KEY_ENV_VARS = (
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_API_KEY",
)
OPENAI_VENDOR_PREFIX = "openai/"
DEFAULT_CONTEXT_TOKENS = 128 * 1024
CONTEXT_TOKENS_ENV_VAR = "AI_SCIENTIST_CONTEXT_TOKENS"
DEFAULT_MAX_OUTPUT_TOKENS = 8 * 1024
MAX_OUTPUT_TOKENS_ENV_VAR = "AI_SCIENTIST_MAX_OUTPUT_TOKENS"


@dataclass(frozen=True)
class OpenAIClientSpec:
    client: openai.OpenAI
    model: str
    route: str


_http_client_pid = None
_http_client_cache = None

def _build_http_client() -> httpx.Client:
    """Build httpx client with process isolation and timeout protection.

    Recreates client after fork to avoid SSL connection state conflicts
    in multi-process environments (ProcessPoolExecutor).
    """
    global _http_client_pid, _http_client_cache
    current_pid = os.getpid()

    # Recreate client if process ID changed (after fork)
    if _http_client_pid != current_pid:
        _http_client_cache = openai.DefaultHttpxClient(
            trust_env=False,
            timeout=httpx.Timeout(
                connect=10.0,   # 10s connect timeout
                read=300.0,     # 5min read timeout for long responses
                write=10.0,     # 10s write timeout
                pool=5.0,       # 5s pool timeout
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )
        _http_client_pid = current_pid

    return _http_client_cache

def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def context_token_limit() -> int:
    raw_value = os.environ.get(CONTEXT_TOKENS_ENV_VAR)
    return _parse_positive_int(
        raw_value,
        default=DEFAULT_CONTEXT_TOKENS,
        env_var=CONTEXT_TOKENS_ENV_VAR,
    )


def max_output_token_limit() -> int:
    raw_value = os.environ.get(MAX_OUTPUT_TOKENS_ENV_VAR)
    return _parse_positive_int(
        raw_value,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        env_var=MAX_OUTPUT_TOKENS_ENV_VAR,
    )


def _parse_positive_int(raw_value: str | None, default: int, env_var: str) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{env_var} must be an integer, got: {raw_value}"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"{env_var} must be positive, got: {raw_value}"
        )
    return value


def normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.path in ("", "/"):
        parsed = parsed._replace(path="/v1")
        return urlunparse(parsed).rstrip("/")
    return base_url.rstrip("/")


load_project_env()


def openai_compatible_base_url() -> str | None:
    base_url = _first_env(*OPENAI_COMPATIBLE_BASE_ENV_VARS)
    if base_url is None:
        return None
    return normalize_base_url(base_url)


def openai_compatible_api_key() -> str:
    api_key = _first_env(*OPENAI_COMPATIBLE_KEY_ENV_VARS)
    if api_key is None:
        raise RuntimeError(
            "OpenAI-compatible base URL is configured, but no API key was found. "
            f"Set one of: {', '.join(OPENAI_COMPATIBLE_KEY_ENV_VARS)}."
        )
    return api_key


def has_openai_compatible_base_url() -> bool:
    return openai_compatible_base_url() is not None


def is_openai_reasoning_model(model: str) -> bool:
    leaf = model.strip().split("/")[-1]
    return leaf.startswith("o1") or leaf.startswith("o3")


def normalize_openai_model_name(model: str) -> str:
    return model.strip()


def normalize_openai_model_for_route(model: str, route: str) -> str:
    normalized = normalize_openai_model_name(model)
    if route == "openai" and normalized.startswith(OPENAI_VENDOR_PREFIX):
        return normalized[len(OPENAI_VENDOR_PREFIX) :]
    return normalized


def create_openai_client(model: str, max_retries: int = 2) -> OpenAIClientSpec:
    compat_base_url = openai_compatible_base_url()
    if compat_base_url is not None:
        client = openai.OpenAI(
            api_key=openai_compatible_api_key(),
            base_url=compat_base_url,
            max_retries=max_retries,
            http_client=_build_http_client(),
        )
        return OpenAIClientSpec(
            client=client,
            model=normalize_openai_model_for_route(model, "openai-compatible"),
            route="openai-compatible",
        )

    client = openai.OpenAI(max_retries=max_retries, http_client=_build_http_client())
    return OpenAIClientSpec(
        client=client,
        model=normalize_openai_model_for_route(model, "openai"),
        route="openai",
    )
