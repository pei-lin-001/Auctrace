from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import openai

from ai_scientist.project_env import load_project_env

from .models import ModelCapabilities, capabilities_for

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
MINIMAX_VENDOR_PREFIX = "minimax/"
MINIMAX_BASE_ENV_VARS = ("MINIMAX_BASE_URL",)
MINIMAX_KEY_ENV_VARS = ("MINIMAX_API_KEY",)
MINIMAX_REGION_ENV_VAR = "MINIMAX_REGION"
MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_ROUTE = "minimax-openai-compatible"

DEFAULT_CONTEXT_TOKENS = 128 * 1024
CONTEXT_TOKENS_ENV_VAR = "AI_SCIENTIST_CONTEXT_TOKENS"
DEFAULT_MAX_OUTPUT_TOKENS = 8 * 1024
MAX_OUTPUT_TOKENS_ENV_VAR = "AI_SCIENTIST_MAX_OUTPUT_TOKENS"


@dataclass(frozen=True)
class ClientSpec:
    client: openai.OpenAI
    model: str
    route: str
    capabilities: ModelCapabilities


_http_client_pid = None
_http_client_cache = None


def _build_http_client() -> httpx.Client:
    """Build httpx client with process isolation and timeout protection.

    Recreates the client after fork to avoid SSL connection state conflicts in
    multi-process environments (ProcessPoolExecutor).
    """

    global _http_client_pid, _http_client_cache
    current_pid = os.getpid()
    if _http_client_pid != current_pid:
        _http_client_cache = openai.DefaultHttpxClient(
            trust_env=False,
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,
                write=10.0,
                pool=5.0,
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
        raise RuntimeError(f"{env_var} must be an integer, got: {raw_value}") from exc
    if value <= 0:
        raise RuntimeError(f"{env_var} must be positive, got: {raw_value}")
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


def minimax_base_url() -> str:
    configured = _first_env(*MINIMAX_BASE_ENV_VARS)
    if configured is not None:
        return normalize_base_url(configured)
    region = os.environ.get(MINIMAX_REGION_ENV_VAR, "global").strip().lower()
    if region == "cn":
        return MINIMAX_CN_BASE_URL
    return MINIMAX_GLOBAL_BASE_URL


def minimax_api_key() -> str:
    api_key = _first_env(*MINIMAX_KEY_ENV_VARS)
    if api_key is None:
        raise RuntimeError(
            "MiniMax route requested, but no API key was found. "
            f"Set one of: {', '.join(MINIMAX_KEY_ENV_VARS)}."
        )
    return api_key


def has_minimax_api_key() -> bool:
    return _first_env(*MINIMAX_KEY_ENV_VARS) is not None


def has_openai_compatible_base_url() -> bool:
    return openai_compatible_base_url() is not None


def normalize_openai_model_name(model: str) -> str:
    return (model or "").strip()


def _split_provider_prefix(model: str) -> tuple[str | None, str]:
    normalized = normalize_openai_model_name(model)
    if "/" not in normalized:
        return None, normalized
    provider, remainder = normalized.split("/", 1)
    provider = provider.strip()
    provider_lower = provider.lower()
    if provider_lower in {
        OPENAI_VENDOR_PREFIX.rstrip("/"),
        MINIMAX_VENDOR_PREFIX.rstrip("/"),
    }:
        return provider_lower, remainder
    return None, normalized


def is_minimax_model(model: str) -> bool:
    _, leaf = _split_provider_prefix(model)
    return leaf.lower().startswith("minimax-")


def normalize_model_for_route(model: str, route: str) -> str:
    normalized = normalize_openai_model_name(model)
    provider, leaf = _split_provider_prefix(normalized)

    if route == MINIMAX_ROUTE:
        leaf = leaf.strip()
        if leaf.lower().startswith("minimax-"):
            suffix = leaf[len("minimax-") :]
            if suffix.lower().startswith("m"):
                suffix = "M" + suffix[1:]
            return "MiniMax-" + suffix
        return leaf

    # IMPORTANT: do not trim/strip vendor prefix for openai-compatible routes.
    if route == "openai-compatible":
        return normalized

    if route == "openai" and provider == OPENAI_VENDOR_PREFIX.rstrip("/"):
        return leaf

    return normalized


def apply_minimax_request_defaults(request_kwargs: dict) -> dict:
    effective = dict(request_kwargs)
    extra_body = dict(effective.get("extra_body") or {})
    extra_body.setdefault("reasoning_split", True)
    effective["extra_body"] = extra_body
    return effective


def response_message_to_history_dict(message: object) -> dict:
    if hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        payload = {k: v for k, v in message.items() if v is not None}
    else:
        raise TypeError(
            f"Unsupported message type for history preservation: {type(message)!r}"
        )
    payload.setdefault("role", "assistant")
    return payload


def tag_client(client: Any, spec: "ClientSpec") -> None:
    """Attach route/capability metadata to an OpenAI client object (best-effort)."""
    try:
        setattr(client, "_ai_scientist_route", spec.route)
        setattr(client, "_ai_scientist_capabilities", spec.capabilities)
    except Exception:
        return


def spec_from_client(client: Any, model: str) -> "ClientSpec":
    """Reconstruct a ClientSpec from a tagged client, falling back to URL sniffing."""
    route = getattr(client, "_ai_scientist_route", None)
    capabilities = getattr(client, "_ai_scientist_capabilities", None)
    if isinstance(route, str) and capabilities is not None:
        return ClientSpec(client=client, model=model, route=route, capabilities=capabilities)

    # Legacy fallback: infer route from base_url string.
    base_url = str(getattr(client, "base_url", "")).lower()
    if "api.minimax.io" in base_url or "api.minimaxi.com" in base_url:
        route = MINIMAX_ROUTE
    elif base_url:
        route = "openai-compatible"
    else:
        route = "openai"
    return ClientSpec(
        client=client,
        model=model,
        route=route,
        capabilities=capabilities_for(model, route),
    )


def create_client_spec(model: str, max_retries: int = 3) -> ClientSpec:
    provider, _ = _split_provider_prefix(model)
    if provider == MINIMAX_VENDOR_PREFIX.rstrip("/"):
        client = openai.OpenAI(
            api_key=minimax_api_key(),
            base_url=minimax_base_url(),
            max_retries=max_retries,
            http_client=_build_http_client(),
        )
        normalized_model = normalize_model_for_route(model, MINIMAX_ROUTE)
        return ClientSpec(
            client=client,
            model=normalized_model,
            route=MINIMAX_ROUTE,
            capabilities=capabilities_for(normalized_model, MINIMAX_ROUTE),
        )

    if is_minimax_model(model) and has_minimax_api_key():
        client = openai.OpenAI(
            api_key=minimax_api_key(),
            base_url=minimax_base_url(),
            max_retries=max_retries,
            http_client=_build_http_client(),
        )
        normalized_model = normalize_model_for_route(model, MINIMAX_ROUTE)
        return ClientSpec(
            client=client,
            model=normalized_model,
            route=MINIMAX_ROUTE,
            capabilities=capabilities_for(normalized_model, MINIMAX_ROUTE),
        )

    compat_base_url = openai_compatible_base_url()
    if compat_base_url is not None:
        client = openai.OpenAI(
            api_key=openai_compatible_api_key(),
            base_url=compat_base_url,
            max_retries=max_retries,
            http_client=_build_http_client(),
        )
        normalized_model = normalize_model_for_route(model, "openai-compatible")
        return ClientSpec(
            client=client,
            model=normalized_model,
            route="openai-compatible",
            capabilities=capabilities_for(normalized_model, "openai-compatible"),
        )

    client = openai.OpenAI(max_retries=max_retries, http_client=_build_http_client())
    normalized_model = normalize_model_for_route(model, "openai")
    return ClientSpec(
        client=client,
        model=normalized_model,
        route="openai",
        capabilities=capabilities_for(normalized_model, "openai"),
    )

