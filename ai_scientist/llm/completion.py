from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any

import openai
from ai_scientist.utils.token_tracker import token_tracker

from .client import (
    ClientSpec,
    apply_minimax_request_defaults,
    max_output_token_limit,
    normalize_model_for_route,
)
from .rate_limiter import wait_for_api_rate_limit

logger = logging.getLogger("ai-scientist")

INTERNAL_FORCE_NON_STREAM_FLAG = "_ai_scientist_force_non_stream"
HARD_MAX_OUTPUT_TOKENS_ENV_VAR = "AI_SCIENTIST_HARD_MAX_OUTPUT_TOKENS"
DEFAULT_HARD_MAX_OUTPUT_TOKENS = 64 * 1024
TRANSPORT_RETRIES_ENV_VAR = "AI_SCIENTIST_LLM_TRANSPORT_RETRIES"
DEFAULT_TRANSPORT_RETRIES = 2

RETRYABLE_MODEL_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.APIError,
)

_MAX_TOKENS_LIMIT_RE = re.compile(
    r"(?:max_total_tokens|max_model_len)\s*=\s*(\d+)",
    re.IGNORECASE,
)
_MAX_CONTEXT_LENGTH_RE = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens",
    re.IGNORECASE,
)

_warned_hard_cap = False


def _dedupe_models(primary: str, fallback: str | None) -> list[str]:
    models = [primary]
    if fallback and fallback not in models:
        models.append(fallback)
    return models


def _jitter(seconds: float) -> float:
    return seconds + random.uniform(0, max(0.0, seconds * 0.1))


def _rate_limit_backoff_seconds(attempt: int) -> float:
    # 1.0, 1.7, 2.9, 4.9, ... capped at 60s
    base = 1.0 * (1.7**attempt)
    return min(60.0, _jitter(base))


def _transport_backoff_seconds(attempt: int) -> float:
    # 1.0, 2.0, 4.0, 8.0, ... capped at 30s
    base = 1.0 * (2.0**attempt)
    return min(30.0, _jitter(base))


def _should_stream(spec: ClientSpec, request_kwargs: dict[str, Any]) -> bool:
    if spec.route not in {"openai", "openai-compatible"}:
        return False
    return True


def _normalize_request_kwargs(
    spec: ClientSpec,
    request_kwargs: dict[str, Any],
) -> dict[str, Any]:
    effective = dict(request_kwargs)

    # seed is deprecated across OpenAI-style APIs; keep it out of requests.
    effective.pop("seed", None)

    token_limit = None
    if "max_completion_tokens" in effective and effective["max_completion_tokens"] is not None:
        token_limit = int(effective.pop("max_completion_tokens"))
    if "max_tokens" in effective and effective["max_tokens"] is not None:
        token_limit = int(effective.pop("max_tokens")) if token_limit is None else token_limit

    if token_limit is None:
        token_limit = max_output_token_limit()

    token_limit = _apply_output_token_caps(spec, token_limit)

    if spec.capabilities.is_reasoning:
        effective["max_completion_tokens"] = token_limit
        effective.pop("temperature", None)
        if effective.get("n", 1) != 1:
            effective["n"] = 1
    elif spec.route == "openai-compatible":
        effective["max_completion_tokens"] = token_limit
        effective.pop("max_tokens", None)
    else:
        effective["max_tokens"] = token_limit
        effective.pop("max_completion_tokens", None)
        temp_range = spec.capabilities.temperature_range
        if temp_range is not None and "temperature" in effective:
            lo, hi = temp_range
            raw_temp = float(effective["temperature"])
            clamped = max(lo + 1e-9, min(hi, raw_temp))
            if clamped != raw_temp:
                logger.warning(
                    "Temperature %.4f outside provider range (%.4f, %.4f]; clamping to %.4f.",
                    raw_temp, lo, hi, clamped,
                )
                effective["temperature"] = clamped

    if spec.route == "minimax-openai-compatible":
        effective = apply_minimax_request_defaults(effective)

    return effective


def _prepare_request_kwargs(
    request_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    effective = dict(request_kwargs)
    force_non_stream = bool(effective.pop(INTERNAL_FORCE_NON_STREAM_FLAG, False))
    return effective, force_non_stream


def _hard_max_output_tokens() -> int:
    raw_value = (os.environ.get(HARD_MAX_OUTPUT_TOKENS_ENV_VAR) or "").strip()
    if not raw_value:
        return DEFAULT_HARD_MAX_OUTPUT_TOKENS
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{HARD_MAX_OUTPUT_TOKENS_ENV_VAR} must be an integer, got: {raw_value}"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"{HARD_MAX_OUTPUT_TOKENS_ENV_VAR} must be positive, got: {raw_value}"
        )
    return value


def _transport_retries() -> int:
    raw_value = (os.environ.get(TRANSPORT_RETRIES_ENV_VAR) or "").strip()
    if not raw_value:
        return DEFAULT_TRANSPORT_RETRIES
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{TRANSPORT_RETRIES_ENV_VAR} must be an integer, got: {raw_value}"
        ) from exc
    if value < 0:
        raise RuntimeError(
            f"{TRANSPORT_RETRIES_ENV_VAR} must be >= 0, got: {raw_value}"
        )
    return value


def _apply_output_token_caps(spec: ClientSpec, requested: int) -> int:
    global _warned_hard_cap
    limit = int(requested)
    cap = spec.capabilities.max_output_tokens
    if cap is not None:
        limit = min(limit, int(cap))

    hard_cap = _hard_max_output_tokens()
    if limit > hard_cap:
        if not _warned_hard_cap:
            logger.warning(
                "Requested output tokens=%d exceeds hard cap=%d; clamping (set %s to override).",
                limit,
                hard_cap,
                HARD_MAX_OUTPUT_TOKENS_ENV_VAR,
            )
            _warned_hard_cap = True
        limit = hard_cap
    return limit


def _parse_max_total_tokens_from_bad_request(exc: Exception) -> int | None:
    text = str(exc)
    match = _MAX_TOKENS_LIMIT_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    match = _MAX_CONTEXT_LENGTH_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def create_chat_completion_with_fallback(
    spec: ClientSpec,
    *,
    messages: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    fallback_model: str | None = None,
    rate_limit_retries: int = 8,
) -> tuple[openai.types.chat.ChatCompletion, str]:
    """Create a chat completion with an optional fallback model.

    Notes:
    - Connection retries are handled by the OpenAI SDK (ClientSpec uses max_retries).
    - We only add explicit backoff for 429 RateLimitError to avoid tight retry loops.
    - Default to non-stream calls for non-OpenAI routes for stability.
    """

    if not messages:
        raise ValueError("chat.completions messages must not be empty")

    normalized_fallback = None
    if fallback_model:
        normalized_fallback = normalize_model_for_route(fallback_model, spec.route)
    models = _dedupe_models(spec.model, normalized_fallback)

    last_error: Exception | None = None
    for index, candidate_model in enumerate(models):
        try:
            completion = _create_with_rate_limit_backoff(
                spec,
                messages=messages,
                model=candidate_model,
                request_kwargs=request_kwargs,
                rate_limit_retries=rate_limit_retries,
            )
            _track_completion_usage(completion)
            return completion, candidate_model
        except (openai.RateLimitError,) + RETRYABLE_MODEL_EXCEPTIONS as exc:
            last_error = exc
            if index + 1 < len(models):
                logger.warning(
                    "LLM request failed on model %s; switching to fallback model %s: %s",
                    candidate_model,
                    models[index + 1],
                    exc,
                )
                continue
            raise

    if last_error is None:
        raise RuntimeError("No chat completion models were provided.")
    raise last_error


def _create_with_rate_limit_backoff(
    spec: ClientSpec,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
    rate_limit_retries: int,
) -> openai.types.chat.ChatCompletion:
    prepared_kwargs, force_non_stream = _prepare_request_kwargs(request_kwargs)
    effective_kwargs = _normalize_request_kwargs(spec, prepared_kwargs)

    attempt = 0
    transport_attempt = 0
    transport_retries = _transport_retries()
    token_clamp_attempted = False
    while True:
        try:
            wait_for_api_rate_limit(model)
            if not force_non_stream and _should_stream(spec, effective_kwargs):
                return _create_streamed_chat_completion(
                    spec,
                    messages=messages,
                    model=model,
                    request_kwargs=effective_kwargs,
                )
            return _create_non_stream_chat_completion(
                spec,
                messages=messages,
                model=model,
                request_kwargs=effective_kwargs,
            )
        except openai.LengthFinishReasonError as exc:
            # Some openai-compatible providers cause the SDK streaming parser to raise
            # LengthFinishReasonError. Retry once with a non-stream request.
            if force_non_stream or spec.route != "openai" or not _should_stream(spec, effective_kwargs):
                raise RuntimeError(str(exc)) from None
            logger.warning(
                "Streaming completion parse failed with LengthFinishReasonError; retrying with non-stream call: %s",
                exc,
            )
            force_non_stream = True
            continue
        except openai.BadRequestError as exc:
            if not token_clamp_attempted:
                max_total_tokens = _parse_max_total_tokens_from_bad_request(exc)
                if max_total_tokens is not None and max_total_tokens > 0:
                    token_field = (
                        "max_completion_tokens"
                        if spec.capabilities.is_reasoning
                        else "max_tokens"
                    )
                    current_value = effective_kwargs.get(token_field)
                    try:
                        current_int = int(current_value) if current_value is not None else None
                    except (TypeError, ValueError):
                        current_int = None
                    if current_int is not None and current_int > max_total_tokens:
                        logger.warning(
                            "Provider rejected %s=%d; clamping to %d and retrying once.",
                            token_field,
                            current_int,
                            max_total_tokens,
                        )
                        effective_kwargs[token_field] = max_total_tokens
                        token_clamp_attempted = True
                        continue
            raise
        except openai.RateLimitError:
            attempt += 1
            if attempt > rate_limit_retries:
                raise
            wait_seconds = _rate_limit_backoff_seconds(attempt - 1)
            logger.warning(
                "RateLimitError on attempt %d/%d for model %s; retrying in %.1fs",
                attempt,
                rate_limit_retries,
                model,
                wait_seconds,
            )
            time.sleep(wait_seconds)
        except RETRYABLE_MODEL_EXCEPTIONS as exc:
            transport_attempt += 1
            if transport_attempt > transport_retries:
                raise
            wait_seconds = _transport_backoff_seconds(transport_attempt - 1)
            logger.warning(
                "%s on attempt %d/%d for model %s; retrying in %.1fs: %s",
                type(exc).__name__,
                transport_attempt,
                transport_retries,
                model,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)


def _create_non_stream_chat_completion(
    spec: ClientSpec,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
) -> openai.types.chat.ChatCompletion:
    return spec.client.chat.completions.create(
        model=model,
        messages=messages,
        **request_kwargs,
    )


def _create_streamed_chat_completion(
    spec: ClientSpec,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
) -> openai.types.chat.ChatCompletion:
    stream_kwargs = dict(request_kwargs)
    stream_options = stream_kwargs.pop("stream_options", None)
    if stream_options is None:
        stream_options = {"include_usage": True}
    with spec.client.chat.completions.stream(
        model=model,
        messages=messages,
        stream_options=stream_options,
        **stream_kwargs,
    ) as stream:
        for _ in stream:
            pass
        return stream.get_final_completion()


def _track_completion_usage(completion: openai.types.chat.ChatCompletion) -> None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    completion_details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = 0
    if completion_details is not None:
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = 0
    if prompt_details is not None:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    model = getattr(completion, "model", None) or "unknown"
    token_tracker.add_tokens(
        model,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens,
        cached_tokens,
    )
