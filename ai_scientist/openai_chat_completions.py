from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import httpx
import openai

from .openai_compatible import is_minimax_client

logger = logging.getLogger("ai-scientist")

DEFAULT_CONNECTION_RETRIES = 4
CONNECTION_BACKOFF_BASE_SECONDS = 2.0
INTERNAL_FORCE_NON_STREAM_FLAG = "_ai_scientist_force_non_stream"

RETRYABLE_MODEL_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.APIError,
)
RETRYABLE_CONNECTION_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.APIError,
    httpx.TransportError,
)


def _dedupe_models(primary: str, fallback: str | None) -> list[str]:
    models = [primary]
    if fallback and fallback not in models:
        models.append(fallback)
    return models


def _connection_backoff_seconds(attempt: int) -> float:
    return CONNECTION_BACKOFF_BASE_SECONDS**attempt


def _require_messages(messages: list[dict[str, Any]]) -> None:
    if messages:
        return
    raise ValueError("chat.completions messages must not be empty")
def _prepare_request_kwargs(
    request_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    effective_kwargs = dict(request_kwargs)
    force_non_stream = bool(
        effective_kwargs.pop(INTERNAL_FORCE_NON_STREAM_FLAG, False)
    )
    return effective_kwargs, force_non_stream


def create_chat_completion_with_fallback(
    client: openai.OpenAI,
    *,
    messages: list[dict[str, Any]],
    model: str,
    fallback_model: str | None,
    request_kwargs: dict[str, Any],
    connection_retries: int = DEFAULT_CONNECTION_RETRIES,
) -> tuple[openai.types.chat.ChatCompletion, str]:
    _require_messages(messages)
    models = _dedupe_models(model, fallback_model)
    last_error: Exception | None = None

    for index, candidate_model in enumerate(models):
        try:
            completion = _create_with_connection_retries(
                client,
                messages=messages,
                model=candidate_model,
                request_kwargs=request_kwargs,
                connection_retries=connection_retries,
            )
            return completion, candidate_model
        except RETRYABLE_MODEL_EXCEPTIONS as exc:
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


def _create_with_connection_retries(
    client: openai.OpenAI,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
    connection_retries: int,
) -> openai.types.chat.ChatCompletion:
    retries = max(1, connection_retries)
    for attempt in range(retries):
        try:
            return _create_chat_completion(
                client,
                messages=messages,
                model=model,
                request_kwargs=request_kwargs,
            )
        except RETRYABLE_CONNECTION_EXCEPTIONS as exc:
            if attempt + 1 >= retries:
                raise
            wait_seconds = _connection_backoff_seconds(attempt)
            cause = exc.__cause__ or exc.__context__
            cause_text = ""
            if cause is not None:
                cause_text = f" cause={type(cause).__name__}: {cause}"
            logger.warning(
                "%s on attempt %d/%d for model %s; retrying in %.1fs: %s%s",
                type(exc).__name__,
                attempt + 1,
                retries,
                model,
                wait_seconds,
                exc,
                cause_text,
            )
            time.sleep(wait_seconds)


def _create_chat_completion(
    client: openai.OpenAI,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
) -> openai.types.chat.ChatCompletion:
    effective_kwargs, force_non_stream = _prepare_request_kwargs(request_kwargs)
    if _should_use_non_stream_chat_completion(
        client,
        effective_kwargs,
        force_non_stream,
    ):
        return _create_non_stream_chat_completion(
            client,
            messages=messages,
            model=model,
            request_kwargs=effective_kwargs,
        )
    return _create_streamed_chat_completion(
        client,
        messages=messages,
        model=model,
        request_kwargs=effective_kwargs,
    )


def _should_use_non_stream_chat_completion(
    client: openai.OpenAI,
    request_kwargs: dict[str, Any],
    force_non_stream: bool,
) -> bool:
    if force_non_stream or request_kwargs.get("tools"):
        return True
    if is_minimax_client(client):
        return True
    return False


def _create_non_stream_chat_completion(
    client: openai.OpenAI,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
) -> openai.types.chat.ChatCompletion:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **request_kwargs,
    )


def _create_streamed_chat_completion(
    client: openai.OpenAI,
    *,
    messages: list[dict[str, Any]],
    model: str,
    request_kwargs: dict[str, Any],
) -> openai.types.chat.ChatCompletion:
    stream_kwargs = dict(request_kwargs)
    stream_options = stream_kwargs.pop("stream_options", None)
    if stream_options is None:
        stream_options = {"include_usage": True}

    with client.chat.completions.stream(
        model=model,
        messages=messages,
        stream_options=stream_options,
        **stream_kwargs,
    ) as stream:
        for _ in stream:
            pass
        return stream.get_final_completion()
