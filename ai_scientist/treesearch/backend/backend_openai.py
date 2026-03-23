import json
import logging
import time
from typing import Any

from funcy import notnone, select_values

from .utils import FunctionSpec, OutputType
from ai_scientist.llm.client import (
    ClientSpec,
    create_client_spec,
    normalize_model_for_route,
    tag_client,
)
from ai_scientist.llm.completion import create_chat_completion_with_fallback
from ai_scientist.llm.messages import build_messages

logger = logging.getLogger("ai-scientist")


def get_ai_client(model: str, max_retries: int = 3):
    spec = create_client_spec(model, max_retries=max_retries)
    tag_client(spec.client, spec)
    return spec.client


def _require_non_empty_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("Refusing to call chat.completions with empty messages.")
    for idx, message in enumerate(messages):
        content = message.get("content")
        if content is None:
            raise ValueError(
                "Refusing to call chat.completions with empty message content: "
                f"index={idx}, role={message.get('role')!r}"
            )
        if isinstance(content, str) and not content.strip():
            raise ValueError(
                "Refusing to call chat.completions with empty message content: "
                f"index={idx}, role={message.get('role')!r}"
            )
        if isinstance(content, list) and not content:
            raise ValueError(
                "Refusing to call chat.completions with empty message content list: "
                f"index={idx}, role={message.get('role')!r}"
            )


def _prepare_request(
    spec: ClientSpec,
    *,
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_kwargs = dict(model_kwargs)

    messages = build_messages(
        spec,
        system_message=system_message,
        history=None,
        user_content=user_message,
    )
    _require_non_empty_messages(messages)

    if func_spec is not None:
        request_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        request_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        request_kwargs["parallel_tool_calls"] = False

    # Keep old behaviour: if a reasoning model is requested on the OpenAI route,
    # default to a reasonably strong reasoning effort unless explicitly set.
    if spec.capabilities.is_reasoning and request_kwargs.get("reasoning_effort") is None:
        request_kwargs["reasoning_effort"] = "high"

    return messages, request_kwargs


def _create_completion(
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict[str, Any],
):
    primary_model = model_kwargs.get("model")
    fallback_model = model_kwargs.get("fallback_model")
    if not primary_model:
        raise ValueError("OpenAI backend requires model to be provided.")

    request_model_kwargs = dict(model_kwargs)
    request_model_kwargs.pop("model", None)
    request_model_kwargs.pop("fallback_model", None)

    primary_spec = create_client_spec(primary_model, max_retries=3)
    normalized_fallback = None
    if fallback_model:
        normalized_fallback = normalize_model_for_route(fallback_model, primary_spec.route)

    messages, request_kwargs = _prepare_request(
        primary_spec,
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        model_kwargs=request_model_kwargs,
    )

    completion, used_model = create_chat_completion_with_fallback(
        primary_spec,
        messages=messages,
        request_kwargs=request_kwargs,
        fallback_model=normalized_fallback,
    )
    used_spec = ClientSpec(
        client=primary_spec.client,
        model=used_model,
        route=primary_spec.route,
        capabilities=primary_spec.capabilities,
    )
    return completion, used_spec


def query(
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict[str, Any] = select_values(notnone, model_kwargs)  # type: ignore

    t0 = time.time()
    completion, used_spec = _create_completion(
        system_message,
        user_message,
        func_spec,
        filtered_kwargs,
    )
    req_time = time.time() - t0

    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise RuntimeError("OpenAI-compatible completion returned no choices.")
    choice = choices[0]

    if func_spec is None:
        output: OutputType = choice.message.content
    else:
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        if not tool_calls:
            raise RuntimeError(
                f"function_call is empty, it is not a function call: {choice.message}"
            )
        if tool_calls[0].function.name != func_spec.name:
            raise RuntimeError("Function name mismatch")
        try:
            output = json.loads(tool_calls[0].function.arguments)
        except json.JSONDecodeError as exc:
            logger.error(
                "Error decoding the function arguments: %s",
                tool_calls[0].function.arguments,
            )
            raise exc

    usage = getattr(completion, "usage", None)
    in_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": getattr(completion, "model", None),
        "created": getattr(completion, "created", None),
        "route": used_spec.route,
        "requested_model": model_kwargs.get("model"),
        "fallback_model": model_kwargs.get("fallback_model"),
        "fallback_used": used_spec.model
        != normalize_model_for_route(model_kwargs.get("model") or "", used_spec.route),
    }

    return output, req_time, in_tokens, out_tokens, info
