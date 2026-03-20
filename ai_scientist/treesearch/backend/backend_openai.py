import json
import logging
import time
from typing import Any

from .utils import FunctionSpec, OutputType, opt_messages_to_list
from funcy import notnone, select_values
import openai
from rich import print
from ai_scientist.openai_chat_completions import create_chat_completion_with_fallback
from ai_scientist.openai_compatible import (
    OpenAIClientSpec,
    apply_minimax_request_defaults,
    create_openai_client,
    is_openai_reasoning_model,
    is_minimax_route,
)

logger = logging.getLogger("ai-scientist")
OPENAI_SDK_MAX_RETRIES = 3


def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    spec = create_openai_client(model, max_retries=max_retries)
    return spec.client


def _route_requires_user_message(route: str) -> bool:
    return route == "openai-compatible" or is_minimax_route(route)


def _normalize_messages_for_route(
    system_message: str | None,
    user_message: str | Any | None,
    route: str,
) -> tuple[str | None, str | Any | None]:
    if _route_requires_user_message(route) and system_message and user_message is None:
        return None, system_message
    return system_message, user_message


def _require_choices(completion: openai.types.chat.ChatCompletion):
    if completion.choices is not None:
        return completion.choices
    base_resp = getattr(completion, "base_resp", {}) or {}
    status_code = base_resp.get("status_code")
    status_msg = base_resp.get("status_msg")
    raise RuntimeError(
        "OpenAI-compatible completion returned no choices: "
        f"status_code={status_code}, status_msg={status_msg}"
    )


def _prepare_request(
    model: str,
    route: str,
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict,
) -> tuple[list[dict[str, str]], dict]:
    request_kwargs = dict(model_kwargs)
    system_message, user_message = _normalize_messages_for_route(
        system_message,
        user_message,
        route,
    )
    messages = opt_messages_to_list(system_message, user_message)
    for idx, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Refusing to call chat.completions with empty message content: "
                f"index={idx}, role={message.get('role')!r}"
            )

    if func_spec is not None:
        request_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        request_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        request_kwargs["parallel_tool_calls"] = False

    if is_openai_reasoning_model(model):
        if system_message is not None:
            if user_message is not None and not isinstance(user_message, str):
                raise TypeError(
                    "Reasoning models require a plain-text user_message when system_message is provided."
                )
            merged_message = f"{system_message}\n\n{user_message or ''}".strip()
            messages = opt_messages_to_list(None, merged_message)
        request_kwargs.pop("temperature", None)
        request_kwargs.pop("max_tokens", None)

    if is_minimax_route(route):
        request_kwargs = apply_minimax_request_defaults(request_kwargs)

    return messages, request_kwargs


def _create_completion(
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None,
    model_kwargs: dict,
) -> tuple[openai.types.chat.ChatCompletion, OpenAIClientSpec]:
    primary_model = model_kwargs.get("model")
    fallback_model = model_kwargs.get("fallback_model")
    if not primary_model:
        raise ValueError("OpenAI backend requires model to be provided.")

    request_model_kwargs = dict(model_kwargs)
    request_model_kwargs.pop("model", None)
    request_model_kwargs.pop("fallback_model", None)

    primary_spec = create_openai_client(primary_model, max_retries=OPENAI_SDK_MAX_RETRIES)
    normalized_fallback_model = None
    if fallback_model:
        normalized_fallback_model = create_openai_client(fallback_model, max_retries=0).model
    messages, request_kwargs = _prepare_request(
        model=primary_spec.model,
        route=primary_spec.route,
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        model_kwargs=request_model_kwargs,
    )
    completion, used_model = create_chat_completion_with_fallback(
        primary_spec.client,
        messages=messages,
        model=primary_spec.model,
        fallback_model=normalized_fallback_model,
        request_kwargs=request_kwargs,
    )
    used_spec = OpenAIClientSpec(
        client=primary_spec.client,
        model=used_model,
        route=primary_spec.route,
    )
    return completion, used_spec


def query(
    system_message: str | None,
    user_message: str | Any | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    t0 = time.time()
    completion, used_spec = _create_completion(
        system_message,
        user_message,
        func_spec,
        filtered_kwargs,
    )
    req_time = time.time() - t0

    choices = _require_choices(completion)
    choice = choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
        "route": used_spec.route,
        "requested_model": model_kwargs.get("model"),
        "fallback_model": model_kwargs.get("fallback_model"),
        "fallback_used": used_spec.model != create_openai_client(
            model_kwargs.get("model") or "",
            max_retries=0,
        ).model,
    }

    return output, req_time, in_tokens, out_tokens, info
