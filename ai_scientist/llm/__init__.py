from __future__ import annotations

from typing import Any

from .client import ClientSpec, create_client_spec, max_output_token_limit, tag_client, spec_from_client
from .completion import create_chat_completion_with_fallback
from .messages import build_messages, preserve_assistant_message
from .models import capabilities_for
from .parsing import extract_json_between_markers, request_structured_output


def create_client(model: str) -> tuple[Any, str]:
    spec = create_client_spec(model)
    tag_client(spec.client, spec)
    print(f"Using {spec.route} API with model {spec.model}.")
    return spec.client, spec.model


def get_response_from_llm(
    prompt: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
) -> tuple[str, list[dict[str, Any]]]:
    if msg_history is None:
        msg_history = []

    spec = spec_from_client(client, model)
    messages = build_messages(
        spec,
        system_message=system_message,
        history=msg_history,
        user_content=prompt,
    )
    request_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_output_token_limit(),
        "n": 1,
        "stop": None,
    }
    if spec.capabilities.is_reasoning:
        request_kwargs.setdefault("reasoning_effort", "high")

    completion, _ = create_chat_completion_with_fallback(
        spec,
        messages=messages,
        request_kwargs=request_kwargs,
    )
    content = completion.choices[0].message.content
    assistant_message = preserve_assistant_message(
        spec,
        completion.choices[0].message,
        content,
    )
    new_msg_history = msg_history + [{"role": "user", "content": prompt}, assistant_message]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def get_batch_responses_from_llm(
    prompt: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    n_responses: int = 1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if msg_history is None:
        msg_history = []

    spec = spec_from_client(client, model)
    if n_responses > 1 and not spec.capabilities.supports_n_responses:
        contents: list[str] = []
        histories: list[list[dict[str, Any]]] = []
        for _ in range(n_responses):
            content, hist = get_response_from_llm(
                prompt,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            contents.append(content)
            histories.append(hist)
        return contents, histories

    messages = build_messages(
        spec,
        system_message=system_message,
        history=msg_history,
        user_content=prompt,
    )
    request_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_output_token_limit(),
        "n": n_responses,
        "stop": None,
    }
    if spec.capabilities.is_reasoning:
        request_kwargs.setdefault("reasoning_effort", "high")

    completion, _ = create_chat_completion_with_fallback(
        spec,
        messages=messages,
        request_kwargs=request_kwargs,
    )

    base_history = msg_history + [{"role": "user", "content": prompt}]
    contents = [choice.message.content for choice in completion.choices]
    histories = [
        base_history
        + [
            preserve_assistant_message(
                spec,
                choice.message,
                choice.message.content,
            )
        ]
        for choice in completion.choices
    ]

    if print_debug and histories:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(histories[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(contents[0] if contents else "")
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return contents, histories



__all__ = [
    "create_client",
    "extract_json_between_markers",
    "get_batch_responses_from_llm",
    "get_response_from_llm",
    "request_structured_output",
]
