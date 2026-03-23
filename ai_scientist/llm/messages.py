from __future__ import annotations

from typing import Any

from .client import ClientSpec, response_message_to_history_dict


def preserve_assistant_message(
    spec: ClientSpec,
    message: Any,
    content: str,
) -> dict[str, Any]:
    if spec.route == "minimax-openai-compatible":
        return response_message_to_history_dict(message)
    return {"role": "assistant", "content": content}


def build_messages(
    spec: ClientSpec,
    *,
    system_message: str | None,
    history: list[dict[str, Any]] | None,
    user_content: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Build OpenAI-style `messages` with provider/model quirks handled."""

    history = list(history or [])

    # MiniMax requires a user message on some system-only requests; keep that shim there.
    route_requires_user = spec.route == "minimax-openai-compatible"
    if route_requires_user and system_message and user_content is None and not history:
        user_content = system_message
        system_message = None

    messages: list[dict[str, Any]] = []

    if system_message:
        if spec.capabilities.is_reasoning:
            if spec.capabilities.supports_developer_role:
                messages.append({"role": "developer", "content": system_message})
            else:
                # Provider does not support the developer role; merge into the first user message.
                if user_content is None:
                    user_content = system_message
                elif isinstance(user_content, str):
                    user_content = f"{system_message}\n\n{user_content}".strip()
                else:
                    messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})

    messages.extend(history)

    if user_content is not None:
        messages.append({"role": "user", "content": user_content})

    return messages
