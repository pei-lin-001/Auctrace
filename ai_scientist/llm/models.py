from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapabilities:
    is_reasoning: bool
    supports_vision: bool
    supports_structured_output: bool
    supports_n_responses: bool
    supports_system_role: bool
    supports_developer_role: bool
    max_output_tokens: int | None = None
    temperature_range: tuple[float, float] | None = None


def model_leaf_name(model: str) -> str:
    normalized = (model or "").strip()
    if not normalized:
        return ""
    return normalized.split("/")[-1].strip()


def is_reasoning_model(model: str) -> bool:
    leaf = model_leaf_name(model).lower()
    return (
        leaf.startswith("o1")
        or leaf.startswith("o3")
        or leaf.startswith("o4")
        or leaf.startswith("gpt-5")
    )


def capabilities_for(model: str, route: str) -> ModelCapabilities:
    route = (route or "").strip().lower()
    reasoning = is_reasoning_model(model)

    supports_developer_role = route in {"openai", "openai-compatible"}
    supports_system_role = True
    supports_structured_output = route in {"openai", "openai-compatible"}
    supports_vision = route != "minimax-openai-compatible"

    supports_n_responses = not reasoning and route != "minimax-openai-compatible"
    temperature_range = (0.0, 1.0) if route == "minimax-openai-compatible" else None

    return ModelCapabilities(
        is_reasoning=reasoning,
        supports_vision=supports_vision,
        supports_structured_output=supports_structured_output,
        supports_n_responses=supports_n_responses,
        supports_system_role=supports_system_role,
        supports_developer_role=supports_developer_role,
        temperature_range=temperature_range,
    )
