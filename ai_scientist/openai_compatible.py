from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import openai

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HUGGINGFACE_DEEPCODER_BASE_URL = (
    "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview"
)
OPENAI_COMPATIBLE_BASE_ENV_VARS = (
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_BASE_URL",
)
OPENAI_COMPATIBLE_KEY_ENV_VARS = (
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_API_KEY",
)
MODEL_ALIASES = {
    "deepseek-coder-v2-0724": "deepseek-coder",
    "deepcoder-14b": "agentica-org/DeepCoder-14B-Preview",
    "llama3.1-405b": "meta-llama/llama-3.1-405b-instruct",
}


@dataclass(frozen=True)
class OpenAIClientSpec:
    client: openai.OpenAI
    model: str
    route: str


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.path in ("", "/"):
        parsed = parsed._replace(path="/v1")
        return urlunparse(parsed).rstrip("/")
    return base_url.rstrip("/")


def openai_compatible_base_url() -> str | None:
    base_url = _first_env(*OPENAI_COMPATIBLE_BASE_ENV_VARS)
    if base_url is None:
        return None
    return normalize_base_url(base_url)


def openai_compatible_api_key() -> str:
    return _first_env(*OPENAI_COMPATIBLE_KEY_ENV_VARS) or "EMPTY"


def has_openai_compatible_base_url() -> bool:
    return openai_compatible_base_url() is not None


def is_ollama_model(model: str) -> bool:
    return model.startswith("ollama/")


def is_explicit_anthropic_model(model: str) -> bool:
    if model.startswith("bedrock/") and "claude" in model:
        return True
    if model.startswith("vertex_ai/") and "claude" in model:
        return True
    if model.startswith("anthropic/") and "claude" in model:
        return True
    return model.startswith("claude-")


def is_openai_reasoning_model(model: str) -> bool:
    normalized = normalize_openai_model_name(model)
    return normalized.startswith("o1") or normalized.startswith("o3")


def normalize_openai_model_name(model: str) -> str:
    if model.startswith("ollama/"):
        model = model.replace("ollama/", "", 1)
    if model.startswith("openai/"):
        model = model.replace("openai/", "", 1)
    return MODEL_ALIASES.get(model, model)


def create_openai_client(model: str, max_retries: int = 2) -> OpenAIClientSpec:
    if is_ollama_model(model):
        client = openai.OpenAI(
            api_key=os.environ.get("OLLAMA_API_KEY", ""),
            base_url=os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            max_retries=max_retries,
        )
        return OpenAIClientSpec(client=client, model=normalize_openai_model_name(model), route="ollama")

    compat_base_url = openai_compatible_base_url()
    if compat_base_url is not None:
        client = openai.OpenAI(
            api_key=openai_compatible_api_key(),
            base_url=compat_base_url,
            max_retries=max_retries,
        )
        return OpenAIClientSpec(
            client=client,
            model=normalize_openai_model_name(model),
            route="openai-compatible",
        )

    normalized_model = normalize_openai_model_name(model)
    if normalized_model == "deepseek-coder":
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
            max_retries=max_retries,
        )
        return OpenAIClientSpec(client=client, model=normalized_model, route="deepseek")
    if normalized_model == "agentica-org/DeepCoder-14B-Preview":
        client = openai.OpenAI(
            api_key=os.environ["HUGGINGFACE_API_KEY"],
            base_url=HUGGINGFACE_DEEPCODER_BASE_URL,
            max_retries=max_retries,
        )
        return OpenAIClientSpec(client=client, model=normalized_model, route="huggingface")
    if normalized_model == "meta-llama/llama-3.1-405b-instruct":
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
            max_retries=max_retries,
        )
        return OpenAIClientSpec(client=client, model=normalized_model, route="openrouter")
    if normalized_model.startswith("gemini"):
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=GEMINI_BASE_URL,
            max_retries=max_retries,
        )
        return OpenAIClientSpec(client=client, model=normalized_model, route="gemini")

    client = openai.OpenAI(max_retries=max_retries)
    return OpenAIClientSpec(client=client, model=normalized_model, route="openai")
