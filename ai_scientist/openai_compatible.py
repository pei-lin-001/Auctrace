import os
from functools import lru_cache

import openai

OPENAI_COMPATIBLE_BASE_URL_ENV = "OPENAI_COMPATIBLE_BASE_URL"
OPENAI_COMPATIBLE_API_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"
AIDER_OPENAI_API_BASE_ENV = "AIDER_OPENAI_API_BASE"
AIDER_OPENAI_API_KEY_ENV = "AIDER_OPENAI_API_KEY"
DEFAULT_MODEL_ENV = "AI_SCIENTIST_MODEL"
DEFAULT_REVIEW_MODEL_ENV = "AI_SCIENTIST_REVIEW_MODEL"
AUTO_MODEL_SENTINEL = "auto"
MODEL_DISCOVERY_TIMEOUT_SECONDS = 15


def is_openai_compatible_enabled():
    return os.getenv(OPENAI_COMPATIBLE_BASE_URL_ENV) is not None


def _candidate_base_urls(base_url):
    normalized = base_url.rstrip("/")
    candidates = [normalized]
    if not normalized.endswith("/v1"):
        candidates.append(f"{normalized}/v1")
    return candidates


def get_openai_compatible_settings():
    base_url = os.getenv(OPENAI_COMPATIBLE_BASE_URL_ENV)
    if not base_url:
        return None

    api_key = os.getenv(OPENAI_COMPATIBLE_API_KEY_ENV) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"{OPENAI_COMPATIBLE_BASE_URL_ENV} is set, but neither "
            f"{OPENAI_COMPATIBLE_API_KEY_ENV} nor OPENAI_API_KEY is available."
        )
    resolved_base_url = resolve_openai_compatible_base_url(base_url, api_key)
    return {"api_key": api_key, "base_url": resolved_base_url}

def _fetch_model_ids_once(base_url, api_key):
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
    )
    payload = client.models.list()
    data = getattr(payload, "data", None)
    if not isinstance(data, list):
        raise ValueError(
            "OpenAI-compatible /models response is invalid: expected a top-level "
            "'data' array."
        )
    model_ids = [
        getattr(item, "id", "").strip() for item in data if getattr(item, "id", None)
    ]
    if not model_ids:
        raise ValueError("OpenAI-compatible /models returned no model IDs.")
    return tuple(model_ids)


@lru_cache(maxsize=8)
def resolve_openai_compatible_base_url(base_url, api_key):
    last_error = None
    for candidate in _candidate_base_urls(base_url):
        try:
            _fetch_model_ids_once(candidate, api_key)
            if candidate != base_url.rstrip("/"):
                print(
                    "Normalized OpenAI-compatible base URL from "
                    f"{base_url.rstrip('/')} to {candidate} after probing /models."
                )
            return candidate
        except Exception as exc:
            last_error = exc
    raise ValueError(
        "Failed to discover an OpenAI-compatible /models endpoint from "
        f"{base_url}: {last_error}"
    ) from last_error


@lru_cache(maxsize=8)
def _fetch_model_ids(base_url, api_key):
    return _fetch_model_ids_once(base_url, api_key)


def fetch_openai_compatible_model_ids():
    settings = get_openai_compatible_settings()
    if settings is None:
        raise ValueError(
            f"{OPENAI_COMPATIBLE_BASE_URL_ENV} is not set, cannot fetch models."
        )
    return list(_fetch_model_ids(settings["base_url"], settings["api_key"]))


def resolve_requested_model(model):
    requested_model = model or os.getenv(DEFAULT_MODEL_ENV)
    if requested_model and requested_model.lower() != AUTO_MODEL_SENTINEL:
        return requested_model
    if not is_openai_compatible_enabled():
        raise ValueError(
            "No model specified. Pass --model, set AI_SCIENTIST_MODEL, or enable "
            "OPENAI_COMPATIBLE_BASE_URL to auto-select from /models."
        )

    selected_model = fetch_openai_compatible_model_ids()[0]
    print(
        "No model specified. Using the first model returned by the "
        f"OpenAI-compatible /models endpoint: {selected_model}."
    )
    return selected_model


def ensure_model_available(model):
    model_ids = fetch_openai_compatible_model_ids()
    if model not in model_ids:
        raise ValueError(
            f"Model {model} was not returned by the OpenAI-compatible /models "
            f"endpoint. Available models: {', '.join(model_ids)}"
        )
    return model


def is_model_available_via_openai_compatible(model):
    return is_openai_compatible_enabled() and model in fetch_openai_compatible_model_ids()


def configure_aider_openai_compatible():
    settings = get_openai_compatible_settings()
    if settings is None:
        return
    os.environ[AIDER_OPENAI_API_BASE_ENV] = settings["base_url"]
    os.environ[AIDER_OPENAI_API_KEY_ENV] = settings["api_key"]
