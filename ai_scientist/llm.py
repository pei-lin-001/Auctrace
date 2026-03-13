import json
import os
import re

import anthropic
import backoff
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

MAX_NUM_TOKENS = 4096
OPENAI_COMPATIBLE_BASE_URL_ENV = "OPENAI_COMPATIBLE_BASE_URL"
OPENAI_COMPATIBLE_API_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"
AIDER_OPENAI_API_BASE_ENV = "AIDER_OPENAI_API_BASE"
AIDER_OPENAI_API_KEY_ENV = "AIDER_OPENAI_API_KEY"
OPENAI_PROVIDER = "openai"
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
ANTHROPIC_PROVIDER = "anthropic"
DEEPSEEK_PROVIDER = "deepseek"
OPENROUTER_PROVIDER = "openrouter"
GEMINI_PROVIDER = "gemini"

DEEPSEEK_MODEL_ALIASES = {
    "deepseek-coder-v2-0724": "deepseek-coder",
}

OPENROUTER_MODEL_ALIASES = {
    "llama3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3-1-405b-instruct": "meta-llama/llama-3.1-405b-instruct",
}

AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # OpenRouter models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "deepseek-coder-v2-0724",
    # Google Gemini models
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
    "llama-3-1-405b-instruct",
]


def _tag_client(client, provider):
    setattr(client, "_ai_scientist_provider", provider)
    return client


def _get_openai_compatible_settings():
    base_url = os.getenv(OPENAI_COMPATIBLE_BASE_URL_ENV)
    if not base_url:
        return None

    api_key = os.getenv(OPENAI_COMPATIBLE_API_KEY_ENV) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"{OPENAI_COMPATIBLE_BASE_URL_ENV} is set, but neither "
            f"{OPENAI_COMPATIBLE_API_KEY_ENV} nor OPENAI_API_KEY is available."
        )
    return {"api_key": api_key, "base_url": base_url}


def _uses_openai_compatible(model):
    if model.startswith("claude-") or model.startswith("bedrock"):
        return False
    if model.startswith("vertex_ai"):
        return False
    if "gpt" in model or "o1" in model or "o3" in model:
        return False
    if model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]:
        return False
    if model in DEEPSEEK_MODEL_ALIASES:
        return False
    if model in OPENROUTER_MODEL_ALIASES:
        return False
    if "gemini" in model:
        return False
    return os.getenv(OPENAI_COMPATIBLE_BASE_URL_ENV) is not None


def _configure_aider_openai_compatible(model):
    if not _uses_openai_compatible(model):
        return
    settings = _get_openai_compatible_settings()
    os.environ[AIDER_OPENAI_API_BASE_ENV] = settings["base_url"]
    os.environ[AIDER_OPENAI_API_KEY_ENV] = settings["api_key"]


def resolve_aider_model_name(model):
    if model in DEEPSEEK_MODEL_ALIASES:
        return f"deepseek/{DEEPSEEK_MODEL_ALIASES[model]}"
    if model == "deepseek-reasoner":
        return "deepseek/deepseek-reasoner"
    if model in OPENROUTER_MODEL_ALIASES:
        return f"openrouter/{OPENROUTER_MODEL_ALIASES[model]}"
    if _uses_openai_compatible(model):
        _configure_aider_openai_compatible(model)
        return f"openai/{model}"
    return model


def _get_api_style(client, model):
    provider = getattr(client, "_ai_scientist_provider", None)
    if provider == ANTHROPIC_PROVIDER or model.startswith("claude-"):
        return ANTHROPIC_PROVIDER
    if "o1" in model or "o3" in model:
        return "openai-reasoning"
    if model == "deepseek-reasoner":
        return "deepseek-reasoner"
    if provider in {
        OPENAI_PROVIDER,
        OPENAI_COMPATIBLE_PROVIDER,
        DEEPSEEK_PROVIDER,
        OPENROUTER_PROVIDER,
        GEMINI_PROVIDER,
    }:
        return "openai-chat"
    raise ValueError(f"Model {model} is not mapped to a supported API style.")


def _resolve_api_model_name(model):
    if model in DEEPSEEK_MODEL_ALIASES:
        return DEEPSEEK_MODEL_ALIASES[model]
    if model in OPENROUTER_MODEL_ALIASES:
        return OPENROUTER_MODEL_ALIASES[model]
    return model


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
        n_responses=1,
):
    if msg_history is None:
        msg_history = []

    api_style = _get_api_style(client, model)
    request_model = _resolve_api_model_name(model)

    if api_style == "openai-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
        msg,
        client,
        model,
        system_message,
        print_debug=False,
        msg_history=None,
        temperature=0.75,
):
    if msg_history is None:
        msg_history = []

    api_style = _get_api_style(client, model)
    request_model = _resolve_api_model_name(model)

    if api_style == ANTHROPIC_PROVIDER:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=request_model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif api_style == "openai-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif api_style == "openai-reasoning":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif api_style == "deepseek-reasoner":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=request_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model):
    if model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return _tag_client(anthropic.Anthropic(), ANTHROPIC_PROVIDER), model
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return _tag_client(anthropic.AnthropicBedrock(), ANTHROPIC_PROVIDER), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return _tag_client(anthropic.AnthropicVertex(), ANTHROPIC_PROVIDER), client_model
    elif model in ["deepseek-chat", "deepseek-reasoner", "deepseek-coder"]:
        print(f"Using DeepSeek API with model {model}.")
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        return _tag_client(client, DEEPSEEK_PROVIDER), model
    elif model in DEEPSEEK_MODEL_ALIASES:
        client_model = DEEPSEEK_MODEL_ALIASES[model]
        print(f"Using DeepSeek API with model {client_model} (requested {model}).")
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        return _tag_client(client, DEEPSEEK_PROVIDER), client_model
    elif model in OPENROUTER_MODEL_ALIASES:
        client_model = OPENROUTER_MODEL_ALIASES[model]
        print(f"Using OpenRouter API with model {client_model}.")
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        return _tag_client(client, OPENROUTER_PROVIDER), client_model
    elif "gemini" in model:
        print(f"Using Gemini OpenAI endpoint with model {model}.")
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return _tag_client(client, GEMINI_PROVIDER), model
    elif _uses_openai_compatible(model):
        settings = _get_openai_compatible_settings()
        print(
            f"Using OpenAI-compatible API with model {model} via "
            f"{settings['base_url']}."
        )
        client = openai.OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
        )
        return _tag_client(client, OPENAI_COMPATIBLE_PROVIDER), model
    elif 'gpt' in model or "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        return _tag_client(openai.OpenAI(), OPENAI_PROVIDER), model
    else:
        raise ValueError(
            f"Model {model} not supported. Set {OPENAI_COMPATIBLE_BASE_URL_ENV} "
            "to use an arbitrary OpenAI-compatible model."
        )
