import json
import os
import re

import anthropic
import backoff
import openai

from ai_scientist.openai_compatible import (
    OPENAI_COMPATIBLE_BASE_URL_ENV,
    configure_aider_openai_compatible,
    get_openai_compatible_settings,
    is_model_available_via_openai_compatible,
    resolve_requested_model,
)

MAX_NUM_TOKENS = 4096
OPENAI_PROVIDER = "openai"
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"
ANTHROPIC_PROVIDER = "anthropic"
DEEPSEEK_PROVIDER = "deepseek"
OPENROUTER_PROVIDER = "openrouter"
GEMINI_PROVIDER = "gemini"
REASONING_MODEL_PREFIXES = ("o1", "o3")
def _tag_client(client, provider):
    setattr(client, "_ai_scientist_provider", provider)
    return client
def _is_anthropic_model(model):
    return model.startswith("claude-")
def _is_bedrock_claude_model(model):
    return model.startswith("bedrock/") and "claude" in model
def _is_vertex_claude_model(model):
    return model.startswith("vertex_ai/") and "claude" in model
def _is_openai_native_model(model):
    lowered = model.lower()
    return lowered.startswith("gpt") or lowered.startswith(REASONING_MODEL_PREFIXES)
def _is_openrouter_model(model):
    return model.startswith("openrouter/")
def _is_deepseek_model(model):
    return model.startswith("deepseek")
def _is_gemini_model(model):
    return model.startswith("gemini")
def _should_use_openai_compatible(model):
    if _is_anthropic_model(model):
        return False
    if _is_bedrock_claude_model(model) or _is_vertex_claude_model(model):
        return False
    if _is_openai_native_model(model):
        return False
    if _is_openrouter_model(model):
        return False
    return is_model_available_via_openai_compatible(model)
def _strip_provider_prefix(model):
    if _is_bedrock_claude_model(model) or _is_vertex_claude_model(model):
        return model.split("/", 1)[1]
    if _is_openrouter_model(model):
        return model.split("/", 1)[1]
    return model
def resolve_aider_model_name(model):
    resolved_model = resolve_requested_model(model)
    if _should_use_openai_compatible(resolved_model):
        configure_aider_openai_compatible()
        return f"openai/{resolved_model}"
    if _is_deepseek_model(resolved_model):
        return f"deepseek/{resolved_model}"
    if _is_openrouter_model(resolved_model):
        return resolved_model
    return resolved_model
def _get_api_style(client, model):
    provider = getattr(client, "_ai_scientist_provider", None)
    if provider == ANTHROPIC_PROVIDER:
        return ANTHROPIC_PROVIDER
    if provider == OPENAI_PROVIDER and model.lower().startswith(REASONING_MODEL_PREFIXES):
        return "openai-reasoning"
    if provider == DEEPSEEK_PROVIDER and "reasoner" in model:
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
    if api_style == "openai-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=_strip_provider_prefix(model),
            messages=[{"role": "system", "content": system_message}, *new_msg_history],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [choice.message.content for choice in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": item}]
            for item in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            item, history = get_response_from_llm(
                msg=msg,
                client=client,
                model=model,
                system_message=system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(item)
            new_msg_history.append(history)
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for idx, history_item in enumerate(new_msg_history[0]):
            print(f'{idx}, {history_item["role"]}: {history_item["content"]}')
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
    request_model = _strip_provider_prefix(model)
    if api_style == ANTHROPIC_PROVIDER:
        new_msg_history = msg_history + [{"role": "user", "content": [{"type": "text", "text": msg}]}]
        response = client.messages.create(
            model=request_model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [{"role": "assistant", "content": [{"type": "text", "text": content}]}]
    elif api_style == "openai-chat":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=request_model,
            messages=[{"role": "system", "content": system_message}, *new_msg_history],
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
            messages=[{"role": "user", "content": system_message}, *new_msg_history],
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
            messages=[{"role": "system", "content": system_message}, *new_msg_history],
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
        for idx, history_item in enumerate(new_msg_history):
            print(f'{idx}, {history_item["role"]}: {history_item["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history
def extract_json_between_markers(llm_output):
    matches = re.findall(r"```json(.*?)```", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{.*?\}", llm_output, re.DOTALL)
    for json_string in matches:
        candidate = json_string.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(re.sub(r"[\x00-\x1F\x7F]", "", candidate))
            except json.JSONDecodeError:
                continue
    return None
def create_client(model):
    resolved_model = resolve_requested_model(model)
    if _is_bedrock_claude_model(resolved_model):
        client_model = _strip_provider_prefix(resolved_model)
        print(f"Using Amazon Bedrock with model {client_model}.")
        return _tag_client(anthropic.AnthropicBedrock(), ANTHROPIC_PROVIDER), client_model
    if _is_vertex_claude_model(resolved_model):
        client_model = _strip_provider_prefix(resolved_model)
        print(f"Using Vertex AI with model {client_model}.")
        return _tag_client(anthropic.AnthropicVertex(), ANTHROPIC_PROVIDER), client_model
    if _is_openrouter_model(resolved_model):
        client_model = _strip_provider_prefix(resolved_model)
        print(f"Using OpenRouter API with model {client_model}.")
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        return _tag_client(client, OPENROUTER_PROVIDER), client_model
    if _should_use_openai_compatible(resolved_model):
        settings = get_openai_compatible_settings()
        print(
            f"Using OpenAI-compatible API with model {resolved_model} via "
            f"{settings['base_url']}."
        )
        client = openai.OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
        )
        return _tag_client(client, OPENAI_COMPATIBLE_PROVIDER), resolved_model
    if _is_anthropic_model(resolved_model):
        print(f"Using Anthropic API with model {resolved_model}.")
        return _tag_client(anthropic.Anthropic(), ANTHROPIC_PROVIDER), resolved_model
    if _is_deepseek_model(resolved_model):
        print(f"Using DeepSeek API with model {resolved_model}.")
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        return _tag_client(client, DEEPSEEK_PROVIDER), resolved_model
    if _is_gemini_model(resolved_model):
        print(f"Using Gemini OpenAI endpoint with model {resolved_model}.")
        client = openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        return _tag_client(client, GEMINI_PROVIDER), resolved_model
    if _is_openai_native_model(resolved_model):
        print(f"Using OpenAI API with model {resolved_model}.")
        return _tag_client(openai.OpenAI(), OPENAI_PROVIDER), resolved_model
    raise ValueError(
        f"Model {resolved_model} has no configured provider. Use a native OpenAI "
        "model name, a provider prefix such as openrouter/<model-id>, or set "
        f"{OPENAI_COMPATIBLE_BASE_URL_ENV} to use OpenAI-compatible discovery."
    )
