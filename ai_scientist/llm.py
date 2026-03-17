import json
import re
from typing import Any

import backoff
import openai

from ai_scientist.openai_chat_completions import create_chat_completion_with_fallback
from ai_scientist.openai_compatible import (
    create_openai_client,
    is_openai_reasoning_model,
    max_output_token_limit,
)
from ai_scientist.utils.token_tracker import track_token_usage

MAX_NUM_TOKENS = max_output_token_limit()


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if is_openai_reasoning_model(model):
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
    else:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = _batch_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
            n_responses=n_responses,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def _batch_llm_call(client, model, temperature, system_message, prompt, n_responses):
    messages = [
        {"role": "system", "content": system_message},
        *prompt,
    ]
    request_kwargs = {
        "temperature": temperature,
        "max_tokens": MAX_NUM_TOKENS,
        "n": n_responses,
        "stop": None,
    }
    completion, _ = create_chat_completion_with_fallback(
        client,
        messages=messages,
        model=model,
        fallback_model=None,
        request_kwargs=request_kwargs,
    )
    return completion


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if is_openai_reasoning_model(model):
        messages = [
            {"role": "user", "content": system_message},
            *prompt,
        ]
        request_kwargs = {
            "temperature": 1,
            "n": 1,
            "seed": 0,
        }
    else:
        messages = [
            {"role": "system", "content": system_message},
            *prompt,
        ]
        request_kwargs = {
            "temperature": temperature,
            "max_tokens": MAX_NUM_TOKENS,
            "n": 1,
            "stop": None,
        }

    completion, _ = create_chat_completion_with_fallback(
        client,
        messages=messages,
        model=model,
        fallback_model=None,
        request_kwargs=request_kwargs,
    )
    return completion


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = make_llm_call(
        client,
        model,
        temperature,
        system_message=system_message,
        prompt=new_msg_history,
    )
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None: 
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


def create_client(model) -> tuple[Any, str]:
    spec = create_openai_client(model)
    print(f"Using {spec.route} API with model {spec.model}.")
    return spec.client, spec.model
