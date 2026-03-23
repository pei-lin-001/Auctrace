from __future__ import annotations

import json
import re


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Best-effort JSON extraction for legacy prompts.

    Prefer Structured Outputs when the provider supports it. This helper remains
    as a compatibility fallback for providers without schema enforcement.
    """

    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue
    return None


def request_structured_output(json_schema: dict, *, name: str = "output") -> dict | None:
    """Return an OpenAI-style `response_format` dict for Structured Outputs.

    Callers should only use this when the selected model/provider supports
    Structured Outputs. For providers that do not support it, return None and
    fall back to text+regex parsing.
    """

    if not isinstance(json_schema, dict) or not json_schema:
        raise ValueError("json_schema must be a non-empty dict")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": json_schema,
            "strict": True,
        },
    }

