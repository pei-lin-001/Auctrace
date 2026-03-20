from __future__ import annotations

import re
from typing import Any


_SUSPICIOUS_NUMERIC_TEXT_RE = re.compile(
    # - decimals like 0.83 or 12.3
    r"\b\d+\.\d+\b"
    # - scientific like 1e-3 / 1.0E+2
    r"|\b\d+(?:\.\d+)?[eE][+-]?\d+\b"
    # - percents like 83% or 83\%
    r"|\b\d+(?:\.\d+)?\s*(?:%|\\%)"
)


def redact_numeric_text(text: str) -> str:
    """Redact likely experimental numeric values embedded in a text string.

    This is only used for prompt hygiene in symbolic-facts mode. It is not a
    truth source and is intentionally conservative: it targets common result
    formats (decimals/scientific/percent) and avoids touching plain integers.
    """

    return _SUSPICIOUS_NUMERIC_TEXT_RE.sub("<NUM>", text)


def redact_numeric_values(obj: Any) -> Any:
    """Recursively replace int/float (but not bool) values with a marker string.

    This is used to prevent numeric leakage from summaries into the writeup LLM
    when running in symbolic-facts mode.
    """

    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return "<NUM>"
    if isinstance(obj, str):
        return redact_numeric_text(obj)
    if isinstance(obj, list):
        return [redact_numeric_values(v) for v in obj]
    if isinstance(obj, dict):
        return {k: redact_numeric_values(v) for k, v in obj.items()}
    return str(obj)
