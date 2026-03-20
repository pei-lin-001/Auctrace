from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .errors import SymbolicLatexError
from .facts import FactStore, format_fact_value_for_latex
from .numeric_lint import NumericLiteralFinding, find_unanchored_numeric_literal_spans
from .params import ParamStore, format_param_value_for_latex


@dataclass(frozen=True)
class NumericBackanchorResult:
    latex_text: str
    replacements: Dict[str, int]


def backanchor_numeric_literals(
    symbolic_tex: str,
    *,
    store: FactStore,
    param_store: ParamStore,
) -> NumericBackanchorResult:
    """Replace unanchored numeric literals with \\fact{KEY}/\\param{KEY} when unambiguous.

    This is deterministic and refuses to guess:
    - Every suspicious numeric literal must map to exactly one fact *or* param value.
    - If any literal is missing or ambiguous, the function raises SymbolicLatexError.
    """

    findings = find_unanchored_numeric_literal_spans(symbolic_tex)
    if not findings:
        return NumericBackanchorResult(
            latex_text=symbolic_tex,
            replacements={},
        )

    fact_value_to_keys = _value_to_fact_keys(store)
    param_value_to_keys = _value_to_param_keys(param_store)

    replacements: Dict[str, int] = {}
    spans: list[tuple[int, int, str]] = []
    for finding in findings:
        literal = symbolic_tex[finding.start : finding.end]
        replacement = _replacement_for_literal(
            literal,
            fact_value_to_keys=fact_value_to_keys,
            param_value_to_keys=param_value_to_keys,
        )
        spans.append((finding.start, finding.end, replacement))
        replacements[literal] = replacements.get(literal, 0) + 1

    out = symbolic_tex
    for start, end, repl in sorted(spans, key=lambda x: x[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return NumericBackanchorResult(
        latex_text=out,
        replacements=replacements,
    )


def _value_to_fact_keys(store: FactStore) -> Mapping[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for key, record in store.facts.items():
        rendered = format_fact_value_for_latex(record)
        mapping.setdefault(rendered, []).append(key)
    return mapping


def _value_to_param_keys(store: ParamStore) -> Mapping[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for key, record in store.params.items():
        rendered = format_param_value_for_latex(record)
        mapping.setdefault(rendered, []).append(key)
    return mapping


def _replacement_for_literal(
    literal: str,
    *,
    fact_value_to_keys: Mapping[str, list[str]],
    param_value_to_keys: Mapping[str, list[str]],
) -> str:
    fact_keys = fact_value_to_keys.get(literal, [])
    param_keys = param_value_to_keys.get(literal, [])
    candidates = [(k, "fact") for k in fact_keys] + [(k, "param") for k in param_keys]
    if len(candidates) != 1:
        raise SymbolicLatexError(
            "Numeric backanchor failed: literal is missing or ambiguous.\n"
            f"- literal: {literal!r}\n"
            f"- fact_key_candidates: {sorted(fact_keys)[:8]}\n"
            f"- param_key_candidates: {sorted(param_keys)[:8]}\n"
        )
    key, kind = candidates[0]
    if kind == "fact":
        return f"\\fact{{{key}}}"
    return f"\\param{{{key}}}"

