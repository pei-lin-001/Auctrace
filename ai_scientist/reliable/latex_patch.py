from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import SymbolicLatexError
from .latex_scaffold import editable_suffix, scaffold_prefix

_FACT_KEY_RE = re.compile(r"\\fact\{([^{}]+)\}")
_PARAM_KEY_RE = re.compile(r"\\param\{([^{}]+)\}")
_ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.DOTALL)
_SECTION_HEADER_RE = re.compile(r"^\\section\*?\{(?P<title>[^{}]+)\}\s*$", re.MULTILINE)
_LABEL_LINE_RE = re.compile(r"^\\label\{(?P<label>[^{}]+)\}\s*$", re.MULTILINE)


@dataclass(frozen=True)
class LatexPatchApplyResult:
    latex_text: str
    applied_ops: int
    notes: list[str]


def apply_latex_patch_ops(
    base_tex: str,
    patch: Mapping[str, Any],
    *,
    protect_placeholders: bool = True,
) -> LatexPatchApplyResult:
    ops = patch.get("ops")
    if not isinstance(ops, list):
        raise SymbolicLatexError("Patch JSON must include 'ops' as a list.")

    before_fact_keys, before_param_keys = _extract_placeholder_keys(base_tex)
    suffix = editable_suffix(base_tex)
    notes: list[str] = []
    for idx, raw_op in enumerate(ops):
        if not isinstance(raw_op, Mapping):
            raise SymbolicLatexError(f"Patch op #{idx} must be an object.")
        suffix, note = _apply_single_op(suffix, raw_op)
        notes.append(note)

    stitched = scaffold_prefix(base_tex) + suffix
    if protect_placeholders:
        _require_placeholder_keys_preserved(
            before_fact_keys,
            before_param_keys,
            stitched,
        )

    return LatexPatchApplyResult(
        latex_text=stitched,
        applied_ops=len(ops),
        notes=notes,
    )


def _extract_placeholder_keys(tex: str) -> tuple[set[str], set[str]]:
    facts = {m.group(1).strip() for m in _FACT_KEY_RE.finditer(tex) if m.group(1).strip()}
    params = {m.group(1).strip() for m in _PARAM_KEY_RE.finditer(tex) if m.group(1).strip()}
    return facts, params


def _require_placeholder_keys_preserved(
    before_fact_keys: set[str],
    before_param_keys: set[str],
    after_tex: str,
) -> None:
    after_fact_keys, after_param_keys = _extract_placeholder_keys(after_tex)
    missing_facts = sorted(before_fact_keys - after_fact_keys)
    missing_params = sorted(before_param_keys - after_param_keys)
    if missing_facts or missing_params:
        lines = [
            "Reflection patch is not allowed to remove existing \\fact{...} or \\param{...} placeholders.",
        ]
        if missing_facts:
            lines.append(f"Missing fact keys (showing up to 10): {missing_facts[:10]}")
        if missing_params:
            lines.append(f"Missing param keys (showing up to 10): {missing_params[:10]}")
        raise SymbolicLatexError("\n".join(lines))


def _apply_single_op(text: str, op: Mapping[str, Any]) -> tuple[str, str]:
    op_name = str(op.get("op") or "").strip()
    if not op_name:
        raise SymbolicLatexError("Patch op is missing required field 'op'.")

    if op_name == "replace_abstract":
        body = _require_text_field(op, "text")
        return _replace_abstract(text, body), "replace_abstract"
    if op_name == "replace_section":
        body = _require_text_field(op, "text")
        section_label = _optional_str(op.get("section_label"))
        section_title = _optional_str(op.get("section_title"))
        return (
            _replace_section(text, body, section_label=section_label, section_title=section_title),
            f"replace_section:{section_label or section_title or '?'}",
        )
    if op_name == "replace_between":
        start = _require_text_field(op, "start")
        end = _require_text_field(op, "end")
        replacement = _require_text_field(op, "text")
        return _replace_between(text, start=start, end=end, replacement=replacement), "replace_between"
    if op_name == "insert_after":
        anchor = _require_text_field(op, "anchor")
        insertion = _require_text_field(op, "text")
        return _insert_after(text, anchor=anchor, insertion=insertion), "insert_after"
    if op_name == "delete_between":
        start = _require_text_field(op, "start")
        end = _require_text_field(op, "end")
        return _replace_between(text, start=start, end=end, replacement=""), "delete_between"

    raise SymbolicLatexError(f"Unsupported patch op: {op_name!r}")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text_field(op: Mapping[str, Any], field: str) -> str:
    value = op.get(field)
    if value is None:
        raise SymbolicLatexError(f"Patch op missing required field: {field!r}")
    if not isinstance(value, str):
        raise SymbolicLatexError(f"Patch field {field!r} must be a string.")
    text = value.strip("\n")
    if not text.strip():
        raise SymbolicLatexError(f"Patch field {field!r} must not be empty.")
    return text


def _replace_abstract(tex: str, body: str) -> str:
    match = _ABSTRACT_RE.search(tex)
    if not match:
        raise SymbolicLatexError("Cannot find abstract environment to replace.")
    start, end = match.span("body")
    return tex[:start] + "\n" + body.strip() + "\n" + tex[end:]


def _replace_section(
    tex: str,
    body: str,
    *,
    section_label: str | None,
    section_title: str | None,
) -> str:
    start, end = _locate_section_body_span(
        tex,
        section_label=section_label,
        section_title=section_title,
    )
    replacement = "\n" + body.strip() + "\n"
    return tex[:start] + replacement + tex[end:]


def _locate_section_body_span(
    tex: str,
    *,
    section_label: str | None,
    section_title: str | None,
) -> tuple[int, int]:
    header_match = _locate_section_header(
        tex,
        section_label=section_label,
        section_title=section_title,
    )
    body_start = header_match["body_start"]
    next_header = _SECTION_HEADER_RE.search(tex, body_start)
    appendix = _find_next_line_anchor(tex, body_start, r"\\appendix\b")
    bibliography = _find_next_line_anchor(tex, body_start, r"\\bibliography\b")
    end_document = _find_next_line_anchor(tex, body_start, r"\\end\{document\}")
    candidates = [m.start() for m in (next_header, appendix, bibliography, end_document) if m is not None]
    body_end = min(candidates) if candidates else len(tex)
    return body_start, body_end


def _locate_section_header(
    tex: str,
    *,
    section_label: str | None,
    section_title: str | None,
) -> dict[str, int]:
    if section_label:
        label_re = re.compile(rf"^\\label\{{{re.escape(section_label)}\}}\s*$", re.MULTILINE)
        label_match = label_re.search(tex)
        if not label_match:
            raise SymbolicLatexError(f"Section label not found: {section_label!r}")
        header_match = _find_previous_section_header(tex, label_match.start())
        body_start = label_match.end()
        return {"header_start": header_match.start(), "body_start": body_start}

    if section_title:
        header_re = re.compile(
            rf"^\\section\*?\{{{re.escape(section_title)}\}}\s*$",
            re.MULTILINE,
        )
        matches = list(header_re.finditer(tex))
        if len(matches) != 1:
            raise SymbolicLatexError(
                f"Expected exactly one section with title {section_title!r}, got {len(matches)}."
            )
        header_match = matches[0]
        body_start = header_match.end()
        label_match = _LABEL_LINE_RE.search(tex, body_start)
        if label_match and label_match.start() == _line_start(tex, label_match.start()):
            if label_match.start() - body_start <= 8:
                body_start = label_match.end()
        return {"header_start": header_match.start(), "body_start": body_start}

    raise SymbolicLatexError("replace_section requires either section_label or section_title.")


def _find_previous_section_header(tex: str, pos: int) -> re.Match[str]:
    matches = list(_SECTION_HEADER_RE.finditer(tex, 0, pos))
    if not matches:
        raise SymbolicLatexError("Unable to locate section header before label.")
    return matches[-1]


def _replace_between(tex: str, *, start: str, end: str, replacement: str) -> str:
    start_index = tex.find(start)
    if start_index < 0:
        raise SymbolicLatexError("replace_between could not find start anchor.")
    end_index = tex.find(end, start_index + len(start))
    if end_index < 0:
        raise SymbolicLatexError("replace_between could not find end anchor.")
    if end_index <= start_index:
        raise SymbolicLatexError("replace_between start anchor occurs after end anchor.")
    return tex[: start_index + len(start)] + replacement + tex[end_index:]


def _insert_after(tex: str, *, anchor: str, insertion: str) -> str:
    first = tex.find(anchor)
    if first < 0:
        raise SymbolicLatexError("insert_after could not find anchor.")
    second = tex.find(anchor, first + len(anchor))
    if second >= 0:
        raise SymbolicLatexError("insert_after anchor is not unique; refuses to apply.")
    insert_at = first + len(anchor)
    return tex[:insert_at] + "\n" + insertion.strip() + "\n" + tex[insert_at:]


def _find_next_line_anchor(tex: str, pos: int, pattern: str) -> re.Match[str] | None:
    regex = re.compile(rf"^\\s*{pattern}", re.MULTILINE)
    return regex.search(tex, pos)


def _line_start(tex: str, pos: int) -> int:
    prev = tex.rfind("\n", 0, pos)
    return 0 if prev < 0 else prev + 1

