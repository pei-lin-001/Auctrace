from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .errors import SymbolicLatexError


_FACT_MACRO_RE = re.compile(r"\\fact\{[^{}]+\}")
_PARAM_MACRO_RE = re.compile(r"\\param\{[^{}]+\}")

# \begin{filecontents}{references.bib} ... \end{filecontents}
# \begin{filecontents*}{...} ... \end{filecontents*}
_FILECONTENTS_RE = re.compile(
    r"\\begin\{filecontents\*?\}\{[^}]+\}.*?\\end\{filecontents\*?\}",
    re.DOTALL,
)

# Common URL commands where numeric literals are not experimental results.
_URL_RE = re.compile(r"\\url\{[^{}]*\}")
_HREF_URL_PREFIX_RE = re.compile(r"\\href\{[^{}]*\}\{")

# - 0.1234, 12.34, 1.00 (>=2 decimal places)
_DECIMAL_RE = re.compile(r"\b\d+\.\d{2,}\b")

# - 1e-3, 1.0E+2
_SCIENTIFIC_RE = re.compile(r"\b\d+(?:\.\d+)?[eE][+-]?\d+\b")

# - 83%, 83\%, 83.2%, 83.2\%
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|\\%)")


_ALLOWED_COMMAND_SUFFIXES = (
    r"\textwidth",
    r"\linewidth",
    r"\columnwidth",
    r"\paperwidth",
    r"\paperheight",
    r"\baselineskip",
    r"\hsize",
    r"\vsize",
)

_ALLOWED_UNIT_SUFFIX_RE = re.compile(r"^(?:cm|mm|in|pt|em|ex)\b")
_SAFE_VERSION_PREFIX_RE = re.compile(
    r"(?:ubuntu|python|cuda|cudnn|macos|linux|pytorch|torch)\s*$",
    re.IGNORECASE,
)
_SAFE_SETUP_CONTEXT_RE = re.compile(
    r"(?:\\beta|learning rate|weight decay|dropout|temperature|momentum|epsilon|"
    r"label smoothing|fault injection|probabilit(?:y|ies)|hyperparameters?|defaults?)",
    re.IGNORECASE,
)
_SAFE_SPLIT_SUFFIX_RE = re.compile(r"^\s*(?:training|train|validation|val|dev|test)\b", re.IGNORECASE)
_RISKY_RESULT_CONTEXT_RE = re.compile(
    r"(?:accuracy|validation|test|dev|loss|score|wbr|f1|precision|recall|auc|"
    r"improv(?:e|ement)|best|final|mean)",
    re.IGNORECASE,
)
_CITE_COMMAND_RE = re.compile(r"\\cite[a-zA-Z]*\{")
_METRIC_CONTEXT_RE = re.compile(
    r"(?:accuracy|validation|test|dev|loss|score|wbr|f1|precision|recall|auc)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericLiteralFinding:
    literal: str
    start: int
    end: int
    context: str


def _strip_fact_macros(tex: str) -> str:
    return _FACT_MACRO_RE.sub("FACT", tex)


def _strip_param_macros(tex: str) -> str:
    return _PARAM_MACRO_RE.sub("PARAM", tex)


def _strip_filecontents_blocks(tex: str) -> str:
    # Embedded BibTeX (references.bib) frequently contains DOI/arXiv identifiers like 10.1145 or 1606.12345.
    # Those are not experimental results and should not be linted.
    return _FILECONTENTS_RE.sub("FILECONTENTS", tex)


def _strip_url_commands(tex: str) -> str:
    tex = _URL_RE.sub(r"\\url{URL}", tex)
    # Only remove the URL part; keep visible link text (second argument) intact.
    return _HREF_URL_PREFIX_RE.sub(r"\\href{URL}{", tex)


def _is_in_latex_comment(tex: str, pos: int) -> bool:
    line_start = tex.rfind("\n", 0, pos)
    line_start = 0 if line_start < 0 else line_start + 1
    prefix = tex[line_start:pos]
    for i, ch in enumerate(prefix):
        if ch != "%":
            continue
        # Count consecutive backslashes immediately before '%'.
        bs = 0
        j = i - 1
        while j >= 0 and prefix[j] == "\\":
            bs += 1
            j -= 1
        # Unescaped '%' starts a LaTeX comment.
        if bs % 2 == 0:
            return True
    return False


def _is_allowed_suffix(tex: str, end_index: int) -> bool:
    suffix = tex[end_index : end_index + 64]
    if any(suffix.startswith(cmd) for cmd in _ALLOWED_COMMAND_SUFFIXES):
        return True
    if suffix.startswith("\\"):
        # Unknown command suffix: treat as suspicious by default.
        return False
    if _ALLOWED_UNIT_SUFFIX_RE.match(suffix):
        return True
    return False


def _is_allowed_version_literal(tex: str, start: int) -> bool:
    prefix = tex[max(0, start - 24) : start]
    return bool(_SAFE_VERSION_PREFIX_RE.search(prefix))


def _is_allowed_setup_literal(tex: str, start: int, end: int) -> bool:
    context = tex[max(0, start - 80) : min(len(tex), end + 80)]
    if not _SAFE_SETUP_CONTEXT_RE.search(context):
        return False
    return not _RISKY_RESULT_CONTEXT_RE.search(context)


def _is_allowed_split_literal(tex: str, start: int, end: int) -> bool:
    prefix = tex[max(0, start - 64) : start].lower()
    suffix = tex[end : end + 24]
    return "split" in prefix and bool(_SAFE_SPLIT_SUFFIX_RE.match(suffix))


def _is_allowed_cited_literal(tex: str, start: int, end: int) -> bool:
    """Allow numeric literals that are clearly part of cited related-work claims.

    Symbolic-facts mode aims to prevent the LLM from sneaking in our experimental
    result values as raw numbers. Related work sometimes includes numeric claims;
    if they are immediately accompanied by citations and not in a metric context,
    we treat them as less risky and allow them to pass.
    """

    window = tex[max(0, start - 160) : min(len(tex), end + 160)]
    if not _CITE_COMMAND_RE.search(window):
        return False
    metric_window = tex[max(0, start - 80) : min(len(tex), end + 80)]
    return not _METRIC_CONTEXT_RE.search(metric_window)


def _iter_numeric_literals(tex: str, patterns: Iterable[re.Pattern[str]]):
    for pat in patterns:
        for m in pat.finditer(tex):
            yield m


def find_unanchored_numeric_literals(symbolic_tex: str) -> list[NumericLiteralFinding]:
    """Find numeric literals that look like experimental values.

    We deliberately avoid over-fitting to LaTeX formatting. This is a pragmatic
    "don't let LLM sneak in results numbers" lint:
    - It ignores anything inside \\fact{...}.
    - It flags decimals with >=2 decimal places and scientific notation.
    - It allows common layout suffixes like 0.5\\textwidth and unit suffixes like 0.25cm.
    """

    tex = _strip_fact_macros(symbolic_tex)
    tex = _strip_param_macros(tex)
    tex = _strip_filecontents_blocks(tex)
    tex = _strip_url_commands(tex)
    findings: list[NumericLiteralFinding] = []
    for m in _iter_numeric_literals(tex, (_DECIMAL_RE, _SCIENTIFIC_RE, _PERCENT_RE)):
        if _is_in_latex_comment(tex, m.start()):
            continue
        if _is_allowed_suffix(tex, m.end()):
            continue
        if _is_allowed_version_literal(tex, m.start()):
            continue
        if _is_allowed_setup_literal(tex, m.start(), m.end()):
            continue
        if _is_allowed_split_literal(tex, m.start(), m.end()):
            continue
        if _is_allowed_cited_literal(tex, m.start(), m.end()):
            continue
        start = max(0, m.start() - 40)
        end = min(len(tex), m.end() + 40)
        context = tex[start:end].replace("\n", "\\n")
        findings.append(
            NumericLiteralFinding(
                literal=m.group(0),
                start=m.start(),
                end=m.end(),
                context=context,
            )
        )
    return findings


def find_unanchored_numeric_literal_spans(symbolic_tex: str) -> list[NumericLiteralFinding]:
    """Like find_unanchored_numeric_literals, but returns spans in the original text.

    The original lint implementation strips \\fact/\\param and filecontents blocks to
    avoid false positives, but that shifts indices. For deterministic fixups we
    need stable spans in the original LaTeX string.
    """

    ignored = _ignored_spans(symbolic_tex)
    findings: list[NumericLiteralFinding] = []
    for m in _iter_numeric_literals(symbolic_tex, (_DECIMAL_RE, _SCIENTIFIC_RE, _PERCENT_RE)):
        if _is_in_spans(m.start(), ignored):
            continue
        if _is_in_latex_comment(symbolic_tex, m.start()):
            continue
        if _is_allowed_suffix(symbolic_tex, m.end()):
            continue
        if _is_allowed_version_literal(symbolic_tex, m.start()):
            continue
        if _is_allowed_setup_literal(symbolic_tex, m.start(), m.end()):
            continue
        if _is_allowed_split_literal(symbolic_tex, m.start(), m.end()):
            continue
        if _is_allowed_cited_literal(symbolic_tex, m.start(), m.end()):
            continue
        start = max(0, m.start() - 40)
        end = min(len(symbolic_tex), m.end() + 40)
        context = symbolic_tex[start:end].replace("\n", "\\n")
        findings.append(
            NumericLiteralFinding(
                literal=m.group(0),
                start=m.start(),
                end=m.end(),
                context=context,
            )
        )
    return findings


def _ignored_spans(tex: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    spans.extend(_collect_spans(_FACT_MACRO_RE, tex))
    spans.extend(_collect_spans(_PARAM_MACRO_RE, tex))
    spans.extend(_collect_spans(_FILECONTENTS_RE, tex))
    spans.extend(_collect_spans(_URL_RE, tex))
    spans.extend(_collect_spans(_HREF_URL_PREFIX_RE, tex))
    return spans


def _collect_spans(pattern: re.Pattern[str], tex: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(tex)]


def _is_in_spans(pos: int, spans: Iterable[tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= pos < end:
            return True
    return False


def require_no_unanchored_numeric_literals(symbolic_tex: str, *, max_findings: int = 8) -> None:
    findings = find_unanchored_numeric_literals(symbolic_tex)
    if not findings:
        return

    shown = findings[:max_findings]
    lines = [
        "Symbolic-facts mode forbids unanchored numeric literals (likely hallucinated or hand-copied results).",
        "Use \\fact{KEY} placeholders for experimental values. (Layout numbers like 0.5\\textwidth are allowed.)",
        f"Found {len(findings)} suspicious numeric literals. Showing up to {len(shown)}:",
    ]
    for f in shown:
        lines.append(f"- {f.literal!r} at [{f.start}:{f.end}] context={f.context!r}")
    raise SymbolicLatexError("\n".join(lines))
