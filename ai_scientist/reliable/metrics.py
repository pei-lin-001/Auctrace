from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NumericToken:
    text: str
    start: int
    end: int
    kind: str
    context: str


_FACT_MACRO_RE = re.compile(r"\\fact\{[^{}]+\}")
_PARAM_MACRO_RE = re.compile(r"\\param\{[^{}]+\}")

# \begin{filecontents}{references.bib} ... \end{filecontents}
_FILECONTENTS_RE = re.compile(
    r"\\begin\{filecontents\*?\}\{[^}]+\}.*?\\end\{filecontents\*?\}",
    re.DOTALL,
)

_URL_RE = re.compile(r"\\url\{[^{}]*\}")
_HREF_URL_PREFIX_RE = re.compile(r"\\href\{[^{}]*\}\{")

# We evaluate NAS on LaTeX source. Strip includegraphics to avoid leaking "accuracy"
# keywords from filenames into integer-context detection.
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*\}")

_DECIMAL_RE = re.compile(r"\b\d+\.\d{2,}\b")
_SCIENTIFIC_RE = re.compile(r"\b\d+(?:\.\d+)?[eE][+-]?\d+\b")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|\\%)")
_INT_RE = re.compile(r"\b\d+\b")

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
    r"(?:ubuntu|python|cuda|cudnn|macos|linux|pytorch|torch|version|v)\s*$",
    re.IGNORECASE,
)
_SAFE_SETUP_CONTEXT_RE = re.compile(
    r"(?:\\beta|learning rate|weight decay|dropout|temperature|momentum|epsilon|"
    r"label smoothing|fault injection|probabilit(?:y|ies)|hyperparameters?|defaults?|"
    r"random seeds?|seeds?)",
    re.IGNORECASE,
)
_SAFE_SPLIT_SUFFIX_RE = re.compile(
    r"^\s*(?:training|train|validation|val|dev|test)\b", re.IGNORECASE
)
_CITE_COMMAND_RE = re.compile(r"\\cite[a-zA-Z]*\{")
_REF_COMMAND_RE = re.compile(r"\\ref\{")
_METRIC_CONTEXT_RE = re.compile(
    r"(?:accuracy|validation|test|dev|loss|score|wbr|f1|precision|recall|auc)\b",
    re.IGNORECASE,
)

# v3: result-context integers, word-boundary matched.
_RESULT_INDICATOR_RE = re.compile(
    r"\b(?:accuracy|loss|score|precision|recall|f1|candidates|samples|count|total|num|number)\b",
    re.IGNORECASE,
)

# v3: overlap heuristic for NIA. We intentionally avoid 'achieves'/'demonstrates'
# alone, because they are often used in non-comparative numeric statements.
_COMPARATIVE_LANG_RE = re.compile(
    r"\b(?:higher|lower|outperform(?:s|ed|ing)?|improv(?:e|es|ed|ing)|"
    r"reduc(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
    r"better|worse|comparable|similar|surpass(?:es|ed|ing)?|exceed(?:s|ed|ing)?|"
    r"degrad(?:e|es|ed|ing)|maintain(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)


def contains_comparative_language(text: str) -> bool:
    if not text:
        return False
    return bool(_COMPARATIVE_LANG_RE.search(text))


def _strip_non_text(tex: str) -> str:
    tex = _FACT_MACRO_RE.sub("FACT", tex)
    tex = _PARAM_MACRO_RE.sub("PARAM", tex)
    tex = _FILECONTENTS_RE.sub("FILECONTENTS", tex)
    tex = _URL_RE.sub(r"\\url{URL}", tex)
    tex = _HREF_URL_PREFIX_RE.sub(r"\\href{URL}{", tex)
    tex = _INCLUDEGRAPHICS_RE.sub(r"\\includegraphics{FIG}", tex)
    return tex


def _is_in_latex_comment(tex: str, pos: int) -> bool:
    line_start = tex.rfind("\n", 0, pos)
    line_start = 0 if line_start < 0 else line_start + 1
    prefix = tex[line_start:pos]
    for i, ch in enumerate(prefix):
        if ch != "%":
            continue
        bs = 0
        j = i - 1
        while j >= 0 and prefix[j] == "\\":
            bs += 1
            j -= 1
        if bs % 2 == 0:
            return True
    return False


def _is_allowed_suffix(tex: str, end_index: int) -> bool:
    suffix = tex[end_index : end_index + 64]
    if any(suffix.startswith(cmd) for cmd in _ALLOWED_COMMAND_SUFFIXES):
        return True
    if suffix.startswith("\\"):
        return False
    if _ALLOWED_UNIT_SUFFIX_RE.match(suffix):
        return True
    return False


def _is_allowed_version_literal(tex: str, start: int) -> bool:
    prefix = tex[max(0, start - 24) : start]
    return bool(_SAFE_VERSION_PREFIX_RE.search(prefix))


def _is_allowed_setup_literal(tex: str, start: int, end: int) -> bool:
    context = tex[max(0, start - 80) : min(len(tex), end + 80)]
    return bool(_SAFE_SETUP_CONTEXT_RE.search(context))


def _is_allowed_split_literal(tex: str, start: int, end: int) -> bool:
    suffix = tex[end : end + 80]
    return bool(_SAFE_SPLIT_SUFFIX_RE.match(suffix))


def _is_allowed_cited_or_ref_literal(tex: str, start: int, end: int) -> bool:
    window = tex[max(0, start - 160) : min(len(tex), end + 160)]
    if _REF_COMMAND_RE.search(window):
        return True
    if not _CITE_COMMAND_RE.search(window):
        return False
    metric_window = tex[max(0, start - 80) : min(len(tex), end + 80)]
    return not _METRIC_CONTEXT_RE.search(metric_window)


def _is_list_enum_index(tex: str, start: int, end: int) -> bool:
    left = tex[start - 1] if start > 0 else ""
    right = tex[end] if end < len(tex) else ""
    if left == "(" and right == ")":
        return True
    if left == "[" and right == "]":
        return True
    return False


def extract_result_numeric_tokens(tex: str) -> list[NumericToken]:
    """Extract numeric tokens as defined by MAC v3's T_result rule.

    This is used for NAS computation on LaTeX source (template.tex and template.rendered.tex).
    """

    stripped = _strip_non_text(tex)
    blocked_spans: list[tuple[int, int]] = []
    tokens: list[NumericToken] = []

    for kind, pat in (
        ("decimal", _DECIMAL_RE),
        ("scientific", _SCIENTIFIC_RE),
        ("percent", _PERCENT_RE),
    ):
        for m in pat.finditer(stripped):
            if _is_in_latex_comment(stripped, m.start()):
                continue
            if _is_allowed_suffix(stripped, m.end()):
                continue
            if _is_allowed_version_literal(stripped, m.start()):
                continue
            if _is_allowed_setup_literal(stripped, m.start(), m.end()):
                continue
            if _is_allowed_split_literal(stripped, m.start(), m.end()):
                continue
            if _is_allowed_cited_or_ref_literal(stripped, m.start(), m.end()):
                continue
            context = stripped[max(0, m.start() - 40) : min(len(stripped), m.end() + 40)]
            context = context.replace("\n", "\\n")
            tokens.append(
                NumericToken(
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    kind=kind,
                    context=context,
                )
            )
            blocked_spans.append((m.start(), m.end()))

    def _overlaps(span: tuple[int, int], other: tuple[int, int]) -> bool:
        return not (span[1] <= other[0] or span[0] >= other[1])

    for m in _INT_RE.finditer(stripped):
        span = (m.start(), m.end())
        if any(_overlaps(span, other) for other in blocked_spans):
            continue
        # Skip pieces of decimals like 0.7 => tokenizes into '0' and '7' otherwise.
        if (m.start() > 0 and stripped[m.start() - 1] == ".") or (
            m.end() < len(stripped) and stripped[m.end()] == "."
        ):
            continue
        if _is_list_enum_index(stripped, m.start(), m.end()):
            continue
        if _is_in_latex_comment(stripped, m.start()):
            continue
        if _is_allowed_suffix(stripped, m.end()):
            continue
        if _is_allowed_version_literal(stripped, m.start()):
            continue
        if _is_allowed_setup_literal(stripped, m.start(), m.end()):
            continue
        if _is_allowed_split_literal(stripped, m.start(), m.end()):
            continue
        if _is_allowed_cited_or_ref_literal(stripped, m.start(), m.end()):
            continue
        window = stripped[max(0, m.start() - 80) : min(len(stripped), m.end() + 80)]
        if not _RESULT_INDICATOR_RE.search(window):
            continue
        context = stripped[max(0, m.start() - 40) : min(len(stripped), m.end() + 40)]
        context = context.replace("\n", "\\n")
        tokens.append(
            NumericToken(
                text=m.group(0),
                start=m.start(),
                end=m.end(),
                kind="int",
                context=context,
            )
        )

    tokens.sort(key=lambda t: (t.start, t.end))
    return tokens


def compute_nas_from_latex(
    *,
    symbolic_tex: str,
    rendered_tex: str,
) -> dict[str, Any]:
    """Compute NAS by comparing symbolic vs rendered LaTeX."""

    unanchored = extract_result_numeric_tokens(symbolic_tex)
    total = extract_result_numeric_tokens(rendered_tex)
    total_count = len(total)
    unanchored_count = len(unanchored)
    nas = None if total_count == 0 else 1.0 - (unanchored_count / total_count)
    return {
        "NAS": nas,
        "T_result_count": total_count,
        "unanchored_count": unanchored_count,
        "unanchored_sample": [
            {
                "text": t.text,
                "kind": t.kind,
                "context": t.context,
            }
            for t in unanchored[:20]
        ],
    }


def compute_eru(
    *,
    fact_store: Mapping[str, Any],
    used_facts: Mapping[str, Any],
) -> dict[str, Any]:
    facts = fact_store.get("facts", [])
    total = len(facts)
    used_keys = set(used_facts.get("used_keys", []) or [])
    used_count = len(used_keys)
    eru = None if total == 0 else used_count / total

    total_by_stage = Counter()
    for f in facts:
        stage = ((f.get("provenance") or {}).get("stage_name")) or "unknown"
        total_by_stage[stage] += 1

    used_by_stage = Counter()
    for rec in used_facts.get("facts", []) or []:
        stage = ((rec.get("provenance") or {}).get("stage_name")) or "unknown"
        used_by_stage[stage] += 1

    eru_by_stage = {}
    for stage, denom in total_by_stage.items():
        numer = used_by_stage.get(stage, 0)
        eru_by_stage[stage] = None if denom == 0 else numer / denom

    return {
        "ERU": eru,
        "used_keys": used_count,
        "total_keys": total,
        "used_by_stage": dict(used_by_stage),
        "total_by_stage": dict(total_by_stage),
        "ERU_by_stage": dict(eru_by_stage),
    }


def compute_ics_auto(
    outsider_audit: Mapping[str, Any],
    *,
    k: int = 15,
    severity_weights: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    weights = dict(severity_weights or {"high": 3, "medium": 2, "low": 1})
    issues = outsider_audit.get("issues", []) or []
    weighted = 0
    counted: list[dict[str, Any]] = []
    for it in issues:
        category = str(it.get("category") or "")
        if category not in {"contradiction", "unsupported_claim"}:
            continue
        severity = str(it.get("severity") or "")
        w = weights.get(severity, 0)
        weighted += w
        counted.append(
            {
                "issue_id": it.get("issue_id"),
                "severity": severity,
                "category": category,
                "weight": w,
            }
        )
    ics = 1.0 - min(weighted / k, 1.0) if k > 0 else None
    return {
        "ICS_auto": ics,
        "weighted_issues": weighted,
        "K": k,
        "counted_issues": counted,
    }


def compute_ceb(
    *,
    claim_ledger: Mapping[str, Any],
    fact_store: Mapping[str, Any] | None,
    artifact_manifest: Mapping[str, Any] | None,
    claim_trace_index: Mapping[str, Any] | None,
) -> dict[str, Any]:
    claims = claim_ledger.get("claims", []) or []
    total_claims = len(claims)
    if total_claims == 0:
        return {
            "CEB": None,
            "total_claims": 0,
            "bound_claims": 0,
            "by_type": {},
            "failure_samples": [],
        }

    facts = set()
    if fact_store is not None:
        facts = {f.get("key") for f in fact_store.get("facts", []) or [] if f.get("key")}

    artifacts = set()
    if artifact_manifest is not None:
        artifacts = {
            a.get("artifact_id")
            for a in artifact_manifest.get("artifacts", []) or []
            if a.get("artifact_id")
        }

    sections = set()
    labels = set()
    if claim_trace_index is not None:
        sections = set(claim_trace_index.get("sections", []) or [])
        labels = set(claim_trace_index.get("labels", []) or [])

    by_type = defaultdict(lambda: {"total": 0, "bound": 0})
    failures: list[dict[str, Any]] = []

    for c in claims:
        claim_id = c.get("claim_id")
        claim_type = str(c.get("claim_type") or "other")
        by_type[claim_type]["total"] += 1

        location = c.get("location_hints") or {}
        section = str(location.get("section") or "")
        latex_anchor = str(location.get("latex_anchor") or "")

        section_ok = bool(section) and (not sections or section in sections)
        # We accept missing labels index for older runs, but if present require anchor membership.
        anchor_ok = True
        if labels and latex_anchor:
            anchor_ok = latex_anchor in labels

        arts = c.get("supporting_artifacts") or []
        artifacts_ok = bool(arts) and (not artifacts or all(a in artifacts for a in arts))

        sfacts = c.get("supporting_facts") or []
        facts_ok = bool(sfacts) and (not facts or all(k in facts for k in sfacts))

        bound = False
        reasons: list[str] = []
        if claim_type in {"numeric", "comparative"}:
            if not facts_ok:
                reasons.append("missing_or_invalid_supporting_facts")
            if not artifacts_ok:
                reasons.append("missing_or_invalid_supporting_artifacts")
            if not section_ok:
                reasons.append("missing_or_unknown_section")
            if not anchor_ok:
                reasons.append("missing_or_unknown_latex_anchor")
            bound = facts_ok and artifacts_ok and section_ok and anchor_ok
        elif claim_type == "qualitative":
            if not artifacts_ok:
                reasons.append("missing_or_invalid_supporting_artifacts")
            if not section_ok:
                reasons.append("missing_or_unknown_section")
            if not anchor_ok:
                reasons.append("missing_or_unknown_latex_anchor")
            bound = artifacts_ok and section_ok and anchor_ok
        elif claim_type == "citation":
            if not section_ok:
                reasons.append("missing_or_unknown_section")
            if not anchor_ok:
                reasons.append("missing_or_unknown_latex_anchor")
            bound = section_ok and anchor_ok
        else:
            if not artifacts_ok:
                reasons.append("missing_or_invalid_supporting_artifacts")
            if not section_ok:
                reasons.append("missing_or_unknown_section")
            if not anchor_ok:
                reasons.append("missing_or_unknown_latex_anchor")
            bound = artifacts_ok and section_ok and anchor_ok

        if bound:
            by_type[claim_type]["bound"] += 1
        else:
            failures.append(
                {
                    "claim_id": claim_id,
                    "claim_type": claim_type,
                    "reasons": reasons,
                    "claim_text": (c.get("claim_text") or "")[:200],
                }
            )

    bound_claims = sum(v["bound"] for v in by_type.values())
    ceb = bound_claims / total_claims if total_claims else None

    by_type_out = {}
    for t, d in by_type.items():
        total = d["total"]
        bound = d["bound"]
        by_type_out[t] = {
            "bound": bound,
            "total": total,
            "CEB_type": None if total == 0 else bound / total,
        }

    return {
        "CEB": ceb,
        "total_claims": total_claims,
        "bound_claims": bound_claims,
        "by_type": dict(by_type_out),
        "failure_samples": failures[:20],
    }


def compute_ptc(
    *,
    used_facts: Mapping[str, Any],
    claim_trace_index: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    used_keys = used_facts.get("used_keys", []) or []
    used_keys = [k for k in used_keys if isinstance(k, str) and k.strip()]
    if not used_keys:
        return {
            "PTC": None,
            "used_keys": 0,
            "covered_keys": 0,
            "missing_keys": [],
        }

    sections = set(claim_trace_index.get("sections", []) or [])
    artifacts = {
        a.get("artifact_id")
        for a in artifact_manifest.get("artifacts", []) or []
        if a.get("artifact_id")
    }
    traces = claim_trace_index.get("traces", []) or []

    fact_to_traces: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for t in traces:
        for k in t.get("supporting_facts", []) or []:
            if isinstance(k, str) and k.strip():
                fact_to_traces[k].append(t)

    def _full_chain(t: Mapping[str, Any]) -> bool:
        arts = t.get("supporting_artifacts") or []
        if not arts:
            return False
        loc = t.get("location_hints") or {}
        sec = str(loc.get("section") or "")
        if not sec or (sections and sec not in sections):
            return False
        for aid in arts:
            if aid not in artifacts:
                return False
        return True

    covered = 0
    missing: list[str] = []
    for k in used_keys:
        ok_any = False
        for t in fact_to_traces.get(k, []):
            if _full_chain(t):
                ok_any = True
                break
        if ok_any:
            covered += 1
        else:
            missing.append(k)

    ptc = covered / len(used_keys)
    return {
        "PTC": ptc,
        "used_keys": len(used_keys),
        "covered_keys": covered,
        "missing_keys": missing[:50],
    }


def overlap_numeric_claims_for_nia(
    claim_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    claims = claim_ledger.get("claims", []) or []
    overlap_ids: list[str] = []
    for c in claims:
        if str(c.get("claim_type") or "") != "numeric":
            continue
        text = str(c.get("claim_text") or "")
        if contains_comparative_language(text):
            overlap_ids.append(str(c.get("claim_id") or ""))
    overlap_ids = [cid for cid in overlap_ids if cid]
    return {
        "overlap_numeric_comparative_count": len(overlap_ids),
        "overlap_numeric_comparative_ids": overlap_ids,
    }
