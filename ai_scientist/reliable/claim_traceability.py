from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any, Mapping, Sequence

from .errors import ReliableWriteupError

CLAIM_TRACE_INDEX_SCHEMA = "auctrace.claim_trace_index.v1"
_SECTION_RE = re.compile(r"\\section\{([^{}]+)\}")
_LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")


class ClaimTraceabilityError(ReliableWriteupError):
    pass


def _section_names(symbolic_tex: str) -> set[str]:
    return {m.group(1).strip() for m in _SECTION_RE.finditer(symbolic_tex) if m.group(1).strip()}


def _latex_labels(symbolic_tex: str) -> set[str]:
    return {m.group(1).strip() for m in _LABEL_RE.finditer(symbolic_tex) if m.group(1).strip()}


def build_claim_trace_index(
    *,
    ledger: Mapping[str, Any],
    symbolic_tex: str,
) -> dict[str, Any]:
    claims = ledger.get("claims")
    if not isinstance(claims, list):
        raise ClaimTraceabilityError("claim_ledger.claims must be a list.")
    sections = _section_names(symbolic_tex)
    labels = _latex_labels(symbolic_tex)
    traces: list[dict[str, Any]] = []
    for idx, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise ClaimTraceabilityError(f"claim_ledger.claims[{idx}] must be an object.")
        location_hints = claim.get("location_hints")
        if not isinstance(location_hints, Mapping):
            raise ClaimTraceabilityError(
                f"claim_ledger.claims[{idx}].location_hints must be an object."
            )
        section = str(location_hints.get("section") or "").strip()
        latex_anchor = str(location_hints.get("latex_anchor") or "").strip()
        if section and sections and section not in sections:
            raise ClaimTraceabilityError(
                f"claim_ledger.claims[{idx}] references unknown section {section!r}."
            )
        if latex_anchor and latex_anchor not in labels:
            raise ClaimTraceabilityError(
                f"claim_ledger.claims[{idx}] references missing latex anchor {latex_anchor!r}."
            )
        traces.append(
            {
                "claim_id": str(claim.get("claim_id") or "").strip(),
                "claim_type": str(claim.get("claim_type") or "").strip(),
                "supporting_facts": [
                    str(item).strip()
                    for item in claim.get("supporting_facts") or []
                    if str(item).strip()
                ],
                "supporting_artifacts": [
                    str(item).strip()
                    for item in claim.get("supporting_artifacts") or []
                    if str(item).strip()
                ],
                "location_hints": {
                    "section": section,
                    "latex_anchor": latex_anchor,
                },
            }
        )
    return {
        "schema": CLAIM_TRACE_INDEX_SCHEMA,
        "sections": sorted(sections),
        "labels": sorted(labels),
        "traces": traces,
    }


def validate_used_fact_traceability(
    trace_index: Mapping[str, Any],
    *,
    used_keys: Sequence[str],
) -> None:
    traces = trace_index.get("traces")
    if not isinstance(traces, list):
        raise ClaimTraceabilityError("claim_trace_index.traces must be a list.")
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for item in traces:
        if not isinstance(item, Mapping):
            continue
        for key in item.get("supporting_facts") or []:
            fact_key = str(key).strip()
            if fact_key:
                by_key.setdefault(fact_key, []).append(item)

    missing = [key for key in used_keys if key not in by_key]
    if missing:
        raise ClaimTraceabilityError(
            "Missing claim traceability for used fact keys: "
            f"{missing[:20]}"
        )

    weak = []
    for key in used_keys:
        traces_for_key = by_key.get(key, [])
        if any(
            (item.get("supporting_artifacts") or [])
            and str((item.get("location_hints") or {}).get("section") or "").strip()
            for item in traces_for_key
        ):
            continue
        weak.append(key)
    if weak:
        raise ClaimTraceabilityError(
            "Used fact keys must resolve to at least one claim with artifact and location bindings: "
            f"{weak[:20]}"
        )


def save_claim_trace_index(path: str, payload: Mapping[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=True)
