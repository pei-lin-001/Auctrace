from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any, Mapping, Sequence

WRITEUP_CONTEXT_PACK_SCHEMA = "auctrace.writeup_context_pack.v2"
CLAIM_CONTEXT_PACK_SCHEMA = "auctrace.claim_context_pack.v1"
_MAX_STAGE_FACT_REFS = 12
_MAX_ABLATION_FACT_REFS = 6
_MAX_PREFERRED_FACT_REFS = 16
_SECTION_RE = re.compile(r"\\section\{([^{}]+)\}")
_LABEL_RE = re.compile(r"\\label\{([^{}]+)\}")


def save_context_pack(path: str, payload: Mapping[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=True)


def _base_writeup_context_pack(
    *,
    symbolic_facts: bool,
    summaries_for_prompt: Mapping[str, Any],
    facts_index: str,
    params_index: str = "",
    artifact_manifest_summary: Mapping[str, Any],
    current_latex: str,
) -> dict[str, Any]:
    return {
        "schema": WRITEUP_CONTEXT_PACK_SCHEMA,
        "mode": "symbolic" if symbolic_facts else "non_symbolic",
        "summaries": dict(summaries_for_prompt),
        "facts_index": facts_index,
        "params_index": params_index,
        "artifact_manifest_summary": dict(artifact_manifest_summary),
        "current_latex": current_latex,
    }


def _compact_stage_summary_payload(stage_payload: Any) -> dict[str, Any]:
    if not isinstance(stage_payload, Mapping):
        return {}
    compact = {
        "schema": stage_payload.get("schema"),
        "stage_kind": stage_payload.get("stage_kind"),
        "stage_name": stage_payload.get("stage_name"),
        "narrative": stage_payload.get("narrative"),
        "included_plots": _compact_included_plots(stage_payload.get("included_plots")),
    }
    fact_refs = stage_payload.get("fact_refs")
    if isinstance(fact_refs, list):
        compact["fact_refs"] = _compact_fact_refs(
            fact_refs,
            limit=_MAX_STAGE_FACT_REFS,
        )
        compact["fact_ref_count"] = len(fact_refs)
    return compact


def _compact_ablation_summary_payload(stage_payload: Any) -> dict[str, Any]:
    if not isinstance(stage_payload, Mapping):
        return {}
    entries = []
    raw_entries = stage_payload.get("entries")
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if not isinstance(item, Mapping):
                continue
            entries.append(
                {
                    "ablation_name": item.get("ablation_name"),
                    "analysis": item.get("analysis"),
                    "plot_count": _plot_count(item.get("plot_paths")),
                    "fact_refs": _compact_fact_refs(
                        item.get("fact_refs"),
                        limit=_MAX_ABLATION_FACT_REFS,
                    ),
                    "fact_ref_count": _fact_ref_count(item.get("fact_refs")),
                }
            )
    compact = {
        "schema": stage_payload.get("schema"),
        "stage_kind": stage_payload.get("stage_kind"),
        "stage_name": stage_payload.get("stage_name"),
        "entries": entries,
    }
    fact_refs = stage_payload.get("fact_refs")
    if isinstance(fact_refs, list):
        compact["fact_refs"] = _compact_fact_refs(
            fact_refs,
            limit=_MAX_ABLATION_FACT_REFS,
        )
        compact["fact_ref_count"] = len(fact_refs)
    return compact


def _compact_fact_refs(payload: Any, *, limit: int) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    compact: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "").strip()
        meaning = str(item.get("meaning") or "").strip()
        if not key or not meaning:
            continue
        compact.append(
            {
                "key": key,
                "meaning": meaning,
            }
        )
        if len(compact) >= limit:
            break
    return compact


def _fact_ref_count(payload: Any) -> int:
    return len(payload) if isinstance(payload, list) else 0


def _compact_included_plots(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                "description": item.get("description"),
                "analysis": item.get("analysis"),
            }
        )
    return compact


def _plot_count(payload: Any) -> int:
    return len(payload) if isinstance(payload, list) else 0


def _compact_symbolic_summaries_for_prompt(
    summaries_for_prompt: Mapping[str, Any],
) -> dict[str, Any]:
    compact = dict(summaries_for_prompt)
    symbolic = compact.get("WRITEUP_SYMBOLIC_SUMMARY")
    if not isinstance(symbolic, Mapping):
        return compact
    compact["WRITEUP_SYMBOLIC_SUMMARY"] = {
        "schema": symbolic.get("schema"),
        "notes": symbolic.get("notes"),
        "draft_summary": _compact_stage_summary_payload(
            symbolic.get("draft_summary")
        ),
        "baseline_summary": _compact_stage_summary_payload(
            symbolic.get("baseline_summary")
        ),
        "research_summary": _compact_stage_summary_payload(
            symbolic.get("research_summary")
        ),
        "ablation_summary": _compact_ablation_summary_payload(
            symbolic.get("ablation_summary")
        ),
    }
    return compact


def _extend_unique_fact_refs(
    dest: list[dict[str, str]],
    seen: set[str],
    payload: Any,
    *,
    limit: int,
) -> None:
    for item in _compact_fact_refs(payload, limit=limit):
        key = item["key"]
        if key in seen:
            continue
        seen.add(key)
        dest.append(item)
        if len(dest) >= _MAX_PREFERRED_FACT_REFS:
            return


def _preferred_fact_refs(symbolic_summary: Mapping[str, Any]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    _extend_unique_fact_refs(
        selected,
        seen,
        symbolic_summary.get("research_summary", {}).get("fact_refs"),
        limit=_MAX_STAGE_FACT_REFS,
    )
    if len(selected) < _MAX_PREFERRED_FACT_REFS:
        _extend_unique_fact_refs(
            selected,
            seen,
            symbolic_summary.get("baseline_summary", {}).get("fact_refs"),
            limit=_MAX_STAGE_FACT_REFS,
        )
    if len(selected) < _MAX_PREFERRED_FACT_REFS:
        _extend_unique_fact_refs(
            selected,
            seen,
            symbolic_summary.get("ablation_summary", {}).get("fact_refs"),
            limit=_MAX_ABLATION_FACT_REFS,
        )
    return selected[:_MAX_PREFERRED_FACT_REFS]


def build_writeup_context_pack(
    *,
    symbolic_facts: bool,
    summaries_for_prompt: Mapping[str, Any],
    facts_index: str,
    params_index: str = "",
    artifact_manifest_summary: Mapping[str, Any],
    current_latex: str,
    plot_names: Sequence[str],
    plot_descriptions: Mapping[str, str],
) -> dict[str, Any]:
    prompt_summaries = (
        _compact_symbolic_summaries_for_prompt(summaries_for_prompt)
        if symbolic_facts
        else summaries_for_prompt
    )
    payload = _base_writeup_context_pack(
        symbolic_facts=symbolic_facts,
        summaries_for_prompt=prompt_summaries,
        facts_index=facts_index,
        params_index=params_index,
        artifact_manifest_summary=artifact_manifest_summary,
        current_latex=current_latex,
    )
    if symbolic_facts:
        symbolic_summary = summaries_for_prompt.get("WRITEUP_SYMBOLIC_SUMMARY")
        if isinstance(symbolic_summary, Mapping):
            payload["preferred_fact_refs"] = _preferred_fact_refs(symbolic_summary)
        return payload
    payload["plot_inventory"] = list(plot_names)
    payload["plot_descriptions"] = dict(plot_descriptions)
    return payload


def build_claim_context_pack(
    *,
    symbolic_tex: str,
    used_facts: Mapping[str, Any],
    artifact_manifest_summary: Mapping[str, Any],
) -> dict[str, Any]:
    section_inventory = sorted(
        {m.group(1).strip() for m in _SECTION_RE.finditer(symbolic_tex) if m.group(1).strip()}
    )
    latex_labels = sorted(
        {m.group(1).strip() for m in _LABEL_RE.finditer(symbolic_tex) if m.group(1).strip()}
    )
    return {
        "schema": CLAIM_CONTEXT_PACK_SCHEMA,
        "symbolic_tex": symbolic_tex,
        "used_facts": dict(used_facts),
        "artifact_manifest_summary": dict(artifact_manifest_summary),
        "section_inventory": section_inventory,
        "latex_labels": latex_labels,
    }
