from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Dict, Mapping, Sequence

from .extract_facts import build_fact_store_for_run
from .fact_extraction import build_ablation_fact_prefix
from .facts import FactRecord, FactStore, validate_fact_key
from .sanitize import redact_numeric_text

STAGE_SCHEMA = "auctrace.stage_summary_symbolic.v1"
BUNDLE_SCHEMA = "auctrace.writeup_summary_symbolic.v2"


def _summarize_provenance(record: FactRecord) -> Dict[str, Any]:
    prov = record.provenance or {}
    return {
        "stage_name": prov.get("stage_name"),
        "node_id": prov.get("node_id"),
        "exp_results_dir": prov.get("exp_results_dir"),
    }


def _facts_for_prefix(store: FactStore, prefix: str) -> list[FactRecord]:
    records = [rec for key, rec in store.facts.items() if key.startswith(prefix)]
    records.sort(key=lambda rec: rec.key)
    return records


def _pack_fact_refs(records: Sequence[FactRecord]) -> list[dict[str, Any]]:
    return [
        {
            "key": rec.key,
            "meaning": rec.meaning,
            "provenance": _summarize_provenance(rec),
        }
        for rec in records
    ]


def _text(value: Any) -> str:
    text = str(value or "").strip()
    return redact_numeric_text(text) if text else ""


def _pack_narrative(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        "experiment_description": _text(summary.get("Experiment_description")),
        "significance": _text(summary.get("Significance")),
        "description": _text(summary.get("Description")),
    }


def _pack_plots(plots: Any) -> list[dict[str, str]]:
    if not isinstance(plots, list):
        return []
    out: list[dict[str, str]] = []
    for item in plots:
        if not isinstance(item, Mapping):
            continue
        out.append(
            {
                "path": str(item.get("path") or "").strip(),
                "description": _text(item.get("description")),
                "analysis": _text(item.get("analysis")),
            }
        )
    return out


def _build_draft_summary_symbolic(raw_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": STAGE_SCHEMA,
        "stage_kind": "draft",
        "stage_name": str(raw_summary.get("stage_name") or "1_draft"),
        "narrative": _pack_narrative(raw_summary),
        "included_plots": _pack_plots(raw_summary.get("List_of_included_plots")),
        "fact_refs": [],
    }


def _build_best_stage_symbolic(
    *,
    stage_kind: str,
    raw_summary: Mapping[str, Any],
    store: FactStore,
    prefix: str,
) -> dict[str, Any]:
    return {
        "schema": STAGE_SCHEMA,
        "stage_kind": stage_kind,
        "stage_name": str(raw_summary.get("stage_name") or stage_kind),
        "narrative": _pack_narrative(raw_summary),
        "included_plots": _pack_plots(raw_summary.get("List_of_included_plots")),
        "fact_refs": _pack_fact_refs(_facts_for_prefix(store, prefix)),
    }


def _build_ablation_entry(raw_entry: Mapping[str, Any], store: FactStore) -> dict[str, Any]:
    ablation_name = str(raw_entry.get("ablation_name") or "").strip()
    prefix = build_ablation_fact_prefix(ablation_name) if ablation_name else ""
    return {
        "stage_name": str(raw_entry.get("stage_name") or "4_ablation"),
        "ablation_name": ablation_name,
        "analysis": _text(raw_entry.get("analysis")),
        "plot_paths": [str(p).strip() for p in raw_entry.get("plot_paths") or [] if str(p).strip()],
        "exp_results_dir": str(raw_entry.get("exp_results_dir") or "").strip(),
        "fact_refs": _pack_fact_refs(_facts_for_prefix(store, prefix)) if prefix else [],
    }


def _build_ablation_summary_symbolic(raw_summary: Any, store: FactStore) -> dict[str, Any]:
    entries = []
    if isinstance(raw_summary, list):
        for item in raw_summary:
            if isinstance(item, Mapping):
                entries.append(_build_ablation_entry(item, store))
    return {
        "schema": STAGE_SCHEMA,
        "stage_kind": "ablation",
        "stage_name": "4_ablation",
        "entries": entries,
        "fact_refs": _pack_fact_refs(_facts_for_prefix(store, "stage4.ablation.")),
    }


def build_writeup_summary_symbolic(
    store: FactStore,
    draft_summary: Mapping[str, Any] | None = None,
    baseline_summary: Mapping[str, Any] | None = None,
    research_summary: Mapping[str, Any] | None = None,
    ablation_summary: Any | None = None,
) -> dict[str, Any]:
    draft_payload = _build_draft_summary_symbolic(draft_summary or {})
    baseline_payload = _build_best_stage_symbolic(
        stage_kind="baseline",
        raw_summary=baseline_summary or {},
        store=store,
        prefix="stage2.best.",
    )
    research_payload = _build_best_stage_symbolic(
        stage_kind="research",
        raw_summary=research_summary or {},
        store=store,
        prefix="stage3.best.",
    )
    ablation_payload = _build_ablation_summary_symbolic(ablation_summary or [], store)
    return {
        "schema": BUNDLE_SCHEMA,
        "notes": [
            "This file is numeric-free by design.",
            "When writing the manuscript, use \\fact{KEY} to cite experimental values.",
            "Narrative text may retain stage names and artifact identifiers, but not direct experimental numbers.",
        ],
        "draft_summary": draft_payload,
        "baseline_summary": baseline_payload,
        "research_summary": research_payload,
        "ablation_summary": ablation_payload,
    }


def _validate_fact_refs(fact_refs: Any, store: FactStore, *, path: str) -> None:
    if not isinstance(fact_refs, list):
        raise ValueError(f"{path}.fact_refs must be a list.")
    for idx, item in enumerate(fact_refs):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}.fact_refs[{idx}] must be an object.")
        key = str(item.get("key") or "").strip()
        validate_fact_key(key)
        store.get(key)


def _validate_stage_payload(stage_payload: Any, store: FactStore, *, path: str) -> None:
    if not isinstance(stage_payload, Mapping):
        raise ValueError(f"{path} must be an object.")
    if str(stage_payload.get("schema") or "").strip() != STAGE_SCHEMA:
        raise ValueError(f"{path}.schema must be {STAGE_SCHEMA!r}.")
    _validate_fact_refs(stage_payload.get("fact_refs"), store, path=path)


def validate_writeup_summary_symbolic(payload: Mapping[str, Any], store: FactStore) -> None:
    if str(payload.get("schema") or "").strip() != BUNDLE_SCHEMA:
        raise ValueError(f"writeup_summary.schema must be {BUNDLE_SCHEMA!r}.")
    for stage_key in ("draft_summary", "baseline_summary", "research_summary", "ablation_summary"):
        _validate_stage_payload(payload.get(stage_key), store, path=f"writeup_summary.{stage_key}")
    ablation = payload.get("ablation_summary") or {}
    entries = ablation.get("entries")
    if not isinstance(entries, list):
        raise ValueError("writeup_summary.ablation_summary.entries must be a list.")
    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(f"writeup_summary.ablation_summary.entries[{idx}] must be an object.")
        _validate_fact_refs(
            item.get("fact_refs"),
            store,
            path=f"writeup_summary.ablation_summary.entries[{idx}]",
        )


def save_writeup_summary_symbolic(path: str, payload: dict[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def save_symbolic_summary_artifacts(log_dir: str, payload: Mapping[str, Any]) -> None:
    targets = {
        "writeup_summary_symbolic.json": dict(payload),
        "draft_summary_symbolic.json": payload.get("draft_summary", {}),
        "baseline_summary_symbolic.json": payload.get("baseline_summary", {}),
        "research_summary_symbolic.json": payload.get("research_summary", {}),
        "ablation_summary_symbolic.json": payload.get("ablation_summary", {}),
    }
    for filename, data in targets.items():
        save_writeup_summary_symbolic(osp.join(log_dir, filename), dict(data))


def export_writeup_summary_symbolic(
    *,
    base_folder: str,
    log_dir: str,
    draft_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    research_summary: Mapping[str, Any],
    ablation_summary: Any,
) -> dict[str, Any]:
    fact_store_path = osp.join(log_dir, "fact_store.json")
    store = (
        FactStore.load_json(fact_store_path)
        if osp.exists(fact_store_path)
        else build_fact_store_for_run(base_folder)
    )
    if not osp.exists(fact_store_path):
        store.save_json(fact_store_path)
    payload = build_writeup_summary_symbolic(
        store,
        draft_summary=draft_summary,
        baseline_summary=baseline_summary,
        research_summary=research_summary,
        ablation_summary=ablation_summary,
    )
    validate_writeup_summary_symbolic(payload, store)
    save_symbolic_summary_artifacts(log_dir, payload)
    return payload
