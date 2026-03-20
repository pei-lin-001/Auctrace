from __future__ import annotations

import json
import os
import os.path as osp
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm

from .errors import ReliableWriteupError
from .facts import FactStore, validate_fact_key


class OutsiderAuditError(ReliableWriteupError):
    pass


_SECTION_RE = re.compile(r"\\section\{([^{}]+)\}")
_ABSTRACT_BEGIN = "\\begin{abstract}"
_ABSTRACT_END = "\\end{abstract}"


@dataclass(frozen=True)
class OutsiderAuditInputs:
    symbolic_tex_excerpt: str
    used_fact_keys: list[str]
    used_fact_table: list[dict[str, Any]]
    claim_ledger: dict[str, Any]
    artifact_manifest_summary: dict[str, Any]


def _extract_abstract(tex: str) -> str:
    start = tex.find(_ABSTRACT_BEGIN)
    if start < 0:
        return ""
    after = tex[start + len(_ABSTRACT_BEGIN) :]
    end = after.find(_ABSTRACT_END)
    if end < 0:
        return ""
    return after[:end].strip()


def _extract_section(tex: str, title: str) -> str:
    needle = f"\\section{{{title}}}"
    start = tex.find(needle)
    if start < 0:
        return ""
    after = tex[start + len(needle) :]
    m = _SECTION_RE.search(after)
    end = len(after) if m is None else m.start()
    return after[:end].strip()


def _used_fact_table(
    store: FactStore, used_keys: Sequence[str], used_facts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    # Prefer meaning/provenance from used_facts.json, and value from FactStore.
    facts = used_facts.get("facts")
    facts_list = facts if isinstance(facts, list) else []
    by_key: dict[str, dict[str, Any]] = {}
    for item in facts_list:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        by_key[key] = {
            "key": key,
            "meaning": item.get("meaning"),
            "format": item.get("format"),
            "provenance": item.get("provenance"),
        }

    out: list[dict[str, Any]] = []
    for key in used_keys:
        validate_fact_key(key)
        rec = store.get(key)
        base = dict(by_key.get(key) or {"key": key, "meaning": rec.meaning})
        base["value"] = rec.value
        base["provenance"] = base.get("provenance") or rec.provenance
        out.append(base)
    return out


def build_outsider_audit_inputs(
    *,
    symbolic_tex: str,
    used_facts: Mapping[str, Any],
    store: FactStore,
    claim_ledger: Mapping[str, Any],
    artifact_manifest_summary: Mapping[str, Any] | None = None,
    max_excerpt_chars: int = 8000,
) -> OutsiderAuditInputs:
    used_keys = used_facts.get("used_keys") if isinstance(used_facts, Mapping) else None
    used_keys = used_keys if isinstance(used_keys, list) else []
    used_keys = [str(k).strip() for k in used_keys if str(k).strip()]

    abstract = _extract_abstract(symbolic_tex)
    experiments = _extract_section(symbolic_tex, "Experiments")
    excerpt_parts = []
    if abstract:
        excerpt_parts.append("\\begin{abstract}\n" + abstract + "\n\\end{abstract}")
    if experiments:
        excerpt_parts.append("\\section{Experiments}\n" + experiments)
    if not excerpt_parts:
        excerpt_parts.append(symbolic_tex)

    excerpt = "\n\n".join(excerpt_parts)
    if len(excerpt) > max_excerpt_chars:
        excerpt = excerpt[:max_excerpt_chars] + "\n% [truncated]\n"

    fact_table = _used_fact_table(store, used_keys, used_facts)
    return OutsiderAuditInputs(
        symbolic_tex_excerpt=excerpt,
        used_fact_keys=used_keys,
        used_fact_table=fact_table,
        claim_ledger=dict(claim_ledger),
        artifact_manifest_summary=dict(artifact_manifest_summary or {}),
    )


def _outsider_audit_system_message() -> str:
    return (
        "You are an outsider auditor for an automated ML research pipeline. "
        "Your job is to check internal consistency and evidentiary support, "
        "using only the provided fact table and claim ledger. "
        "Be strict and concrete. Do not hallucinate missing experiments."
    )


def _outsider_audit_payload(inputs: OutsiderAuditInputs) -> dict[str, Any]:
    return {
        "schema": "auctrace.outsider_audit_input.v1",
        "notes": [
            "The manuscript is symbolic LaTeX (facts are referenced via \\fact{KEY}).",
            "The fact table contains only USED keys and includes numeric values.",
            "Use fact keys and claim_ids as evidence anchors; do not invent new keys.",
        ],
        "manuscript_excerpt": inputs.symbolic_tex_excerpt,
        "used_fact_keys": inputs.used_fact_keys,
        "used_fact_table": inputs.used_fact_table,
        "claim_ledger": inputs.claim_ledger,
        "artifact_manifest_summary": inputs.artifact_manifest_summary,
    }


def build_outsider_audit_context_pack(inputs: OutsiderAuditInputs) -> dict[str, Any]:
    return _outsider_audit_payload(inputs)


def _outsider_audit_prompt(payload: Mapping[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True, indent=2)
    return f"""
You are given a compact input bundle (JSON below).

Task:
- Audit the manuscript excerpt and claim ledger against the used fact table.
- Use the artifact manifest summary to interpret figure/table identities when relevant.
- Flag contradictions, unsupported claims, missing evidence links, or unclear experimental statements.
- If a claim is not verifiable from the provided fact keys, flag it as "unverifiable".

Output ONLY valid JSON inside a ```json block with this schema:
```json
{{
  "schema": "auctrace.outsider_audit.v1",
  "summary": "One short paragraph.",
  "issues": [
    {{
      "issue_id": "I001",
      "severity": "low|medium|high",
      "category": "contradiction|unsupported_claim|unverifiable|clarity|citation|other",
      "description": "What is wrong.",
      "evidence": {{
        "claim_ids": ["C001"],
        "fact_keys": ["stage2.best.xxx.yyy.best"],
        "manuscript_quote": "Short excerpt"
      }},
      "suggested_fix": "Concrete fix suggestion."
    }}
  ]
}}
```

Important rules:
- Do NOT add experimental numeric values in the text unless they come from the provided used_fact_table.
- Prefer citing fact_keys instead of repeating numbers.
- If there are no issues, return an empty issues list.

Input bundle:
```json
{payload_json}
```
"""


def generate_outsider_audit(
    *,
    inputs: OutsiderAuditInputs,
    client: Any,
    model: str,
) -> dict[str, Any]:
    sys_msg = _outsider_audit_system_message()
    payload = _outsider_audit_payload(inputs)
    prompt = _outsider_audit_prompt(payload)
    text, _ = get_response_from_llm(
        prompt,
        client,
        model,
        sys_msg,
        print_debug=False,
        msg_history=None,
        temperature=0.2,
    )
    parsed = extract_json_between_markers(text)
    if not isinstance(parsed, dict):
        raise OutsiderAuditError("Failed to parse outsider audit JSON from LLM output.")
    return parsed


def validate_outsider_audit(audit: Mapping[str, Any]) -> None:
    if str(audit.get("schema", "")).strip() != "auctrace.outsider_audit.v1":
        raise OutsiderAuditError("outsider_audit.schema must be 'auctrace.outsider_audit.v1'.")
    summary = str(audit.get("summary", "")).strip()
    if not summary:
        raise OutsiderAuditError("outsider_audit.summary is required.")
    issues = audit.get("issues")
    if issues is None:
        raise OutsiderAuditError("outsider_audit.issues is required (can be empty list).")
    if not isinstance(issues, list):
        raise OutsiderAuditError("outsider_audit.issues must be a list.")

    seen_ids: set[str] = set()
    for idx, item in enumerate(issues):
        if not isinstance(item, Mapping):
            raise OutsiderAuditError(f"outsider_audit.issues[{idx}] must be an object.")
        issue_id = str(item.get("issue_id", "")).strip()
        if not issue_id:
            raise OutsiderAuditError(f"outsider_audit.issues[{idx}].issue_id is required.")
        if issue_id in seen_ids:
            raise OutsiderAuditError(f"Duplicate outsider audit issue_id not allowed: {issue_id!r}")
        seen_ids.add(issue_id)

        severity = str(item.get("severity", "")).strip()
        if severity not in {"low", "medium", "high"}:
            raise OutsiderAuditError(
                f"outsider_audit.issues[{idx}].severity must be low|medium|high, got {severity!r}"
            )

        category = str(item.get("category", "")).strip()
        if not category:
            raise OutsiderAuditError(f"outsider_audit.issues[{idx}].category is required.")

        desc = str(item.get("description", "")).strip()
        if not desc:
            raise OutsiderAuditError(f"outsider_audit.issues[{idx}].description is required.")

        evidence = item.get("evidence")
        if not isinstance(evidence, Mapping):
            raise OutsiderAuditError(f"outsider_audit.issues[{idx}].evidence must be an object.")

        suggested_fix = str(item.get("suggested_fix", "")).strip()
        if not suggested_fix:
            raise OutsiderAuditError(
                f"outsider_audit.issues[{idx}].suggested_fix is required."
            )


def save_outsider_audit(path: str, audit: Mapping[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(audit), f, indent=2, ensure_ascii=True)
