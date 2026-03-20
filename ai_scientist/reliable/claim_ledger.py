from __future__ import annotations

import json
import os.path as osp
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm

from .artifact_manifest import artifact_id_set
from .errors import ReliableWriteupError
from .facts import FactStore, validate_fact_key


class ClaimLedgerError(ReliableWriteupError):
    pass


CLAIM_LEDGER_SCHEMA_V1 = "auctrace.claim_ledger.v1"
CLAIM_LEDGER_SCHEMA_V2 = "auctrace.claim_ledger.v2"

_SECTION_RE = re.compile(r"\\section\{([^{}]+)\}")
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_CLAIM_TOKEN_WITH_DIGIT_RE = re.compile(r"[A-Za-z0-9_.:+/%-]*\d[A-Za-z0-9_.:+/%-]*")
_IDENTIFIER_LIKE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+/-]*\d[A-Za-z0-9_.:+/-]*$")
_FACT_PLACEHOLDER_RE = re.compile(r"\\fact\{([^}]+)\}")


def _extract_section(tex: str, title: str) -> str:
    """Extract a \\section{title} ... block. Returns empty string if not found."""
    # Find the first \section{title}
    needle = f"\\section{{{title}}}"
    start = tex.find(needle)
    if start < 0:
        return ""
    after = tex[start + len(needle) :]
    # Find next section header.
    m = _SECTION_RE.search(after)
    end = len(after) if m is None else m.start()
    return after[:end].strip()


def _extract_includegraphics_files(tex: str) -> list[str]:
    files: list[str] = []
    for m in _INCLUDEGRAPHICS_RE.finditer(tex):
        files.append(m.group(1).strip())
    # Preserve order but de-dup.
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _section_inventory(tex: str) -> list[str]:
    return sorted({m.group(1).strip() for m in _SECTION_RE.finditer(tex) if m.group(1).strip()})


def _latex_labels(tex: str) -> list[str]:
    return sorted({m.group(1).strip() for m in re.finditer(r"\\label\{([^{}]+)\}", tex) if m.group(1).strip()})


def _facts_index_for_used(used_facts: Mapping[str, Any]) -> str:
    facts = used_facts.get("facts")
    if not isinstance(facts, list):
        return ""
    lines: list[str] = []
    for item in facts:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key", "")).strip()
        meaning = str(item.get("meaning", "")).strip().replace("\n", " ")
        if key:
            lines.append(f"- {key}: {meaning}")
    return "\n".join(lines)


def _validate_supporting_facts(store: FactStore, facts: Sequence[Any]) -> list[str]:
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
        raise ClaimLedgerError("Claim supporting_facts must be a list of fact key strings.")
    keys: list[str] = []
    for item in facts:
        key = str(item).strip()
        validate_fact_key(key)
        store.get(key)  # validates existence
        keys.append(key)
    return keys


def _find_claim_numeric_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _CLAIM_TOKEN_WITH_DIGIT_RE.finditer(text):
        token = match.group(0).strip()
        if not token:
            continue
        if "%" in token or "\\%" in token:
            tokens.append(token)
            continue
        if _IDENTIFIER_LIKE_TOKEN_RE.match(token):
            continue
        tokens.append(token)
    return tokens


def _validate_claim_text_no_numbers(text: str) -> None:
    bad_tokens = _find_claim_numeric_tokens(text)
    if bad_tokens:
        raise ClaimLedgerError(
            "Claim text must not contain standalone numeric result tokens "
            "(use supporting_facts instead). Identifier-like names such as "
            "SST-2 or GPT-4o are allowed. "
            f"Found examples: {bad_tokens[:4]}"
        )


def _extract_fact_placeholders(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FACT_PLACEHOLDER_RE.finditer(text) if m.group(1).strip()]


def _validate_supporting_artifacts(
    artifacts: Any,
    *,
    idx: int,
    allowed_artifact_ids: set[str],
) -> list[str]:
    if not isinstance(artifacts, list):
        raise ClaimLedgerError(
            f"claim_ledger.claims[{idx}].supporting_artifacts must be a list."
        )
    out: list[str] = []
    for item in artifacts:
        artifact = str(item).strip()
        if not artifact:
            raise ClaimLedgerError(
                f"claim_ledger.claims[{idx}].supporting_artifacts contains an empty entry."
            )
        if allowed_artifact_ids and artifact not in allowed_artifact_ids:
            raise ClaimLedgerError(
                f"claim_ledger.claims[{idx}].supporting_artifacts contains unknown artifact_id {artifact!r}."
            )
        out.append(artifact)
    return out


def _validate_claim_text_template(
    template: Any,
    *,
    idx: int,
    supporting_facts: Sequence[str],
) -> None:
    text = str(template or "").strip()
    if not text:
        raise ClaimLedgerError(f"claim_ledger.claims[{idx}].claim_text_template is required.")
    placeholders = _extract_fact_placeholders(text)
    unknown = [key for key in placeholders if key not in supporting_facts]
    if unknown:
        raise ClaimLedgerError(
            f"claim_ledger.claims[{idx}].claim_text_template references fact keys not listed in supporting_facts: {unknown[:10]}"
        )


def _validate_location_hints(location_hints: Any, *, idx: int) -> None:
    if not isinstance(location_hints, Mapping):
        raise ClaimLedgerError(
            f"claim_ledger.claims[{idx}].location_hints must be an object."
        )
    section = str(location_hints.get("section", "")).strip()
    if not section:
        raise ClaimLedgerError(
            f"claim_ledger.claims[{idx}].location_hints.section is required."
        )
    latex_anchor = location_hints.get("latex_anchor")
    if latex_anchor is None:
        return
    if not str(latex_anchor).strip():
        raise ClaimLedgerError(
            f"claim_ledger.claims[{idx}].location_hints.latex_anchor must be non-empty when provided."
        )


def validate_claim_ledger(
    ledger: Mapping[str, Any],
    *,
    store: FactStore,
    used_keys: Sequence[str],
    artifact_manifest: Mapping[str, Any] | None = None,
) -> None:
    schema = str(ledger.get("schema", "")).strip()
    if schema not in {CLAIM_LEDGER_SCHEMA_V1, CLAIM_LEDGER_SCHEMA_V2}:
        raise ClaimLedgerError(
            f"claim_ledger.schema must be {CLAIM_LEDGER_SCHEMA_V1!r} or {CLAIM_LEDGER_SCHEMA_V2!r}."
        )
    claims = ledger.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ClaimLedgerError("claim_ledger.claims must be a non-empty list.")

    all_supporting: set[str] = set()
    seen_ids: set[str] = set()
    allowed_artifact_ids = artifact_id_set(artifact_manifest or {})
    for idx, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise ClaimLedgerError(f"claim_ledger.claims[{idx}] must be an object.")
        claim_id = str(claim.get("claim_id", "")).strip()
        if not claim_id:
            raise ClaimLedgerError(f"claim_ledger.claims[{idx}].claim_id is required.")
        if claim_id in seen_ids:
            raise ClaimLedgerError(f"Duplicate claim_id not allowed: {claim_id!r}")
        seen_ids.add(claim_id)

        claim_type = str(claim.get("claim_type", "")).strip()
        if claim_type not in {"numeric", "comparative", "qualitative", "citation"}:
            raise ClaimLedgerError(
                f"claim_ledger.claims[{idx}].claim_type must be one of "
                f"numeric|comparative|qualitative|citation, got {claim_type!r}"
            )

        claim_text = str(claim.get("claim_text", "")).strip()
        if not claim_text:
            raise ClaimLedgerError(f"claim_ledger.claims[{idx}].claim_text is required.")
        _validate_claim_text_no_numbers(claim_text)

        keys = _validate_supporting_facts(store, claim.get("supporting_facts") or [])
        all_supporting.update(keys)
        claim_text_template = claim.get("claim_text_template")
        if schema == CLAIM_LEDGER_SCHEMA_V2 or claim_text_template is not None:
            _validate_claim_text_template(
                claim_text_template,
                idx=idx,
                supporting_facts=keys,
            )
        audit_status = claim.get("audit_status")
        if schema == CLAIM_LEDGER_SCHEMA_V2 and audit_status is None:
            raise ClaimLedgerError(f"claim_ledger.claims[{idx}].audit_status is required.")
        if audit_status is not None:
            audit_status_str = str(audit_status).strip()
            if audit_status_str not in {"unknown", "pass", "fail"}:
                raise ClaimLedgerError(
                    f"claim_ledger.claims[{idx}].audit_status must be unknown|pass|fail, got {audit_status_str!r}"
                )
        _validate_supporting_artifacts(
            claim.get("supporting_artifacts"),
            idx=idx,
            allowed_artifact_ids=allowed_artifact_ids,
        )
        _validate_location_hints(claim.get("location_hints"), idx=idx)

    missing = [k for k in used_keys if k not in all_supporting]
    if missing:
        raise ClaimLedgerError(
            "Claim ledger coverage failed: every used \\fact{KEY} must be listed under some claim.supporting_facts. "
            f"Missing keys: {missing[:20]}"
        )


def generate_claim_ledger(
    *,
    symbolic_tex: str,
    used_facts: Mapping[str, Any],
    client: Any,
    model: str,
    artifact_manifest_summary: Mapping[str, Any] | None = None,
    context_pack: Mapping[str, Any] | None = None,
    remediation_instructions: str = "",
) -> dict[str, Any]:
    used_keys = used_facts.get("used_keys") if isinstance(used_facts, Mapping) else None
    used_keys = used_keys if isinstance(used_keys, list) else []
    pack = dict(context_pack or {})
    if not pack:
        facts_index = _facts_index_for_used(used_facts)
        experiments = _extract_section(symbolic_tex, "Experiments") or symbolic_tex
        figures = _extract_includegraphics_files(symbolic_tex)
        pack = {
            "schema": "auctrace.claim_context_pack.v1",
            "symbolic_tex_excerpt": experiments,
            "used_fact_keys": used_keys,
            "facts_index": facts_index,
            "figures_in_manuscript": figures,
            "artifact_manifest_summary": dict(artifact_manifest_summary or {}),
            "section_inventory": _section_inventory(symbolic_tex),
            "latex_labels": _latex_labels(symbolic_tex),
        }
    else:
        pack.setdefault("section_inventory", _section_inventory(symbolic_tex))
        pack.setdefault("latex_labels", _latex_labels(symbolic_tex))

    sys_msg = (
        "You are an audit-oriented ML researcher. "
        "Build a claim ledger that links manuscript claims to fact keys. "
        "You must NOT include any numeric experimental values in claim_text."
    )
    prompt = f"""
You are given:
1) A claim context pack JSON.

Your task:
- Produce a JSON claim ledger.
- claim_type must be exactly one of: numeric|comparative|qualitative|citation (do NOT output other values like "summary").
- Each claim must reference one or more fact keys under supporting_facts.
- claim_text must NOT contain standalone numeric result tokens.
- Identifier-like names such as SST-2, GPT-4o, or Qwen2.5 are allowed.
- Standalone values like 2, 0.83, 1e-3, 83\\%, and 2x are forbidden in claim_text.
- Use only the provided fact keys.
- Every claim must include supporting_artifacts (list, can be empty) and
  location_hints with at least a section name.
- location_hints.section must be an exact string from claim_context_pack.section_inventory.
- Do not invent section aliases or synonyms; if the manuscript section is "Experiments", do not output "Results".
- Prefer artifact_ids from artifact_manifest_summary over raw file paths when applicable.
- For every key in claim_context_pack.used_facts.used_keys, at least one covering claim must have non-empty supporting_artifacts and a non-empty location_hints.section.
- For claims backed by manuscript-used fact keys, do not leave supporting_artifacts empty; provide a latex_anchor whenever the claim is tied to a figure or table.
- If location_hints.latex_anchor is provided, use an exact string from claim_context_pack.latex_labels.
- Every claim must include claim_text_template and audit_status.
- claim_text_template may contain \\fact{{KEY}} placeholders, but any placeholder key must also appear in supporting_facts.
- Set audit_status to "unknown" for newly generated claims.

{remediation_instructions}

Claim context pack:
```json
{json.dumps(pack, ensure_ascii=True, indent=2)}
```

    Respond with:
    ```json
    {{
      "schema": "auctrace.claim_ledger.v2",
      "claims": [
        {{
          "claim_id": "C001",
          "claim_type": "numeric",
          "claim_text": "A claim sentence without numeric literals.",
          "claim_text_template": "A claim sentence with optional \\fact{{stage3.best.ag_news.validation_accuracy.best}} placeholders.",
          "audit_status": "unknown",
          "supporting_facts": ["stage3.best.ag_news.validation_accuracy.best"],
          "supporting_artifacts": ["figure.main_plot"],
          "location_hints": {{"section": "Experiments", "latex_anchor": "tab:main_results"}}
        }}
      ]
    }}
    ```
"""
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
        raise ClaimLedgerError("Failed to parse JSON claim ledger from LLM output.")
    return parsed


def save_claim_ledger(path: str, ledger: Mapping[str, Any]) -> None:
    parent = osp.dirname(path)
    if parent:
        import os

        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(ledger), f, indent=2, ensure_ascii=True)
