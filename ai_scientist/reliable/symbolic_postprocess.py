from __future__ import annotations

import json
import os.path as osp
import traceback
from typing import Any, Mapping

from ai_scientist.env_utils import env_bool, env_str
from ai_scientist.llm import create_client

from .artifact_manifest import artifact_manifest_summary, ensure_artifact_manifest
from .claim_ledger import generate_claim_ledger, save_claim_ledger, validate_claim_ledger
from .claim_traceability import (
    build_claim_trace_index,
    save_claim_trace_index,
    validate_used_fact_traceability,
)
from .context_packs import (
    build_claim_context_pack,
    save_context_pack,
)
from .latex_compile import compile_symbolic_latex_project
from .outsider_audit import (
    build_outsider_audit_context_pack,
    build_outsider_audit_inputs,
    generate_outsider_audit,
    raise_if_blocking_audit_issues,
    save_outsider_audit,
    validate_outsider_audit,
)
from .params import ensure_param_store_for_run
from .remediation import (
    build_remediation_prompt_block,
    classify_remediation_failure,
    print_remediation_report,
    save_remediation_report,
)
from .writeup_inputs import ensure_symbolic_writeup_inputs
from .writeup_validation import validate_generated_writeup


def _load_existing_symbolic_paths(base_folder: str) -> tuple[str, str, str]:
    latex_folder = osp.join(base_folder, "latex")
    writeup_file = osp.join(latex_folder, "template.tex")
    used_facts_path = osp.join(latex_folder, "used_facts.json")
    if not osp.exists(writeup_file):
        raise FileNotFoundError(f"Missing symbolic writeup draft: {writeup_file}")
    return latex_folder, writeup_file, used_facts_path


def _ensure_existing_symbolic_compile(
    *,
    base_folder: str,
    latex_folder: str,
    fact_store_path: str,
    param_store_path: str | None,
    used_facts_path: str,
    pdf_file: str,
) -> None:
    if osp.exists(used_facts_path) and osp.exists(pdf_file):
        return
    compile_symbolic_latex_project(
        latex_folder=latex_folder,
        pdf_file=pdf_file,
        fact_store_path=fact_store_path,
        param_store_path=param_store_path,
        rendered_tex_artifact_path=osp.join(latex_folder, "template.rendered.tex"),
        used_facts_artifact_path=used_facts_path,
    )


def _run_claim_and_audit_pipeline(
    *,
    base_folder: str,
    latex_folder: str,
    writeup_file: str,
    used_facts_path: str,
    big_model: str,
    small_model: str,
    remediation_context: Mapping[str, Any] | None,
) -> None:
    store, _ = ensure_symbolic_writeup_inputs(base_folder)
    param_store = ensure_param_store_for_run(base_folder)
    with open(writeup_file, "r", encoding="utf-8") as f:
        symbolic_tex = f.read()
    manifest = ensure_artifact_manifest(
        base_folder=base_folder,
        store=store,
        tex_path=writeup_file,
    )
    validate_generated_writeup(
        symbolic_tex,
        symbolic_facts=True,
        store=store,
        param_store=param_store,
        artifact_manifest=manifest,
    )
    with open(used_facts_path, "r", encoding="utf-8") as f:
        used_facts = json.load(f)
    claim_context_pack = build_claim_context_pack(
        symbolic_tex=symbolic_tex,
        used_facts=used_facts,
        artifact_manifest_summary=artifact_manifest_summary(manifest),
    )
    save_context_pack(osp.join(latex_folder, "claim_context_pack.json"), claim_context_pack)
    big_client, big_client_model = create_client(big_model)
    ledger = generate_claim_ledger(
        symbolic_tex=symbolic_tex,
        used_facts=used_facts,
        client=big_client,
        model=big_client_model,
        artifact_manifest_summary=artifact_manifest_summary(manifest),
        context_pack=claim_context_pack,
        remediation_instructions=build_remediation_prompt_block(remediation_context),
    )
    used_keys = used_facts.get("used_keys")
    used_keys = used_keys if isinstance(used_keys, list) else []
    validate_claim_ledger(ledger, store=store, used_keys=used_keys, artifact_manifest=manifest)
    save_claim_ledger(osp.join(latex_folder, "claim_ledger.json"), ledger)
    claim_trace_index = build_claim_trace_index(ledger=ledger, symbolic_tex=symbolic_tex)
    validate_used_fact_traceability(claim_trace_index, used_keys=used_keys)
    save_claim_trace_index(osp.join(latex_folder, "claim_trace_index.json"), claim_trace_index)
    if not env_bool("AI_SCIENTIST_OUTSIDER_AUDIT", True):
        return
    audit_model_name = env_str("AI_SCIENTIST_MODEL_OUTSIDER_AUDIT", small_model)
    audit_client, audit_model = create_client(audit_model_name)
    rendered_tex_path = osp.join(latex_folder, "template.rendered.tex")
    rendered_tex: str | None = None
    if osp.exists(rendered_tex_path):
        with open(rendered_tex_path, "r", encoding="utf-8") as f:
            rendered_tex = f.read()
    audit_inputs = build_outsider_audit_inputs(
        symbolic_tex=symbolic_tex,
        used_facts=used_facts,
        store=store,
        claim_ledger=ledger,
        artifact_manifest_summary=artifact_manifest_summary(manifest),
        rendered_tex=rendered_tex,
    )
    save_context_pack(
        osp.join(latex_folder, "audit_context_pack.json"),
        build_outsider_audit_context_pack(audit_inputs),
    )
    audit = generate_outsider_audit(
        inputs=audit_inputs,
        client=audit_client,
        model=audit_model,
    )
    validate_outsider_audit(audit)
    save_outsider_audit(osp.join(latex_folder, "outsider_audit.json"), audit)
    # Default to audit-as-evidence instead of audit-as-hard-block.
    # The outsider audit is itself LLM-generated, so treating every high-severity
    # contradiction as fatal is too brittle for the main writeup pipeline.
    if env_bool("AI_SCIENTIST_BLOCK_ON_OUTSIDER_AUDIT", False):
        raise_if_blocking_audit_issues(audit)


def perform_symbolic_postprocess_retry(
    *,
    base_folder: str,
    big_model: str,
    small_model: str,
    remediation_context: Mapping[str, Any] | None = None,
) -> bool:
    latex_folder, writeup_file, used_facts_path = _load_existing_symbolic_paths(base_folder)
    fact_store_path = osp.join(base_folder, "logs", "0-run", "fact_store.json")
    param_store_path = osp.join(base_folder, "logs", "0-run", "param_store.json")
    try:
        ensure_param_store_for_run(base_folder)
    except Exception:
        param_store_path = None
    pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}.pdf")
    try:
        _ensure_existing_symbolic_compile(
            base_folder=base_folder,
            latex_folder=latex_folder,
            fact_store_path=fact_store_path,
            param_store_path=param_store_path,
            used_facts_path=used_facts_path,
            pdf_file=pdf_file,
        )
        _run_claim_and_audit_pipeline(
            base_folder=base_folder,
            latex_folder=latex_folder,
            writeup_file=writeup_file,
            used_facts_path=used_facts_path,
            big_model=big_model,
            small_model=small_model,
            remediation_context=remediation_context,
        )
        return osp.exists(pdf_file)
    except Exception as exc:
        trace_text = traceback.format_exc()
        report = classify_remediation_failure(
            phase="perform_symbolic_postprocess_retry",
            exc=exc,
            traceback_text=trace_text,
        )
        save_remediation_report(
            osp.join(latex_folder, "remediation_failure.json"),
            report,
        )
        print("EXCEPTION in perform_symbolic_postprocess_retry:")
        print(trace_text)
        print_remediation_report(report)
        return False
