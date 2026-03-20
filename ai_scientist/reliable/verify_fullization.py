from __future__ import annotations

import json
import os
import tempfile
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai_scientist.reliable.artifact_manifest import (
    artifact_manifest_summary,
    ensure_artifact_manifest,
)
from ai_scientist.reliable.claim_ledger import validate_claim_ledger
from ai_scientist.reliable.claim_traceability import (
    build_claim_trace_index,
    validate_used_fact_traceability,
)
from ai_scientist.reliable.context_packs import (
    build_claim_context_pack,
    build_writeup_context_pack,
)
from ai_scientist.reliable.facts import FactRecord, FactStore
from ai_scientist.reliable.outsider_audit import (
    OutsiderAuditInputs,
    build_outsider_audit_context_pack,
    build_outsider_audit_inputs,
)
from ai_scientist.reliable.errors import UnknownFactKeyError
from ai_scientist.reliable.gates import require_includegraphics_paths_from_manifest
from ai_scientist.reliable.numeric_lint import find_unanchored_numeric_literals
from ai_scientist.reliable.remediation import (
    classify_remediation_failure,
    should_retry_writeup,
)
from ai_scientist.reliable.writeup_inputs import ensure_symbolic_writeup_inputs
from ai_scientist.reliable.writeup_summary import export_writeup_summary_symbolic


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as base:
        log_dir = os.path.join(base, "logs", "0-run")
        latex_dir = os.path.join(base, "latex")
        figures_dir = os.path.join(base, "figures")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(latex_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        store = FactStore()
        store.add(FactRecord("stage2.best.acc", "Best baseline accuracy", 0.91, "float:3", {"stage_name": "2_baseline", "node_id": "n1", "exp_results_dir": os.path.join(base, "experiment_results", "baseline")}))  # noqa: E501
        store.add(FactRecord("stage3.best.acc", "Best research accuracy", 0.93, "float:3", {"stage_name": "3_research", "node_id": "n2", "exp_results_dir": os.path.join(base, "experiment_results", "research")}))  # noqa: E501
        store.save_json(os.path.join(log_dir, "fact_store.json"))

        _write_json(os.path.join(log_dir, "draft_summary.json"), {"Experiment_description": "draft 12.3%"})
        _write_json(os.path.join(log_dir, "baseline_summary.json"), {"stage_name": "2_baseline", "Experiment_description": "baseline 91.2%"})  # noqa: E501
        _write_json(os.path.join(log_dir, "research_summary.json"), {"stage_name": "3_research", "Experiment_description": "research 93.1%"})  # noqa: E501
        _write_json(os.path.join(log_dir, "ablation_summary.json"), [])

        open(os.path.join(figures_dir, "main_plot.png"), "wb").close()
        tex_path = os.path.join(latex_dir, "template.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\\section{Experiments} \\includegraphics{figures/main_plot.png} \\begin{table}\\label{tab:main}\\end{table}")  # noqa: E501

        export_writeup_summary_symbolic(
            base_folder=base,
            log_dir=log_dir,
            draft_summary={"Experiment_description": "draft 12.3%"},
            baseline_summary={"stage_name": "2_baseline", "Experiment_description": "baseline 91.2%"},
            research_summary={"stage_name": "3_research", "Experiment_description": "research 93.1%"},
            ablation_summary=[],
        )
        ensured_store, writeup_summary = ensure_symbolic_writeup_inputs(base)
        manifest = ensure_artifact_manifest(base_folder=base, store=ensured_store, tex_path=tex_path)
        manifest_summary = artifact_manifest_summary(manifest)

        writeup_pack = build_writeup_context_pack(
            symbolic_facts=True,
            summaries_for_prompt={"WRITEUP_SYMBOLIC_SUMMARY": writeup_summary},
            facts_index="- stage2.best.acc: Best baseline accuracy",
            artifact_manifest_summary=manifest_summary,
            current_latex="\\section{Experiments}",
            plot_names=["main_plot.png"],
            plot_descriptions={"main_plot.png": "main result plot"},
        )
        claim_pack = build_claim_context_pack(
            symbolic_tex="\\section{Experiments} \\fact{stage2.best.acc}",
            used_facts={"used_keys": ["stage2.best.acc"], "facts": [{"key": "stage2.best.acc", "meaning": "Best baseline accuracy"}]},  # noqa: E501
            artifact_manifest_summary=manifest_summary,
        )
        symbolic_summary = writeup_pack.get("summaries", {}).get("WRITEUP_SYMBOLIC_SUMMARY", {})
        symbolic_baseline = symbolic_summary.get("baseline_summary", {})
        preferred_fact_refs = writeup_pack.get("preferred_fact_refs", [])
        ledger = {
            "schema": "auctrace.claim_ledger.v2",
            "claims": [
                {
                    "claim_id": "C001",
                    "claim_type": "numeric",
                    "claim_text": "Baseline improves on the benchmark.",
                    "claim_text_template": "Baseline reaches \\fact{stage2.best.acc} on the benchmark.",
                    "audit_status": "unknown",
                    "supporting_facts": ["stage2.best.acc"],
                    "supporting_artifacts": ["table.tab:main"],
                    "location_hints": {"section": "Experiments", "latex_anchor": "tab:main"},
                }
            ],
        }
        validate_claim_ledger(
            ledger,
            store=ensured_store,
            used_keys=["stage2.best.acc"],
            artifact_manifest=manifest,
        )
        trace_index = build_claim_trace_index(
            ledger=ledger,
            symbolic_tex="\\section{Experiments} \\fact{stage2.best.acc} \\begin{table}\\label{tab:main}\\end{table}",
        )
        validate_used_fact_traceability(
            trace_index,
            used_keys=["stage2.best.acc"],
        )

        audit_inputs = build_outsider_audit_inputs(
            symbolic_tex="\\section{Experiments} \\fact{stage2.best.acc}",
            used_facts={"used_keys": ["stage2.best.acc"], "facts": [{"key": "stage2.best.acc", "meaning": "Best baseline accuracy"}]},  # noqa: E501
            store=ensured_store,
            claim_ledger=ledger,
            artifact_manifest_summary=manifest_summary,
        )
        audit_pack = build_outsider_audit_context_pack(
            OutsiderAuditInputs(
                symbolic_tex_excerpt=audit_inputs.symbolic_tex_excerpt,
                used_fact_keys=audit_inputs.used_fact_keys,
                used_fact_table=audit_inputs.used_fact_table,
                claim_ledger=audit_inputs.claim_ledger,
                artifact_manifest_summary=audit_inputs.artifact_manifest_summary,
            )
        )
        remediation = classify_remediation_failure(
            phase="verify_fullization",
            exc=FileNotFoundError("missing artifact"),
            traceback_text="traceback",
        )
        unknown_fact_remediation = classify_remediation_failure(
            phase="verify_fullization",
            exc=UnknownFactKeyError("Unknown fact key: 'stage3.best.synthetic.wasted_budget_ratio.best'"),
            traceback_text="traceback",
        )
        safe_setup_tex = (
            "Ubuntu 22.04 with Adam defaults "
            "($\\beta_{1}=0.9$, $\\beta_{2}=0.999$, learning rate $10^{-3}$)."
        )
        risky_result_tex = "Validation accuracy reached 0.9123 on the held-out split."
        safe_findings = find_unanchored_numeric_literals(safe_setup_tex)
        risky_findings = find_unanchored_numeric_literals(risky_result_tex)
        require_includegraphics_paths_from_manifest(
            "\\includegraphics{main_plot.png}",
            manifest,
        )
        try:
            require_includegraphics_paths_from_manifest(
                "\\includegraphics{experiments/old-run/synthetic_loss_curves.png}",
                manifest,
            )
        except Exception as exc:
            figure_gate_remediation = classify_remediation_failure(
                phase="verify_fullization",
                exc=exc,
                traceback_text="traceback",
            )
        else:
            raise AssertionError("Expected invalid figure path gate to fail")

        print(
            json.dumps(
                {
                    "writeup_summary_schema": writeup_summary["schema"],
                    "manifest_schema": manifest["schema"],
                    "writeup_context_pack_schema": writeup_pack["schema"],
                    "claim_context_pack_schema": claim_pack["schema"],
                    "audit_context_pack_schema": audit_pack["schema"],
                    "claim_trace_index_schema": trace_index["schema"],
                    "remediation_failure_code": remediation.failure_code,
                    "unknown_fact_retryable": should_retry_writeup(unknown_fact_remediation),
                    "artifact_ids": [item["artifact_id"] for item in manifest["artifacts"]],
                    "manifest_has_includegraphics_targets": bool(
                        manifest_summary.get("includegraphics_targets")
                    ),
                    "symbolic_writer_has_plot_inventory": "plot_inventory" in writeup_pack,
                    "symbolic_summary_exposes_fact_refs": bool(
                        symbolic_baseline.get("fact_refs")
                    ),
                    "symbolic_writer_has_preferred_fact_refs": bool(preferred_fact_refs),
                    "numeric_lint_allows_setup_literals": not safe_findings,
                    "numeric_lint_blocks_result_literals": bool(risky_findings),
                    "invalid_figure_path_retryable": should_retry_writeup(
                        figure_gate_remediation
                    ),
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
